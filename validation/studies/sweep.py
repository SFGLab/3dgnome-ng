"""EV/confinement hyperparameter sweep, scored by Hi-C correlation and physical sanity.

Implements the plan in docs/validation-sweep-plan.md. For each config and region it runs an
ensemble through the public gnome3d API, scores it with the structure metrics from
validation/metrics/structure.py and Hi-C correlation against a 4DN .mcool from validation/metrics/hic.py, then
picks the winning EV/confinement config. The default objective is constrained min-overlap. It
minimises median overlap among configs whose radius of gyration does not inflate past baseline and
whose polymer scaling stays sane. The scc objective instead maximises median Hi-C SCC among
configs whose overlaps do not exceed the no-feature baseline.

Results are cached per config, region and budget so the sweep is resumable and re-scoring is free.
Prefer --search, a successive-halving mode that cheaply screens all configs, expands the top
survivors, then validates the winner at full quality with n=100. It is about 7x cheaper than the
flat grid.

    python -m validation sweep --cell GM12878 --search \\
        --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool --chroms chr1 --n-regions 20 \\
        --binsize 10000 --out out/sweep/GM12878_search.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gnome3d.data import ContactData
from gnome3d.settings import Settings
from validation import metrics
from validation.core.config import apply_flags as _apply_flags
from validation.core.config import settings_for_cell
from validation.core.data import load_contacts
from validation.core.ensemble import run_ensemble, summarize
from validation.core.regions import enumerate_regions
from validation.core.regions import parse_region_arg as _chrs_and_region
from validation.metrics import hic as contacts
from validation.studies import Context, Study, register


def _cfg(
    name: str, ev: float | None = None, conf: float | None = None, ev_radius: float | None = None
) -> dict[str, object]:
    """A grid config as explicit flag overrides. Passing ev=None turns EV off and conf=None turns
    confinement off, so baseline definitively disables both and overrides the canonical on-at-0.1
    default."""
    d: dict[str, object] = {
        "_name": name,
        "use_excluded_volume": ev is not None,
        "exclusion_apply_to_smooth": True,
        "use_confinement": conf is not None,
        "confinement_apply_to_smooth": True,
    }
    if ev is not None:
        d["exclusion_weight"] = ev
    if ev_radius is not None:
        d["exclusion_auto_factor_smooth"] = ev_radius
    if conf is not None:
        d["confinement_packing_factor_smooth"] = conf
    return d


# EV/confinement search grid centered on the subordinate range. EV must stay below the distance
# restraint at dist_weight = 1.0 so it acts as a gentle correction rather than a dominant term that
# over-expands or distorts. The original 3dgnome used about 0.05. The grid spans the subordinate
# range 0.05, 0.1, 0.2, 0.3, 0.5, all at or below 0.5 and under 1.0. The min-overlap objective's Rg
# guard via --rg-tol rejects any config that reduces overlaps by over-expanding. Overlap is scored
# resolution-normalized as overlap_frac_norm so it is not confounded by bead density. Big-N,
# multi-region or 3-cell search runs on CUDA per the RUNBOOK, and --max-configs subsets the grid.
GRID: list[dict[str, object]] = [
    _cfg("baseline"),
    _cfg("ev0.05", ev=0.05),  # original 3dgnome magnitude
    _cfg("ev0.1", ev=0.1),
    _cfg("ev0.2", ev=0.2),
    _cfg("ev0.3", ev=0.3),
    _cfg("ev0.5", ev=0.5),  # top of the subordinate range
    # EV-radius bump gives more de-clash reach at fixed weight, watched by the Rg guard
    _cfg("ev0.3_r0.7", ev=0.3, ev_radius=0.7),
    # confinement-only compacts and is here for contrast, plus gentle EV and confinement combos
    _cfg("conf_p0.75", conf=0.75),
    _cfg("ev0.2+conf0.75", ev=0.2, conf=0.75),
    _cfg("ev0.5+conf0.75", ev=0.5, conf=0.75),
]


def score_config(
    s: Settings,
    data: ContactData,
    chrs_list: list[str],
    bed_region: object,
    contacts_list: list[tuple[int, int, float]],
    n: int,
    radius: float,
    skip: int,
    hic_path: str,
    region: str,
    binsize: int,
) -> dict[str, float]:
    """Run one config's ensemble and score structure metrics and Hi-C correlation."""
    ens = run_ensemble(s, data, chrs_list, bed_region, n)  # type: ignore[arg-type]
    m = summarize(ens, contacts_list, radius, skip)
    # Hi-C is a population average so correlate the ensemble-summed simulated map, which is denoised.
    coords_list, mids_list = [], []
    for beads in ens:
        coords, mids = metrics.to_arrays(beads)
        coords_list.append(coords)
        mids_list.append(mids)
    r = contacts.ensemble_hic_correlation(coords_list, mids_list, hic_path, region, binsize, radius)
    m["hic_scc"] = r["scc"]
    m["hic_pearson"] = r["pearson"]  # decay-stripped via off-diagonal log1p
    m["hic_insulation"] = r["insulation"]
    # MultiMM's metric approach, a faithful (d+1)^-3 simulated map against the ICE-balanced observed,
    # consistent with compare_reference and hic_tune.
    try:
        cobs_bal, bstarts = contacts.observed_hic(hic_path, region, binsize, balance=True)
    except Exception:  # noqa: BLE001  mcool may lack balance weights
        cobs_bal, bstarts = contacts.observed_hic(hic_path, region, binsize)
    eff = int(bstarts[1] - bstarts[0]) if len(bstarts) > 1 else binsize
    m["hic_multimm"] = contacts.multimm_faithful_pearson(
        coords_list, mids_list[0], cobs_bal, bstarts, eff
    )
    return m


def _key(name: str, region: str, tag: str | None) -> str:
    return f"{name}::{region}::{tag}" if tag else f"{name}::{region}"


def run_budget(
    base: Settings,
    configs: list[dict[str, object]],
    regions: list[str],
    n: int,
    tag: str | None,
    ctx: Context,
    args: argparse.Namespace,
    cache: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Run every config and region at one budget. base is already built for the budget's quality
    and n is its ensemble size. Results are cached under tag-scoped keys so different budgets do not
    collide and the run is resumable."""
    for region in regions:
        chrs_list, bed_region = _chrs_and_region(region)
        data = ContactData.from_files(base, chrs_list, bed_region)
        clist = load_contacts(base, chrs_list, bed_region)
        radius_key = (
            f"__radius__::{region}"  # budget-independent since bead spacing is roughly constant
        )
        if radius_key in cache:
            radius = cache[radius_key]["radius"]
        else:
            ens0 = run_ensemble(base, data, chrs_list, bed_region, 1)
            radius = float(np.median(metrics.bond_lengths(metrics.to_arrays(ens0[0])[0])))
            cache[radius_key] = {"radius": radius}
            out_path.write_text(json.dumps(cache, indent=2))
        todo = [c for c in configs if _key(str(c["_name"]), region, tag) not in cache]
        if not todo:
            print(f"[sweep] region {region} ({tag or 'flat'}): all cached")
            continue
        print(
            f"\n[sweep] region {region} ({tag or 'flat'}) radius={radius:.2f} contacts={len(clist)}"
        )
        for cfg in todo:
            name = str(cfg["_name"])
            flags = {k: v for k, v in cfg.items() if k != "_name"}
            s = _apply_flags(base, flags)  # type: ignore[arg-type]
            m = score_config(
                s,
                data,
                chrs_list,
                bed_region,
                clist,
                n,
                radius,
                args.skip_neighbors,
                ctx.hic,  # type: ignore[arg-type]
                region,
                args.binsize,
            )
            cache[_key(name, region, tag)] = m
            out_path.write_text(json.dumps(cache, indent=2))
            print(
                f"  [done] {name:<16} SCC={m['hic_scc']:+.3f} pear={m['hic_pearson']:+.3f} "
                f"overlap={m['overlap_frac']:.4f} dscale={m['dist_scaling_exp']:+.3f}"
            )


def _aggregate(
    cache: dict[str, dict[str, float]], names: list[str], regions: list[str], tag: str | None
) -> dict[str, dict[str, float]]:
    """Median of each metric per config across the regions that completed at this budget."""
    agg: dict[str, dict[str, float]] = {}
    for name in names:
        rows = [cache[_key(name, r, tag)] for r in regions if _key(name, r, tag) in cache]
        if not rows:
            continue
        med = lambda k, rows=rows: float(np.nanmedian([row[k] for row in rows]))
        medg = lambda k, rows=rows: float(  # tolerant of older caches missing the key
            np.nanmedian([row.get(k, float("nan")) for row in rows])
        )
        agg[name] = {
            "scc": med("hic_scc"),
            "pearson": med("hic_pearson"),
            "multimm": medg("hic_multimm"),
            "overlap": medg("overlap_frac_norm"),  # resolution-normalized, this is the objective
            "overlap_raw": med("overlap_frac"),
            "rg": med("rg"),
            "dscale": med("dist_scaling_exp"),
            "cprob": med("contact_prob_exp"),
            "diversity": med("diversity_dab"),
        }
    return agg


def _passes(a: dict[str, float], base: dict[str, float], objective: str, rg_tol: float) -> bool:
    """Guardrails a config must clear to be eligible. Every objective requires sane polymer scaling
    and non-collapsed diversity. The overlap objective additionally requires that the structure does
    not blow up, so Rg must stay at or below baseline times (1 + rg_tol). Otherwise EV could win by
    simply expanding the chain to de-clash. Hi-C SCC is deliberately not gated, since validation
    showed it is noise with respect to these knobs. The scc objective additionally requires overlaps
    at or below baseline."""
    if not (0.0 <= a["dscale"] <= 1.0 and -2.5 <= a["cprob"] <= -0.05 and a["diversity"] > 1e-6):
        return False
    if objective == "overlap":
        return not (
            np.isfinite(base["rg"]) and np.isfinite(a["rg"]) and a["rg"] > base["rg"] * (1 + rg_tol)
        )
    return a["overlap"] <= base["overlap"] + 1e-9  # scc objective gates overlaps and maximises scc


def _obj_value(a: dict[str, float], objective: str) -> float:
    """Sort key where smaller is better. The overlap objective uses overlap directly and the scc
    objective uses the negated scc."""
    return a["overlap"] if objective == "overlap" else -a["scc"]


def select(
    agg: dict[str, dict[str, float]], objective: str, rg_tol: float
) -> tuple[list[str], str | None]:
    """Return the non-baseline configs passing the guardrails ordered best-first by the objective,
    along with the winner."""
    base = agg.get("baseline", {"scc": float("nan"), "rg": float("nan"), "overlap": float("inf")})
    passers = [n for n, a in agg.items() if n != "baseline" and _passes(a, base, objective, rg_tol)]
    passers.sort(key=lambda n: _obj_value(agg[n], objective))
    return passers, (passers[0] if passers else None)


def report(
    agg: dict[str, dict[str, float]],
    names: list[str],
    title: str,
    objective: str,
    rg_tol: float,
) -> str | None:
    """Print the per-config table and return the winner by the objective under the guardrails."""
    base = agg.get("baseline")
    print(f"\n{'=' * 84}\n  {title}   [objective: {objective}]\n{'=' * 84}")
    print(
        f"  {'config':<16}{'overlap':>9}{'HiC SCC':>9}{'Pearson':>9}{'MultiMM':>9}{'Rg':>8}"
        f"{'dscale':>8}{'divers':>8}  ok?"
    )
    for name in names:
        if name not in agg:
            continue
        a = agg[name]
        ok = name == "baseline" or (base is not None and _passes(a, base, objective, rg_tol))
        print(
            f"  {name:<16}{a['overlap']:>9.4f}{a['scc']:>9.3f}{a['pearson']:>9.3f}"
            f"{a.get('multimm', float('nan')):>9.3f}{a['rg']:>8.2f}"
            f"{a['dscale']:>8.3f}{a['diversity']:>8.3f}   {'✓' if ok else '·'}"
        )
    _, winner = select(agg, objective, rg_tol)
    if base is not None:
        print(
            f"\n  baseline: overlap={base['overlap']:.4f}  SCC={base['scc']:.3f}  Rg={base['rg']:.2f}"
        )
    if winner:
        w = agg[winner]
        print(
            f"  winner ({'min-overlap' if objective == 'overlap' else 'max-SCC'}): {winner}  "
            f"overlap={w['overlap']:.4f}  SCC={w['scc']:.3f}  Rg={w['rg']:.2f}"
            + (f"  (baseline overlap {base['overlap']:.4f})" if base else "")
        )
    else:
        print("  no feature config cleared the guardrails")
    return winner


class Sweep(Study):
    name = "sweep"
    help = "excluded-volume and confinement hyperparameter search scored by Hi-C correlation"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--chroms",
            default=None,
            help="comma-separated chromosomes (default: all in the breakpoints file)",
        )
        p.add_argument("--n-regions", type=int, default=20)
        p.add_argument("--min-ibs", type=int, default=2, help="min segments (≈IBs) a region spans")
        p.add_argument("--max-ibs", type=int, default=6, help="max segments (≈IBs) a region spans")
        p.add_argument("--max-mb", type=float, default=6.0, help="cap region size (Mb)")
        p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
        p.add_argument(
            "--max-configs", type=int, default=None, help="flat mode: first N grid configs"
        )
        p.add_argument("--skip-neighbors", type=int, default=1)
        p.add_argument(
            "--out", default="out/sweep/sweep.json", help="resumable results cache (JSON)"
        )
        # successive-halving search that screens all configs cheap, expands top-K, then validates the
        # winner at full quality with n=100
        p.add_argument(
            "--search",
            action="store_true",
            help="3-tier search: screen all configs cheap, expand survivors, validate winner",
        )
        p.add_argument("--screen-regions", type=int, default=5, help="tier-1 region subset")
        p.add_argument("--search-n", type=int, default=30, help="tier-1/2 ensemble size")
        p.add_argument(
            "--search-quality",
            default="balanced",
            choices=["fast", "balanced", "full"],
            help="tier-1/2 MC schedule",
        )
        p.add_argument("--keep", type=int, default=4, help="configs surviving tier 1")
        p.add_argument("--final-n", type=int, default=100, help="tier-3 winner ensemble size")
        p.add_argument(
            "--final-quality",
            default="full",
            choices=["fast", "balanced", "full"],
            help="tier-3 winner MC schedule",
        )
        # --- objective ---
        p.add_argument(
            "--objective",
            default="overlap",
            choices=["overlap", "scc"],
            help="overlap minimises overlaps subject to Rg not inflating. This is the default. Hi-C is not gated, it is "
            "noise with respect to these knobs. scc maximises Hi-C SCC subject to overlaps at or below baseline",
        )
        p.add_argument(
            "--rg-tol",
            type=float,
            default=0.30,
            help="overlap objective, max allowed Rg inflation vs baseline. 0.30 is moderate "
            "de-compaction. The reference is over-compact so some EV-driven expansion is legitimate",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        if ctx.hic is None:
            sys.exit("[error] sweep requires --hic, an observed Hi-C .mcool")

        grid = GRID[: args.max_configs] if args.max_configs else GRID
        names = [str(c["_name"]) for c in grid]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache: dict[str, dict[str, float]] = {}
        if out_path.exists():
            cache = json.loads(out_path.read_text())
            print(f"[sweep] resuming: {len(cache)} cached results")

        # region enumeration uses the breakpoints file, which is schedule-independent
        s_meta = settings_for_cell(ctx.cell, ctx.data_root)
        bp = s_meta.data_path(s_meta.data_segment_split)
        chroms = args.chroms.split(",") if args.chroms else None
        regions = enumerate_regions(
            bp,
            args.n_regions,
            chroms=chroms,
            min_ibs=args.min_ibs,
            max_ibs=args.max_ibs,
            max_mb=args.max_mb,
        )
        if not regions:
            sys.exit(f"[error] no regions found on {args.chroms} in size band")

        if not args.search:
            base = settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
            print(f"[sweep] flat: {len(regions)} regions x {len(grid)} configs x n={ctx.n}")
            run_budget(base, grid, regions, ctx.n, None, ctx, args, cache, out_path)
            report(
                _aggregate(cache, names, regions, None),
                names,
                f"SWEEP RESULTS (median over {len(regions)} regions)",
                args.objective,
                args.rg_tol,
            )
            return

        # --- search mode ---
        screen = regions[: args.screen_regions]
        search_tag = f"n{args.search_n}_{args.search_quality}"
        final_tag = f"n{args.final_n}_{args.final_quality}"
        base_search = settings_for_cell(ctx.cell, ctx.data_root, args.search_quality)
        print(f"[search] tier1: {len(grid)} configs x {len(screen)} regions @ {search_tag}")
        run_budget(base_search, grid, screen, args.search_n, search_tag, ctx, args, cache, out_path)
        agg1 = _aggregate(cache, names, screen, search_tag)
        report(
            agg1,
            names,
            f"tier 1, screen ({len(screen)} regions, {search_tag})",
            args.objective,
            args.rg_tol,
        )

        ranked1, _ = select(agg1, args.objective, args.rg_tol)
        survivors = ["baseline"] + ranked1[: args.keep]
        survivor_cfgs = [c for c in grid if str(c["_name"]) in survivors]
        print(f"\n[search] survivors -> tier2: {survivors}")

        print(
            f"[search] tier2: {len(survivor_cfgs)} configs x {len(regions)} regions @ {search_tag}"
        )
        run_budget(
            base_search,
            survivor_cfgs,
            regions,
            args.search_n,
            search_tag,
            ctx,
            args,
            cache,
            out_path,
        )
        agg2 = _aggregate(cache, survivors, regions, search_tag)
        winner = report(
            agg2,
            survivors,
            f"tier 2, expand ({len(regions)} regions, {search_tag})",
            args.objective,
            args.rg_tol,
        )
        if not winner:
            sys.exit("[search] no config cleared the guardrails; nothing to validate")

        final_cfgs = [c for c in grid if str(c["_name"]) in (winner, "baseline")]
        base_final = settings_for_cell(ctx.cell, ctx.data_root, args.final_quality)
        print(
            f"\n[search] tier3: validate '{winner}' vs baseline x {len(regions)} regions @ {final_tag}"
        )
        run_budget(
            base_final, final_cfgs, regions, args.final_n, final_tag, ctx, args, cache, out_path
        )
        aggf = _aggregate(cache, [winner, "baseline"], regions, final_tag)
        report(
            aggf,
            [winner, "baseline"],
            f"tier 3, validate winner ({len(regions)} regions, {final_tag})",
            args.objective,
            args.rg_tol,
        )
        print(f"\n[search] FINAL WINNER for {ctx.cell}: {winner}")


register(Sweep())
