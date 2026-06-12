#!/usr/bin/env python3
"""EV/confinement hyperparameter sweep, scored by Hi-C correlation + physical sanity.

Implements the plan in ``docs/validation-sweep-plan.md``: for each (config, region) it runs an
ensemble through the public gnome3d API, scores it with the structure metrics
(``validation/metrics.py``) AND Hi-C correlation against a 4DN ``.mcool``
(``validation/contacts.py``), then picks the winning EV/confinement config by **constrained
max-SCC**: maximise median Hi-C SCC among configs whose overlaps don't exceed the no-feature
baseline and whose polymer scaling stays sane.

Results are cached per (config, region, budget) so the sweep is resumable and re-scoring is
free. Prefer ``--search`` (successive halving: cheap screen of all configs -> expand the top
survivors -> validate the winner at full x n=100), which is ~7x cheaper than the flat grid.

    python -m validation.sweep --cell GM12878 --search \\
        --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool --chrom chr1 --n-regions 20 \\
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
from validation import contacts, metrics
from validation.cell_config import settings_for_cell
from validation.validate import (
    _apply_flags,
    _chrs_and_region,
    load_contacts,
    run_ensemble,
    summarize,
)


def _cfg(
    name: str, ev: float | None = None, conf: float | None = None, ev_radius: float | None = None
) -> dict[str, object]:
    """A grid config as explicit flag overrides. ev=None -> EV off; conf=None -> confinement off
    (so 'baseline' definitively disables both, overriding the canonical on-at-0.1)."""
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


# EV/confinement search grid, aimed at MINIMISING overlaps (the lever that actually moves;
# validation showed Hi-C SCC is insensitive to these knobs). Axes that reduce overlaps: EV
# weight ↑, EV radius (auto_factor) ↑ (EV acts at larger separation). Confinement compacts
# (adds overlaps) but is searched in combos since EV can soak its overlaps up. The min-overlap
# objective's Rg/Hi-C guardrails keep over-strong EV from "winning" by inflating the structure.
# Big-N / multi-region / 3-cell search runs on CUDA (validation/RUNBOOK.md); --max-configs subsets.
GRID: list[dict[str, object]] = [
    _cfg("baseline"),
    # EV weight scan (default radius)
    _cfg("ev0.5", ev=0.5),
    _cfg("ev1.0", ev=1.0),
    _cfg("ev2.0", ev=2.0),
    _cfg("ev4.0", ev=4.0),
    # EV radius scan (auto_factor ↑ => EV active farther => fewer overlaps) at w=2.0
    _cfg("ev2.0_r0.7", ev=2.0, ev_radius=0.7),
    _cfg("ev2.0_r1.0", ev=2.0, ev_radius=1.0),
    # confinement-only (compacts; usually adds overlaps — kept for contrast)
    _cfg("conf_p0.75", conf=0.75),
    _cfg("conf_p0.5", conf=0.5),
    # EV + confinement combos (compaction + de-clash) — the user's target space
    _cfg("ev1.0+conf0.75", ev=1.0, conf=0.75),
    _cfg("ev2.0+conf0.75", ev=2.0, conf=0.75),
    _cfg("ev2.0+conf0.5", ev=2.0, conf=0.5),
    _cfg("ev4.0+conf0.5", ev=4.0, conf=0.5),
]


def enumerate_regions(
    breakpoints_path: str, chrom: str, n: int, lo_mb: float = 1.5, hi_mb: float = 3.0
) -> list[str]:
    """Consecutive-breakpoint segments on ``chrom`` within [lo_mb, hi_mb] Mb → 'chr:start-end'."""
    pts: list[int] = []
    with open(breakpoints_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[0] == chrom:
                pts.append(int(p[1]))
    pts.sort()
    out: list[str] = []
    for a, b in zip(pts, pts[1:], strict=False):
        span = (b - a) / 1e6
        if lo_mb <= span <= hi_mb:
            out.append(f"{chrom}:{a}-{b}")
        if len(out) >= n:
            break
    return out


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
    """Run one config's ensemble and score structure metrics + Hi-C correlation."""
    ens = run_ensemble(s, data, chrs_list, bed_region, n)  # type: ignore[arg-type]
    m = summarize(ens, contacts_list, radius, skip)
    # Hi-C is a population average → correlate the ensemble-summed simulated map (denoised).
    coords_list, mids_list = [], []
    for beads in ens:
        coords, mids = metrics.to_arrays(beads)
        coords_list.append(coords)
        mids_list.append(mids)
    r = contacts.ensemble_hic_correlation(coords_list, mids_list, hic_path, region, binsize, radius)
    m["hic_scc"] = r["scc"]
    m["hic_pearson"] = r["pearson"]
    m["hic_insulation"] = r["insulation"]
    return m


def _key(name: str, region: str, tag: str | None) -> str:
    return f"{name}::{region}::{tag}" if tag else f"{name}::{region}"


def run_budget(
    base: Settings,
    configs: list[dict[str, object]],
    regions: list[str],
    n: int,
    tag: str | None,
    args: argparse.Namespace,
    cache: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Run every (config, region) at one budget (``base`` already built for the budget's
    quality; ``n`` its ensemble size). Results cached under tag-scoped keys so different
    budgets don't collide and the whole thing is resumable."""
    for region in regions:
        chrs_list, bed_region = _chrs_and_region(region)
        data = ContactData.from_files(base, chrs_list, bed_region)
        clist = load_contacts(base, chrs_list, bed_region)
        radius_key = f"__radius__::{region}"  # budget-independent (bead spacing ~ constant)
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
                args.hic,
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
    """Median of each metric per config across the regions that completed (at this budget)."""
    agg: dict[str, dict[str, float]] = {}
    for name in names:
        rows = [cache[_key(name, r, tag)] for r in regions if _key(name, r, tag) in cache]
        if not rows:
            continue
        med = lambda k, rows=rows: float(np.nanmedian([row[k] for row in rows]))
        agg[name] = {
            "scc": med("hic_scc"),
            "pearson": med("hic_pearson"),
            "overlap": med("overlap_frac"),
            "rg": med("rg"),
            "dscale": med("dist_scaling_exp"),
            "cprob": med("contact_prob_exp"),
            "diversity": med("diversity_dab"),
        }
    return agg


def _passes(a: dict[str, float], base: dict[str, float], objective: str, rg_tol: float) -> bool:
    """Guardrails a config must clear to be eligible. Always: sane polymer scaling + non-collapsed
    diversity. For the **overlap** objective, additionally: the structure must not blow up
    (Rg ≤ baseline × (1+rg_tol)) — else EV could 'win' by simply expanding the chain to de-clash.
    Hi-C SCC is deliberately NOT gated (validation showed it's noise w.r.t. these knobs). For the
    **scc** objective: overlaps ≤ baseline."""
    if not (0.0 <= a["dscale"] <= 1.0 and -2.5 <= a["cprob"] <= -0.05 and a["diversity"] > 1e-6):
        return False
    if objective == "overlap":
        return not (
            np.isfinite(base["rg"]) and np.isfinite(a["rg"]) and a["rg"] > base["rg"] * (1 + rg_tol)
        )
    return a["overlap"] <= base["overlap"] + 1e-9  # scc objective: overlaps gated, scc maximised


def _obj_value(a: dict[str, float], objective: str) -> float:
    """Sort key (smaller = better): overlap directly; scc as its negation."""
    return a["overlap"] if objective == "overlap" else -a["scc"]


def select(
    agg: dict[str, dict[str, float]], objective: str, rg_tol: float
) -> tuple[list[str], str | None]:
    """Non-baseline configs passing the guardrails, ordered best-first by the objective; winner."""
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
    """Print the per-config table; return the winner by ``objective`` under the guardrails."""
    base = agg.get("baseline")
    print(f"\n{'=' * 84}\n  {title}   [objective: {objective}]\n{'=' * 84}")
    print(
        f"  {'config':<16}{'overlap':>9}{'HiC SCC':>9}{'Pearson':>9}{'Rg':>8}"
        f"{'dscale':>8}{'divers':>8}  ok?"
    )
    for name in names:
        if name not in agg:
            continue
        a = agg[name]
        ok = name == "baseline" or (base is not None and _passes(a, base, objective, rg_tol))
        print(
            f"  {name:<16}{a['overlap']:>9.4f}{a['scc']:>9.3f}{a['pearson']:>9.3f}{a['rg']:>8.2f}"
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


def main() -> None:
    p = argparse.ArgumentParser(description="EV/confinement sweep scored by Hi-C correlation")
    p.add_argument("--cell", required=True, help="cell line — settings wired from canonical params")
    p.add_argument("--data-root", default="data", help="root holding <cell>/ data (default: data)")
    p.add_argument(
        "--quality",
        default="full",
        choices=["fast", "balanced", "full"],
        help="flat-mode MC schedule (ignored in --search)",
    )
    p.add_argument("--hic", required=True, help="4DN .mcool path (observed Hi-C)")
    p.add_argument("--chrom", default="chr1")
    p.add_argument("--n-regions", type=int, default=20)
    p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=100,
        help="flat-mode ensemble size (search uses --search-n / --final-n)",
    )
    p.add_argument("--max-configs", type=int, default=None, help="flat mode: first N grid configs")
    p.add_argument("--skip-neighbors", type=int, default=1)
    p.add_argument("--out", default="out/sweep/sweep.json", help="resumable results cache (JSON)")
    # --- successive-halving search (cheap screen -> expand top-K -> validate winner full x100) ---
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
        help="overlap = minimise overlaps s.t. Rg not inflating (default; Hi-C NOT gated — it's "
        "noise w.r.t. these knobs); scc = maximise Hi-C SCC s.t. overlaps <= baseline",
    )
    p.add_argument(
        "--rg-tol",
        type=float,
        default=0.10,
        help="overlap objective: max allowed Rg inflation vs baseline (blocks 'expand to de-clash')",
    )
    args = p.parse_args()

    grid = GRID[: args.max_configs] if args.max_configs else GRID
    names = [str(c["_name"]) for c in grid]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, float]] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text())
        print(f"[sweep] resuming: {len(cache)} cached results")

    # region enumeration uses the breakpoints file (schedule-independent)
    s_meta = settings_for_cell(args.cell, args.data_root)
    bp = s_meta.data_path(s_meta.data_segment_split)
    regions = enumerate_regions(bp, args.chrom, args.n_regions)
    if not regions:
        sys.exit(f"[error] no regions found on {args.chrom} in size band")

    if not args.search:
        base = settings_for_cell(args.cell, args.data_root, args.quality)
        print(f"[sweep] flat: {len(regions)} regions x {len(grid)} configs x n={args.n_structures}")
        run_budget(base, grid, regions, args.n_structures, None, args, cache, out_path)
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
    base_search = settings_for_cell(args.cell, args.data_root, args.search_quality)
    print(f"[search] tier1: {len(grid)} configs x {len(screen)} regions @ {search_tag}")
    run_budget(base_search, grid, screen, args.search_n, search_tag, args, cache, out_path)
    agg1 = _aggregate(cache, names, screen, search_tag)
    report(
        agg1,
        names,
        f"TIER 1 — screen ({len(screen)} regions, {search_tag})",
        args.objective,
        args.rg_tol,
    )

    ranked1, _ = select(agg1, args.objective, args.rg_tol)
    survivors = ["baseline"] + ranked1[: args.keep]
    survivor_cfgs = [c for c in grid if str(c["_name"]) in survivors]
    print(f"\n[search] survivors -> tier2: {survivors}")

    print(f"[search] tier2: {len(survivor_cfgs)} configs x {len(regions)} regions @ {search_tag}")
    run_budget(
        base_search, survivor_cfgs, regions, args.search_n, search_tag, args, cache, out_path
    )
    agg2 = _aggregate(cache, survivors, regions, search_tag)
    winner = report(
        agg2,
        survivors,
        f"TIER 2 — expand ({len(regions)} regions, {search_tag})",
        args.objective,
        args.rg_tol,
    )
    if not winner:
        sys.exit("[search] no config cleared the guardrails; nothing to validate")

    final_cfgs = [c for c in grid if str(c["_name"]) in (winner, "baseline")]
    base_final = settings_for_cell(args.cell, args.data_root, args.final_quality)
    print(
        f"\n[search] tier3: validate '{winner}' vs baseline x {len(regions)} regions @ {final_tag}"
    )
    run_budget(base_final, final_cfgs, regions, args.final_n, final_tag, args, cache, out_path)
    aggf = _aggregate(cache, [winner, "baseline"], regions, final_tag)
    report(
        aggf,
        [winner, "baseline"],
        f"TIER 3 — VALIDATE winner ({len(regions)} regions, {final_tag})",
        args.objective,
        args.rg_tol,
    )
    print(f"\n[search] FINAL WINNER for {args.cell}: {winner}")


if __name__ == "__main__":
    main()
