#!/usr/bin/env python3
"""EV/confinement hyperparameter sweep, scored by Hi-C correlation + physical sanity.

Implements the plan in ``docs/validation-sweep-plan.md``: for each (config, region) it runs an
ensemble through the public gnome3d API, scores it with the structure metrics
(``validation/metrics.py``) AND Hi-C correlation against a 4DN ``.mcool``
(``validation/contacts.py``), then picks the winning EV/confinement config by **constrained
max-SCC**: maximise median Hi-C SCC among configs whose overlaps don't exceed the no-feature
baseline and whose polymer scaling stays sane.

Results are cached per (cell_line, region, config) so the sweep is resumable and re-scoring is
free — extend the same cache on a CUDA box for the full run.

    python -m validation.sweep --config data/GM12878/config_dryrun.ini --data-dir data/GM12878 \\
        --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool --chrom chr1 --n-regions 4 \\
        --binsize 25000 -n 3 --out out/sweep/GM12878.json
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


# EV/confinement search grid. Spans the REAL default weight (0.1) up through 2.0 — the lean pass
# suggested EV correlation peaks well below 2.0. Confinement sampled at engaging packing factors
# (1.5 was inactive). Plus combos and two EV-radius probes. Big-N / multi-region / 3-cell-line
# search runs on CUDA (see validation/RUNBOOK.md); --max-configs subsets for a quick local pass.
GRID: list[dict[str, object]] = [
    _cfg("baseline"),
    _cfg("ev0.1", ev=0.1),
    _cfg("ev0.25", ev=0.25),
    _cfg("ev0.5", ev=0.5),
    _cfg("ev1.0", ev=1.0),
    _cfg("ev2.0", ev=2.0),
    _cfg("conf_p1.0", conf=1.0),
    _cfg("conf_p0.75", conf=0.75),
    _cfg("conf_p0.5", conf=0.5),
    _cfg("ev0.1+conf0.75", ev=0.1, conf=0.75),
    _cfg("ev0.5+conf1.0", ev=0.5, conf=1.0),
    _cfg("ev1.0+conf1.0", ev=1.0, conf=1.0),
    _cfg("ev1.0+conf0.75", ev=1.0, conf=0.75),
    _cfg("ev1.0_r0.3", ev=1.0, ev_radius=0.3),
    _cfg("ev1.0_r0.7", ev=1.0, ev_radius=0.7),
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


def main() -> None:
    p = argparse.ArgumentParser(description="EV/confinement sweep scored by Hi-C correlation")
    p.add_argument(
        "--cell",
        required=True,
        help="cell line (e.g. GM12878) — settings wired from canonical params",
    )
    p.add_argument("--data-root", default="data", help="root holding <cell>/ data (default: data)")
    p.add_argument(
        "--quality",
        default="full",
        choices=["fast", "balanced", "full"],
        help="MC schedule: fast/balanced for quick local checks, full for real runs",
    )
    p.add_argument("--hic", required=True, help="4DN .mcool path (observed Hi-C)")
    p.add_argument("--chrom", default="chr1")
    p.add_argument("--n-regions", type=int, default=4)
    p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=100,
        help="ensemble size — 3dgnome ensembles need >=100 (2016 paper); use small n only for "
        "quick pipeline checks, not real verdicts",
    )
    p.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="use only the first N grid configs (lean pass)",
    )
    p.add_argument("--skip-neighbors", type=int, default=1)
    p.add_argument("--out", default="out/sweep/sweep.json", help="resumable results cache (JSON)")
    args = p.parse_args()

    base = settings_for_cell(args.cell, args.data_root, args.quality)

    bp = base.data_path(base.data_segment_split)
    regions = enumerate_regions(bp, args.chrom, args.n_regions)
    if not regions:
        sys.exit(f"[error] no regions found on {args.chrom} in size band")
    grid = GRID[: args.max_configs] if args.max_configs else GRID
    print(f"[sweep] {len(regions)} regions x {len(grid)} configs x n={args.n_structures}")
    print(f"[sweep] regions: {regions}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, float]] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text())
        print(f"[sweep] resuming: {len(cache)} cached (config,region) results")

    for region in regions:
        chrs_list, bed_region = _chrs_and_region(region)
        data = ContactData.from_files(base, chrs_list, bed_region)
        clist = load_contacts(base, chrs_list, bed_region)
        # radius fixed per region from a defaults run (fair across configs)
        radius_key = f"__radius__::{region}"
        if radius_key in cache:
            radius = cache[radius_key]["radius"]
        else:
            ens0 = run_ensemble(base, data, chrs_list, bed_region, 1)
            coords0, _ = metrics.to_arrays(ens0[0])
            radius = float(np.median(metrics.bond_lengths(coords0)))
            cache[radius_key] = {"radius": radius}
            out_path.write_text(json.dumps(cache, indent=2))
        print(f"\n[sweep] region {region}  radius={radius:.2f}  contacts={len(clist)}")

        for cfg in grid:
            name = str(cfg["_name"])
            key = f"{name}::{region}"
            if key in cache:
                print(f"  [cached] {name}")
                continue
            flags = {k: v for k, v in cfg.items() if k != "_name"}
            s = _apply_flags(base, flags)  # type: ignore[arg-type]
            m = score_config(
                s,
                data,
                chrs_list,
                bed_region,
                clist,
                args.n_structures,
                radius,
                args.skip_neighbors,
                args.hic,
                region,
                args.binsize,
            )
            cache[key] = m
            out_path.write_text(json.dumps(cache, indent=2))
            print(
                f"  [done] {name:<16} SCC={m['hic_scc']:+.3f} pear={m['hic_pearson']:+.3f} "
                f"overlap={m['overlap_frac']:.4f} dscale={m['dist_scaling_exp']:+.3f}"
            )

    report(cache, regions, grid)


def report(
    cache: dict[str, dict[str, float]], regions: list[str], grid: list[dict[str, object]]
) -> None:
    """Aggregate per config across regions; pick winner by constrained max-SCC."""
    names = [str(c["_name"]) for c in grid]
    agg: dict[str, dict[str, float]] = {}
    for name in names:
        rows = [cache[f"{name}::{r}"] for r in regions if f"{name}::{r}" in cache]
        if not rows:
            continue
        med = lambda k, rows=rows: float(np.nanmedian([row[k] for row in rows]))
        agg[name] = {
            "scc": med("hic_scc"),
            "pearson": med("hic_pearson"),
            "overlap": med("overlap_frac"),
            "dscale": med("dist_scaling_exp"),
            "cprob": med("contact_prob_exp"),
            "diversity": med("diversity_dab"),
        }
    base_overlap = agg.get("baseline", {}).get("overlap", float("inf"))
    print(f"\n{'=' * 78}\n  SWEEP RESULTS (median over {len(regions)} regions)\n{'=' * 78}")
    print(
        f"  {'config':<16}{'HiC SCC':>9}{'Pearson':>9}{'overlap':>9}{'dscale':>8}"
        f"{'cprob':>8}{'divers':>8}  ok?"
    )
    winner, best_scc = None, -2.0
    for name in names:
        if name not in agg:
            continue
        a = agg[name]
        ok = (
            a["overlap"] <= base_overlap + 1e-9
            and 0.0 <= a["dscale"] <= 1.0
            and -2.5 <= a["cprob"] <= -0.05
            and a["diversity"] > 1e-6
        )
        flag = "✓" if ok else "·"
        print(
            f"  {name:<16}{a['scc']:>9.3f}{a['pearson']:>9.3f}{a['overlap']:>9.4f}"
            f"{a['dscale']:>8.3f}{a['cprob']:>8.3f}{a['diversity']:>8.3f}   {flag}"
        )
        if ok and np.isfinite(a["scc"]) and a["scc"] > best_scc and name != "baseline":
            winner, best_scc = name, a["scc"]
    print(f"\n  baseline SCC = {agg.get('baseline', {}).get('scc', float('nan')):.3f}")
    if winner:
        print(
            f"  WINNER (constrained max-SCC): {winner}  SCC={best_scc:.3f} "
            f"(vs baseline {agg['baseline']['scc']:+.3f})"
        )
    else:
        print("  no feature config satisfied the constraints with SCC > baseline")


if __name__ == "__main__":
    main()
