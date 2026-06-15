#!/usr/bin/env python3
"""Fast Hi-C correlation tuning — reach the MultiMM metric without re-running MC per config.

The EV/confinement sweep is slow because it runs a fresh ensemble for every config. But Hi-C
correlation is a **readout on fixed coordinates**: SCC depends only on the contact radius, and the
MultiMM inverse-distance Pearson depends only on ``sim_power``/binsize — none of them need new MC.

So this tool generates **one tuned ensemble per region** (the only expensive step) and then sweeps
the readout parameters for free on the saved coords:

  * ``contact_radius`` (× the median bond length) -> hicrep **SCC** + decay-stripped Pearson;
  * the ensemble inverse-distance map -> **MultiMM Pearson** (decay retained, sim_power 3/2),
    the number directly comparable to MultiMM's ≈0.70 (random <0.40).

It then picks the **single** contact-radius factor that maximises the *median* SCC across all
regions (a generalizing readout, not per-region cherry-picking) and prints how the MultiMM Pearson
compares to 0.70. Tuned config = ``compare_reference.TUNED_FEATURES`` (EV 1.0 + confinement +
dynamic sub-anchors + IB-MC + arcs repulsion cutoff). Resumable: per-region results are cached.

    python -m validation.hic_tune --cell GM12878 \\
        --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \\
        --chroms chr1,chr17 --n 100 --n-regions 8 --binsize 25000 \\
        --out out/sweep/GM12878_hic_tune.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gnome3d.data import ContactData

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation import contacts, metrics  # noqa: E402
from validation.cell_config import settings_for_cell  # noqa: E402
from validation.compare_reference import TUNED_FEATURES  # noqa: E402
from validation.sweep import enumerate_regions  # noqa: E402
from validation.validate import _apply_flags, _chrs_and_region, run_ensemble  # noqa: E402

DEFAULT_FACTORS = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
MULTIMM_TARGET = 0.70  # MultiMM's reported inverse-distance Pearson (random structures < 0.40)


def tune_region(
    region: str,
    cell: str,
    data_root: str,
    quality: str,
    n: int,
    hic_path: str,
    binsize: int,
    factors: list[float],
) -> dict[str, object]:
    """Generate ONE tuned ensemble for ``region`` and sweep the Hi-C readout on its coords.

    Returns {base_radius, multimm, n_bins, scc:{factor->val}, pearson:{factor->val}}.
    """
    tuned = _apply_flags(settings_for_cell(cell, data_root, quality), TUNED_FEATURES)
    chrs_list, bed_region = _chrs_and_region(region)
    data = ContactData.from_files(tuned, chrs_list, bed_region)
    ens = run_ensemble(tuned, data, chrs_list, bed_region, n)  # the only expensive step

    coords_list, mids_list = [], []
    bond_meds = []
    for beads in ens:
        coords, mids = metrics.to_arrays(beads)
        coords_list.append(coords)
        mids_list.append(mids)
        bond_meds.append(float(np.median(metrics.bond_lengths(coords))))
    base_radius = float(np.median(bond_meds))

    c_obs, bin_starts = contacts.observed_hic(hic_path, region, binsize)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize

    # MultiMM Pearson: decay-retained inverse-distance map, radius-free -> computed ONCE.
    inv = contacts.inverse_distance_heatmap(coords_list, mids_list[0], bin_starts, eff)
    multimm = contacts.multimm_pearson(inv, c_obs)

    # SCC / decay-stripped Pearson: rebuild the hard-radius contact map per factor (cheap).
    scc: dict[str, float] = {}
    pear: dict[str, float] = {}
    for f in factors:
        radius = f * base_radius
        c_sim = np.zeros_like(c_obs)
        for coords, mids in zip(coords_list, mids_list, strict=True):
            c_sim += contacts.simulated_contacts(coords, mids, bin_starts, eff, radius)
        cc = contacts.contact_correlation(c_sim, c_obs)
        scc[f"{f:g}"] = cc["scc"]
        pear[f"{f:g}"] = cc["pearson"]
    return {
        "base_radius": base_radius,
        "multimm": multimm,
        "n_bins": int(len(bin_starts)),
        "scc": scc,
        "pearson": pear,
    }


def _median(vals: list[float]) -> float:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def report(cache: dict[str, dict], regions: list[str], factors: list[float]) -> None:
    rows = [cache[r] for r in regions if r in cache]
    if not rows:
        print("[hic_tune] no results yet")
        return
    print(f"\n{'='*64}\nHi-C READOUT TUNING — median over {len(rows)} regions\n{'='*64}")
    print(f"{'radius_factor':>14} {'median_SCC':>12} {'median_pearson':>16}")
    best_f, best_scc = None, -np.inf
    for f in factors:
        key = f"{f:g}"
        msc = _median([row["scc"].get(key) for row in rows])
        mpe = _median([row["pearson"].get(key) for row in rows])
        flag = ""
        if np.isfinite(msc) and msc > best_scc:
            best_scc, best_f = msc, f
        print(f"{f:>14g} {msc:>12.4f} {mpe:>16.4f}{flag}")
    mm = _median([row["multimm"] for row in rows])
    print(f"\nBest contact-radius factor (max median SCC): {best_f:g}  -> SCC {best_scc:.4f}")
    print(f"{'-'*64}")
    print(f"MultiMM inverse-distance Pearson (decay retained, sim_power 3/2): {mm:.4f}")
    delta = mm - MULTIMM_TARGET
    verdict = "AT/ABOVE" if delta >= -0.02 else "below"
    print(f"  vs MultiMM target ~{MULTIMM_TARGET:.2f}: {verdict} (Δ {delta:+.3f}); random < 0.40")
    print(f"{'='*64}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True)
    p.add_argument("--hic", required=True, help="observed Hi-C .mcool")
    p.add_argument("--data-root", default="data")
    p.add_argument("--chroms", default=None, help="comma-separated; default all in breakpoints file")
    p.add_argument("--n", type=int, default=100, help="ensemble size (3dgnome standard >=100)")
    p.add_argument("--n-regions", type=int, default=8)
    p.add_argument("--quality", default="full", help="MC schedule (fast/balanced/full)")
    p.add_argument("--binsize", type=int, default=25000)
    p.add_argument("--min-ibs", type=int, default=2)
    p.add_argument("--max-ibs", type=int, default=6)
    p.add_argument("--max-mb", type=float, default=6.0)
    p.add_argument("--factors", default=None, help="comma-separated radius factors (× median bond)")
    p.add_argument("--out", required=True, help="results cache JSON (resumable)")
    args = p.parse_args()

    factors = [float(x) for x in args.factors.split(",")] if args.factors else DEFAULT_FACTORS
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}
    if cache:
        print(f"[hic_tune] resuming: {len(cache)} cached regions")

    s_meta = settings_for_cell(args.cell, args.data_root)
    bp = s_meta.data_path(s_meta.data_segment_split)
    chroms = args.chroms.split(",") if args.chroms else None
    regions = enumerate_regions(
        bp, args.n_regions, chroms=chroms, min_ibs=args.min_ibs, max_ibs=args.max_ibs, max_mb=args.max_mb
    )
    if not regions:
        sys.exit("[error] no regions found in size band")

    for region in regions:
        if region in cache:
            print(f"[hic_tune] {region}: cached")
            continue
        print(f"\n[hic_tune] {region}: generating n={args.n} tuned ensemble ({args.quality})...")
        cache[region] = tune_region(
            region, args.cell, args.data_root, args.quality, args.n, args.hic, args.binsize, factors
        )
        out_path.write_text(json.dumps(cache, indent=2))
        r = cache[region]
        best = max(r["scc"].items(), key=lambda kv: (kv[1] if kv[1] is not None and np.isfinite(kv[1]) else -np.inf))
        print(f"  [done] best SCC {best[1]:.3f} @ factor {best[0]} | MultiMM {r['multimm']:.3f}")

    report(cache, regions, factors)


if __name__ == "__main__":
    main()
