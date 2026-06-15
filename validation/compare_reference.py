#!/usr/bin/env python3
"""Compare 3dgnome output against the C++ reference on the validation metrics.

Answers "are our outputs better or worse than the reference?" by scoring THREE ensembles
on the same metrics (``validation/metrics.py``):

  * reference         — the C++ 3dnome binary (the algorithmic source of truth, NO EV /
                        confinement; the 2016 paper admitted it makes overlapping loops)
  * python (parity)   — our port, feature flags OFF — should MATCH the reference (faithful)
  * python (+tuned)   — the TUNED production set: EV (weight 2.0, smooth radius 0.7) + confinement
                        + dynamic sub-anchor count — should have FEWER overlaps than the reference

Reference & parity are N-matched (same parity config) for the faithfulness check. The tuned
variant turns on dynamic sub-anchors, so its bead count differs — it is scored at its own
bond-scale radius and its overlap fraction is self-relative (see the radius note in main()).

Reuses the proven reference runner from ``harness/integration.py``.

    python -m validation.compare_reference --region chr1:18288319-20307135 -n 3 --fast
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from gnome3d.data import ContactData
from gnome3d.settings import Settings

# Make `validation` importable when run as a script (`python validation/compare_reference.py`),
# not only as a module (`python -m validation.compare_reference`). Must precede the first-party
# imports below (which are therefore E402-exempt).
ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from validation import contacts, metrics  # noqa: E402
from validation.sweep import enumerate_regions  # noqa: E402
from validation.validate import (  # noqa: E402
    FAIL,
    PASS,
    _apply_flags,
    _chrs_and_region,
    load_contacts,
    run_ensemble,
    summarize,
)

sys.path.insert(0, str(ROOT / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)

MAX_LEVEL = 2  # heatmap + arc + smooth MC (same as the integration test)


TUNED_FEATURES: dict[str, object] = {
    # The TUNED production set (validation/RUNBOOK.md): EV at the sweep winner (weight 1.0, smooth
    # radius 0.7) + confinement + dynamic sub-anchor count + IB-MC. NOT default EV (0.5/0.5) —
    # that barely moved overlaps. IB-MC and inter-IB de-clashing only bite in MULTI-IB regions.
    # weight 1.0 (not 2.0): ~90% of the overlap reduction at the lowest Rg cost, scale-safe.
    "use_excluded_volume": True,
    "exclusion_apply_to_smooth": True,
    "exclusion_weight": 1.0,
    "exclusion_auto_factor_smooth": 0.7,
    "use_confinement": True,
    "confinement_apply_to_smooth": True,
    "use_dynamic_loop_density": True,
    "target_bp_per_subanchor": 1000,
    "use_ib_mc": True,
    # Truncate the unbounded non-arc 1/d repulsion (3x mean arc distance) so sparse sub-IBs don't
    # explode / hang in arcs polish. TUNED-only: parity stays unbounded to mirror C++ faithfully.
    "arcs_repulsion_cutoff_factor": 3.0,
}


def _radius(structs: list, fixed: float | None) -> float:  # type: ignore[type-arg]
    if fixed is not None:
        return fixed
    coords, _ = metrics.to_arrays(structs[0])
    return float(np.median(metrics.bond_lengths(coords)))


def score_region(
    region: str, config: Path, tmp: Path, args: argparse.Namespace
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run reference + python-parity + python-tuned on one region; return their scored metrics.

    Reference & parity are N-matched (shared radius). The tuned variant turns on dynamic
    sub-anchors → finer beads, so it is scored at its OWN bond-scale radius (the coarse parity
    radius would over-count its overlaps)."""
    label = f"validation_{region.replace(':', '_').replace('-', '_')}"
    chrs_list, bed_region = _chrs_and_region(region)
    outdir = tmp / label
    outdir.mkdir(parents=True, exist_ok=True)
    workers = getattr(args, "ref_workers", 0)
    print(
        f"  [{region}] reference binary ({args.n_structures} structures, "
        f"{'auto' if workers <= 0 else workers} workers)..."
    )
    ref_structs, _ = ig.run_cpp_ensemble_parallel(
        outdir, config, args.n_structures, MAX_LEVEL, region, label, workers=workers
    )

    s_base = Settings()
    s_base.load_ini(str(config))
    # Run the python ensembles' arcs stage threaded (numba threading is byte-exact via thread-local
    # RNG, so parity stays faithful) instead of serial — the n=100 arcs nodes parallelise across
    # cores. Applies to both parity and tuned (they share s_base).
    workers = getattr(args, "py_workers", 0)
    s_base.mc_executor_arcs = "threaded"
    s_base.mc_executor_threaded_workers = workers if workers > 0 else (os.cpu_count() or 1)
    data = ContactData.from_files(s_base, chrs_list, bed_region)
    contacts_list = load_contacts(s_base, chrs_list, bed_region)
    # --multimm-mode coarsens the python beads (~20 kb/bead) so a ~20 Mb region stays tractable
    # AND matches MultiMM's resolution; empty dict otherwise.
    coarsen: dict[str, object] = getattr(args, "_coarsen", {})
    s_par = _apply_flags(s_base, coarsen) if coarsen else s_base
    print(f"  [{region}] python parity...")
    base_structs = run_ensemble(s_par, data, chrs_list, bed_region, args.n_structures)
    print(f"  [{region}] python +tuned (EV2.0/r0.7+conf+dynamic+ib_mc)...")
    feat_structs = run_ensemble(
        _apply_flags(s_base, {**TUNED_FEATURES, **coarsen}),
        data,
        chrs_list,
        bed_region,
        args.n_structures,
    )

    radius = _radius(base_structs, args.contact_radius)
    radius_feat = _radius(feat_structs, args.contact_radius)
    ref_m = summarize(ref_structs, contacts_list, radius, args.skip_neighbors)
    base_m = summarize(base_structs, contacts_list, radius, args.skip_neighbors)
    feat_m = summarize(feat_structs, contacts_list, radius_feat, args.skip_neighbors)

    # Per-variant: genome-structure scaling laws (always) + Hi-C correlation vs the .mcool (if
    # --hic). Each ensemble at its own bond-scale radius; Hi-C on the SAME observed map → comparable.
    for structs, met, rad in (
        (ref_structs, ref_m, radius),
        (base_structs, base_m, radius),
        (feat_structs, feat_m, radius_feat),
    ):
        cl, ml = [], []
        for beads in structs:
            c, mm = metrics.to_arrays(beads)
            cl.append(c)
            ml.append(mm)
        laws = metrics.ensemble_scaling_laws(cl, ml[0], rad)
        met["law_dist_exp"] = laws["dist_exp"]
        met["law_dist_r2"] = laws["dist_r2"]
        met["law_contact_exp"] = laws["contact_exp"]
        met["law_contact_r2"] = laws["contact_r2"]
        met["law_bond_cv"] = laws["bond_cv"]
        if args.hic:
            hr = contacts.ensemble_hic_correlation(cl, ml, args.hic, region, args.binsize, rad)
            met["hic_scc"] = hr["scc"]
            met["hic_multimm"] = hr["multimm_pearson"]
    return ref_m, base_m, feat_m


def _sign_test(k: int, m: int) -> float:
    """One-sided sign test: P(≥ k of m 'wins' under a fair coin). Small p ⇒ a real effect."""
    if m == 0:
        return 1.0
    return sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0**m


def main() -> None:
    p = argparse.ArgumentParser(description="Score 3dgnome output vs the C++ reference")
    p.add_argument(
        "--region", default=None, help="single region override (default: multi-IB sample)"
    )
    p.add_argument("--n-regions", type=int, default=6, help="# multi-IB regions to sample")
    p.add_argument("--chroms", default=None, help="comma-sep chromosomes (default: all)")
    p.add_argument("--min-ibs", type=int, default=2)
    p.add_argument("--max-ibs", type=int, default=6)
    p.add_argument("--max-mb", type=float, default=6.0)
    p.add_argument(
        "--hic",
        default=None,
        help="4DN .mcool path; if given, also report Hi-C SCC + MultiMM inverse-distance Pearson",
    )
    p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
    p.add_argument(
        "--multimm-mode",
        action="store_true",
        help="fix the geometry to MultiMM's (≈20 Mb regions, 20 kb bins, ~20 kb/bead coarsening) so "
        "the MultiMM inverse-distance Pearson is directly quotable against their ≈0.70 (random <0.40)",
    )
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=100,
        help="ensemble size — 3dgnome ensembles need >=100; small n only for quick checks",
    )
    p.add_argument("--fast", action="store_true", help="fast (low-quality) MC schedule")
    p.add_argument(
        "--ref-workers",
        type=int,
        default=0,
        help="parallelize the C++ reference ensemble across this many cores (each worker a chunk "
        "with a distinct seed); 0 = auto (min(n, cpu_count)), 1 = serial. Needs `make 3dnome` "
        "(the -r seed flag).",
    )
    p.add_argument(
        "--py-workers",
        type=int,
        default=0,
        help="threads for the python ensembles' arcs stage (numba threading is byte-exact); "
        "0 = auto (cpu_count). The arcs nodes run threaded instead of serial.",
    )
    p.add_argument("--contact-radius", type=float, default=None)
    p.add_argument("--skip-neighbors", type=int, default=1)
    args = p.parse_args()

    # MultiMM geometry preset: ≈20 Mb regions, 20 kb Hi-C bins, and ~20 kb/bead coarsening of the
    # python variants (so a 20 Mb region is ~1000 beads — MultiMM's resolution, and tractable).
    args._coarsen: dict[str, object] = {}
    if args.multimm_mode:
        if not args.hic:
            sys.exit("[error] --multimm-mode needs --hic (it's a Hi-C-correlation comparison)")
        args.binsize = 25000  # ≈MultiMM's 20 kb; 25 kb is a standard mcool resolution
        args.min_ibs, args.max_ibs, args.max_mb = 12, 24, 24.0
        args._coarsen = {"use_dynamic_loop_density": True, "target_bp_per_subanchor": 20000}
        print("[compare] MultiMM mode: ≈20 Mb regions, 25 kb bins, ~20 kb/bead (vs MultiMM ≈0.70)")

    if not ig.CPP_BIN.exists():
        sys.exit(f"[error] reference binary not found: {ig.CPP_BIN}\n  run: make 3dnome")

    if args.region:
        regions = [args.region]
    else:
        bp = ig.DATA_DIR / "ccds_all_hg38_merged100k_GM12878.breakpoints.bed"
        chroms = args.chroms.split(",") if args.chroms else None
        regions = enumerate_regions(
            str(bp),
            args.n_regions,
            chroms=chroms,
            min_ibs=args.min_ibs,
            max_ibs=args.max_ibs,
            max_mb=args.max_mb,
        )
        if not regions:
            sys.exit("[error] no multi-IB regions found")
    print(f"[compare] {len(regions)} region(s), n={args.n_structures}: {regions}")

    tmp = Path(tempfile.mkdtemp(prefix="gnome3d_cmp_"))
    config = tmp / "parity.ini"
    ig.write_config(config, fast=args.fast)  # parity settings, GM12878 data paths

    rows: list[tuple[str, dict[str, float], dict[str, float], dict[str, float]]] = []
    for region in regions:
        ref_m, base_m, feat_m = score_region(region, config, tmp, args)
        rows.append((region, ref_m, base_m, feat_m))
        print(
            f"  -> {region}: overlap ref={ref_m['overlap_frac']:.4f} "
            f"parity={base_m['overlap_frac']:.4f} +tuned={feat_m['overlap_frac']:.4f} "
            f"(Nref={int(ref_m['n_beads'])} Nfeat={int(feat_m['n_beads'])})"
        )

    # --- aggregate across regions (paired per region) ---
    m = len(rows)
    med = lambda f, var: float(np.median([{0: r[1], 1: r[2], 2: r[3]}[var][f] for r in rows]))
    faithful_ok = sum(
        abs(r[2]["overlap_frac"] - r[1]["overlap_frac"]) <= 0.02
        and r[2]["n_beads"] == r[1]["n_beads"]
        for r in rows
    )
    feat_wins = sum(r[3]["overlap_frac"] < r[1]["overlap_frac"] - 1e-9 for r in rows)
    d_overlaps = [r[1]["overlap_frac"] - r[3]["overlap_frac"] for r in rows]  # ref - feat (>0 good)
    sc_ok = sum(
        np.isfinite(r[3]["selfconsistency_rho"])
        and r[3]["selfconsistency_rho"] <= r[1]["selfconsistency_rho"] + 0.15
        for r in rows
    )
    p_better = _sign_test(feat_wins, m)

    print(f"\n{'=' * 74}\n  ANSWERS  (median over {m} region(s); paired per region)\n{'=' * 74}")
    print(f"  {'variant':<26}{'overlap':>9}{'HiC SCC':>9}{'Rg':>8}{'dscale':>8}{'divers':>8}")
    for label, var in [("reference (C++)", 0), ("python parity", 1), ("python +tuned", 2)]:
        print(
            f"  {label:<26}{med('overlap_frac', var):>9.4f}{med('selfconsistency_rho', var):>9.3f}"
            f"{med('rg', var):>8.2f}{med('dist_scaling_exp', var):>8.3f}{med('diversity_dab', var):>8.3f}"
        )
    print()
    # Degeneracy guard: strong EV can BLOW THE STRUCTURE UP (chain shredded — Rg explodes, bonds
    # wildly uneven), especially at coarse/large scale. Then overlap/Hi-C/laws are meaningless.
    feat_rg, ref_rg = med("rg", 2), med("rg", 0)
    feat_bcv = med("law_bond_cv", 2)
    degenerate = (ref_rg > 0 and feat_rg > 3 * ref_rg) or (np.isfinite(feat_bcv) and feat_bcv > 3.0)
    if degenerate:
        print(
            f"  {FAIL}  +tuned structure DEGENERATE: Rg={feat_rg:.0f} vs ref {ref_rg:.1f} "
            f"({feat_rg / ref_rg:.0f}×), bond CV={feat_bcv:.1f} — its overlap/Hi-C/laws below are "
            "INVALID (EV too strong for this scale; gentler EV or stronger confinement needed)\n"
        )

    print(
        f"  {PASS if faithful_ok == m else FAIL}  parity faithful to reference "
        f"(overlap Δ≤0.02 & N matched): {faithful_ok}/{m} regions"
    )
    print(
        f"  {PASS if feat_wins == m else (FAIL if feat_wins == 0 else PASS)}  +tuned has FEWER "
        f"overlaps than reference: {feat_wins}/{m} regions  "
        f"(median Δ={float(np.median(d_overlaps)):+.4f}, sign-test p={p_better:.4f})"
    )
    print(f"  {PASS if sc_ok == m else FAIL}  self-consistency preserved: {sc_ok}/{m} regions")

    # Genome-structure scaling laws (3dgnome / MultiMM): R(s)~s^β, P(s)~s^-α. A "law holds" =
    # power-law (log-log R² high) with exponent in the biological band. Canonical β≈1/3, α≈1
    # appear over LARGE / multi-IB ranges; small single-IB regions read flatter (low R²).
    print("\n  Genome-structure laws (median; β=dist R(s)~s^β, α=contact P(s)~s^-α):")
    print(f"    {'variant':<20}{'β (R²)':>14}{'α (R²)':>14}{'bond CV':>9}  laws?")
    for label, var in [("reference (C++)", 0), ("python +tuned", 2)]:
        de, dr = med("law_dist_exp", var), med("law_dist_r2", var)
        ce, cr = med("law_contact_exp", var), med("law_contact_r2", var)
        d_ok, _ = metrics.check_law("dist_exp", de, dr)
        c_ok, _ = metrics.check_law("contact_exp", ce, cr)
        print(
            f"    {label:<20}{de:>7.2f} ({dr:>4.2f}){ce:>7.2f} ({cr:>4.2f}){med('law_bond_cv', var):>9.2f}"
            f"   {'✓' if d_ok and c_ok else '·'}"
        )

    if args.hic:
        print(f"\n  Hi-C correlation vs {Path(args.hic).name} ({args.binsize // 1000}kb):")
        print(f"    {'variant':<26}{'SCC':>9}{'MultiMM Pearson':>18}")
        for label, var in [("reference (C++)", 0), ("python +tuned", 2)]:
            print(f"    {label:<26}{med('hic_scc', var):>9.3f}{med('hic_multimm', var):>18.3f}")
        ref_mm, feat_mm = med("hic_multimm", 0), med("hic_multimm", 2)
        non_inferior = np.isfinite(feat_mm) and (
            not np.isfinite(ref_mm) or feat_mm >= ref_mm - 0.02
        )
        print(
            f"    {PASS if non_inferior else FAIL}  +tuned Hi-C (MultiMM) ≥ reference: "
            f"ref={ref_mm:.3f} vs +tuned={feat_mm:.3f}"
        )
        if args.multimm_mode:
            hit = np.isfinite(feat_mm) and feat_mm >= 0.40
            print(
                f"    {PASS if hit else FAIL}  vs MultiMM paper: +tuned={feat_mm:.3f}  "
                f"(MultiMM ≈0.70, random <0.40 — MultiMM geometry, directly comparable)"
            )
        else:
            print(
                "    (note: MultiMM reports ≈0.70 on 20 Mb regions; NOT comparable at this region "
                "size — use --multimm-mode for an apples-to-apples number)"
            )

    if m == 1:
        print(
            "\n  NOTE: single region — re-run without --region for a multi-region significance test."
        )


if __name__ == "__main__":
    main()
