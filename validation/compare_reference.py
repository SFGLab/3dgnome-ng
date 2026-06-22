#!/usr/bin/env python3
"""Compare 3dgnome output against the C++ reference on the validation metrics.

Answers "are our outputs better or worse than the reference?" by scoring THREE ensembles
on the same metrics (``validation/metrics.py``):

  * reference         — the C++ 3dnome binary (the algorithmic source of truth, NO EV /
                        confinement; the 2016 paper admitted it makes overlapping loops)
  * python (parity)   — our port, feature flags OFF — should MATCH the reference (faithful)
  * python (+tuned)   — the UNIFIED canonical config (validation/cell_config.py): EV + confinement
                        + dynamic sub-anchors + IB-MC, the SAME config sweep/hic_tune use

Reference & parity share the parity.ini base (C++-faithfulness check); the tuned variant is the
canonical config and turns on dynamic sub-anchors, so its bead count differs — it is scored at its
own bond-scale radius and its overlap fraction is self-relative (see the radius note in main()).

Reuses the proven reference runner from ``harness/integration.py``.

    python -m validation.compare_reference --region chr1:18288319-20307135 -n 3 --fast
"""

from __future__ import annotations

import argparse
import math
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
from validation.cell_config import settings_for_cell, with_arcs_executor  # noqa: E402
from validation.sweep import enumerate_regions  # noqa: E402
from validation.validate import (  # noqa: E402
    FAIL,
    PASS,
    _chrs_and_region,
    load_contacts,
    run_ensemble,
    summarize,
)

sys.path.insert(0, str(ROOT / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)

MAX_LEVEL = 2  # heatmap + arc + smooth MC (same as the integration test)


# UNIFIED CONFIG: the "tuned" production settings are the SINGLE canonical config from
# validation/cell_config.py (settings_for_cell) — the same config sweep / hic_tune /
# self_correlation use. There is no separate TUNED_FEATURES flag-overlay any more: layering a
# partial overlay on the bare parity.ini base left confinement/EV at wrong defaults and exploded
# at scale. The C++ reference and the python "parity" variant keep the parity.ini base (their job
# is C++ faithfulness, features OFF); only the "tuned" variant uses the full canonical config.


def _tuned_settings(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """The unified production config (canonical) for the tuned variant, with the arcs executor
    honoured. Coarsening is NEVER applied to the MC — it happens after the fact on the binned
    contact/distance matrices in the metrics."""
    quality = "fast" if args.fast else "full"
    s = settings_for_cell(args.cell, args.data_root, quality)
    return with_arcs_executor(s, args.py_arcs, getattr(args, "py_workers", 0))


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
    # Arcs-stage executor for the python ensembles (both parity & tuned share s_base):
    #   batch    = JAX/GPU region-batch the n=100 arcs nodes (default; use on a CUDA box)
    #   threaded = numba across `--py-workers` cores (byte-exact via thread-local RNG)
    #   serial   = one node at a time
    # Both batch and threaded preserve parity (the kernels are the same MC); pick per hardware.
    s_base = with_arcs_executor(s_base, args.py_arcs, getattr(args, "py_workers", 0))
    data = ContactData.from_files(s_base, chrs_list, bed_region)
    contacts_list = load_contacts(s_base, chrs_list, bed_region)
    print(f"  [{region}] python parity (parity.ini, features off)...")
    base_structs = run_ensemble(s_base, data, chrs_list, bed_region, args.n_structures)
    print(f"  [{region}] python +tuned (unified canonical config)...")
    feat_structs = run_ensemble(_tuned_settings(args), data, chrs_list, bed_region, args.n_structures)

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
            # Hi-C correlation via MultiMM's APPROACH (faithful (d+1)^-3 vs ICE-balanced observed);
            # reported as our standard Hi-C number, NOT chased against their 0.70.
            try:
                cobs_bal, bstarts = contacts.observed_hic(args.hic, region, args.binsize, balance=True)
            except Exception:  # noqa: BLE001 (mcool may lack balance weights)
                cobs_bal, bstarts = contacts.observed_hic(args.hic, region, args.binsize)
            eff = int(bstarts[1] - bstarts[0]) if len(bstarts) > 1 else args.binsize
            met["hic_multimm"] = contacts.multimm_faithful_pearson(cl, ml[0], cobs_bal, bstarts, eff)
    return ref_m, base_m, feat_m


def score_law_region(region: str, config: Path, tmp: Path, args: argparse.Namespace) -> dict:
    """Scaling-law pass on ONE large region: reference (C++) vs python +tuned, both fit for
    R(s)~s^β and P(s)~s^-α. The python side is COARSENED (~20 kb/bead) so a ≥20 Mb region is
    tractable; the reference runs at its native resolution. At this scale there are several
    decades of genomic separation, so a real power-law window (high log-log R²) exists — unlike
    the small overlap/Hi-C regions. Returns {'ref': laws, 'feat': laws, 'n_ref', 'n_feat'}."""
    chrs_list, bed_region = _chrs_and_region(region)
    outdir = tmp / ("law_" + region.replace(":", "_").replace("-", "_"))
    outdir.mkdir(parents=True, exist_ok=True)
    n = args.law_n if args.law_n > 0 else args.n_structures
    workers = getattr(args, "ref_workers", 0)

    print(f"  [laws @ {region}] reference binary ({n} structures, native res)...")
    ref_structs, _ = ig.run_cpp_ensemble_parallel(
        outdir, config, n, MAX_LEVEL, region, "law", workers=workers
    )

    s_base = Settings()
    s_base.load_ini(str(config))  # parity.ini base only for loading the region's contact data
    data = ContactData.from_files(s_base, chrs_list, bed_region)
    # tuned = the unified canonical config, run at NATIVE resolution (no coarsening). Scaling-law
    # exponents are resolution-robust (the windowed fit excludes sub-resolution), so comparing the
    # tuned (fine) vs reference (coarse) exponents is valid; coarsening the MC is never done.
    print(f"  [laws @ {region}] python +tuned (unified canonical config, native res)...")
    feat_structs = run_ensemble(_tuned_settings(args), data, chrs_list, bed_region, n)

    # Resolution-normalize the β/α fit to a common bp grid so ref (~4.5 kb beads) and tuned
    # (dynamic ~1 kb) are compared at the SAME resolution (P(s)/α is bead-density-dependent
    # otherwise — the same confound we fixed for overlap). Rg/bond_cv stay native (per scaling_laws).
    def laws_of(structs: list) -> dict:
        cl, ml = [], []
        for beads in structs:
            c, mm = metrics.to_arrays(beads)
            cl.append(c)
            ml.append(mm)
        rad = float(np.median(metrics.bond_lengths(cl[0])))
        return metrics.ensemble_scaling_laws(cl, ml[0], rad, resolution_bp=args.law_resolution_bp)

    return {
        "ref": laws_of(ref_structs),
        "feat": laws_of(feat_structs),
        "n_ref": len(ref_structs[0]),
        "n_feat": len(feat_structs[0]),
    }


def _span(region: str) -> int:
    """Genomic span (bp) of a 'chr:a-b' region string."""
    a, b = region.split(":")[1].split("-")
    return int(b) - int(a)


def _sign_test(k: int, m: int) -> float:
    """One-sided sign test: P(≥ k of m 'wins' under a fair coin). Small p ⇒ a real effect."""
    if m == 0:
        return 1.0
    return sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0**m


def main() -> None:
    p = argparse.ArgumentParser(description="Score 3dgnome output vs the C++ reference")
    p.add_argument("--cell", default="GM12878", help="cell line for the unified tuned config")
    p.add_argument("--data-root", default="data")
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
        help="4DN .mcool path; if given, also report Hi-C SCC + the faithful inverse-distance "
        "Pearson (MultiMM's metric approach, (d+1)^-3 vs ICE-balanced observed)",
    )
    p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
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
        "--py-arcs",
        choices=["batch", "threaded", "serial"],
        default="batch",
        help="executor for the python ensembles' arcs stage: batch = JAX/GPU (default, for a CUDA "
        "box), threaded = numba across --py-workers cores, serial = one node at a time. smooth + "
        "estimate_dist already use the GPU batch path automatically when JAX is present.",
    )
    p.add_argument(
        "--py-workers",
        type=int,
        default=0,
        help="threads for --py-arcs threaded (numba, byte-exact); 0 = auto (cpu_count)",
    )
    p.add_argument("--contact-radius", type=float, default=None)
    p.add_argument("--skip-neighbors", type=int, default=1)
    p.add_argument(
        "--law-region",
        default=None,
        help="region for the scaling-law pass; default = auto-pick one large (≥--law-mb) region. "
        "The fractal-globule laws (β≈1/3, α≈1) only have a power-law window at this scale — they "
        "are NOT meaningful on the small overlap/Hi-C regions, so they are measured separately here.",
    )
    p.add_argument("--law-mb", type=float, default=20.0, help="min size (Mb) of the law region")
    p.add_argument(
        "--law-resolution-bp",
        type=int,
        default=25000,
        help="coarse-grain both variants to this bp grid before fitting β/α, so different bead "
        "resolutions (ref vs tuned) are comparable; 0 = native (confounded by bead density)",
    )
    p.add_argument(
        "--law-n",
        type=int,
        default=20,
        help="ensemble size for the law pass (exponents need far fewer structures than overlaps); "
        "0 = reuse -n",
    )
    p.add_argument("--no-laws", action="store_true", help="skip the large-region scaling-law pass")
    args = p.parse_args()

    # No coarsening — all variants run at native resolution (coarsening to a different resolution
    # made comparisons meaningless and exploded at full schedule).
    args._coarsen: dict[str, object] = {}

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
    # tuned vs ref overlap uses the RESOLUTION-NORMALIZED metric (tuned has finer beads → raw
    # overlap is density-inflated and not comparable). Parity-vs-ref above stays raw (N-matched).
    feat_wins = sum(r[3]["overlap_frac_norm"] < r[1]["overlap_frac_norm"] - 1e-9 for r in rows)
    d_overlaps = [r[1]["overlap_frac_norm"] - r[3]["overlap_frac_norm"] for r in rows]  # >0 good
    sc_ok = sum(
        np.isfinite(r[3]["selfconsistency_rho"])
        and r[3]["selfconsistency_rho"] <= r[1]["selfconsistency_rho"] + 0.15
        for r in rows
    )
    p_better = _sign_test(feat_wins, m)

    print(f"\n{'=' * 74}\n  ANSWERS  (median over {m} region(s); paired per region)\n{'=' * 74}")
    print(
        f"  {'variant':<26}{'overlap':>9}{'ovlp_norm':>10}{'HiC SCC':>9}{'Rg':>8}"
        f"{'dscale':>8}{'divers':>8}"
    )
    for label, var in [("reference (C++)", 0), ("python parity", 1), ("python +tuned", 2)]:
        print(
            f"  {label:<26}{med('overlap_frac', var):>9.4f}{med('overlap_frac_norm', var):>10.4f}"
            f"{med('selfconsistency_rho', var):>9.3f}{med('rg', var):>8.2f}"
            f"{med('dist_scaling_exp', var):>8.3f}{med('diversity_dab', var):>8.3f}"
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
        f"overlaps than reference (resolution-normalized): {feat_wins}/{m} regions  "
        f"(median Δ={float(np.median(d_overlaps)):+.4f}, sign-test p={p_better:.4f})"
    )
    print(f"  {PASS if sc_ok == m else FAIL}  self-consistency preserved: {sc_ok}/{m} regions")
    print(
        "  (scaling laws are NOT gated here — the fractal-globule regime needs a large region; "
        "see the dedicated law pass below)"
    )

    if args.hic:
        print(f"\n  Hi-C correlation vs {Path(args.hic).name} ({args.binsize // 1000}kb):")
        print(f"    {'variant':<26}{'SCC':>9}{'invdist Pearson':>18}")
        for label, var in [("reference (C++)", 0), ("python +tuned", 2)]:
            print(f"    {label:<26}{med('hic_scc', var):>9.3f}{med('hic_multimm', var):>18.3f}")
        ref_mm, feat_mm = med("hic_multimm", 0), med("hic_multimm", 2)
        non_inferior = np.isfinite(feat_mm) and (
            not np.isfinite(ref_mm) or feat_mm >= ref_mm - 0.02
        )
        print(
            f"    {PASS if non_inferior else FAIL}  +tuned Hi-C (inv-dist Pearson) ≥ reference: "
            f"ref={ref_mm:.3f} vs +tuned={feat_mm:.3f}"
        )
        print("    (faithful (d+1)^-3 vs ICE-balanced Hi-C — MultiMM's metric approach; "
              "value scales with region size, compared ref-vs-tuned at the SAME geometry)")

    # --- scaling-law pass on ONE large region (where the fractal-globule regime exists) ---
    if not args.no_laws:
        law_region = args.law_region
        if not law_region:
            bp_law = ig.DATA_DIR / "ccds_all_hg38_merged100k_GM12878.breakpoints.bed"
            chroms_law = args.chroms.split(",") if args.chroms else None
            cands = enumerate_regions(
                str(bp_law), 40, chroms=chroms_law, min_ibs=6, max_ibs=400, max_mb=args.law_mb
            )
            law_region = max(cands, key=_span) if cands else None
        if not law_region:
            print(f"\n  [laws] no region up to {args.law_mb} Mb found — pass --law-region explicitly")
        else:
            print(
                f"\n{'=' * 74}\n  SCALING LAWS @ {law_region} ({_span(law_region) / 1e6:.1f} Mb — "
                f"large enough for the power-law window)\n{'=' * 74}"
            )
            lw = score_law_region(law_region, config, tmp, args)
            print(f"  {'variant':<22}{'β (R²)':>14}{'α (R²)':>14}{'bond CV':>9}{'beads':>8}  laws?")
            for label, key, nkey in [
                ("reference (C++)", "ref", "n_ref"),
                ("python +tuned", "feat", "n_feat"),
            ]:
                lo = lw[key]
                d_ok, _ = metrics.check_law("dist_exp", lo["dist_exp"], lo["dist_r2"])
                c_ok, _ = metrics.check_law("contact_exp", lo["contact_exp"], lo["contact_r2"])
                # bond CV >> 1 => the chain is shredded (uneven bonds), so β/α are meaningless.
                degenerate = lo["bond_cv"] > 3.0
                verdict = "DEGENERATE" if degenerate else ("✓" if d_ok and c_ok else "·")
                print(
                    f"  {label:<22}{lo['dist_exp']:>7.2f} ({lo['dist_r2']:>4.2f})"
                    f"{lo['contact_exp']:>7.2f} ({lo['contact_r2']:>4.2f}){lo['bond_cv']:>9.2f}"
                    f"{lw[nkey]:>8d}   {verdict}"
                )
            print(
                "  (β: fractal globule ~1/3, band 0.15–0.60; α: chromatin ~1.0, band 0.50–1.60; "
                "'laws hold' = in-band AND log-log R² ≥ 0.80; bond CV > 3 = DEGENERATE/exploded, "
                "β/α invalid)"
            )

    if m == 1:
        print(
            "\n  NOTE: single region — re-run without --region for a multi-region significance test."
        )


if __name__ == "__main__":
    main()
