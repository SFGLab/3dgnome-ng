"""Compare 3dgnome output against the reference on the validation metrics.

Answers whether our outputs are better or worse than the reference by scoring three ensembles
on the same metrics in validation/metrics/structure.py.

  * reference. The reference 3dnome binary, the algorithmic source of truth, with no EV or
    confinement. The 2016 paper noted it makes overlapping loops.
  * python (parity). Our port with feature flags off. Should match the reference.
  * python (+tuned). The unified canonical config in validation/core/config.py. EV plus
    confinement plus dynamic sub-anchors plus IB-MC, the same config sweep and hic_tune use.

Reference and parity share the parity.ini base, a reference-faithfulness check. The tuned variant is the
canonical config and turns on dynamic sub-anchors, so its bead count differs. It is scored at its
own bond-scale radius and its overlap fraction is self-relative. See the radius note in run().

Reuses the single reconstruction path in validation/core/variants.py.

    python -m validation compare --region chr1:18288319-20307135 -n 3 --fast
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from gnome3d.data import ContactData
from validation import metrics
from validation.core import variants
from validation.core.data import load_chiapet_contacts
from validation.core.ensemble import summarize
from validation.core.regions import enumerate_regions
from validation.core.regions import parse_region_arg as _chrs_and_region
from validation.core.report import FAIL, PASS, WARN
from validation.metrics import hic as contacts
from validation.studies import Context, Study, register

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)


def _radius(structs: list, fixed: float | None) -> float:  # type: ignore[type-arg]
    if fixed is not None:
        return fixed
    coords, _ = metrics.to_arrays(structs[0])
    return float(np.median(metrics.bond_lengths(coords)))


def _score_region(
    region: str, ctx: Context, args: argparse.Namespace
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run reference, python-parity and python-tuned on one region and return their scored metrics.

    Reference and parity are N-matched, sharing one radius. The tuned variant turns on dynamic
    sub-anchors and so has finer beads. It is scored at its own bond-scale radius. The coarse
    parity radius would over-count its overlaps."""
    label = f"validation_{region.replace(':', '_').replace('-', '_')}"
    chrs_list, bed_region = _chrs_and_region(region)
    workers = ctx.ref_workers
    print(
        f"  [{region}] reference binary ({ctx.n} structures, "
        f"{'auto' if workers <= 0 else workers} workers)..."
    )
    print(f"  [{region}] python parity (parity.ini, features off)...")
    print(f"  [{region}] python +tuned (unified canonical config)...")

    # Arcs-stage executor for the python ensembles. Both parity and tuned share s_base.
    #   batch    JAX/GPU region-batch the n=100 arcs nodes. Default, use on a CUDA box.
    #   threaded numba across --py-workers cores, byte-exact via thread-local RNG.
    #   serial   one node at a time.
    # Both batch and threaded preserve parity since the kernels are the same MC. Pick per hardware.
    s_base = variants.parity_settings(ctx.config, ctx.py_arcs, ctx.py_workers)
    # Self-consistency correlates input interaction frequency against 3D distance. The real IF
    # signal is the cluster loop strengths, with PET counts around 3 to 61828. Singletons-only
    # scores 1 to 2, a near-binary rank signal that goes to noise on small regions and spuriously
    # regresses when a tighter structure narrows the distance spread. Use the full ChIA-PET so
    # self-consistency measures loop reproduction as the paper intends. This only feeds
    # self_consistency. Overlap, Rg and Hi-C are structure-derived and unaffected.
    contacts_list = load_chiapet_contacts(s_base, chrs_list, bed_region)
    data = ContactData.from_files(s_base, chrs_list, bed_region)

    ctx.data = data
    ctx.label = label
    ens = variants.reconstruct_all(["reference", "parity", "tuned"], region, ctx)
    ref_structs, base_structs, feat_structs = ens["reference"], ens["parity"], ens["tuned"]

    radius = _radius(base_structs, args.contact_radius)
    radius_feat = _radius(feat_structs, args.contact_radius)
    ref_m = summarize(ref_structs, contacts_list, radius, args.skip_neighbors)
    base_m = summarize(base_structs, contacts_list, radius, args.skip_neighbors)
    feat_m = summarize(feat_structs, contacts_list, radius_feat, args.skip_neighbors)

    # Per variant. Genome-structure scaling laws always, plus Hi-C correlation against the .mcool
    # when --hic is set. Each ensemble uses its own bond-scale radius. Hi-C uses the same observed
    # map so the variants stay comparable.
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
        if ctx.hic:
            hr = contacts.ensemble_hic_correlation(cl, ml, ctx.hic, region, args.binsize, rad)
            met["hic_scc"] = hr["scc"]
            # Hi-C correlation via MultiMM's approach, faithful (d+1)^-3 against ICE-balanced
            # observed. Reported as our standard Hi-C number, not chased against their 0.70.
            try:
                cobs_bal, bstarts = contacts.observed_hic(ctx.hic, region, args.binsize, balance=True)
            except Exception:  # noqa: BLE001  mcool may lack balance weights
                cobs_bal, bstarts = contacts.observed_hic(ctx.hic, region, args.binsize)
            eff = int(bstarts[1] - bstarts[0]) if len(bstarts) > 1 else args.binsize
            met["hic_multimm"] = contacts.multimm_faithful_pearson(cl, ml[0], cobs_bal, bstarts, eff)
    return ref_m, base_m, feat_m


def _score_law_region(region: str, ctx: Context, args: argparse.Namespace) -> dict:  # type: ignore[type-arg]
    """Scaling-law pass on one large region. Reference against python +tuned, both fit for
    R(s)~s^β and P(s)~s^-α. Both run at their native resolution. At this scale there are several
    decades of genomic separation, so a real power-law window with high log-log R² exists, unlike
    the small overlap and Hi-C regions. Returns {'ref': laws, 'feat': laws, 'n_ref', 'n_feat'}."""
    n = args.law_n if args.law_n > 0 else ctx.n
    chrs_list, bed_region = _chrs_and_region(region)
    # parity.ini base, used just to load the region's contact data.
    s_base = variants.parity_settings(ctx.config, ctx.py_arcs, ctx.py_workers)
    data = ContactData.from_files(s_base, chrs_list, bed_region)

    print(f"  [laws @ {region}] reference binary ({n} structures, native res)...")
    # tuned is the unified canonical config, run at native resolution with no coarsening.
    # Scaling-law exponents are resolution-robust since the windowed fit excludes sub-resolution,
    # so comparing the tuned fine exponents against the reference coarse exponents is valid.
    print(f"  [laws @ {region}] python +tuned (unified canonical config, native res)...")
    law_ctx = dataclasses.replace(ctx, n=n, data=data, label="law")
    ens = variants.reconstruct_all(["reference", "tuned"], region, law_ctx)
    ref_structs, feat_structs = ens["reference"], ens["tuned"]

    # Resolution-normalize the β/α fit to a common bp grid so ref beads around 4.5 kb and tuned
    # beads around dynamic 1 kb are compared at the same resolution. Otherwise P(s) and α depend on
    # bead density, the same confound we fixed for overlap. Rg and bond_cv stay native per scaling_laws.
    def laws_of(structs: list) -> dict:  # type: ignore[type-arg]
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
    """One-sided sign test. P(≥ k of m wins) under a fair coin. Small p implies a real effect."""
    if m == 0:
        return 1.0
    return sum(math.comb(m, i) for i in range(k, m + 1)) / 2.0**m


def _wilcoxon(deltas: list[float]) -> float:
    """One-sided Wilcoxon signed-rank p for H1 that paired deltas are greater than 0. Uses the delta
    magnitudes rather than just their signs like the sign test, so it resolves an effect with far
    fewer regions. NaN if scipy is missing or too few non-zero deltas."""
    d = [x for x in deltas if np.isfinite(x) and abs(x) > 1e-12]
    if len(d) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d, alternative="greater").pvalue)
    except (ImportError, ValueError):
        return float("nan")


class Compare(Study):
    name = "compare"
    help = "reference vs parity vs tuned across regions, overlaps, Hi-C, cross-data correlation, scaling laws"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--region", default=None, help="single region override (default: multi-IB sample)"
        )
        p.add_argument("--n-regions", type=int, default=6, help="# multi-IB regions to sample")
        p.add_argument("--chroms", default=None, help="comma-sep chromosomes (default: all)")
        p.add_argument("--min-ibs", type=int, default=2)
        p.add_argument("--max-ibs", type=int, default=6)
        p.add_argument("--max-mb", type=float, default=6.0)
        p.add_argument("--binsize", type=int, default=25000, help="Hi-C bin size for correlation")
        p.add_argument(
            "--cross-data-binsize",
            type=int,
            default=1000000,
            help="Hi-C bin size for the cross-data correlation of ChIA-PET vs Hi-C. 1 Mb is the paper's "
            "own resolution for the intra-chromosomal Fig. 2B correlation, Chr 3 @ 1 Mb, ρ=0.67. This "
            "is the apples-to-apples match for our intra-chromosomal region-level cross-data correlation. "
            "Coarse is correct here. A CTCF ChIA-PET map fills only ~7%% of 100kb bin-pairs, correlation "
            "underpowered on a 93%%-empty matrix, versus ~72%% at 1 Mb. At 1 Mb our log1p Pearson is ~0.76 "
            "median across GM12878 regions, matching or beating the paper's 0.67. The O/E correlation is "
            "roughly res-invariant.",
        )
        p.add_argument("--contact-radius", type=float, default=None)
        p.add_argument("--skip-neighbors", type=int, default=1)
        p.add_argument(
            "--law-region",
            default=None,
            help="region for the scaling-law pass; default = auto-pick one large (≥--law-mb) region. "
            "The fractal-globule laws β≈1/3 and α≈1 only have a power-law window at this scale. They "
            "are not meaningful on the small overlap and Hi-C regions, so they are measured separately here.",
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

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
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
        print(f"[compare] {len(regions)} region(s), n={ctx.n}: {regions}")

        tmp = Path(tempfile.mkdtemp(prefix="gnome3d_cmp_"))
        ctx.config = variants.write_parity_ini(tmp, fast=ctx.fast)  # parity settings, GM12878 data paths

        s_v4 = variants.parity_settings(ctx.config, ctx.py_arcs, ctx.py_workers)  # for the cross-data correlation ChIA-PET input contacts
        rows: list[tuple[str, dict[str, float], dict[str, float], dict[str, float]]] = []
        v4_rows: list[dict[str, float]] = []
        for region in regions:
            ref_m, base_m, feat_m = _score_region(region, ctx, args)
            rows.append((region, ref_m, base_m, feat_m))
            if ctx.hic:  # cross-data correlation data-level check. Input ChIA-PET heatmap against Hi-C, no structure.
                chrs_list, bed_region = _chrs_and_region(region)
                chia = load_chiapet_contacts(s_v4, chrs_list, bed_region)
                v4_rows.append(
                    contacts.cross_data_correlation(chia, ctx.hic, region, args.cross_data_binsize)
                )
            print(
                f"  -> {region}: overlap ref={ref_m['overlap_frac']:.4f} "
                f"parity={base_m['overlap_frac']:.4f} +tuned={feat_m['overlap_frac']:.4f} "
                f"(Nref={int(ref_m['n_beads'])} Nfeat={int(feat_m['n_beads'])})"
            )

        # --- aggregate across regions, paired per region ---
        m = len(rows)
        med = lambda f, var: float(np.median([{0: r[1], 1: r[2], 2: r[3]}[var][f] for r in rows]))
        faithful_ok = sum(
            abs(r[2]["overlap_frac"] - r[1]["overlap_frac"]) <= 0.02
            and r[2]["n_beads"] == r[1]["n_beads"]
            for r in rows
        )
        # tuned vs ref overlap uses the resolution-normalized metric. Tuned has finer beads, so raw
        # overlap is density-inflated and not comparable. Parity vs ref above stays raw and N-matched.
        feat_wins = sum(r[3]["overlap_frac_norm"] < r[1]["overlap_frac_norm"] - 1e-9 for r in rows)
        d_overlaps = [r[1]["overlap_frac_norm"] - r[3]["overlap_frac_norm"] for r in rows]  # positive is good
        sc_ok = sum(
            np.isfinite(r[3]["selfconsistency_rho"])
            and r[3]["selfconsistency_rho"] <= r[1]["selfconsistency_rho"] + 0.15
            for r in rows
        )
        p_better = _sign_test(feat_wins, m)

        print(f"\n{'=' * 74}\n  answers, median over {m} region(s), paired per region\n{'=' * 74}")
        print(
            f"  {'variant':<26}{'overlap':>9}{'ovlp_norm':>10}{'selfcons':>9}{'Rg':>8}"
            f"{'dscale':>8}{'divers':>8}"
        )
        for label, var in [("reference", 0), ("python parity", 1), ("python +tuned", 2)]:
            print(
                f"  {label:<26}{med('overlap_frac', var):>9.4f}{med('overlap_frac_norm', var):>10.4f}"
                f"{med('selfconsistency_rho', var):>9.3f}{med('rg', var):>8.2f}"
                f"{med('dist_scaling_exp', var):>8.3f}{med('diversity_dab', var):>8.3f}"
            )
        print()
        # Degeneracy guard. Strong EV can make the structure expand without bound, with Rg exploding
        # and bond lengths wildly uneven, especially at coarse or large scale. Then overlap, Hi-C and
        # laws are meaningless.
        feat_rg, ref_rg = med("rg", 2), med("rg", 0)
        feat_bcv = med("law_bond_cv", 2)
        degenerate = (ref_rg > 0 and feat_rg > 3 * ref_rg) or (np.isfinite(feat_bcv) and feat_bcv > 3.0)
        if degenerate:
            print(
                f"  {FAIL}  +tuned structure degenerate. Rg={feat_rg:.0f} vs ref {ref_rg:.1f} "
                f"({feat_rg / ref_rg:.0f}×), bond CV={feat_bcv:.1f}. Its overlap, Hi-C and laws below are "
                "invalid. EV is too strong for this scale, a gentler EV or stronger confinement is needed\n"
            )

        print(
            f"  {PASS if faithful_ok == m else FAIL}  parity faithful to reference "
            f"(overlap Δ≤0.02 & N matched): {faithful_ok}/{m} regions"
        )
        print(
            f"  {PASS if feat_wins == m else (FAIL if feat_wins == 0 else PASS)}  +tuned has fewer "
            f"overlaps than reference (resolution-normalized): {feat_wins}/{m} regions  "
            f"(median Δ={float(np.median(d_overlaps)):+.4f}, sign-test p={p_better:.4f}, "
            f"Wilcoxon p={_wilcoxon(d_overlaps):.4f})"
        )
        print(f"  {PASS if sc_ok == m else FAIL}  self-consistency preserved: {sc_ok}/{m} regions")
        # Name the offending regions so a FAIL is diagnosable without a re-run. Self-consistency rho
        # is Spearman of input-IF against 3D distance, where negative is good, so tuned regressing
        # means rho went up.
        sc_bad = [
            (r[0], r[1]["selfconsistency_rho"], r[3]["selfconsistency_rho"])
            for r in rows
            if not (
                np.isfinite(r[3]["selfconsistency_rho"])
                and r[3]["selfconsistency_rho"] <= r[1]["selfconsistency_rho"] + 0.15
            )
        ]
        for region, rho_p, rho_t in sc_bad:
            print(f"      ↳ regressed: {region}  parity ρ={rho_p:+.3f} → tuned ρ={rho_t:+.3f}")
        print(
            "  scaling laws are not gated here. The fractal-globule regime needs a large region. "
            "See the dedicated law pass below"
        )

        if ctx.hic:
            print(f"\n  Hi-C correlation vs {Path(ctx.hic).name} ({args.binsize // 1000}kb):")
            print(f"    {'variant':<26}{'SCC':>9}{'invdist Pearson':>18}")
            for label, var in [("reference", 0), ("python +tuned", 2)]:
                print(f"    {label:<26}{med('hic_scc', var):>9.3f}{med('hic_multimm', var):>18.3f}")
            ref_mm, feat_mm = med("hic_multimm", 0), med("hic_multimm", 2)
            non_inferior = np.isfinite(feat_mm) and (
                not np.isfinite(ref_mm) or feat_mm >= ref_mm - 0.02
            )
            print(
                f"    {PASS if non_inferior else FAIL}  +tuned Hi-C (inv-dist Pearson) ≥ reference: "
                f"ref={ref_mm:.3f} vs +tuned={feat_mm:.3f}"
            )
            print("    faithful (d+1)^-3 vs ICE-balanced Hi-C, MultiMM's metric approach. "
                  "Value scales with region size, compared ref against tuned at the same geometry")

            # Cross-data correlation from 3dgnome 2016 Fig. 2. Input ChIA-PET heatmap against Hi-C, data-level, no structure.
            if v4_rows:
                v4med = lambda k: float(np.nanmedian([r[k] for r in v4_rows]))
                print(
                    f"\n  cross-data correlation, input ChIA-PET heatmap vs Hi-C @ {args.cross_data_binsize // 1000}kb "
                    "(data-level, no structure):"
                )
                print(
                    f"    raw  Pearson(log1p) = {v4med('pearson'):.3f}   SCC = {v4med('scc'):.3f}   "
                    "(paper Fig. 2B: intra-chromosomal @ 1Mb, ρ=0.67)"
                )
                print(
                    f"    O/E  Pearson(logO/E) = {v4med('pearson_oe'):.3f}   Spearman(O/E) = "
                    f"{v4med('spearman_oe'):.3f}   (median {v4med('n_pairs_oe'):.0f} shared pairs/region)"
                )
                # A region only spans span/binsize bins, so a coarse bin size on a small region
                # leaves very few bin pairs and the correlation stops being meaningful. Warn rather
                # than let a confident-looking number stand on a handful of points.
                npairs = v4med("n_pairs_oe")
                if npairs < 50:
                    print(
                        f"    {WARN}  only {npairs:.0f} shared bin-pairs per region at "
                        f"{args.cross_data_binsize // 1000}kb. The correlation is unreliable at this "
                        "count. Use larger regions with --max-mb and --min-ibs, or a finer "
                        "--cross-data-binsize, so a region spans enough bins"
                    )
                print(
                    "    compared at the paper's own 1Mb Fig.2B resolution, where a region needs to "
                    "span many 1Mb bins to be meaningful. O/E strips the shared decay for the "
                    "structure-only agreement"
                )

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
                print(f"\n  [laws] no region up to {args.law_mb} Mb found. Pass --law-region explicitly")
            else:
                print(
                    f"\n{'=' * 74}\n  scaling laws @ {law_region}, {_span(law_region) / 1e6:.1f} Mb, "
                    f"large enough for the power-law window\n{'=' * 74}"
                )
                lw = _score_law_region(law_region, ctx, args)
                print(f"  {'variant':<22}{'β (R²)':>14}{'α (R²)':>14}{'bond CV':>9}{'beads':>8}  laws?")
                for label, key, nkey in [
                    ("reference", "ref", "n_ref"),
                    ("python +tuned", "feat", "n_feat"),
                ]:
                    lo = lw[key]
                    d_ok, _ = metrics.check_law("dist_exp", lo["dist_exp"], lo["dist_r2"])
                    c_ok, _ = metrics.check_law("contact_exp", lo["contact_exp"], lo["contact_r2"])
                    # bond CV >> 1 means bonds are very uneven, so β/α are meaningless.
                    degenerate = lo["bond_cv"] > 3.0
                    verdict = "degenerate" if degenerate else ("✓" if d_ok and c_ok else "·")
                    print(
                        f"  {label:<22}{lo['dist_exp']:>7.2f} ({lo['dist_r2']:>4.2f})"
                        f"{lo['contact_exp']:>7.2f} ({lo['contact_r2']:>4.2f}){lo['bond_cv']:>9.2f}"
                        f"{lw[nkey]:>8d}   {verdict}"
                    )
                print(
                    "  β is fractal globule ~1/3, band 0.15 to 0.60. α is chromatin ~1.0, band 0.50 to "
                    "1.60. 'laws hold' means in-band and log-log R² ≥ 0.80. bond CV > 3 means degenerate "
                    "or exploded, β and α invalid"
                )

        if m == 1:
            print(
                "\n  note, single region. Re-run without --region for a multi-region significance test."
            )


register(Compare())
