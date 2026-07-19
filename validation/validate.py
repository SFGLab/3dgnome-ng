#!/usr/bin/env python3
"""Validation harness — prove a 3dgnome run produces sensible structure.

Drives the *public* gnome3d modelling API (``Settings`` / ``ContactData`` / ``simulate``)
exactly as a user would, then scores the output ensemble with the resolution-independent
metrics in ``validation/metrics.py`` (see ``docs/validation.md``). Two modes:

  Report a single config's ensemble:
      python -m validation.validate --config data/GM12878/config.ini \\
          --region chr1:18288319-20307135 -n 5

  Prove a divergence makes sense (run flags-OFF vs flags-ON on the SAME data,
  compare). The headline test for excluded volume / confinement is that they
  REDUCE the "physically impossible overlaps" the 2016 paper admitted, without
  degrading self-consistency or the polymer scaling laws:
      python -m validation.validate --config data/GM12878/config.ini \\
          --region chr1:18288319-20307135 -n 5 --prove ev

  --prove {ev, confinement, dynamic, all}

Settings can also be built from scratch with ``Settings.from_dict`` (no .ini) — see
``--data-dir``/``--anchors``/... below, or use it directly in a notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.io import load_singletons, parse_region
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BedRegion

# Make `validation` importable when run as a script (`python validation/validate.py`), not only
# as a module (`python -m validation.validate`). Must precede the first-party imports below.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation import metrics  # noqa: E402
from validation.cell_config import settings_for_cell  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
PASS, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"

# Which settings attributes each --prove target toggles (smooth-stage gates, where the
# divergences actually act). Baseline forces them all False; treatment forces them True.
PROVE_FLAGS: dict[str, dict[str, bool]] = {
    "ev": {"use_excluded_volume": True, "exclusion_apply_to_smooth": True},
    "confinement": {"use_confinement": True, "confinement_apply_to_smooth": True},
    "dynamic": {"use_dynamic_loop_density": True},
}


def _chrs_and_region(region: str) -> tuple[list[str], BedRegion | None]:
    bed = parse_region(region)
    if bed is None:
        chrom = region.strip()
        if not chrom:
            raise ValueError(f"cannot parse region: {region!r}")
        return [chrom], None
    return [bed.chr], bed


# Config modification is centralized in cell_config; re-exported here under the existing name so
# validate's callers (and sweep) keep working without touching Settings attributes directly.
from validation.cell_config import apply_flags as _apply_flags  # noqa: E402


def run_ensemble(
    s: Settings,
    data: ContactData,
    chrs_list: list[str],
    region: BedRegion | None,
    n: int,
) -> list[list[BeadOut]]:
    """One ensemble via the public ``simulate`` API; beads for the single region's chr."""
    from gnome3d.simulate import simulate

    structures = simulate(s, data, chrs_list, n, region=region)
    return [per_chr[chrs_list[0]] for per_chr in structures]


def load_contacts(
    s: Settings, chrs_list: list[str], region: BedRegion | None
) -> list[tuple[int, int, float]]:
    """Input singleton contacts (genomic pos + score) for V1, via the public loader."""
    path = s.data_path(s.data_singletons)
    raw = load_singletons(path, set(chrs_list), region)
    return [(p1, p2, float(sc)) for _c1, p1, _c2, p2, sc in raw]


def load_chiapet_contacts(
    s: Settings, chrs_list: list[str], region: BedRegion | None
) -> list[tuple[int, int, float]]:
    """FULL input ChIA-PET contacts (singleton weak contacts + cluster loop-arcs) as
    (pos_a, pos_b, score) — the model's input heat map, for the V4 cross-data (ChIA-PET vs Hi-C)
    check. Loops (arcs) are the strong PET clusters; singletons the dense Hi-C-like background."""
    from gnome3d.io import load_arcs

    chr_set = set(chrs_list)
    out = load_contacts(s, chrs_list, region)  # singletons
    arcs, _long = load_arcs(s.data_path(s.data_pet_clusters), chr_set, region)
    for chrom in chrs_list:
        out.extend((a.start, a.end, float(a.score)) for a in arcs.get(chrom, []))
    return out


def summarize(
    ensemble: list[list[BeadOut]],
    contacts: list[tuple[int, int, float]],
    radius: float,
    skip: int,
    overlap_norm_bp: int = 5000,
) -> dict[str, float]:
    """Average the per-structure metrics across the FULL ensemble (+ V3 diversity). Reports BOTH the
    raw ``overlap_frac`` (bead-density-dependent) and ``overlap_frac_norm`` — overlaps after coarse-
    graining to ``overlap_norm_bp`` bins, so structures at different bead resolutions are comparable
    (use the _norm one for ref-vs-tuned overlap claims).

    PERF: the overlap / distance-scaling / contact-probability metrics all reduce the SAME
    upper-triangle pairwise distances, and the genomic separations are constant across the ensemble
    (shared mids). So we compute the separations ONCE and one ``pdist`` per structure (condensed, in
    triu(n,1) order), then derive all three by reduction — instead of 3 full O(N²) ``_pairwise``
    passes per structure. Exact, all structures kept (no subsampling)."""
    from scipy.spatial.distance import pdist

    coords0, mids0 = metrics.to_arrays(ensemble[0])
    n = len(coords0)
    iu, ju = np.triu_indices(n, k=1)
    sep = np.abs(mids0[iu] - mids0[ju]).astype(np.float64)  # constant across ensemble
    nonbond = (ju - iu) > skip  # exclude |i-j| <= skip (bonded neighbours), matches overlap_fraction

    rgs, bonds, overlaps, ov_norm, rhos, dscals, cprobs, extents = [], [], [], [], [], [], [], []
    coords_list = []
    for beads in ensemble:
        coords, mids = metrics.to_arrays(beads)
        coords_list.append(coords)
        rgs.append(metrics.radius_of_gyration(coords))
        bonds.append(float(np.median(metrics.bond_lengths(coords))))
        d = pdist(coords)  # condensed upper-tri distances, SAME order as triu(n,1) — computed ONCE
        overlaps.append(float((d[nonbond] < radius).mean()) if nonbond.any() else 0.0)
        ov_norm.append(metrics.overlap_fraction_binned(coords, mids, overlap_norm_bp, skip_neighbors=skip)[0])
        extents.append(metrics.max_extent(coords))
        rho, _ = metrics.self_consistency(coords, mids, contacts)
        rhos.append(rho)
        dscals.append(metrics._loglog_bins(sep, d, 20)[2])
        cprobs.append(metrics._loglog_bins(sep, (d < radius).astype(np.float64), 20)[2])
    dab = metrics.dab_matrix(coords_list)
    nanmean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")
    return {
        "n_beads": float(len(ensemble[0])),
        "rg": nanmean(rgs),
        "bond": nanmean(bonds),
        "overlap_frac": nanmean(overlaps),
        "overlap_frac_norm": nanmean(ov_norm),
        "max_extent": nanmean(extents),
        "selfconsistency_rho": nanmean(rhos),
        "dist_scaling_exp": nanmean(dscals),
        "contact_prob_exp": nanmean(cprobs),
        "diversity_dab": metrics.ensemble_diversity(dab),
    }


def _fmt(label: str, value: float, unit: str = "") -> str:
    return f"    {label:<26}{value:10.4f} {unit}"


def print_single(name: str, m: dict[str, float]) -> None:
    print(f"\n  [{name}]  {int(m['n_beads'])} beads/structure")
    print(_fmt("Rg", m["rg"]))
    print(_fmt("median bond length", m["bond"]))
    print(_fmt("max extent (centroid)", m["max_extent"]))
    print(_fmt("overlap fraction", m["overlap_frac"], "(non-bonded pairs < radius)"))
    print(_fmt("V1 self-consistency rho", m["selfconsistency_rho"], "(want < 0)"))
    print(_fmt("V2 distance scaling exp", m["dist_scaling_exp"], "(want > 0)"))
    print(_fmt("V2 contact-prob exp", m["contact_prob_exp"], "(want < 0)"))
    print(_fmt("V3 ensemble diversity", m["diversity_dab"], "(median d_AB)"))


def _verdict(ok: bool, warn: bool = False) -> str:
    return WARN if warn else (PASS if ok else FAIL)


def print_comparison(target: str, base: dict[str, float], treat: dict[str, float]) -> bool:
    print(f"\n{'=' * 70}\n  PROVE '{target}': flags OFF (baseline) vs ON (treatment)\n{'=' * 70}")
    print_single("baseline (off)", base)
    print_single("treatment (on)", treat)

    results: list[bool] = []
    print("\n  [verdict]")

    # Each divergence has its OWN success axis:
    #   ev / all  -> excluded volume should REDUCE overlaps (its purpose).
    #   confinement -> a containment envelope: should REDUCE spatial extent (Rg / max extent).
    #                  It is NOT an anti-overlap term — compaction may even raise overlaps
    #                  slightly, so overlaps are reported as info only (pair it with EV).
    #   dynamic -> changes bead spacing; overlaps should at least not inflate.
    ov_b, ov_t = base["overlap_frac"], treat["overlap_frac"]
    ext_b, ext_t = base["max_extent"], treat["max_extent"]
    if target == "confinement":
        ok = ext_t <= ext_b + 1e-9
        print(
            f"  {_verdict(ok)}  max extent {ext_b:.2f} -> {ext_t:.2f}"
            f"  ({'reduced' if ext_t < ext_b else 'not reduced'} — confinement should compact)"
        )
        results.append(ok)
        print(
            f"  {WARN}  overlaps {ov_b:.4f} -> {ov_t:.4f}  (info: confinement alone needn't cut these)"
        )
    elif target in ("ev", "all"):
        ok = ov_t <= ov_b + 1e-9
        print(
            f"  {_verdict(ok)}  overlaps {ov_b:.4f} -> {ov_t:.4f}"
            f"  ({'reduced' if ov_t < ov_b else 'not reduced'} — excluded volume should cut overlaps)"
        )
        results.append(ok)
    else:  # dynamic
        ok = ov_t <= ov_b * 1.5 + 1e-9
        print(f"  {_verdict(ok)}  overlaps {ov_b:.4f} -> {ov_t:.4f}  (should not inflate)")
        results.append(ok)

    # Self-consistency must not degrade (rho stays comparably negative).
    rb, rt = base["selfconsistency_rho"], treat["selfconsistency_rho"]
    if np.isfinite(rb) and np.isfinite(rt):
        ok = rt <= rb + 0.10  # treatment no more than 0.10 worse (less negative)
        print(f"  {_verdict(ok)}  V1 self-consistency {rb:+.3f} -> {rt:+.3f}  (must not degrade)")
        results.append(ok)
    else:
        print(f"  {WARN}  V1 self-consistency unavailable (too few in-region contacts)")

    # Scaling laws stay in sane polymer bands in BOTH runs.
    for key, lo, hi, want in (
        ("dist_scaling_exp", 0.05, 1.0, "distance grows with separation"),
        ("contact_prob_exp", -2.5, -0.2, "contacts decay with separation"),
    ):
        vb, vt = base[key], treat[key]
        ok = bool(np.isfinite(vt) and lo <= vt <= hi)
        print(
            f"  {_verdict(ok, warn=not np.isfinite(vt))}  V2 {key} {vb:+.3f} -> {vt:+.3f}"
            f"  (sane: [{lo}, {hi}]; {want})"
        )
        if np.isfinite(vt):
            results.append(ok)

    # Diversity should not collapse to ~0 or explode.
    db, dt = base["diversity_dab"], treat["diversity_dab"]
    if np.isfinite(dt):
        ok = dt > 1e-6
        print(f"  {_verdict(ok)}  V3 diversity {db:.4f} -> {dt:.4f}  (must not collapse)")
        results.append(ok)

    all_ok = all(results)
    print(f"\n  {'=' * 66}\n  {target}: {PASS if all_ok else FAIL}")
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="3dgnome structure-validation harness")
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
        help="MC schedule: fast/balanced for quick checks, full for real runs",
    )
    p.add_argument("--region", required=True, help="chr:start-end or chr name")
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=100,
        help="ensemble size — 3dgnome ensembles need >=100; use small n only for quick checks",
    )
    p.add_argument(
        "--prove",
        choices=["ev", "confinement", "dynamic", "all"],
        help="comparison mode: run flags OFF vs ON and judge the divergence",
    )
    p.add_argument(
        "--contact-radius",
        type=float,
        default=None,
        help="overlap / contact-prob radius (default: median baseline bond length)",
    )
    p.add_argument(
        "--skip-neighbors", type=int, default=1, help="exclude |i-j|<=this as bonded (default 1)"
    )
    args = p.parse_args()

    chrs_list, region = _chrs_and_region(args.region)

    base_settings = settings_for_cell(args.cell, args.data_root, args.quality)
    log.setup(base_settings.output_level, log_file=base_settings.log_file or None)

    # Load contacts once (independent of the toggled physics flags).
    data = ContactData.from_files(base_settings, chrs_list, region)
    contacts = load_contacts(base_settings, chrs_list, region)
    print(
        f"[validate] region={args.region}  ensemble={args.n_structures}  contacts={len(contacts)}"
    )

    def radius_for(ensemble: list[list[BeadOut]]) -> float:
        if args.contact_radius is not None:
            return args.contact_radius
        coords, _ = metrics.to_arrays(ensemble[0])
        return float(np.median(metrics.bond_lengths(coords)))

    if not args.prove:
        ens = run_ensemble(base_settings, data, chrs_list, region, args.n_structures)
        radius = radius_for(ens)
        print(f"[validate] contact/overlap radius = {radius:.3f}")
        print_single(args.cell, summarize(ens, contacts, radius, args.skip_neighbors))
        return

    targets = ["ev", "confinement", "dynamic"] if args.prove == "all" else [args.prove]
    # Baseline = all proved flags OFF; build once and reuse its radius for both runs.
    off = {a: False for t in targets for a in PROVE_FLAGS[t]}
    base_s = _apply_flags(base_settings, off)
    base_ens = run_ensemble(base_s, data, chrs_list, region, args.n_structures)
    radius = radius_for(base_ens)
    print(f"[validate] contact/overlap radius = {radius:.3f}")
    base_m = summarize(base_ens, contacts, radius, args.skip_neighbors)

    all_ok = True
    for target in targets:
        treat_s = _apply_flags(base_s, PROVE_FLAGS[target])
        treat_ens = run_ensemble(treat_s, data, chrs_list, region, args.n_structures)
        treat_m = summarize(treat_ens, contacts, radius, args.skip_neighbors)
        all_ok &= print_comparison(target, base_m, treat_m)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
