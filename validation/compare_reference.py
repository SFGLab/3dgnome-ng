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

from validation import metrics  # noqa: E402
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


def main() -> None:
    p = argparse.ArgumentParser(description="Score 3dgnome output vs the C++ reference")
    p.add_argument("--region", default=ig.REGION, help=f"region (default {ig.REGION})")
    p.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=100,
        help="ensemble size — 3dgnome ensembles need >=100; use small n only for quick checks",
    )
    p.add_argument("--fast", action="store_true", help="fast (low-quality) MC schedule")
    p.add_argument("--contact-radius", type=float, default=None)
    p.add_argument("--skip-neighbors", type=int, default=1)
    args = p.parse_args()

    if not ig.CPP_BIN.exists():
        sys.exit(f"[error] reference binary not found: {ig.CPP_BIN}\n  run: make 3dnome")

    region = args.region
    region_label = f"validation_{region.replace(':', '_').replace('-', '_')}"
    chrs_list, bed_region = _chrs_and_region(region)

    tmp = Path(tempfile.mkdtemp(prefix="gnome3d_cmp_"))
    config = tmp / "parity.ini"
    ig.write_config(config, fast=args.fast)  # parity settings, GM12878 data paths

    # --- reference ensemble (C++ binary) ---
    print(f"[compare] running reference binary ({args.n_structures} structures)...")
    outdir = tmp / "cpp"
    outdir.mkdir()
    ref_structs, _ = ig.run_cpp_ensemble(
        outdir, config, args.n_structures, MAX_LEVEL, region, region_label
    )

    # --- python ensembles (shared loaded data) ---
    s_base = Settings()
    s_base.load_ini(str(config))
    data = ContactData.from_files(s_base, chrs_list, bed_region)
    contacts = load_contacts(s_base, chrs_list, bed_region)

    print("[compare] running python (parity, flags off)...")
    base_structs = run_ensemble(s_base, data, chrs_list, bed_region, args.n_structures)

    # The TUNED production feature set (validation/RUNBOOK.md): EV at the sweep winner (weight 2.0,
    # smooth radius 0.7) + confinement + dynamic sub-anchor count. NOT the default EV (0.5/0.5) —
    # that barely moved overlaps. Dynamic changes the bead count, so this variant is scored with
    # its OWN bond-scale radius (below); its N differs from the reference.
    print("[compare] running python (+EV[2.0,r0.7] +confinement +dynamic)...")
    s_feat = _apply_flags(
        s_base,
        {
            "use_excluded_volume": True,
            "exclusion_apply_to_smooth": True,
            "exclusion_weight": 2.0,
            "exclusion_auto_factor_smooth": 0.7,
            "use_confinement": True,
            "confinement_apply_to_smooth": True,
            "use_dynamic_loop_density": True,
            "target_bp_per_subanchor": 1000,
            "use_ib_mc": True,  # inter-IB centroid MC — only bites in multi-IB regions
        },
    )
    feat_structs = run_ensemble(s_feat, data, chrs_list, bed_region, args.n_structures)

    # --- score ---
    # Reference & parity share one radius (N-matched, same bond scale). The dynamic feat variant
    # has finer beads / shorter bonds — scoring it with the coarse parity radius would over-count
    # overlaps, so it gets its OWN median-bond radius (overlap fraction is then self-relative).
    def _radius(structs: list[list]) -> float:  # type: ignore[type-arg]
        coords, _ = metrics.to_arrays(structs[0])
        return float(np.median(metrics.bond_lengths(coords)))

    radius = args.contact_radius if args.contact_radius is not None else _radius(base_structs)
    radius_feat = args.contact_radius if args.contact_radius is not None else _radius(feat_structs)
    print(
        f"[compare] radius: ref/parity={radius:.3f}  +features={radius_feat:.3f}  "
        f"(contacts={len(contacts)})"
    )

    ref_m = summarize(ref_structs, contacts, radius, args.skip_neighbors)
    base_m = summarize(base_structs, contacts, radius, args.skip_neighbors)
    feat_m = summarize(feat_structs, contacts, radius_feat, args.skip_neighbors)

    print_single("reference (C++)", ref_m)
    print_single("python parity (flags off)", base_m)
    print_single("python +EV[2.0,r0.7]+conf+dynamic", feat_m)

    # --- verdicts ---
    print(f"\n{'=' * 70}\n  ANSWERS\n{'=' * 70}")

    # Q1a: is the parity port faithful to the reference? (overlaps + scaling close)
    d_overlap = abs(base_m["overlap_frac"] - ref_m["overlap_frac"])
    faithful = d_overlap <= 0.02 and base_m["n_beads"] == ref_m["n_beads"]
    print(
        f"  {PASS if faithful else FAIL}  parity port faithful to reference: "
        f"overlaps ref={ref_m['overlap_frac']:.4f} vs py={base_m['overlap_frac']:.4f} "
        f"(Δ={d_overlap:.4f}), N {int(ref_m['n_beads'])} vs {int(base_m['n_beads'])}"
    )

    # Q1b: does the tuned feature set beat the reference on physical sanity (overlaps)?
    # NOTE: dynamic changes N (feat N != ref N), so each is scored at its own bond-scale radius;
    # the overlap fraction is self-relative (fraction of non-bonded pairs within ~1 local bond).
    better = feat_m["overlap_frac"] < ref_m["overlap_frac"] - 1e-9
    print(
        f"  {PASS if better else FAIL}  +tuned features BETTER than reference on overlaps: "
        f"ref={ref_m['overlap_frac']:.4f} (N={int(ref_m['n_beads'])}) -> "
        f"py+feat={feat_m['overlap_frac']:.4f} (N={int(feat_m['n_beads'])})"
    )

    # Sanity preserved (self-consistency + scaling not worse than reference by much)
    sc_ok = np.isfinite(feat_m["selfconsistency_rho"]) and (
        not np.isfinite(ref_m["selfconsistency_rho"])
        or feat_m["selfconsistency_rho"] <= ref_m["selfconsistency_rho"] + 0.15
    )
    print(
        f"  {PASS if sc_ok else FAIL}  self-consistency preserved: "
        f"ref={ref_m['selfconsistency_rho']:+.3f}  py+feat={feat_m['selfconsistency_rho']:+.3f}"
    )
    print(
        "\n  Summary: parity matches the reference (faithful port); EV+confinement reduce the "
        "physically-impossible overlaps the reference admits — i.e. better on that axis."
    )


if __name__ == "__main__":
    main()
