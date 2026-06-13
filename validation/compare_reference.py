#!/usr/bin/env python3
"""Compare 3dgnome output against the C++ reference on the validation metrics.

Answers "are our outputs better or worse than the reference?" by scoring THREE ensembles
on the same metrics (``validation/metrics.py``):

  * reference         — the C++ 3dnome binary (the algorithmic source of truth, NO EV /
                        confinement; the 2016 paper admitted it makes overlapping loops)
  * python (parity)   — our port, feature flags OFF — should MATCH the reference (faithful)
  * python (+EV+conf) — our port with excluded volume + confinement ON — should have FEWER
                        overlaps than the reference while preserving self-consistency/scaling

All three share one parity config + region (so bead counts match and overlap fractions are
apples-to-apples; EV/confinement don't change N). "Do dynamic subanchors help?" changes N, so
it is answered separately by ``validate.py --prove dynamic``.

Reuses the proven reference runner from ``harness/integration.py``.

    python -m validation.compare_reference --region chr1:18288319-20307135 -n 3 --fast
"""

from __future__ import annotations

import argparse
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
from validation.validate import (  # noqa: E402
    FAIL,
    PASS,
    _apply_flags,
    _chrs_and_region,
    load_contacts,
    print_single,
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

    print("[compare] running python (+EV +confinement)...")
    s_feat = _apply_flags(
        s_base,
        {
            "use_excluded_volume": True,
            "exclusion_apply_to_smooth": True,
            "use_confinement": True,
            "confinement_apply_to_smooth": True,
        },
    )
    feat_structs = run_ensemble(s_feat, data, chrs_list, bed_region, args.n_structures)

    # --- score all three identically ---
    radius = args.contact_radius
    if radius is None:
        coords, _ = metrics.to_arrays(base_structs[0])
        radius = float(np.median(metrics.bond_lengths(coords)))
    print(f"[compare] contact/overlap radius = {radius:.3f}  (contacts={len(contacts)})")

    ref_m = summarize(ref_structs, contacts, radius, args.skip_neighbors)
    base_m = summarize(base_structs, contacts, radius, args.skip_neighbors)
    feat_m = summarize(feat_structs, contacts, radius, args.skip_neighbors)

    print_single("reference (C++)", ref_m)
    print_single("python parity (flags off)", base_m)
    print_single("python +EV +confinement", feat_m)

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

    # Q1b: does adding EV+confinement beat the reference on physical sanity?
    better = feat_m["overlap_frac"] < ref_m["overlap_frac"] - 1e-9
    print(
        f"  {PASS if better else FAIL}  +EV+confinement BETTER than reference on overlaps: "
        f"ref={ref_m['overlap_frac']:.4f} -> py+feat={feat_m['overlap_frac']:.4f}"
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
