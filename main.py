"""
cudaMMC-PyTorch: 3D chromatin structure reconstruction from ChIA-PET data.

Usage:
    python main.py \\
        --anchors anchors.bed \\
        --clusters pet_clusters.bedpe \\
        [--singletons singletons.bedpe] \\
        [--config config.ini] \\
        [--output output/prefix] \\
        [--device cuda|mps|cpu] \\
        [--chromosomes 'chr1-chr22,chrX']
"""

import argparse
import sys

from src.regions import default_regions, parse_region_spec
from src.settings import Settings
from src.solver import LooperSolver


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct 3D chromatin structure from ChIA-PET data."
    )
    p.add_argument("--anchors", required=True,
                   help="BED file with CTCF anchors")
    p.add_argument("--clusters", required=True,
                   help="BEDPE file with PET clusters")
    p.add_argument("--singletons", default=None,
                   help="BEDPE file with singleton reads (optional)")
    p.add_argument("--config", default=None,
                   help="INI configuration file (optional)")
    p.add_argument("--output", default="output/structure",
                   help="Output prefix (default: output/structure)")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Compute device (default: cuda or mps if available, otherwise cpu)")
    p.add_argument("--factor", type=int, default=0,
                   help="Factor index for multi-factor ChIA-PET (default: 0)")
    p.add_argument("--chromosomes", default="chr1-chr22,chrX",
                   metavar="SPEC",
                   help=("Chromosomes or regions to include, comma-separated. "
                         "Supports ranges (chr1-chr22), individual names (chrX), "
                         "and coordinate windows (chr14:1:2500000). "
                         "Default: 'chr1-chr22,chrX'"))
    p.add_argument("--hcm", action="store_true",
                   help="Also save anchor positions in HCM format")
    p.add_argument("--cif", action="store_true",
                   help="Also save positions in mmCIF format for 3D viewers")
    p.add_argument("--cif-all-beads", action="store_true",
                   help="When writing CIF, include all beads (not just anchors)")
    p.add_argument("--debug-dump-stages", default=None, metavar="DIR",
                   help=("Diagnostic: write per-IB CIFs at each MC stage "
                         "(arc, densify, smooth) plus arc/chain distance stats "
                         "into DIR.  Used to localise where the loop-rosette "
                         "structure fails to form."))
    p.add_argument("--debug-max-ibs", type=int, default=0, metavar="N",
                   help=("Diagnostic: stop after processing N interaction "
                         "blocks (0 = no limit).  Speeds up iteration when "
                         "debugging per-IB structure issues."))
    p.add_argument("--debug-max-mc-seconds", type=float, default=0.0,
                   metavar="SEC",
                   help=("Diagnostic: hard wall-time cap (seconds) per MC "
                         "phase per restart (sets mc_max_seconds_arcs and "
                         "mc_max_seconds_smooth).  Lets the pipeline finish "
                         "in a known budget for visual inspection."))
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect ``config.ini`` next to --anchors if --config not provided.
    # cudaMMC always runs with an INI; without one, ``data_segments_split`` is
    # empty and ``find_segments`` Branch B (cpp:964-994) returns gaps unchanged
    # — on dense ChIA-PET data this collapses the entire chromosome into a
    # single IB (arcs_cnt rarely hits zero), producing 2 segments and a
    # 20 k-anchor IB.  Match cudaMMC's intended workflow by loading the
    # adjacent config automatically.
    import os as _os
    cfg_path = args.config
    if cfg_path is None:
        cand = _os.path.join(_os.path.dirname(_os.path.abspath(args.anchors)),
                             "config.ini")
        if _os.path.exists(cand):
            print(f"Auto-loading config: {cand}")
            cfg_path = cand
    if cfg_path:
        settings = Settings.from_ini(cfg_path)
    else:
        settings = Settings()
        print("WARNING: no config.ini found — running with built-in defaults. "
              "Whole-chromosome runs require a `segment_split` breakpoint BED "
              "(see cudaMMC LooperSolver.cpp:911-962).")

    if args.device:
        settings.device = args.device

    if args.debug_dump_stages:
        settings.debug_dump_stages = args.debug_dump_stages
        import os as _os2
        _os2.makedirs(args.debug_dump_stages, exist_ok=True)
        print(f"Debug stage dumps → {args.debug_dump_stages}")
    if args.debug_max_ibs:
        settings.debug_max_ibs = args.debug_max_ibs
        print(f"Debug: stopping after {args.debug_max_ibs} IB(s)")
    if args.debug_max_mc_seconds > 0.0:
        settings.mc_max_seconds_arcs = args.debug_max_mc_seconds
        settings.mc_max_seconds_smooth = args.debug_max_mc_seconds
        print(f"Debug: MC wall-time cap = {args.debug_max_mc_seconds}s/phase")

    try:
        regions = parse_region_spec(args.chromosomes)
    except ValueError as e:
        print(f"Error in --chromosomes: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {settings.device}")
    print(f"Anchors: {args.anchors}")
    print(f"Clusters: {args.clusters}")
    if args.singletons:
        print(f"Singletons: {args.singletons}")
    print(f"Regions: {', '.join(str(r) for r in regions)}")

    solver = LooperSolver(settings, regions=regions)
    solver.run(
        anchors_bed=args.anchors,
        pet_clusters_bedpe=args.clusters,
        singletons_bedpe=args.singletons,
        output_prefix=args.output,
        factor=args.factor,
    )

    if args.hcm:
        solver.save_hcm(args.output)

    if args.cif:
        solver.save_cif(args.output, anchors_only=not args.cif_all_beads)

    print("Done.")


if __name__ == "__main__":
    main()
