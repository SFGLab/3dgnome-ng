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
    p.add_argument("--cif-anchors-only", action="store_true",
                   help="When writing CIF, include only anchor beads (not linkers)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        settings = Settings.from_ini(args.config)
    else:
        settings = Settings()

    if args.device:
        settings.device = args.device

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
        solver.save_cif(args.output, anchors_only=args.cif_anchors_only)

    print("Done.")


if __name__ == "__main__":
    main()
