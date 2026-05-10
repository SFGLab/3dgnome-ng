"""
cudaMMC-PyTorch: 3D chromatin structure reconstruction from ChIA-PET data.

Usage:
    python main.py \\
        --anchors anchors.bed \\
        --clusters pet_clusters.bedpe \\
        [--singletons singletons.bedpe] \\
        [--config config.ini] \\
        [--output output/prefix] \\
        [--device cuda|mlp|cpu]
"""

import argparse
import sys

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
    p.add_argument("--hcm", action="store_true",
                   help="Also save anchor positions in HCM format")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        settings = Settings.from_ini(args.config)
    else:
        settings = Settings()

    if args.device:
        settings.device = args.device

    print(f"Device: {settings.device}")
    print(f"Anchors: {args.anchors}")
    print(f"Clusters: {args.clusters}")
    if args.singletons:
        print(f"Singletons: {args.singletons}")

    solver = LooperSolver(settings)
    solver.run(
        anchors_bed=args.anchors,
        pet_clusters_bedpe=args.clusters,
        singletons_bedpe=args.singletons,
        output_prefix=args.output,
    )

    if args.hcm:
        solver.save_hcm(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
