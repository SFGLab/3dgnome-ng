#!/usr/bin/env python3
"""
main.py  —  Explicit 3dgnome-torch workflow.

Mirrors exactly what src/simulate.run_region() does, but written out
step by step so each stage can be inspected or swapped out.

Usage:
    python main.py --config data/GM12878/config.ini \
                   --region chr1:18288319-20307135  \
                   --n 1                            \
                   --out out/
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.settings import Settings
from src.io import parse_region, write_cif
from src.data import ContactData
from src.solver import Solver
from src.energy import get_device


def main():
    parser = argparse.ArgumentParser(description="3dgnome-torch structure prediction")
    parser.add_argument("--config", required=True, help="Path to config.ini")
    parser.add_argument("--region", required=True,
                        help="Genomic region, e.g. chr1:18288319-20307135")
    parser.add_argument("-n", "--n-structures", type=int, default=1,
                        help="Number of independent structures to generate (default 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override data_dir from config")
    parser.add_argument("--out", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    print(f"[main] device: {get_device()}")

    # 1. Parse region
    bed_region = parse_region(args.region)
    if bed_region is None:
        chrom = args.region.strip()
        if not chrom:
            sys.exit(f"Cannot parse region: {args.region!r}")
        chrs_list = [chrom]
        bed_region = None
    else:
        chrs_list = [bed_region.chr]

    # 2. Load settings
    s = Settings()
    if not s.load_ini(args.config):
        sys.exit(f"Failed to load config: {args.config!r}")
    if args.data_dir:
        s.data_dir = args.data_dir
    print(f"[main] config: {args.config}  data_dir: {s.data_dir}")

    # 3. Load all input data from files
    data = ContactData.from_files(s, chrs_list, bed_region)

    # 4. Build hierarchy (done once, shared across all structures)
    solver = Solver(s)
    solver.load(data, chrs_list, bed_region)

    # 5. Run n independent MC trajectories
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_base = args.region.replace(":", "_").replace("-", "_")

    for i in range(args.n_structures):
        print(f"\n[main] structure {i + 1}/{args.n_structures}")
        solver.reconstruct_heatmap()
        solver.reconstruct_arcs()

        beads = solver.get_leaf_positions(chrs_list[0])
        if not beads:
            sys.exit(f"Structure {i + 1}: no leaf beads returned")

        cif_path = out_dir / f"{entry_base}_s{i + 1}.cif"
        write_cif(str(cif_path), beads, entry_id=f"{entry_base}_s{i + 1}")
        print(f"[main] wrote {cif_path}  ({len(beads)} beads)")

    print(f"\n[main] done — {args.n_structures} structure(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
