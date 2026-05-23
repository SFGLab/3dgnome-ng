"""
Command-line interface for 3dgnome-ng.

gnome3d-ng --config data/GM12878/config.ini \
           --region chr1:18288319-20307135   \
           --n 1                             \
           --out out/
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gnome3d.data import ContactData
from gnome3d.io import parse_region, write_cif
from gnome3d.settings import Settings
from gnome3d.solver import Solver
from gnome3d.types import BedRegion


def _run_structure(
    i: int,
    n: int,
    s: Settings,
    data: ContactData,
    chrs_list: list[str],
    region: BedRegion | None,
    out_dir: Path,
    entry_base: str,
) -> int:
    """Build and write one independent structure. Returns bead count."""
    print(f"\n[main] structure {i + 1}/{n}")
    solver = Solver(s)
    solver.load(data, chrs_list, region)
    solver.reconstruct_heatmap()
    solver.reconstruct_arcs()

    beads = solver.get_leaf_positions(chrs_list[0])
    if not beads:
        raise RuntimeError(f"Structure {i + 1}: no leaf beads returned")

    cif_path = out_dir / f"{entry_base}_s{i + 1}.cif"
    write_cif(str(cif_path), beads, entry_id=f"{entry_base}_s{i + 1}")
    print(f"[main] wrote {cif_path}  ({len(beads)} beads)")
    return len(beads)


def main() -> None:
    parser = argparse.ArgumentParser(description="3dgnome-ng structure prediction")
    parser.add_argument("--config", required=True, help="Path to config.ini")
    parser.add_argument("--region", required=True,
                        help="Genomic region, e.g. chr1:18288319-20307135")
    parser.add_argument("-n", "--n-structures", type=int, default=1,
                        help="Number of independent structures to generate (default 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override data_dir from config")
    parser.add_argument("--out", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    bed_region = parse_region(args.region)
    if bed_region is None:
        chrom = args.region.strip()
        if not chrom:
            sys.exit(f"Cannot parse region: {args.region!r}")
        chrs_list = [chrom]
        bed_region = None
    else:
        chrs_list = [bed_region.chr]

    s = Settings()
    if not s.load_ini(args.config):
        sys.exit(f"Failed to load config: {args.config!r}")
    if args.data_dir:
        s.data_dir = args.data_dir
    print(f"[main] config: {args.config}  data_dir: {s.data_dir}")

    data = ContactData.from_files(s, chrs_list, bed_region)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_base = args.region.replace(":", "_").replace("-", "_")

    # Numba releases the GIL during MC, so threads genuinely overlap.
    n = args.n_structures
    n_workers = min(n, os.cpu_count() or 1)
    print(f"[main] running {n} structure(s) with {n_workers} worker(s)")

    pool = ThreadPoolExecutor(max_workers=n_workers)
    futures = {
        pool.submit(_run_structure, i, n, s, data, chrs_list, bed_region, out_dir, entry_base): i
        for i in range(n)
    }
    try:
        for fut in as_completed(futures):
            fut.result()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    print(f"\n[main] {n} structure(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
