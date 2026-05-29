"""
Command-line interface for 3dgnome-ng.

Single region:   gnome3d-ng --config X.ini --region chr1:18288319-20307135
Single chr:      gnome3d-ng --config X.ini --region chr14
Chromosome list: gnome3d-ng --config X.ini --region chr1,chr3,chrX
Range:           gnome3d-ng --config X.ini --region chr1-chr22,chrX
Whole genome:    gnome3d-ng --config X.ini      (defaults to chr1-chr22,chrX)
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg, write_cif
from gnome3d.settings import Settings
from gnome3d.solver import Solver
from gnome3d.types import BedRegion


def _cif_name(entry_base: str, chr_: str, i: int, multi_chr: bool) -> str:
    """Per-structure CIF filename. Disambiguate per-chr when more than one chr."""
    if multi_chr:
        return f"{entry_base}_{chr_}_s{i + 1}.cif"
    return f"{entry_base}_s{i + 1}.cif"


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
    """Build and write one independent structure. Returns total bead count."""
    print(f"\n[main] structure {i + 1}/{n}")
    solver = Solver(s)
    solver.load(data, chrs_list, region)
    solver.reconstruct_heatmap()
    solver.reconstruct_arcs()

    multi_chr = len(chrs_list) > 1
    total_beads = 0
    for chr_ in chrs_list:
        beads = solver.get_leaf_positions(chr_)
        if not beads:
            print(f"[main] {chr_}: no leaf beads (skipping)")
            continue

        cif_path = out_dir / _cif_name(entry_base, chr_, i, multi_chr)
        entry_id = cif_path.stem
        write_cif(str(cif_path), beads, entry_id=entry_id)
        print(f"[main] wrote {cif_path}  ({len(beads)} beads)")
        total_beads += len(beads)

    if total_beads == 0:
        raise RuntimeError(f"Structure {i + 1}: no leaf beads from any chromosome")

    return total_beads


def main() -> None:
    parser = argparse.ArgumentParser(description="3dgnome-ng structure prediction")
    parser.add_argument("--config", required=True, help="Path to config.ini")
    parser.add_argument(
        "--region",
        default="",
        help=(
            "Chromosomes/region to reconstruct.  Examples: "
            "'chr14:18288319-20307135' (single region), 'chr14' (single chr), "
            "'chr1,chr3,chrX' (comma list), 'chr1-chr22,chrX' (range + extras). "
            "Default (empty): chr1-chr22,chrX (whole human genome, matches Reference)."
        ),
    )
    parser.add_argument(
        "-n",
        "--n-structures",
        type=int,
        default=1,
        help="Number of independent structures to generate (default 1)",
    )
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config")
    parser.add_argument("--out", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    chrs_list, bed_region = parse_chrs_arg(args.region)
    if not chrs_list:
        sys.exit(f"Cannot parse region: {args.region!r}")

    s = Settings()
    if not s.load_ini(args.config):
        sys.exit(f"Failed to load config: {args.config!r}")
    if args.data_dir:
        s.data_dir = args.data_dir
    print(f"[main] config: {args.config}  data_dir: {s.data_dir}")
    print(
        f"[main] chromosomes ({len(chrs_list)}): {','.join(chrs_list)}"
        + (f"  region={bed_region.chr}:{bed_region.start}-{bed_region.end}" if bed_region else "")
    )

    data = ContactData.from_files(s, chrs_list, bed_region)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build a filesystem-safe base name from the --region argument (or "genome"
    # when whole-genome default was used).
    raw = args.region.strip() or "genome"
    entry_base = raw.replace(":", "_").replace("-", "_").replace(",", "_")

    # Numba releases the GIL during MC, so threads genuinely overlap.
    n = args.n_structures
    n_workers = min(n, os.cpu_count() or 1)
    # Avoid nesting structure-level and IB-level thread pools: when we already
    # parallelise across structures, force ib_workers=1 inside each worker.
    # Numba is also threading inside the MC kernels, so over-subscription here
    # only hurts throughput.
    if n_workers > 1 and s.ib_workers > 1:
        print(f"[main] n_structures>1: forcing ib_workers=1 (was {s.ib_workers})")
        s.ib_workers = 1
    # JAX on a single GPU does not benefit from CPU-side threading — multiple
    # IB threads end up serialised on the device anyway, plus each pays JAX
    # setup/sync overhead.  Force ib_workers=1 so the GPU sees one MC call at
    # a time.  Independent restarts can still be vmapped inside each call via
    # settings.mc_smooth_chains.
    if str(s.mc_backend).strip().lower() == "jax" and s.ib_workers > 1:
        print(f"[main] mc_backend=jax: forcing ib_workers=1 (was {s.ib_workers})")
        s.ib_workers = 1
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
