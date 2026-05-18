"""
src/simulate.py - High-level entry point for 3dgnome-ng.

Thin wrappers around the Settings / ContactData / Solver pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_region(
    config_path: str,
    region: str,
    n_structures: int,
    data_dir: str | None = None,
) -> list:
    """
    Run MC reconstruction for the given genomic region.

    Parameters
    ----------
    config_path : str
        Path to the .ini config file.
    region : str
        Genomic region in 'chr:start-end' format, e.g. 'chr1:18288319-20307135'.
    n_structures : int
        Number of independent MC runs to perform.
    data_dir : str, optional
        Override data_dir from the config file.

    Returns
    -------
    list of list of (midpoint_bp, x, y, z)
        One entry per structure, sorted by genomic midpoint.
        Includes both anchor beads and loop_density subanchor beads.
    """
    src_dir = Path(__file__).parent
    repo_root = src_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.settings import Settings
    from src.io import parse_region
    from src.data import ContactData
    from src.solver import Solver
    from src.energy import get_device

    print(f"[simulate] device: {get_device()}")

    bed_region = parse_region(region)
    if bed_region is None:
        chrom = region.strip()
        if not chrom:
            raise ValueError(f"Cannot parse region: {region!r}")
        chrs_list = [chrom]
    else:
        chrs_list = [bed_region.chr]

    s = Settings()
    if not s.load_ini(config_path):
        raise RuntimeError(f"Failed to load config: {config_path!r}")
    if data_dir is not None:
        s.data_dir = str(data_dir)

    data = ContactData.from_files(s, chrs_list, bed_region)

    solver = Solver(s)
    solver.load(data, chrs_list, bed_region)

    structures = []
    for i in range(n_structures):
        print(f"\n[simulate] structure {i + 1}/{n_structures}")
        solver.reconstruct_heatmap()
        solver.reconstruct_arcs()
        beads = solver.get_leaf_positions(chrs_list[0])
        if not beads:
            raise RuntimeError(
                f"Structure {i + 1}: no leaf beads returned - "
                "check hierarchy building or anchor loading."
            )
        structures.append(beads)

    return structures


def run_chromosome(
    config_path: str,
    chrom: str,
    n_structures: int,
    data_dir: str | None = None,
) -> list:
    """Run MC reconstruction for an entire chromosome."""
    return run_region(config_path, chrom, n_structures, data_dir=data_dir)
