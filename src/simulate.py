"""
src/simulate.py  —  Public entry point for 3dgnome-torch.

Exposes run_region(), the interface consumed by harness/integration.py.
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
    Run MC reconstruction for the given genomic region and return n_structures
    independent conformations.

    Parameters
    ----------
    config_path : str
        Path to the .ini config file (same format as the C++ binary uses).
    region : str
        Genomic region in 'chr:start-end' format (e.g. 'chr1:18288319-20307135').
    n_structures : int
        Number of independent MC runs to perform.
    data_dir : str, optional
        Override the data_dir from the config file.  Useful when the config
        was written for a different machine (e.g. the bundled GM12878/config.ini
        has data_dir = /Projects/GM12878/).  If None, uses the value in the config.

    Returns
    -------
    list of list of (midpoint_bp, x, y, z)
        One entry per structure.  Each entry is a list of tuples sorted by
        genomic midpoint, matching the anchor-level leaf beads produced by
        the C++ binary at -v 2.
    """
    # Resolve the src package — when called from harness/ with sys.path pointing
    # at src/, the relative imports inside solver.py etc. use the package name
    # "src".  We need to make sure the parent directory is importable.
    src_dir = Path(__file__).parent
    repo_root = src_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local imports after path fix
    from src.settings import Settings
    from src.io import parse_region
    from src.solver import Solver
    from src.energy import get_device

    print(f"[simulate] device: {get_device()}")

    # 1. Parse region string
    bed_region = parse_region(region)
    if bed_region is None:
        raise ValueError(f"Cannot parse region: {region!r}")
    chrs_list = [bed_region.chr]

    # 2. Load settings
    s = Settings()
    ok = s.load_ini(config_path)
    if not ok:
        raise RuntimeError(f"Failed to load config: {config_path!r}")

    if data_dir is not None:
        s.data_dir = str(data_dir)

    # 3. Load data and build hierarchy once (shared across runs)
    solver = Solver(s)
    solver.set_contact_data(chrs_list, bed_region, s.data_dir)

    # 4. Run n_structures independent MC trajectories
    structures = []
    for i in range(n_structures):
        print(f"\n[simulate] structure {i + 1}/{n_structures}")

        # Reset positions to IB-level positions (re-randomised each run)
        solver.reconstruct_heatmap()
        solver.reconstruct_arcs()

        beads = solver.get_leaf_positions(bed_region.chr)
        if not beads:
            raise RuntimeError(
                f"Structure {i + 1}: no leaf beads returned — "
                "check hierarchy building or anchor loading."
            )
        structures.append(beads)

    return structures
