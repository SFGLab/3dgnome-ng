"""
High-level entry point for 3dgnome-ng.

Thin wrappers around the Settings / ContactData / Solver pipeline.
"""

from __future__ import annotations

from .data import ContactData
from .io import parse_chrs_arg, parse_region
from .settings import Settings
from .solver import Solver
from .types import BeadOut


def run_region(
    config_path: str,
    region: str,
    n_structures: int,
    data_dir: str | None = None,
) -> list[list[BeadOut]]:
    """
    Run MC reconstruction for the given single genomic region or chromosome.

    Parameters
    ----------
    config_path : str
        Path to the .ini config file.
    region : str
        Genomic region in 'chr:start-end' format (e.g. 'chr1:18288319-20307135')
        or a single chromosome name (e.g. 'chr14').  For multi-chromosome runs
        use `run_genome` instead.
    n_structures : int
        Number of independent MC runs to perform.
    data_dir : str, optional
        Override data_dir from the config file.

    Returns
    -------
    list of list of BeadOut = (start_bp, end_bp, x, y, z)
        One entry per structure, sorted by genomic start.
        Includes both anchor beads and loop_density subanchor beads.
    """
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

    structures: list[list[BeadOut]] = []
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
) -> list[list[BeadOut]]:
    """Run MC reconstruction for an entire chromosome."""
    return run_region(config_path, chrom, n_structures, data_dir=data_dir)


def run_genome(
    config_path: str,
    region: str = "",
    n_structures: int = 1,
    data_dir: str | None = None,
) -> list[dict[str, list[BeadOut]]]:
    """
    Run MC reconstruction across multiple chromosomes (or whole genome).

    Parameters
    ----------
    config_path : str
        Path to the .ini config file.
    region : str
        Accepts the same syntax as the CLI --region flag:
          - empty string -> whole human genome (chr1..chr22, chrX)
          - 'chr14'      -> single chromosome
          - 'chr1,chr3,chrX'  -> comma-separated list
          - 'chr1-chr22,chrX' -> range + extras
          - 'chr14:18288319-20307135' -> single sub-chromosomal region
        Matches the Reference `-c` flag's accepted forms.
    n_structures : int
        Number of independent MC runs to perform.
    data_dir : str, optional
        Override data_dir from the config file.

    Returns
    -------
    list of dict[chr -> list[BeadOut]]
        One entry per structure.  Each dict maps chromosome name to its
        sorted bead list.
    """
    chrs_list, bed_region = parse_chrs_arg(region)
    if not chrs_list:
        raise ValueError(f"Cannot parse region: {region!r}")

    s = Settings()
    if not s.load_ini(config_path):
        raise RuntimeError(f"Failed to load config: {config_path!r}")
    if data_dir is not None:
        s.data_dir = str(data_dir)

    data = ContactData.from_files(s, chrs_list, bed_region)

    solver = Solver(s)
    solver.load(data, chrs_list, bed_region)

    structures: list[dict[str, list[BeadOut]]] = []
    for i in range(n_structures):
        print(f"\n[simulate] structure {i + 1}/{n_structures}")
        solver.reconstruct_heatmap()
        solver.reconstruct_arcs()
        per_chr: dict[str, list[BeadOut]] = {}
        any_beads = False
        for chr_ in chrs_list:
            beads = solver.get_leaf_positions(chr_)
            if beads:
                per_chr[chr_] = beads
                any_beads = True
        if not any_beads:
            raise RuntimeError(f"Structure {i + 1}: no leaf beads from any chromosome")
        structures.append(per_chr)

    return structures
