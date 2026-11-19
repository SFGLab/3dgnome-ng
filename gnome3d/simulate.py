"""Entry points of the reconstruction."""

from __future__ import annotations

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg, parse_region
from gnome3d.reconstruct import pick_executor, reconstruct_ensemble
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BedRegion

LOG = log.get("simulate")


def simulate(
    settings: Settings,
    data: ContactData,
    chrs_list: list[str],
    n_structures: int = 1,
    region: BedRegion | None = None,
    seed_offset: int = 0,
) -> list[dict[str, list[BeadOut]]]:
    """Reconstructs `n_structures` structures from built settings and contact data, with no file I/O.

    Returns one dict of chromosome to bead list per structure. Member i is byte identical to a
    single run at `seed_offset = i * MEMBER_SEED_STRIDE`.
    """
    executor = pick_executor(settings)
    with log.step(LOG, f"reconstruct {n_structures} structure(s)"):
        raw = reconstruct_ensemble(
            settings,
            data,
            chrs_list,
            region,
            n=n_structures,
            executor=executor,
            base_seed_offset=seed_offset,
        )

    structures: list[dict[str, list[BeadOut]]] = []
    for i, per_chr in enumerate(raw):
        per_chr = {chr_: beads for chr_, beads in per_chr.items() if beads}
        if not per_chr:
            raise RuntimeError(f"Structure {i + 1}: no leaf beads from any chromosome")
        structures.append(per_chr)
    return structures


def run_region(
    config_path: str,
    region: str,
    n_structures: int,
    data_dir: str | None = None,
) -> list[list[BeadOut]]:
    """Reconstructs one region or one chromosome from a config file.

    Parameters
    ----------
    config_path
        Path of the .ini config file.
    region
        'chr:start-end' or a chromosome such as 'chr14'.
    n_structures
        Number of structures.
    data_dir
        Replaces `data_dir` of the config file.

    Returns
    -------
    list of list of BeadOut
        One list per structure, sorted by genomic start, anchors and subanchors.
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

    log.setup(s.output_level, log_file=s.log_file or None)

    data = ContactData.from_files(s, chrs_list, bed_region)
    structures = simulate(s, data, chrs_list, n_structures, region=bed_region)
    return [per_chr[chrs_list[0]] for per_chr in structures]


def run_chromosome(
    config_path: str,
    chrom: str,
    n_structures: int,
    data_dir: str | None = None,
) -> list[list[BeadOut]]:
    """Reconstructs a whole chromosome from a config file."""
    return run_region(config_path, chrom, n_structures, data_dir=data_dir)


def run_genome(
    config_path: str,
    region: str = "",
    n_structures: int = 1,
    data_dir: str | None = None,
) -> list[dict[str, list[BeadOut]]]:
    """Reconstructs several chromosomes from a config file.

    Parameters
    ----------
    config_path
        Path of the .ini config file.
    region
        '' for chr1 to chr22 and chrX, a chromosome, a list 'chr1,chr3,chrX', a range
        'chr1-chr22,chrX' or one region 'chr14:18288319-20307135'.
    n_structures
        Number of structures.
    data_dir
        Replaces `data_dir` of the config file.

    Returns
    -------
    list of dict
        One dict of chromosome to sorted bead list per structure.
    """
    chrs_list, bed_region = parse_chrs_arg(region)
    if not chrs_list:
        raise ValueError(f"Cannot parse region: {region!r}")

    s = Settings()
    if not s.load_ini(config_path):
        raise RuntimeError(f"Failed to load config: {config_path!r}")
    if data_dir is not None:
        s.data_dir = str(data_dir)

    log.setup(s.output_level, log_file=s.log_file or None)

    data = ContactData.from_files(s, chrs_list, bed_region)
    return simulate(s, data, chrs_list, n_structures, region=bed_region)
