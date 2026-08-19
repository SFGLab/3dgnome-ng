"""
Command-line interface for 3dgnome-ng.

Single region:   gnome3d-ng --config X.ini --region chr1:18288319-20307135
Single chr:      gnome3d-ng --config X.ini --region chr14
Chromosome list: gnome3d-ng --config X.ini --region chr1,chr3,chrX
Range:           gnome3d-ng --config X.ini --region chr1-chr22,chrX
Whole genome:    gnome3d-ng --config X.ini      (defaults to chr1-chr22,chrX)
"""

import argparse
import contextlib
import sys
from pathlib import Path

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg, write_cif
from gnome3d.pipeline.executor import Executor
from gnome3d.reconstruct import MEMBER_SEED_STRIDE, pick_executor, reconstruct
from gnome3d.settings import Settings
from gnome3d.types import BedRegion

LOG = log.get("main")


def parse_members(spec: str) -> list[int]:
    """Structure indices from a comma list of numbers and ``lo-hi`` ranges, in the given
    order and de-duplicated."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


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
    executor: Executor,
) -> int:
    """Reconstruct + write one independent structure via the task-DAG pipeline.
    Returns total bead count.  Per-member seed offset makes an ensemble vary."""
    # Scope per structure only when several run - for a single structure the
    # extra nesting just indents everything for no benefit.
    structure_ctx = log.step(LOG, f"structure {i + 1}/{n}") if n > 1 else contextlib.nullcontext()
    with structure_ctx:
        per_chr = reconstruct(
            s, data, chrs_list, region, executor=executor, seed_offset=i * MEMBER_SEED_STRIDE
        )

        multi_chr = len(chrs_list) > 1
        total_beads = 0
        for chr_ in chrs_list:
            beads = per_chr.get(chr_)
            if not beads:
                LOG.warning("%s: no leaf beads (skipping)", chr_)
                continue

            cif_path = out_dir / _cif_name(entry_base, chr_, i, multi_chr)
            entry_id = cif_path.stem
            write_cif(str(cif_path), beads, entry_id=entry_id)
            LOG.info("wrote %s  (%d beads)", cif_path, len(beads))
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
    parser.add_argument(
        "--members",
        default="",
        help=(
            "Explicit structure indices instead of -n, as a comma list of numbers and "
            "ranges, e.g. '0-9' or '0,4,7-9'. One ensemble can then be split over several "
            "jobs by giving each a disjoint set, and a member whose .cif already exists is "
            "skipped, so a requeued job resumes instead of redoing its work. Index i keeps "
            "the seed offset it would have had under -n, so the ensemble is the same "
            "however it was split."
        ),
    )
    parser.add_argument("--data-dir", default=None, help="Override data_dir from config")
    parser.add_argument("--out", default=".", help="Output directory (default: .)")
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Write a full-detail (DEBUG) structured log here in addition to "
            "stdout. Overrides [main] log_file. Handy for reconstructing "
            "parallel (ib_workers>1 / -n>1) runs."
        ),
    )
    args = parser.parse_args()

    chrs_list, bed_region = parse_chrs_arg(args.region)
    if not chrs_list:
        sys.exit(f"Cannot parse region: {args.region!r}")

    s = Settings()
    if not s.load_ini(args.config):
        sys.exit(f"Failed to load config: {args.config!r}")
    if args.data_dir:
        s.data_dir = args.data_dir

    log.setup(s.output_level, log_file=args.log_file or s.log_file or None)
    log.status(LOG, "config: %s  data_dir: %s", args.config, s.data_dir)
    log.status(
        LOG,
        "chromosomes (%d): %s%s",
        len(chrs_list),
        ",".join(chrs_list),
        f"  region={bed_region.chr}:{bed_region.start}-{bed_region.end}" if bed_region else "",
    )

    data = ContactData.from_files(s, chrs_list, bed_region)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build a filesystem-safe base name from the --region argument (or "genome"
    # when whole-genome default was used).
    raw = args.region.strip() or "genome"
    entry_base = raw.replace(":", "_").replace("-", "_").replace(",", "_")

    # Structures run sequentially: the pipeline parallelises *inside* a
    # reconstruct (IB batching on the JAX BatchExecutor), and its deterministic
    # global-RNG seeding can't be threaded across structures safely.  Each
    # structure gets a distinct per-member seed offset, so an ensemble varies.
    members = parse_members(args.members) if args.members else list(range(args.n_structures))
    if not members:
        sys.exit(f"No structures selected: --members {args.members!r}")

    executor = pick_executor(s)
    n = len(members)
    log.status(LOG, "running %d structure(s)  [%s]", n, type(executor).__name__)

    multi_chr = len(chrs_list) > 1
    written = 0
    for i in members:
        if args.members and all(
            (out_dir / _cif_name(entry_base, c, i, multi_chr)).exists() for c in chrs_list
        ):
            log.status(LOG, "structure %d already written, skipping", i + 1)
            continue
        _run_structure(i, n, s, data, chrs_list, bed_region, out_dir, entry_base, executor)
        written += 1

    log.status(LOG, "%d structure(s) written to %s/", written, out_dir)


if __name__ == "__main__":
    main()
