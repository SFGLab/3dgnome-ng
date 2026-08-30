"""Spike. Derive the segment boundary file from called TAD boundaries.

`validation tracks` writes TAD boundaries as `chr pos pos`, which is the format
`[data] segment_split` reads. Blocks are not derived here. They come from arc gaps, as on the
cell lines, because a TAD boundary knows nothing about the arcs and cuts through them.

Segments are the grouping above blocks, and they must be much coarser than a TAD. Under the default
`refine_scope = segment` a segment holding one block or fewer is skipped, so segments at TAD
scale would skip placement for most of the genome without reporting anything. On GM12878,
thinning at 2 Mb gives 1292 boundaries against the 1298 of the CCDS breakpoints file the
existing cell lines use, at 7.6 TADs per segment.

    python playground/trio/trio_segments.py --samples HG00512
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402

MIN_BLOCKS_PER_SEGMENT = 2.0


def read_boundaries(path: Path) -> dict[str, list[int]]:
    by: dict[str, list[int]] = defaultdict(list)
    for line in path.open():
        f = line.split()
        if len(f) >= 2:
            by[f[0]].append(int(f[1]))
    return by


def thin(positions: list[int], min_span: int) -> list[int]:
    """Keep a boundary only when it is at least min_span from the last kept one."""
    kept: list[int] = []
    for p in sorted(positions):
        if not kept or p - kept[-1] >= min_span:
            kept.append(p)
    return kept


def write(by: dict[str, list[int]], min_span: int, out: Path) -> tuple[int, list[int]]:
    kept_total = 0
    spans: list[int] = []
    with out.open("w") as fh:
        for c in sorted(by):
            kept = thin(by[c], min_span)
            kept_total += len(kept)
            spans.extend(b - a for a, b in zip(kept, kept[1:], strict=False))
            for p in kept:
                fh.write(f"{c}\t{p}\t{p}\n")
            if len(kept) < 2:
                print(f"[segments]   WARNING {c}: {len(kept)} boundaries, too few to segment")
    return kept_total, spans


def build(name: str, tads: Path, root: Path, min_tad: int, min_segment: int) -> None:
    by = read_boundaries(tads)
    raw = sum(len(v) for v in by.values())

    blocks_out = root / f"{name}_blocks.bed"
    n_blocks, block_spans = write(by, min_tad, blocks_out)
    segments_out = root / f"{name}_segments.bed"
    n_segments, seg_spans = write(by, min_segment, segments_out)

    if not block_spans or not seg_spans:
        raise SystemExit(f"no boundaries produced from {tads}")
    per_segment = n_blocks / max(n_segments, 1)
    print(
        f"[segments:{name}] {raw} TAD boundaries\n"
        f"    blocks   {n_blocks:6d} -> {blocks_out.name}   "
        f"median span {statistics.median(block_spans) / 1e3:.0f} kb\n"
        f"    segments {n_segments:6d} -> {segments_out.name}   "
        f"median span {statistics.median(seg_spans) / 1e6:.2f} Mb\n"
        f"    {per_segment:.1f} blocks per segment"
    )
    # A segment holding one block or fewer is skipped by IB placement, silently. Refuse rather
    # than write a pair of files that would model almost nothing.
    if per_segment < MIN_BLOCKS_PER_SEGMENT:
        raise SystemExit(
            f"[segments:{name}] only {per_segment:.1f} blocks per segment. IB placement skips "
            f"segments holding one block or fewer, so raise --min-segment above {min_segment}."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--samples")
    ap.add_argument("--min-tad", type=int, default=100_000, help="merge TADs smaller than this")
    ap.add_argument("--min-segment", type=int, default=2_000_000, help="segment grouping scale")
    args = ap.parse_args()
    for s in trio_samples.resolve(args.samples):
        root = Path(args.data_root) / s.name
        tads = root / f"{s.name}_tads.bed"
        if not tads.is_file():
            raise SystemExit(
                f"{tads} missing, run `python -m validation tracks --cell {s.name} "
                f"--skip-compartments --skip-signal` first"
            )
        build(s.name, tads, root, args.min_tad, args.min_segment)


if __name__ == "__main__":
    main()
