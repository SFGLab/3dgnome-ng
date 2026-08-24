"""Spike. Give anchor intervals a CTCF motif orientation column.

The pipeline reads anchors as chr, start, end, orientation, where orientation is L, R or N.
The trio peak files carry no strand, so orientation comes from JASPAR MA0139.1 motif hits.
An anchor takes the strand of the highest scoring motif that overlaps it, and N when no motif
overlaps it.

The mapping from strand to letter was recovered from the GM12878 anchors already in the repo
rather than assumed. Rerun that check any time with --validate.

    python playground/trio/trio_orient.py anchors.bed out.bed
    python playground/trio/trio_orient.py --validate data/GM12878/GM12878_anchors_3+_oriented.bed
"""

import argparse
import bisect
import gzip
from collections import Counter, defaultdict
from pathlib import Path

MOTIFS = Path("playground/jaspar_MA0139.1_hg38.tsv.gz")
# MA0139.1 is 19 bp. A motif can only overlap an anchor if it starts at or after
# anchor_start - width, so that bounds the scan backwards from the anchor start.
MOTIF_WIDTH = 32
STRAND_TO_ORIENTATION = {"+": "R", "-": "L"}


def load_motifs(path: Path) -> dict[str, tuple[list[int], list[tuple[int, int, str]]]]:
    """Read motif hits per chromosome as sorted starts plus (end, score, strand)."""
    by_chr: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 7:
                continue
            by_chr[f[0]].append((int(f[1]), int(f[2]), int(f[4]), f[6]))
    out: dict[str, tuple[list[int], list[tuple[int, int, str]]]] = {}
    for c, hits in by_chr.items():
        hits.sort()
        out[c] = ([h[0] for h in hits], [(h[1], h[2], h[3]) for h in hits])
    return out


def orient(
    chrom: str, start: int, end: int, motifs: dict[str, tuple[list[int], list[tuple[int, int, str]]]]
) -> str:
    entry = motifs.get(chrom)
    if entry is None:
        return "N"
    starts, rest = entry
    i = bisect.bisect_left(starts, start - MOTIF_WIDTH)
    best_score, best_strand = -1, None
    while i < len(starts) and starts[i] < end:
        m_end, score, strand = rest[i]
        if m_end > start and score > best_score:
            best_score, best_strand = score, strand
        i += 1
    return STRAND_TO_ORIENTATION.get(best_strand or "", "N")


def read_anchors(path: Path) -> list[tuple[str, int, int, str | None]]:
    out: list[tuple[str, int, int, str | None]] = []
    for line in path.open():
        f = line.split()
        if len(f) < 3:
            continue
        out.append((f[0], int(f[1]), int(f[2]), f[3] if len(f) > 3 else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("anchors", nargs="?", help="3 column anchor BED")
    ap.add_argument("out", nargs="?", help="4 column oriented anchor BED")
    ap.add_argument("--motifs", default=str(MOTIFS))
    ap.add_argument("--validate", help="oriented BED to score this recipe against")
    args = ap.parse_args()

    motifs = load_motifs(Path(args.motifs))
    print(f"[orient] {sum(len(v[0]) for v in motifs.values())} motif hits over {len(motifs)} chrs")

    if args.validate:
        anchors = read_anchors(Path(args.validate))
        agree, total = 0, 0
        confusion: Counter[tuple[str, str]] = Counter()
        for c, s, e, truth in anchors:
            if truth is None:
                continue
            got = orient(c, s, e, motifs)
            confusion[(truth, got)] += 1
            total += 1
            agree += truth == got
        print(f"[orient] {agree}/{total} agree ({100 * agree / total:.2f}%) on {args.validate}")
        for (truth, got), n in sorted(confusion.items()):
            mark = "" if truth == got else "   <- disagree"
            print(f"    truth={truth} derived={got}  {n}{mark}")
        return

    if not args.anchors or not args.out:
        raise SystemExit("give anchors and out, or --validate")
    anchors = read_anchors(Path(args.anchors))
    counts: Counter[str] = Counter()
    with Path(args.out).open("w") as fh:
        for c, s, e, _ in anchors:
            o = orient(c, s, e, motifs)
            counts[o] += 1
            fh.write(f"{c}\t{s}\t{e}\t{o}\n")
    total = sum(counts.values())
    share = "  ".join(f"{k}={counts[k]} ({100 * counts[k] / total:.1f}%)" for k in "LRN")
    print(f"[orient] {total} anchors -> {args.out}\n    {share}")


if __name__ == "__main__":
    main()
