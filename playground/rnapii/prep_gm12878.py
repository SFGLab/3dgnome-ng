"""GM12878 RNAPII ChIA-PET loops of Tang et al. 2015 as a second cluster file, and the anchor
set both factors share.

    python playground/rnapii/prep_gm12878.py

Reads the GEO cluster file GSM1872887 on hg19 under data/GM12878/raw, lifts every loop end to
hg38 with the UCSC chain under data/_liftover, keeps loops of three or more PETs whose two
ends lift to the same chromosome, and writes:

    data/GM12878/GM12878_rnapii_clusters_3+.bedpe     seven columns, the pipeline's format
    data/GM12878/GM12878_anchors_ctcf_rnapii.bed      the CTCF anchors verbatim, plus the
                                                      RNAPII loop ends that overlap no CTCF
                                                      anchor, as anchors of no orientation

The RNAPII loop ends are the file's own anchor intervals. An end overlapping a CTCF anchor is
not added, so the CTCF set stays what production runs on; it maps onto that anchor when its
midpoint falls inside, which is the loader's rule, and the script counts the ends it loses.
"""

from __future__ import annotations

import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anchor_union import read_anchors, union_anchors  # noqa: E402
from pyliftover import LiftOver  # noqa: E402

DATA = Path("data/GM12878")
RAW = DATA / "raw" / "GSM1872887_GM12878_RNAPII_PET_clusters.txt.gz"
CHAIN = Path("data/_liftover/hg19ToHg38.over.chain.gz")
CTCF_ANCHORS = DATA / "GM12878_anchors_3+_oriented.bed"
OUT_LOOPS = DATA / "GM12878_rnapii_clusters_3+.bedpe"
OUT_ANCHORS = DATA / "GM12878_anchors_ctcf_rnapii.bed"
MIN_PETS = 3


def lift(lo: LiftOver, chrom: str, start: int, end: int) -> tuple[str, int, int] | None:
    a = lo.convert_coordinate(chrom, start)
    b = lo.convert_coordinate(chrom, end)
    if not a or not b or a[0][0] != b[0][0] or a[0][0] != chrom:
        return None
    s, e = sorted((a[0][1], b[0][1]))
    if e - s < 100 or e - s > 50_000:
        return None
    return chrom, s, e


def main() -> None:
    lo = LiftOver(str(CHAIN))
    stats: Counter[str] = Counter()
    loops: list[tuple[str, int, int, int, int, int]] = []
    with gzip.open(RAW, "rt") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 7 or f[0] != f[3]:
                stats["trans or short"] += 1
                continue
            stats["cis"] += 1
            pets = int(float(f[6]))
            if pets < MIN_PETS:
                stats["under 3 PETs"] += 1
                continue
            a = lift(lo, f[0], int(f[1]), int(f[2]))
            b = lift(lo, f[3], int(f[4]), int(f[5]))
            if a is None or b is None:
                stats["not lifted"] += 1
                continue
            if a[1] > b[1]:
                a, b = b, a
            loops.append((a[0], a[1], a[2], b[1], b[2], pets))
            stats["kept"] += 1
    loops.sort()
    with open(OUT_LOOPS, "w") as out:
        for c, s1, e1, s2, e2, p in loops:
            out.write(f"{c}\t{s1}\t{e1}\t{c}\t{s2}\t{e2}\t{p}\n")
    print("loops:", dict(stats))

    ctcf = read_anchors(CTCF_ANCHORS)
    ends = [(c, s, e) for c, s1, e1, s2, e2, _ in loops for s, e in ((s1, e1), (s2, e2))]
    rows, st = union_anchors(ctcf, ends)
    with open(OUT_ANCHORS, "w") as out:
        for c, s, e, o in rows:
            out.write(f"{c}\t{s}\t{e}\t{o}\n")
    print("RNAPII", st.describe())
    print(f"anchors: {len(ctcf)} CTCF rows + {st.added} RNAPII = {len(rows)}")
    print(f"wrote {OUT_LOOPS} and {OUT_ANCHORS}")


if __name__ == "__main__":
    main()
