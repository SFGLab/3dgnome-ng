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

    # anchors: the CTCF set verbatim, plus RNAPII ends that overlap no CTCF anchor, merged among
    # themselves. An RNAPII end overlapping a CTCF anchor maps onto it when its midpoint falls
    # inside, which is how the loader maps a loop end, and is lost otherwise; both are counted.
    import bisect

    ctcf: list[tuple[str, int, int, str]] = []
    for line in CTCF_ANCHORS.read_text().splitlines():
        f = line.split()
        if len(f) >= 3:
            ctcf.append((f[0], int(f[1]), int(f[2]), f[3] if len(f) > 3 else "N"))
    by_chr: dict[str, list[tuple[int, int]]] = {}
    for c, s, e, _ in ctcf:
        by_chr.setdefault(c, []).append((s, e))
    for lst in by_chr.values():
        lst.sort()
    starts = {c: [s for s, _ in lst] for c, lst in by_chr.items()}

    def ctcf_hit(c: str, s: int, e: int) -> tuple[bool, bool]:
        """Whether [s, e) overlaps a CTCF anchor, and whether its midpoint lies inside one."""
        lst = by_chr.get(c, [])
        k = bisect.bisect_right(starts.get(c, []), e) - 1
        over = inside = False
        mid = (s + e) // 2
        while k >= 0 and lst[k][1] > s - 50_000:
            if lst[k][0] < e and lst[k][1] > s:
                over = True
                if lst[k][0] <= mid <= lst[k][1]:
                    inside = True
            k -= 1
        return over, inside

    ends_over = ends_inside = ends_free = 0
    free: list[tuple[str, int, int]] = []
    for c, s1, e1, s2, e2, _ in loops:
        for s, e in ((s1, e1), (s2, e2)):
            over, inside = ctcf_hit(c, s, e)
            if over:
                ends_over += 1
                ends_inside += int(inside)
            else:
                ends_free += 1
                free.append((c, s, e))
    free.sort()
    merged: list[list[object]] = []
    for c, s, e in free:
        if merged and merged[-1][0] == c and s <= int(merged[-1][2]):
            merged[-1][2] = max(int(merged[-1][2]), e)
        else:
            merged.append([c, s, e])
    rows = [(c, s, e, o) for c, s, e, o in ctcf] + [(str(c), int(s), int(e), "N") for c, s, e in merged]
    rows.sort()
    with open(OUT_ANCHORS, "w") as out:
        for c, s, e, o in rows:
            out.write(f"{c}\t{s}\t{e}\t{o}\n")
    print(
        f"RNAPII loop ends: {ends_over} overlap a CTCF anchor ({ends_inside} with the midpoint inside it, "
        f"{ends_over - ends_inside} lost), {ends_free} on no CTCF anchor -> {len(merged)} new anchors, "
        f"mean width {sum(int(m[2]) - int(m[1]) for m in merged) / max(len(merged), 1) / 1e3:.1f} kb"
    )
    print(f"anchors: {len(ctcf)} CTCF rows + {len(merged)} RNAPII = {len(rows)}")
    print(f"wrote {OUT_LOOPS} and {OUT_ANCHORS}")


if __name__ == "__main__":
    main()
