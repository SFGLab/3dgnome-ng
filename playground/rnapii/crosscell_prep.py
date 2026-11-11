"""The cross line RNAPII inputs of `design/2026-11-11-rnapii-release.md`, Part 1.

Loops of GM12878, H1ESC and HFFc6 are pooled over replicates, filtered on the line's own peaks
and a span cap, and thinned binomially so every line of one factor carries the same PET total.
The rule is the trio rule of `triosformer/pipeline/trio_resample.py`. A CTCF loop needs both
ends on a CTCF peak and an RNAPII loop one end on an RNAPII peak. The span is the gap between
the two ends. For the merged anchor arm an RNAPII loop end within `absorb_bp` of a CTCF anchor
takes that anchor's interval, so the loader maps it onto the anchor and no new bead is made.

A loop is `(chrA, startA, endA, chrB, startB, endB, pets)`, intra chromosomal, end A first.
"""

from __future__ import annotations

import bisect
import gzip
from pathlib import Path

import numpy as np

Loop = tuple[str, int, int, str, int, int, int]
Interval = tuple[int, int]
AnchorRow = tuple[str, int, int, str]

# No CTCF anchor is wider than this, which bounds the scan back from a query position.
MAX_ANCHOR_BP = 50_000


def read_bedpe(path: Path) -> list[Loop]:
    """Intra chromosomal loops of a seven column bedpe, gzip aware, end A first."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out: list[Loop] = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 7 or f[0] != f[3]:
                continue
            a = (int(f[1]), int(f[2]))
            b = (int(f[4]), int(f[5]))
            if b < a:
                a, b = b, a
            out.append((f[0], a[0], a[1], f[0], b[0], b[1], int(float(f[6]))))
    return out


def write_bedpe(path: Path, loops: list[Loop]) -> None:
    """The loops sorted, one tab separated row each."""
    with open(path, "w") as out:
        for c, s1, e1, c2, s2, e2, p in sorted(loops):
            out.write(f"{c}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{p}\n")


def pool_replicates(reps: list[list[Loop]]) -> list[Loop]:
    """One loop set from several. Loops whose two ends both overlap join into one loop with the
    union interval at each end and the PET counts summed."""
    # Rows are [startA, endA, startB, endB, pets]. Input is sorted by end A, so a pooled loop
    # whose end A finishes before the current start can never be joined again.
    pooled: list[tuple[str, list[int]]] = []
    active: dict[str, list[list[int]]] = {}
    for c, s1, e1, _, s2, e2, p in sorted(lp for rep in reps for lp in rep):
        live = [q for q in active.get(c, []) if q[1] > s1]
        hit = next((q for q in live if q[2] < e2 and q[3] > s2), None)
        if hit is None:
            row = [s1, e1, s2, e2, p]
            pooled.append((c, row))
            live.append(row)
        else:
            hit[1] = max(hit[1], e1)
            hit[2] = min(hit[2], s2)
            hit[3] = max(hit[3], e2)
            hit[4] += p
        active[c] = live
    return sorted((c, r[0], r[1], c, r[2], r[3], r[4]) for c, r in pooled)


class PeakIndex:
    """Overlap queries on merged, non overlapping intervals per chromosome."""

    def __init__(self, union: dict[str, list[Interval]]) -> None:
        self.starts = {c: [s for s, _ in lst] for c, lst in union.items()}
        self.ends = {c: [e for _, e in lst] for c, lst in union.items()}

    @classmethod
    def from_beds(cls, paths: list[Path]) -> PeakIndex:
        """The merged union of every peak file, gzip aware."""
        by: dict[str, list[Interval]] = {}
        for path in paths:
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt") as fh:
                for line in fh:
                    f = line.split()
                    if len(f) >= 3 and not line.startswith(("#", "track")):
                        by.setdefault(f[0], []).append((int(float(f[1])), int(float(f[2]))))
        union: dict[str, list[Interval]] = {}
        for c, lst in by.items():
            lst.sort()
            merged: list[list[int]] = []
            for s, e in lst:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            union[c] = [(s, e) for s, e in merged]
        return cls(union)

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        starts = self.starts.get(chrom)
        if not starts:
            return False
        k = bisect.bisect_right(starts, end - 1) - 1
        return k >= 0 and self.ends[chrom][k] > start


def filter_loops(
    loops: list[Loop], peaks: PeakIndex, ends_on_peaks: int, max_span: int
) -> list[Loop]:
    """Loops whose gap is at most `max_span` and with at least `ends_on_peaks` ends on a peak."""
    out: list[Loop] = []
    for lp in loops:
        c, s1, e1, _, s2, e2, _ = lp
        if s2 - e1 > max_span:
            continue
        on = int(peaks.overlaps(c, s1, e1)) + int(peaks.overlaps(c, s2, e2))
        if on >= ends_on_peaks:
            out.append(lp)
    return out


def depth(loops: list[Loop]) -> int:
    """The PET total."""
    return sum(lp[6] for lp in loops)


def thin(loops: list[Loop], p: float, rng: np.random.Generator, min_pets: int = 3) -> list[Loop]:
    """Every loop's PET count drawn from Binomial(n, p), the loops still at `min_pets` kept."""
    if p >= 1.0:
        return list(loops)
    drawn = rng.binomial(np.array([lp[6] for lp in loops], dtype=np.int64), p)
    return [(*lp[:6], int(d)) for lp, d in zip(loops, drawn, strict=True) if d >= min_pets]


def _nearest(
    by_chr: dict[str, list[AnchorRow]], starts: dict[str, list[int]], c: str, s: int, e: int, reach: int
) -> AnchorRow | None:
    """The CTCF anchor with the smallest gap to [s, e) when that gap is at most `reach`."""
    rows = by_chr.get(c)
    if not rows:
        return None
    k = bisect.bisect_right(starts[c], e + reach) - 1
    best: AnchorRow | None = None
    best_gap = reach + 1
    while k >= 0 and rows[k][1] >= s - reach - MAX_ANCHOR_BP:
        a = rows[k]
        gap = max(0, a[1] - e, s - a[2])
        if gap < best_gap or (gap == best_gap and best is not None and a[1] < best[1]):
            best, best_gap = a, gap
        k -= 1
    return best


def absorb_ends(
    loops: list[Loop], ctcf: list[AnchorRow], absorb_bp: int
) -> tuple[list[Loop], int]:
    """Loops with every end within `absorb_bp` of a CTCF anchor moved onto that anchor.

    A loop whose two ends land on one anchor is dropped. Loops that land on the same pair of
    intervals join with their PET counts summed. Returns the loops and the number of ends moved.
    """
    by_chr: dict[str, list[AnchorRow]] = {}
    for row in ctcf:
        by_chr.setdefault(row[0], []).append(row)
    for lst in by_chr.values():
        lst.sort(key=lambda r: (r[1], r[2]))
    starts = {c: [r[1] for r in lst] for c, lst in by_chr.items()}
    moved = 0
    joined: dict[tuple[str, int, int, int, int], int] = {}
    for c, s1, e1, _, s2, e2, p in loops:
        ends = []
        for s, e in ((s1, e1), (s2, e2)):
            a = _nearest(by_chr, starts, c, s, e, absorb_bp)
            if a is not None and (a[1], a[2]) != (s, e):
                moved += 1
                ends.append((a[1], a[2]))
            else:
                ends.append((s, e))
        ends.sort()
        if ends[0] == ends[1]:
            continue
        key = (c, ends[0][0], ends[0][1], ends[1][0], ends[1][1])
        joined[key] = joined.get(key, 0) + p
    out = [(c, a, b, c, x, y, p) for (c, a, b, x, y), p in sorted(joined.items())]
    return out, moved
