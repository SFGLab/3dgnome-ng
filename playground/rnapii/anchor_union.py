"""The anchor set two factors share. The CTCF anchors verbatim, plus the ends of the second
factor's loops that overlap no CTCF anchor, merged among themselves, as anchors of no
orientation.

An end overlapping a CTCF anchor is not added, so the CTCF set stays what production runs on.
It maps onto that anchor when its midpoint falls inside, which is the loader's rule, and is
lost otherwise. Both cases are counted so the loss is visible.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

AnchorRow = tuple[str, int, int, str]
End = tuple[str, int, int]

# No anchor is wider than this, so the scan back from the first anchor starting past an end
# can stop once anchor ends fall this far behind it.
MAX_ANCHOR_BP = 50_000


@dataclass
class UnionStats:
    ends_over: int = 0
    ends_inside: int = 0
    ends_free: int = 0
    added: int = 0
    added_mean_kb: float = 0.0

    def describe(self) -> str:
        return (
            f"{self.ends_over} loop ends overlap a CTCF anchor ({self.ends_inside} with the "
            f"midpoint inside it, {self.ends_over - self.ends_inside} lost), {self.ends_free} on "
            f"no CTCF anchor -> {self.added} new anchors, mean width {self.added_mean_kb:.1f} kb"
        )


def read_anchors(path: Path) -> list[AnchorRow]:
    """A four column anchor BED, orientation N when the column is absent."""
    rows: list[AnchorRow] = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 3:
            rows.append((f[0], int(f[1]), int(f[2]), f[3] if len(f) > 3 else "N"))
    return rows


def union_anchors(ctcf: list[AnchorRow], ends: Iterable[End]) -> tuple[list[AnchorRow], UnionStats]:
    """The CTCF rows plus one anchor per run of overlapping free ends, in genomic order.

    Parameters
    ----------
    ctcf
        The CTCF anchor rows, kept verbatim.
    ends
        The second factor's loop ends as (chromosome, start, end), both ends of every loop.
    """
    by_chr: dict[str, list[tuple[int, int]]] = {}
    for c, s, e, _ in ctcf:
        by_chr.setdefault(c, []).append((s, e))
    for lst in by_chr.values():
        lst.sort()
    starts = {c: [s for s, _ in lst] for c, lst in by_chr.items()}

    def hit(c: str, s: int, e: int) -> tuple[bool, bool]:
        """Whether [s, e) overlaps a CTCF anchor, and whether its midpoint lies inside one."""
        lst = by_chr.get(c, [])
        k = bisect.bisect_right(starts.get(c, []), e) - 1
        over = inside = False
        mid = (s + e) // 2
        while k >= 0 and lst[k][1] > s - MAX_ANCHOR_BP:
            if lst[k][0] < e and lst[k][1] > s:
                over = True
                if lst[k][0] <= mid <= lst[k][1]:
                    inside = True
            k -= 1
        return over, inside

    st = UnionStats()
    free: list[End] = []
    for c, s, e in ends:
        over, inside = hit(c, s, e)
        if over:
            st.ends_over += 1
            st.ends_inside += int(inside)
        else:
            st.ends_free += 1
            free.append((c, s, e))
    free.sort()
    merged: list[list[int | str]] = []
    for c, s, e in free:
        if merged and merged[-1][0] == c and s <= int(merged[-1][2]):
            merged[-1][2] = max(int(merged[-1][2]), e)
        else:
            merged.append([c, s, e])
    st.added = len(merged)
    if merged:
        st.added_mean_kb = sum(int(m[2]) - int(m[1]) for m in merged) / len(merged) / 1e3
    rows = list(ctcf) + [(str(c), int(s), int(e), "N") for c, s, e in merged]
    rows.sort()
    return rows, st
