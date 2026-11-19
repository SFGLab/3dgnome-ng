"""Reference to haplotype coordinates along one chromosome, from the variants applied to it.

The rules and the sequence builder apply a haplotype's variants left to right, each wholly
after the last. A deletion takes its bases out and shifts what follows up by its length, an
insertion shifts what follows down by its length, an inversion mirrors the bases it holds and
shifts nothing. A duplication is tandem. Its bases stay where they are, a second copy follows
them and shifts what follows down by its length. A reference position inside it lifts to the
first copy, and `copies` gives its place in the second. Positions are 0 based.
"""
from __future__ import annotations

import numpy as np

from .variants import SV


class Lift:
    def __init__(self, svs: list[SV]):
        self.events: list[tuple[str, int, int, int]] = []
        shift, last = 0, -1
        for s in sorted(svs, key=lambda v: v.start):
            a = s.start - 1
            if a < last:
                continue
            if s.kind == "DEL":
                b = s.end - 1
                self.events.append(("DEL", a, b, shift))
                shift -= b - a
                last = b
            elif s.kind == "INS":
                self.events.append(("INS", a, a, shift))
                shift += len(s.seq)
                last = a
            elif s.kind == "INV":
                b = s.end - 1
                self.events.append(("INV", a, b, shift))
                last = b
            elif s.kind == "DUP":
                b = s.end - 1
                self.events.append(("DUP", a, b, shift))
                shift += b - a
                last = b
        self.final = shift

    def __call__(self, pos: int) -> int | None:
        """The haplotype position of a reference position, None if it is deleted."""
        for kind, a, b, before in self.events:
            if pos < a:
                return pos + before
            if kind == "DEL" and pos < b:
                return None
            if kind == "INV" and pos < b:
                return a + before + (b - 1 - pos)
            if kind == "DUP" and pos < b:
                return pos + before
        return pos + self.final

    @property
    def duplications(self) -> list[tuple[int, int, int]]:
        """Each duplication as its reference interval and the shift of its first copy. The
        second copy is the first moved down by the interval's length."""
        return [(a, b, before) for kind, a, b, before in self.events if kind == "DUP"]

    def copies(self, pos: np.ndarray) -> np.ndarray:
        """The second copy's haplotype position of each reference position inside a
        duplication, -1 for every other position."""
        pos = np.asarray(pos, dtype=np.int64)
        out = np.full(len(pos), -1, dtype=np.int64)
        for a, b, before in self.duplications:
            m = (pos >= a) & (pos < b)
            out[m] = pos[m] + before + (b - a)
        return out

    def edge(self, pos: int) -> int:
        """The haplotype position of a reference position, or where its deletion joins the
        two sides, for a region's edge that falls in a deletion."""
        x = self(pos)
        if x is not None:
            return x
        for kind, a, b, before in self.events:
            if kind == "DEL" and a <= pos < b:
                return a + before
        raise AssertionError(pos)

    def many(self, pos: np.ndarray) -> np.ndarray:
        """Haplotype positions of many reference positions, -1 where deleted, as __call__ gives
        them one at a time. The events do not overlap, so the one that holds a position is the
        last that starts at or before it."""
        pos = np.asarray(pos, dtype=np.int64)
        if not self.events:
            return pos.copy()
        kind = np.array([e[0] for e in self.events])
        a = np.array([e[1] for e in self.events], dtype=np.int64)
        b = np.array([e[2] for e in self.events], dtype=np.int64)
        before = np.array([e[3] for e in self.events], dtype=np.int64)
        after = np.r_[before[1:], self.final]
        k = np.searchsorted(a, pos, side="right") - 1
        out = pos.copy()
        on = k >= 0
        kk = np.clip(k, 0, None)
        out[on] = pos[on] + after[kk[on]]
        inside = on & (pos < b[kk])
        dele = inside & (kind[kk] == "DEL")
        inv = inside & (kind[kk] == "INV")
        dup = inside & (kind[kk] == "DUP")
        out[inv] = a[kk[inv]] + before[kk[inv]] + (b[kk[inv]] - 1 - pos[inv])
        out[dup] = pos[dup] + before[kk[dup]]
        out[dele] = -1
        return out
