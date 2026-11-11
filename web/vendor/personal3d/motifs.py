"""CTCF motifs in anchors, and the narrowing of the 2019 deletion rule, idea 110.

The rules remove every anchor a deletion touches, by a single base. A merged anchor spans a few
kilobases and a deletion of 50 bp in it mostly leaves its CTCF site and its loops in place,
which the controlled test of idea 108 showed, the person keeping the loops the rules took. Here
a deletion removes an anchor only when it cuts one of the anchor's motifs, a window of the
rules' matrix scoring at least the threshold, or covers half the anchor or more. An anchor it
touches and does not cut is clipped to the side that keeps its best motif, or the longer side,
so the rules leave it, and its reference interval is kept for the read back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hicdiffusion import Reference
from .variants import SV

THRESHOLD = 10.0


def scores(seq: str, P: np.ndarray) -> np.ndarray:
    """The better strand's score of every window of a sequence, N the mean of the four."""
    m = len(P)
    if len(seq) < m:
        return np.zeros(0)
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.array([code.get(c, 4) for c in seq.upper()])
    P5 = np.vstack([P.T, P.mean(axis=1)]).T
    win = np.lib.stride_tricks.sliding_window_view
    fwd = P5[np.arange(m), win(x, m)].sum(axis=1)
    rc = np.where(x == 4, 4, 3 - x)
    rev = P5[np.arange(m), win(rc, m)[:, ::-1]].sum(axis=1)
    return np.maximum(fwd, rev)


def anchor_motifs(ref: Reference, anchors: pd.DataFrame, pwm, threshold: float = THRESHOLD) -> dict[tuple, list[tuple[int, int, float]]]:
    """Each anchor's motifs at or above threshold, as reference intervals with their scores."""
    P = pwm[["A", "C", "G", "T"]].to_numpy()
    m = len(P)
    out = {}
    for r in anchors.itertuples(index=False):
        s = scores(ref.fetch(r.chrom, int(r.start), int(r.end)), P)
        k = np.flatnonzero(s >= threshold)
        out[(r.chrom, int(r.start), int(r.end))] = [(int(r.start) + int(i), int(r.start) + int(i) + m, float(s[i])) for i in k]
    return out


def cuts(sv: SV, anchor: tuple[str, int, int], motifs: list[tuple[int, int, float]]) -> bool:
    """Whether a deletion removes an anchor under the narrowed rule."""
    a, b = sv.start - 1, sv.end - 1
    _, s, e = anchor
    if min(b, e) - max(a, s) >= 0.5 * (e - s):
        return True
    return any(a < me and ms < b for ms, me, _ in motifs)


def narrow(clusters: list[list], svs: list[SV], motifs_of: dict[tuple, list]) -> tuple[list[list], dict[tuple, tuple[int, int]]]:
    """The clusters with every anchor a deletion touches and does not cut clipped clear of it,
    and the map from a clipped anchor to its reference interval."""
    dels = [s for s in svs if s.kind == "DEL"]
    if not dels:
        return clusters, {}
    anchors = {(c[0], c[1], c[2]) for c in clusters} | {(c[3], c[4], c[5]) for c in clusters}
    clip: dict[tuple, tuple[int, int]] = {}
    for an in anchors:
        chrom, s, e = an
        lo, hi = s, e
        for d in dels:
            a, b = d.start - 1, d.end - 1
            if d.chrom != chrom or not (a < hi and lo < b):
                continue
            mot = motifs_of.get(an, [])
            if cuts(d, an, mot):
                lo, hi = s, e
                break
            # The rules take a deletion as start to end, 1 based, end exclusive, and remove an
            # anchor that reaches into it, so the right side starts at its end and not a base
            # before.
            left, right = (lo, min(hi, d.start - 1)), (max(lo, d.end), hi)
            best = max(mot, key=lambda x: x[2]) if mot else None
            if best is not None and left[0] <= best[0] and best[1] <= left[1]:
                lo, hi = left
            elif best is not None and right[0] <= best[0] and best[1] <= right[1]:
                lo, hi = right
            else:
                lo, hi = left if left[1] - left[0] >= right[1] - right[0] else right
            if hi <= lo:
                lo, hi = s, e
                break
        if (lo, hi) != (s, e):
            clip[an] = (lo, hi)
    if not clip:
        return clusters, {}
    out = []
    for c in clusters:
        x, y = (c[0], c[1], c[2]), (c[3], c[4], c[5])
        xs, xe = clip.get(x, (c[1], c[2]))
        ys, ye = clip.get(y, (c[4], c[5]))
        out.append([c[0], xs, xe, c[3], ys, ye, *c[6:]])
    ref_of = {(an[0], *iv): (an[1], an[2]) for an, iv in clip.items()}
    return out, ref_of
