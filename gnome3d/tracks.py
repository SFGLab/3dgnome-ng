"""Per bead arrays from compartment tracks, by binning onto beads and phasing an unsigned
eigenvector.

Bead ranges are inclusive and a zero length subanchor covers one base. Track intervals are half
open, as in BED.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import TypeVar

import numpy as np

from gnome3d import log
from gnome3d.types import (
    Anchor,
    AnchorMap,
    Compartment,
    CompartmentInterval,
    CompartmentMap,
    F32Array,
    I8Array,
    SignalInterval,
    SignalMap,
)

LOG = log.get("tracks")

_IntervalT = TypeVar("_IntervalT", CompartmentInterval, SignalInterval)


def _candidate_range(iv_starts: list[int], iv_ends: list[int], s: int, e: int) -> tuple[int, int]:
    """Returns the half open index range of the intervals that can overlap the bead range `[s, e]`.

    Exact for sorted intervals that do not overlap.
    """
    lo = bisect_right(iv_ends, s)
    hi = bisect_right(iv_starts, e)
    return lo, hi


def bin_compartments(
    intervals: list[CompartmentInterval],
    starts: list[int],
    ends: list[int],
) -> tuple[I8Array, F32Array]:
    """Reduces compartment intervals onto bead ranges.

    The class of a bead is the call that covers most of it and the score is the coverage weighted
    mean. A bead with no interval gets `NONE` and 0.0.

    Parameters
    ----------
    intervals
        Sorted by start, for one chromosome.
    starts, ends
        Inclusive genomic range per bead.

    Returns
    -------
    (cls, score)
        Both of length `len(starts)`.
    """
    n = len(starts)
    cls: I8Array = np.zeros(n, dtype=np.int8)
    score: F32Array = np.zeros(n, dtype=np.float32)
    if not intervals:
        return cls, score

    iv_starts = [iv.start for iv in intervals]
    iv_ends = [iv.end for iv in intervals]

    for i in range(n):
        s, e = starts[i], ends[i] + 1
        lo, hi = _candidate_range(iv_starts, iv_ends, s, e)
        weight_by_cls: dict[int, int] = {}
        total = 0
        acc = 0.0
        for k in range(lo, hi):
            iv = intervals[k]
            w = min(e, iv.end) - max(s, iv.start)
            if w <= 0:
                continue
            weight_by_cls[iv.cls] = weight_by_cls.get(iv.cls, 0) + w
            total += w
            acc += iv.score * w
        if total == 0:
            continue
        cls[i] = max(weight_by_cls.items(), key=lambda kv: kv[1])[0]
        score[i] = acc / total

    return cls, score


def slice_intervals(intervals: list[_IntervalT], lo: int, hi: int) -> list[_IntervalT]:
    """Returns the intervals that overlap the inclusive range `[lo, hi]`."""
    if not intervals:
        return []
    iv_starts = [iv.start for iv in intervals]
    iv_ends = [iv.end for iv in intervals]
    a, b = _candidate_range(iv_starts, iv_ends, lo, hi + 1)
    return intervals[a:b]


def _flip(intervals: list[CompartmentInterval]) -> None:
    for iv in intervals:
        iv.score = -iv.score


def _apply_sign(intervals: list[CompartmentInterval]) -> None:
    for iv in intervals:
        if iv.score > 0.0:
            iv.cls = int(Compartment.A2)
        elif iv.score < 0.0:
            iv.cls = int(Compartment.B1)
        else:
            iv.cls = int(Compartment.NONE)


def _mean_signal_over(
    intervals: list[SignalInterval], iv_starts: list[int], iv_ends: list[int], s: int, e: int
) -> float:
    lo, hi = _candidate_range(iv_starts, iv_ends, s, e)
    total = 0
    acc = 0.0
    for k in range(lo, hi):
        iv = intervals[k]
        w = min(e, iv.end) - max(s, iv.start)
        if w <= 0:
            continue
        total += w
        acc += iv.value * w
    return acc / total if total else math.nan


def _phase_by_signal(comps: list[CompartmentInterval], signal: list[SignalInterval]) -> bool:
    """Orients one chromosome by the signal, A on the side that correlates positively.

    Returns False when the track overlaps no signal or the signal is constant over its intervals.
    """
    if not signal:
        return False
    iv_starts = [iv.start for iv in signal]
    iv_ends = [iv.end for iv in signal]

    xs: list[float] = []
    ys: list[float] = []
    for c in comps:
        m = _mean_signal_over(signal, iv_starts, iv_ends, c.start, c.end)
        if math.isnan(m):
            continue
        xs.append(c.score)
        ys.append(m)

    if len(xs) < 2:
        return False
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return False

    r = float(np.corrcoef(x, y)[0, 1])
    if r != r:
        return False
    if r < 0.0:
        _flip(comps)
    return True


def _phase_by_anchors(comps: list[CompartmentInterval], anchors: list[Anchor]) -> bool:
    """Orients one chromosome by the anchors, A on the side with more of them, as MultiMM does.

    Returns False when no anchor is in a signed interval.
    """
    if not anchors:
        return False
    iv_starts = [c.start for c in comps]
    iv_ends = [c.end for c in comps]

    pos = 0
    neg = 0
    for a in anchors:
        c = a.center
        lo, hi = _candidate_range(iv_starts, iv_ends, c, c + 1)
        for k in range(lo, hi):
            iv = comps[k]
            if iv.start <= c < iv.end:
                if iv.score > 0.0:
                    pos += 1
                elif iv.score < 0.0:
                    neg += 1
                break

    if pos == neg:
        return False
    if pos < neg:
        _flip(comps)
    return True


def phase_compartments(
    comps: CompartmentMap,
    phasing_signal: SignalMap | None = None,
    anchors: AnchorMap | None = None,
) -> CompartmentMap:
    """Fills in A and B calls for compartment tracks that hold values only.

    The sign of a Hi-C eigenvector is arbitrary per chromosome. A chromosome whose intervals carry
    a class is left unchanged. Otherwise the signal decides, then the anchor counts. A chromosome
    that neither decides stays `NONE` and the compartment term does not act on it.

    Mutates and returns `comps`.
    """
    for chr_, lst in comps.items():
        if not lst:
            continue
        if any(iv.cls != int(Compartment.NONE) for iv in lst):
            continue

        sig = phasing_signal.get(chr_, []) if phasing_signal else []
        if _phase_by_signal(lst, sig):
            LOG.info("compartments phased by signal: %s", chr_)
        elif anchors is not None and _phase_by_anchors(lst, anchors.get(chr_, [])):
            LOG.info("compartments phased by anchor counts: %s", chr_)
        else:
            LOG.warning(
                "compartments not phased, leaving %s unassigned: no signal track "
                "and no anchor evidence",
                chr_,
            )
            continue

        _apply_sign(lst)

    return comps
