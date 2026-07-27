"""
Epigenomic track handling.

Turns loaded compartment and signal intervals into the per-bead arrays the MC
terms read.  Three steps live here.  Binning intervals onto bead ranges, phasing
an unsigned eigenvector track into A and B calls, and normalising a raw signal
into the accessibility scale HiP-HoP uses.

Bead ranges are inclusive, so a bead spanning `[s, e]` covers `e - s + 1` bases
and a zero-length subanchor still covers one.  Track intervals are half open,
matching BED.

See docs/epigenome-energy-terms.md.
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

# Offset inside the log used to normalise accessibility.  HiP-HoP's value.
_LOG_EPS = 1e-6

# Either interval flavour, for the helpers that only touch start and end.
_IntervalT = TypeVar("_IntervalT", CompartmentInterval, SignalInterval)


def _candidate_range(iv_starts: list[int], iv_ends: list[int], s: int, e: int) -> tuple[int, int]:
    """
    Half-open index range of intervals that can overlap the bead range `[s, e]`.

    Exact for the sorted non-overlapping intervals a binned track produces.  The
    caller still measures each candidate, so an overlapping source only costs
    wasted comparisons.
    """
    lo = bisect_right(iv_ends, s)
    hi = bisect_right(iv_starts, e)
    return lo, hi


def bin_compartments(
    intervals: list[CompartmentInterval],
    starts: list[int],
    ends: list[int],
) -> tuple[I8Array, F32Array]:
    """
    Reduce compartment intervals onto bead ranges.

    The class is the coverage-weighted majority over the bead, so the call that
    covers most of the bead wins.  The score is the coverage-weighted mean.  A
    bead with no overlapping interval gets `NONE` and 0.0.

    Parameters
    ----------
    intervals : list[CompartmentInterval]
        Sorted by start, for one chromosome.
    starts, ends : list[int]
        Inclusive genomic range per bead.

    Returns
    -------
    (cls, score) : (I8Array, F32Array)
        Both length len(starts).
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


def bin_signal(
    intervals: list[SignalInterval],
    starts: list[int],
    ends: list[int],
) -> F32Array:
    """
    Reduce a continuous track onto bead ranges as a coverage-weighted mean.

    A bead with no overlapping interval gets the mean over the beads that do have
    one, so an unmeasured gap reads as typical rather than as silent.  With no
    overlap anywhere the result is all zeros.

    Parameters
    ----------
    intervals : list[SignalInterval]
        Sorted by start, for one chromosome.
    starts, ends : list[int]
        Inclusive genomic range per bead.
    """
    n = len(starts)
    out: F32Array = np.zeros(n, dtype=np.float32)
    if not intervals:
        return out

    iv_starts = [iv.start for iv in intervals]
    iv_ends = [iv.end for iv in intervals]
    covered: list[int] = []

    for i in range(n):
        s, e = starts[i], ends[i] + 1
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
        if total == 0:
            continue
        out[i] = acc / total
        covered.append(i)

    if covered and len(covered) < n:
        fill = float(out[covered].mean())
        mask = np.ones(n, dtype=np.bool_)
        mask[covered] = False
        out[mask] = fill

    return out


def normalize_accessibility(values: F32Array) -> F32Array:
    """
    Map a raw signal onto the [0, 1] accessibility scale.

    `log(v + 1e-6)` then min-max, following HiP-HoP.  The log compresses the long
    upper tail of a p-value or fold-change track so a handful of strong peaks do
    not flatten everything else to zero.  A constant track maps to all zeros.
    """
    if values.size == 0:
        return values.astype(np.float32)
    v = np.log(np.maximum(values, 0.0).astype(np.float64) + _LOG_EPS)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return ((v - lo) / (hi - lo)).astype(np.float32)


def normalize_signal_map(signal: SignalMap) -> SignalMap:
    """
    Rescale a whole signal map onto [0, 1] in place.

    Normalising once over every loaded interval, rather than per region at scoring
    time, is what makes the bridging strength comparable across regions.  A region
    that is closed throughout would otherwise still produce beads at 1.0 after a
    local min-max.  The extent is whatever was loaded, so a whole-genome run
    normalises genome wide and a single-chromosome run normalises over that
    chromosome.

    Mutates and returns `signal`.
    """
    flat = [iv for lst in signal.values() for iv in lst]
    if not flat:
        return signal
    raw = np.array([iv.value for iv in flat], dtype=np.float32)
    scaled = normalize_accessibility(raw)
    for iv, v in zip(flat, scaled.tolist(), strict=True):
        iv.value = float(v)
    return signal


def slice_intervals(intervals: list[_IntervalT], lo: int, hi: int) -> list[_IntervalT]:
    """
    Intervals overlapping the inclusive genomic range `[lo, hi]`.

    Used to hand one IB just the slice of a chromosome track it can see, so a
    per-IB task stays self-contained without copying the whole chromosome.
    """
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
    """
    Orient one chromosome so A correlates positively with the signal.

    Returns True when the decision was made.  A track that overlaps nothing, or
    whose signal is constant across the compartment intervals, gives no evidence
    and returns False.
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
    """
    Orient one chromosome so the side holding more loop anchors becomes A.

    MultiMM's heuristic.  Returns True when at least one anchor landed in a
    signed interval, and False when the counts give no evidence.
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
    """
    Fill in A and B calls for value-only compartment tracks.

    The sign of a Hi-C eigenvector is arbitrary and can differ per chromosome, so
    a raw PC1 track has to be oriented before it means anything.  Chromosomes
    whose intervals already carry a class from a label source are left alone.

    Rules in order.  Correlate against `phasing_signal` and orient so A is the
    high side.  Otherwise put the side holding more loop anchors on A.  When
    neither applies the chromosome stays `NONE`, which leaves the compartment
    terms inert there rather than segregating it backwards.

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
