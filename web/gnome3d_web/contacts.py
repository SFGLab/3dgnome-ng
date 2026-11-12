"""Contact maps of a job on one grid: the Hi-C, the model as MultiMM's 1/(d+1)^3 map or as
counted contacts, ICE balancing, and the stratum adjusted correlation of the validation battery
(validation/metrics/hic.py), so the site and the battery report one number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CONTACT_RADIUS = 1.5
RESOLUTIONS = (1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
MAX_BINS = 800


def bins_for(start: int, end: int, res: int) -> np.ndarray:
    """The left edges of the bins of a window."""
    return np.arange(start - start % res, end, res)


def median_bond(positions: list[np.ndarray]) -> float:
    """The median distance between consecutive beads over the structures, the battery's contact
    radius."""
    steps = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in positions if len(p) > 1] or [np.array([CONTACT_RADIUS])])
    return float(np.median(steps))


def multimm_map(positions: list[np.ndarray], mids: np.ndarray, start: int, end: int, res: int) -> np.ndarray:
    """MultiMM's map as the battery computes it: per structure the distance between bin centroids
    in units of the median step between neighbouring bins, 1/(d+1)^3, averaged over the
    structures. A bin with no bead stays 0.

    @param positions the bead positions of each structure, in one bead order
    @param mids the genomic midpoints of the beads in that order
    """
    edges = bins_for(start, end, res)
    n = len(edges)
    idx = (mids - edges[0]) // res
    inside = (idx >= 0) & (idx < n)
    filled = np.zeros(n, dtype=bool)
    filled[np.unique(idx[inside])] = True
    out = np.zeros((n, n))
    if filled.sum() < 2:
        return out
    for pos in positions:
        cent = np.zeros((n, 3))
        np.add.at(cent, idx[inside], pos[inside])
        cent[filled] /= np.bincount(idx[inside], minlength=n)[filled][:, None]
        c = cent[filled]
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        scale = float(np.median(step[step > 0])) if np.any(step > 0) else 1.0
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1) / scale
        out[np.ix_(filled, filled)] += 1.0 / (d + 1.0) ** 3
    return out / len(positions)


def model_map(positions: list[np.ndarray], mids: np.ndarray, start: int, end: int, res: int, radius: float = CONTACT_RADIUS) -> np.ndarray:
    """Bead pairs closer than `radius`, summed over the structures, binned by bead midpoint.

    @param positions the bead positions of each structure, in one bead order
    @param mids the genomic midpoints of the beads in that order
    """
    edges = bins_for(start, end, res)
    n = len(edges)
    idx = (mids - edges[0]) // res
    inside = (idx >= 0) & (idx < n)
    m = np.zeros((n, n))
    for pos in positions:
        pairs = cKDTree(pos[inside]).query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            continue
        b = idx[inside][pairs]
        np.add.at(m, (b[:, 0], b[:, 1]), 1)
        off = b[:, 0] != b[:, 1]
        np.add.at(m, (b[off, 1], b[off, 0]), 1)
    return m


def hic_map(singletons: pd.DataFrame, chrom: str, start: int, end: int, res: int) -> np.ndarray:
    """The contact pairs of a BEDPE file in a window, each end placed by its start column."""
    edges = bins_for(start, end, res)
    n = len(edges)
    t = singletons[(singletons[0] == chrom) & (singletons[3] == chrom)]
    i = (t[1].to_numpy() - edges[0]) // res
    j = (t[4].to_numpy() - edges[0]) // res
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    m = np.zeros((n, n))
    w = t[6].to_numpy(dtype=float)[ok]
    np.add.at(m, (i[ok], j[ok]), w)
    off = i[ok] != j[ok]
    np.add.at(m, (j[ok][off], i[ok][off]), w[off])
    return m


def ice(m: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Iterative correction: rows and columns scaled until every non empty row has one sum."""
    out = m.astype(float).copy()
    live = out.sum(axis=1) > 0
    if not live.any():
        return out
    for _ in range(iterations):
        s = out.sum(axis=1)
        s[~live] = 1
        s = s / s[live].mean()
        out = out / s[:, None] / s[None, :]
        if np.abs(s[live] - 1).max() < 1e-6:
            break
    out[~live, :] = 0
    out[:, ~live] = 0
    return out


def scc(a: np.ndarray, b: np.ndarray, res: int, max_bp: int = 2_000_000) -> float | None:
    """The stratum adjusted correlation as the battery computes it, validation/metrics/hic.py
    _scc: the Pearson correlation of each whole diagonal up to max_bp, zeros included, weighted by
    its length and the two standard deviations. None when no diagonal has two maps with
    variance."""
    n = len(a)
    num = den = 0.0
    for k in range(1, min(n - 1, max_bp // res) + 1):
        x, y = np.diagonal(a, k), np.diagonal(b, k)
        if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(r):
            continue
        w = x.size * float(x.std()) * float(y.std())
        num += w * r
        den += w
    return num / den if den > 0 else None


def coarsest_fit(start: int, end: int) -> int:
    """The finest resolution with no more than MAX_BINS bins over a window."""
    return next(r for r in RESOLUTIONS if len(bins_for(start, end, r)) <= MAX_BINS)
