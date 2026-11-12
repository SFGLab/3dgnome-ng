"""Contact maps of a job: the model's bead contacts and the Hi-C the job read, binned on one
grid, ICE balancing, and the stratum adjusted correlation between two maps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

CONTACT_RADIUS = 1.5
RESOLUTIONS = (1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
MAX_BINS = 800


def bins_for(start: int, end: int, res: int) -> np.ndarray:
    """The left edges of the bins of a window."""
    return np.arange(start - start % res, end, res)


def model_map(positions: list[np.ndarray], mids: np.ndarray, start: int, end: int, res: int) -> np.ndarray:
    """Bead pairs closer than 1.5 bead units, summed over the structures, binned by bead midpoint.

    @param positions the bead positions of each structure, in one bead order
    @param mids the genomic midpoints of the beads in that order
    """
    edges = bins_for(start, end, res)
    n = len(edges)
    idx = (mids - edges[0]) // res
    inside = (idx >= 0) & (idx < n)
    m = np.zeros((n, n))
    for pos in positions:
        pairs = cKDTree(pos[inside]).query_pairs(CONTACT_RADIUS, output_type="ndarray")
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
    """The stratum adjusted correlation of HiCRep, Yang et al. 2017, without smoothing: the
    Pearson correlation of each diagonal up to max_bp, weighted by its length and the spread of
    its ranks. None when no diagonal has two maps with variance."""
    n = len(a)
    num = den = 0.0
    for k in range(1, min(n, max_bp // res + 1)):
        x, y = np.diagonal(a, k), np.diagonal(b, k)
        keep = (x > 0) | (y > 0)
        x, y = x[keep], y[keep]
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = len(x) * float(np.sqrt(np.var(rankdata(x) / len(x)) * np.var(rankdata(y) / len(y))))
        num += w * r
        den += w
    return num / den if den > 0 else None


def coarsest_fit(start: int, end: int) -> int:
    """The finest resolution with no more than MAX_BINS bins over a window."""
    return next(r for r in RESOLUTIONS if len(bins_for(start, end, r)) <= MAX_BINS)
