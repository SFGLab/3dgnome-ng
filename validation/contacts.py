"""Hi-C correlation / self-reproduction metric (V1 + V4).

Does a reconstructed structure reproduce the experimental contact map? We build a *simulated*
contact map from the 3D structure (bead-pairs within a contact radius, binned to the Hi-C grid)
and correlate it with the *observed* Hi-C (a 4DN ``.mcool`` read via cooler), using:

  * **SCC** — stratum-adjusted correlation coefficient (hicrep-style; distance-decay-aware) —
    the Hi-C standard and the sweep's primary objective.
  * Pearson (on log1p contacts) and Spearman — secondary.
  * insulation-score correlation — domain-boundary agreement.

Both maps are binned on the SAME genomic grid (the cooler's), so the comparison is exact.
Needs the ``validation`` extra (``pip install -e .[validation]`` — cooler + scipy).
"""

from __future__ import annotations

import numpy as np

from gnome3d.types import F64Array, I64Array


def observed_hic(
    mcool_path: str, region: str, binsize: int, balance: bool = False
) -> tuple[F64Array, I64Array]:
    """Read the observed Hi-C submatrix for ``region`` at ``binsize`` from a .mcool.

    Returns (matrix (B,B) float64, bin_start_bp (B,)). NaNs (unmappable bins) -> 0.
    """
    import cooler

    # Snap to the nearest resolution actually present in the .mcool (not every binsize exists —
    # e.g. mcools often have 25 kb but not 20 kb). Single-resolution coolers open directly.
    avail = sorted(
        int(p.rsplit("/", 1)[-1])
        for p in cooler.fileops.list_coolers(mcool_path)
        if p.rsplit("/", 1)[-1].isdigit()
    )
    if avail:
        res = binsize if binsize in avail else min(avail, key=lambda r: abs(r - binsize))
        if res != binsize:
            print(f"[contacts] binsize {binsize} absent in mcool; using nearest available {res}")
        c = cooler.Cooler(f"{mcool_path}::/resolutions/{res}")
    else:
        c = cooler.Cooler(mcool_path)
    mat = np.asarray(c.matrix(balance=balance).fetch(region), dtype=np.float64)
    starts = c.bins().fetch(region)["start"].to_numpy().astype(np.int64)
    return np.nan_to_num(mat), starts


def simulated_contacts(
    coords: F64Array, mids: I64Array, bin_starts: I64Array, binsize: int, contact_radius: float
) -> F64Array:
    """Simulated contact map on the Hi-C bin grid: count bead-pairs within ``contact_radius``
    (3D), aggregated by the bins their genomic midpoints fall in. Mirrors how Hi-C tallies
    ligations between loci. Symmetric, zero diagonal-self (a bead with itself)."""
    nbins = len(bin_starts)
    last_edge = int(bin_starts[-1]) + binsize
    bidx = np.searchsorted(bin_starts, mids, side="right") - 1
    in_grid = (bidx >= 0) & (bidx < nbins) & (mids < last_edge)

    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))
    close = d < contact_radius
    np.fill_diagonal(close, False)
    ii, jj = np.where(close)
    ok = in_grid[ii] & in_grid[jj]
    C = np.zeros((nbins, nbins), dtype=np.float64)
    np.add.at(C, (bidx[ii][ok], bidx[jj][ok]), 1.0)
    return C


def _scc(A: F64Array, B: F64Array, max_sep_bins: int) -> float:
    """Stratum-adjusted correlation coefficient (hicrep / Wang et al.): per genomic-distance
    stratum (matrix diagonal d), Pearson r_d weighted by N_d * std(A_d) * std(B_d)."""
    num = den = 0.0
    n = A.shape[0]
    for dd in range(1, min(max_sep_bins, n - 1) + 1):
        a, b = np.diagonal(A, dd), np.diagonal(B, dd)
        if a.size < 2:
            continue
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(r):
            continue
        w = a.size * sa * sb
        num += w * r
        den += w
    return num / den if den > 0 else float("nan")


def _insulation(C: F64Array, window_bins: int) -> F64Array:
    """Per-bin insulation score: mean cross-contacts in the square straddling each bin
    (low = boundary). NaN where the window runs off the edge."""
    n = C.shape[0]
    ins = np.full(n, np.nan)
    for i in range(window_bins, n - window_bins):
        ins[i] = C[i - window_bins : i, i + 1 : i + 1 + window_bins].mean()
    return ins


def contact_correlation(
    c_sim: F64Array, c_obs: F64Array, min_sep_bins: int = 1, insulation_window: int = 5
) -> dict[str, float]:
    """Correlate simulated vs observed contact maps. Returns SCC (primary), Pearson (log1p),
    Spearman, and insulation-profile Pearson. Off-diagonal only (genomic-decay-dominated
    diagonals add no information). NaN-safe; returns nan for a metric that can't be computed."""
    n = c_obs.shape[0]
    out: dict[str, float] = {
        "scc": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "insulation": float("nan"),
        "n_bins": float(n),
    }
    if n < 4:
        return out
    iu = np.triu_indices(n, min_sep_bins)
    a, b = c_sim[iu], c_obs[iu]
    out["scc"] = _scc(c_sim, c_obs, n - 1)
    if a.std() > 1e-12 and b.std() > 1e-12:
        out["pearson"] = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
        try:
            from scipy.stats import spearmanr

            out["spearman"] = float(spearmanr(a, b).statistic)
        except ImportError:
            pass
    ia, ib = _insulation(c_sim, insulation_window), _insulation(c_obs, insulation_window)
    ok = np.isfinite(ia) & np.isfinite(ib) & (ia > 0) & (ib > 0)
    if ok.sum() >= 3 and np.log(ia[ok]).std() > 1e-12 and np.log(ib[ok]).std() > 1e-12:
        out["insulation"] = float(np.corrcoef(np.log(ia[ok]), np.log(ib[ok]))[0, 1])
    return out


def hic_correlation(
    coords: F64Array,
    mids: I64Array,
    mcool_path: str,
    region: str,
    binsize: int,
    contact_radius: float,
    balance: bool = False,
) -> dict[str, float]:
    """End-to-end for ONE structure: simulated contacts vs observed Hi-C -> metrics."""
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize
    c_sim = simulated_contacts(coords, mids, bin_starts, eff, contact_radius)
    return contact_correlation(c_sim, c_obs)


def inverse_distance_heatmap(
    coords_list: list[F64Array],
    mids: I64Array,
    bin_starts: I64Array,
    binsize: int,
    eps: float = 1e-6,
) -> F64Array:
    """Ensemble-mean **inverse-distance** heat map on the Hi-C bin grid: per bin-pair, the mean of
    1/d over all bead-pairs (and ensemble members) whose midpoints fall in those bins. This is the
    MultiMM contact surrogate (a smooth, dense 1/d map, vs the sparse hard-radius contact count).
    Computed via a one-hot bin matrix B so the binning is a BLAS matmul (Bᵀ·(1/d)·B)."""
    nbins = len(bin_starts)
    last_edge = int(bin_starts[-1]) + binsize
    bidx = np.searchsorted(bin_starts, mids, side="right") - 1
    in_grid = (bidx >= 0) & (bidx < nbins) & (mids < last_edge)
    bi = bidx[in_grid]
    n = bi.size
    if n < 2:
        return np.zeros((nbins, nbins))
    onehot = np.zeros((n, nbins))
    onehot[np.arange(n), bi] = 1.0
    counts = onehot.sum(0)
    pair_cnt = np.outer(counts, counts)  # bead-pairs per bin-pair (off-diagonal exact; diag unused)
    acc = np.zeros((nbins, nbins))
    for coords in coords_list:
        c = coords[in_grid]
        diff = c[:, None, :] - c[None, :, :]
        inv = 1.0 / np.maximum(np.sqrt((diff * diff).sum(axis=2)), eps)
        np.fill_diagonal(inv, 0.0)
        acc += onehot.T @ inv @ onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pair_cnt > 0, acc / (pair_cnt * len(coords_list)), 0.0)


def multimm_pearson(
    inv_map: F64Array, c_obs: F64Array, min_sep_bins: int = 1, sim_power: float = 1.5
) -> float:
    """MultiMM-style Hi-C correlation: Pearson of the simulated inverse-distance heat map (raised
    to ``sim_power`` = 3/2, per the paper) vs observed Hi-C, **main diagonal excluded but the
    genomic distance-decay retained**. MultiMM reports ≈0.70 here (random structures <0.40). This
    is the literature-comparable metric — unlike SCC, which strips the decay (see docs/validation.md
    §1bis)."""
    n = c_obs.shape[0]
    if n < 4:
        return float("nan")
    iu = np.triu_indices(n, min_sep_bins)
    a = np.power(np.maximum(inv_map[iu], 0.0), sim_power)
    b = c_obs[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ensemble_hic_correlation(
    coords_list: list[F64Array],
    mids_list: list[I64Array],
    mcool_path: str,
    region: str,
    binsize: int,
    contact_radius: float,
    balance: bool = False,
) -> dict[str, float]:
    """Correlate the ENSEMBLE simulated maps vs observed Hi-C. Hi-C is a population average, so we
    aggregate over the whole ensemble (our "cells") before correlating. Returns the strict hicrep
    **SCC** + decay-free Pearson/insulation (``contact_correlation``) AND the literature-comparable
    MultiMM **inverse-distance Pearson** (decay retained) under ``multimm_pearson``.
    """
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize  # actual res used
    c_sim = np.zeros_like(c_obs)
    for coords, mids in zip(coords_list, mids_list, strict=True):
        c_sim += simulated_contacts(coords, mids, bin_starts, eff, contact_radius)
    out = contact_correlation(c_sim, c_obs)
    inv = inverse_distance_heatmap(coords_list, mids_list[0], bin_starts, eff)
    out["multimm_pearson"] = multimm_pearson(inv, c_obs)
    return out
