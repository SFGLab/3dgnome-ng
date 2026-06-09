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

    c = cooler.Cooler(f"{mcool_path}::/resolutions/{binsize}")
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
    c_sim = simulated_contacts(coords, mids, bin_starts, binsize, contact_radius)
    return contact_correlation(c_sim, c_obs)


def ensemble_hic_correlation(
    coords_list: list[F64Array],
    mids_list: list[I64Array],
    mcool_path: str,
    region: str,
    binsize: int,
    contact_radius: float,
    balance: bool = False,
) -> dict[str, float]:
    """Correlate the ENSEMBLE-summed simulated contact map vs observed Hi-C.

    Hi-C is a population average over cells, so the right comparison aggregates the simulated
    contacts over the whole ensemble (our "cells") into one map, then correlates once — more
    faithful and far less noisy than averaging per-structure correlations.
    """
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    c_sim = np.zeros_like(c_obs)
    for coords, mids in zip(coords_list, mids_list, strict=True):
        c_sim += simulated_contacts(coords, mids, bin_starts, binsize, contact_radius)
    return contact_correlation(c_sim, c_obs)
