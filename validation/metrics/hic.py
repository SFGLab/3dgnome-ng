"""Hi-C contact-map correlation metrics.

These check whether a reconstructed structure reproduces the experimental contact map. We build a
simulated contact map from the 3D structure by counting bead-pairs within a contact radius and
binning them to the Hi-C grid, then correlate it with the observed Hi-C, a 4DN .mcool read via
cooler. The metrics are:

  * SCC, the stratum-adjusted correlation coefficient. Hicrep-style and distance-decay-aware.
  * Pearson on log1p contacts and Spearman, both secondary.
  * insulation-score correlation, for domain-boundary agreement.
  * multimm_faithful_pearson, the decay-retained inverse-distance (d+1)^-3 Pearson. MultiMM's
    metric approach reproduced faithfully. See docs/multimm/README.md.

Both maps are binned on the same genomic grid, the cooler's, so the comparison is exact.
Needs the validation extra via pip install -e .[validation], which is cooler, cooltools and scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnome3d.types import F64Array, I64Array


def observed_hic(
    mcool_path: str, region: str, binsize: int, balance: bool = False
) -> tuple[F64Array, I64Array]:
    """Read the observed Hi-C submatrix for region at binsize from a .mcool.

    Returns a tuple of the (B,B) float64 matrix and the (B,) bin_start_bp array. NaNs from
    unmappable bins become 0.
    """
    import cooler

    # Snap to the nearest resolution actually present in the .mcool, since not every binsize
    # exists. For example mcools often have 25 kb but not 20 kb. Single-resolution coolers open
    # directly.
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
    """Simulated contact map on the Hi-C bin grid. Count bead-pairs within contact_radius in 3D,
    aggregated by the bins their genomic midpoints fall in. This mirrors how Hi-C tallies ligations
    between loci. Symmetric with a zero self-diagonal."""
    from scipy.spatial import cKDTree

    nbins = len(bin_starts)
    last_edge = int(bin_starts[-1]) + binsize
    bidx = np.searchsorted(bin_starts, mids, side="right") - 1
    in_grid = (bidx >= 0) & (bidx < nbins) & (mids < last_edge)

    C = np.zeros((nbins, nbins), dtype=np.float64)
    # Only the within-radius pairs matter, so a KD-tree finds them in about O(N log N) instead of
    # the full O(N²) distance matrix, which is the post-MC Hi-C hotspot. query_pairs gives
    # unordered i<j, so we add both directions to keep C symmetric.
    pairs = cKDTree(coords).query_pairs(contact_radius, output_type="ndarray")
    if pairs.size == 0:
        return C
    ii, jj = pairs[:, 0], pairs[:, 1]
    ok = in_grid[ii] & in_grid[jj]
    bi, bj = bidx[ii][ok], bidx[jj][ok]
    np.add.at(C, (bi, bj), 1.0)
    np.add.at(C, (bj, bi), 1.0)
    return C


def _scc(A: F64Array, B: F64Array, max_sep_bins: int) -> float:
    """Stratum-adjusted correlation coefficient from hicrep and Wang et al. For each
    genomic-distance stratum, that is matrix diagonal d, the Pearson r_d is weighted by
    N_d * std(A_d) * std(B_d)."""
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
    """Per-bin insulation score. Mean cross-contacts in the square straddling each bin, where a low
    value marks a boundary. NaN where the window runs off the edge."""
    n = C.shape[0]
    ins = np.full(n, np.nan)
    for i in range(window_bins, n - window_bins):
        ins[i] = C[i - window_bins : i, i + 1 : i + 1 + window_bins].mean()
    return ins


def contact_correlation(
    c_sim: F64Array, c_obs: F64Array, min_sep_bins: int = 1, insulation_window: int = 5
) -> dict[str, float]:
    """Correlate simulated against observed contact maps. Returns SCC as the primary, Pearson on
    log1p, Spearman, and insulation-profile Pearson. Off-diagonal only, since the genomic-decay
    dominated diagonals add no information. NaN-safe, returning nan for a metric that cannot be
    computed."""
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
    """End-to-end for one structure. Simulated contacts against observed Hi-C give the metrics."""
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize
    c_sim = simulated_contacts(coords, mids, bin_starts, eff, contact_radius)
    return contact_correlation(c_sim, c_obs)


# --------------------------------------------------------------------------- cross-data correlation (ChIA-PET vs Hi-C)


def contact_list_heatmap(
    contacts: list[tuple[int, int, float]], bin_starts: I64Array, binsize: int
) -> F64Array:
    """Bin a list of (pos_a, pos_b, score) genomic contacts into a symmetric frequency heatmap on
    the given bin grid. For example the input ChIA-PET contact map of clusters and singletons for
    the cross-data correlation check."""
    nbins = len(bin_starts)
    last_edge = int(bin_starts[-1]) + binsize
    C = np.zeros((nbins, nbins))
    starts = np.asarray(bin_starts)
    for pa, pb, sc in contacts:
        if pa >= last_edge or pb >= last_edge or pa < starts[0] or pb < starts[0]:
            continue
        ia = int(np.searchsorted(starts, pa, side="right")) - 1
        ib = int(np.searchsorted(starts, pb, side="right")) - 1
        if 0 <= ia < nbins and 0 <= ib < nbins:
            C[ia, ib] += sc
            if ia != ib:
                C[ib, ia] += sc
    return C


def observed_over_expected(matrix: F64Array, min_sep_bins: int = 1) -> F64Array:
    """Distance-decay-normalized map, observed over expected. Each entry is divided by the mean of
    its genomic-distance diagonal, so the monotone distance-decay both Hi-C and ChIA-PET share is
    stripped and what remains is structure, meaning loops and compartments. This is the standard
    confound removal. Without it a raw Pearson mostly measures that both fall off with distance.
    Diagonals with no signal become 0. Entries within min_sep_bins of the diagonal are left at 0.
    Returns a copy."""
    m = np.asarray(matrix, dtype=np.float64)
    n = m.shape[0]
    oe = np.zeros_like(m)
    for d in range(max(min_sep_bins, 1), n):
        idx = np.arange(n - d)
        diag = m[idx, idx + d]
        pos = diag > 0
        if not pos.any():
            continue
        v = diag / float(diag[pos].mean())
        oe[idx, idx + d] = v
        oe[idx + d, idx] = v
    return oe


def _oe_correlation(a_oe: F64Array, b_oe: F64Array, min_sep_bins: int) -> dict[str, float]:
    """Pearson of log O/E and Spearman of O/E over the upper-triangle bin-pairs where both maps
    have signal, the informative overlap of a sparse ChIA-PET map and dense Hi-C. n_pairs_oe
    reports how many survive. A small count is itself the finding, meaning the data is too sparse
    to correlate."""
    out = {"pearson_oe": float("nan"), "spearman_oe": float("nan"), "n_pairs_oe": 0.0}
    n = a_oe.shape[0]
    if n < 4:
        return out
    iu = np.triu_indices(n, max(min_sep_bins, 1))
    a, b = a_oe[iu], b_oe[iu]
    both = (a > 0) & (b > 0)
    out["n_pairs_oe"] = float(both.sum())
    if both.sum() >= 4:
        la, lb = np.log(a[both]), np.log(b[both])
        if la.std() > 1e-12 and lb.std() > 1e-12:
            out["pearson_oe"] = float(np.corrcoef(la, lb)[0, 1])
        try:
            from scipy.stats import spearmanr

            out["spearman_oe"] = float(spearmanr(a[both], b[both]).statistic)
        except ImportError:
            pass
    return out


def cross_data_correlation(
    chiapet_contacts: list[tuple[int, int, float]],
    mcool_path: str,
    region: str,
    binsize: int,
    balance: bool = True,
    min_sep_bins: int = 1,
) -> dict[str, float]:
    """Cross-data correlation from 3dgnome 2016, Fig. 2. Correlate the model's input ChIA-PET contact heatmap against
    the observed Hi-C on a common bin grid, a data-level check with no structure. Returns the raw
    decay-retained log1p Pearson and SCC together with the decay-stripped O/E correlation, namely
    pearson_oe, spearman_oe and n_pairs_oe. The raw number is confounded by shared distance-decay.
    The O/E number is the structure-vs-structure agreement the paper's ρ≈0.67–0.73 is really about.
    Hi-C is ICE-balanced by default, which is needed for a meaningful O/E."""
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize
    c_chia = contact_list_heatmap(chiapet_contacts, bin_starts, eff)
    out = contact_correlation(c_chia, c_obs, min_sep_bins=min_sep_bins)
    out.update(
        _oe_correlation(
            observed_over_expected(c_chia, min_sep_bins),
            observed_over_expected(c_obs, min_sep_bins),
            min_sep_bins,
        )
    )
    return out


def remove_diagonals(matrix: F64Array, n_diag: int) -> F64Array:
    """MultiMM's diagonal neutralizer, per src/multimm/validation.py::remove_diagonals. Set the
    main diagonal and its n_diag neighbours on both sides to the matrix mean, so the dominant
    near-diagonal band cannot drive the Pearson. Returns a copy."""
    m = matrix.copy()
    n = m.shape[0]
    mean = float(m.mean())
    for d in range(n_diag + 1):
        idx = np.arange(n - d)
        m[idx, idx + d] = mean
        m[idx + d, idx] = mean
    return m


def _bin_centroids(
    coords: F64Array, mids: I64Array, bin_starts: I64Array, binsize: int
) -> F64Array:
    """Mean 3D position of the beads whose genomic midpoint falls in each Hi-C bin, with a NaN row
    for an empty bin. MultiMM mean_downsamples the bead array to the matrix size. Binning by
    genomic midpoint is the correct analogue for our non-uniformly-spaced subanchor beads."""
    nbins = len(bin_starts)
    last_edge = int(bin_starts[-1]) + binsize
    bidx = np.searchsorted(bin_starts, mids, side="right") - 1
    in_grid = (bidx >= 0) & (bidx < nbins) & (mids < last_edge)
    cent = np.full((nbins, 3), np.nan)
    bi, cg = bidx[in_grid], coords[in_grid]
    for b in range(nbins):
        sel = bi == b
        if sel.any():
            cent[b] = cg[sel].mean(0)
    return cent


def multimm_faithful_pearson(
    coords_list: list[F64Array],
    mids: I64Array,
    c_obs: F64Array,
    bin_starts: I64Array,
    binsize: int,
    n_diag: int = 5,
) -> float:
    """MultiMM's actual Hi-C correlation, reproduced faithfully per src/multimm/validation.py,
    rather than the paper's loose "^3/2" wording. The simulated heatmap is 1/(d+1)³ between per-bin
    structure centroids. Note that structure_to_heatmap writes 1/(d+1)**3/2, which by Python
    precedence is 1/(2·(d+1)³), that is a cube with a +1 offset and the ½ a constant. Distances are
    scaled so the median genomically-adjacent centroid step is 1, so the +1 offset is one bead-step
    and matches their bead-spacing units. Averaged over the ensemble. The main and n_diag diagonals
    are set to the mean on both maps, and they use 5, then plain Pearson over the full matrix.
    c_obs should be ICE or balance-normalised, and they use ICE-norm. Empty bins are dropped from
    both."""
    cent0 = _bin_centroids(coords_list[0], mids, bin_starts, binsize)
    valid = ~np.isnan(cent0[:, 0])
    nv = int(valid.sum())
    if nv < 4:
        return float("nan")
    sim = np.zeros((nv, nv))
    for coords in coords_list:
        cent = _bin_centroids(coords, mids, bin_starts, binsize)[valid]
        step = np.linalg.norm(np.diff(cent, axis=0), axis=1)
        scale = float(np.median(step[step > 0])) if np.any(step > 0) else 1.0
        diff = cent[:, None, :] - cent[None, :, :]
        d = np.sqrt((diff * diff).sum(-1)) / scale
        sim += 1.0 / (d + 1.0) ** 3
    sim /= len(coords_list)
    obs = np.asarray(c_obs)[np.ix_(valid, valid)]
    a = remove_diagonals(sim, n_diag).flatten()
    b = remove_diagonals(obs, n_diag).flatten()
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
    """Correlate the ensemble simulated maps against observed Hi-C. Hi-C is a population average, so
    we aggregate over the whole ensemble, our cells, before correlating. Returns the strict hicrep
    SCC plus decay-free Pearson and insulation from contact_correlation. The decay-retained
    inverse-distance Pearson is a separate metric, multimm_faithful_pearson.
    """
    c_obs, bin_starts = observed_hic(mcool_path, region, binsize, balance=balance)
    eff = (
        int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize
    )  # actual resolution used
    c_sim = np.zeros_like(c_obs)
    for coords, mids in zip(coords_list, mids_list, strict=True):
        c_sim += simulated_contacts(coords, mids, bin_starts, eff, contact_radius)
    return contact_correlation(c_sim, c_obs)


# --- A/B compartments -------------------------------------------------------
#
# MultiMM's second validation: the first eigenvector of the structure-derived
# correlation matrix against the eigenvector of the experimental Hi-C. See
# docs/multimm/README.md and docs/epigenome-energy-terms.md.


def compartment_eigenvector(contacts: F64Array, phasing: F64Array | None = None) -> F64Array:
    """First compartment eigenvector of a dense cis contact map.

    Delegates to `cooltools.api.eigdecomp.cis_eig`, the same routine `eigs_cis`
    runs on experimental Hi-C. Using it for the simulated map too means both sides
    of a comparison get identical treatment, which is the only way the correlation
    between them means anything. A hand-rolled O/E-and-PCA reimplementation agreed
    with cooltools at only |r| = 0.39 on real chr1.

    `clip_percentile=99.9` matters and is not optional. It is `eigs_cis`'s own
    default, and without it a handful of outlier pixels dominate the decomposition:
    the top three eigenvalues come out nearly degenerate and the leading vector
    correlates with the real one at only |r| = 0.41. With it, |r| = 1.0000.

    The sign is arbitrary unless `phasing` is given, so callers compare on absolute
    correlation. Unmappable bins come back as 0.
    """
    from cooltools.api import eigdecomp

    n = contacts.shape[0]
    if n < 8:
        return np.zeros(n, dtype=np.float64)

    a = np.asarray(contacts, dtype=np.float64).copy()
    # cis_eig expects a balanced map with unmappable bins as NaN, not zero.
    dead = a.sum(axis=1) <= 0.0
    a[dead, :] = np.nan
    a[:, dead] = np.nan

    try:
        _eigvals, eigvecs = eigdecomp.cis_eig(
            a, n_eigs=1, phasing_track=phasing, clip_percentile=99.9
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.zeros(n, dtype=np.float64)

    out = np.nan_to_num(np.asarray(eigvecs[0], dtype=np.float64))
    return out


def compartment_correlation(
    c_sim: F64Array, c_obs: F64Array, track: F64Array | None = None
) -> dict[str, float]:
    """Compare the structure's compartment eigenvector against the observed one.

    Both eigenvector signs are arbitrary, so the headline number is the absolute
    Pearson correlation. `agreement` is the fraction of bins the two put in the
    same compartment after orienting them to agree on average, which is the more
    interpretable figure.

    `track` optionally supplies the input compartment call, so a run can be checked
    against what it was told rather than only against the Hi-C it never saw.
    """
    e_sim = compartment_eigenvector(c_sim)
    e_obs = compartment_eigenvector(c_obs)
    both = (e_sim != 0.0) & (e_obs != 0.0)
    out: dict[str, float] = {"n_bins": float(int(both.sum()))}
    if int(both.sum()) < 3:
        return {**out, "eig_pearson_abs": float("nan"), "agreement": float("nan")}

    a, b = e_sim[both], e_obs[both]
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")
    out["eig_pearson_abs"] = abs(r) if r == r else float("nan")
    # Orient sim to obs before scoring per-bin agreement.
    sgn = 1.0 if (r == r and r >= 0) else -1.0
    out["agreement"] = float(np.mean(np.sign(sgn * a) == np.sign(b)))

    if track is not None:
        t = np.asarray(track, dtype=np.float64)
        m = both & (t != 0.0)
        if int(m.sum()) >= 3 and e_sim[m].std() > 1e-12 and t[m].std() > 1e-12:
            rt = float(np.corrcoef(e_sim[m], t[m])[0, 1])
            out["eig_vs_input_abs"] = abs(rt)
            out["input_agreement"] = float(
                np.mean(np.sign((1.0 if rt >= 0 else -1.0) * e_sim[m]) == np.sign(t[m]))
            )
    return out


def compartment_saddle(
    contacts: F64Array, track: F64Array, n_quantiles: int = 5
) -> dict[str, float]:
    """Compartment enrichment, the saddle statistic.

    Sorts bins by `track`, a signed compartment value, into quantiles, then
    averages observed-over-expected contact within each quantile pair. Strength is
    the enrichment of same-compartment contact over cross-compartment contact,
    using the extreme quantiles:

        strength = (AA + BB) / (2 * AB)

    Above 1 means A sits near A and B near B. This is what the compartment energy
    term acts on directly, so it is a far more sensitive probe of that term than a
    whole-region eigenvector correlation, where the baseline varies by 0.445 across
    regions of one chromosome while the term moves it by 0.007.

    `track` sorts the bins and is normally the input compartment call, the thing
    the model was told. Passing the same track for every arm keeps the quantile
    definition fixed, so a difference in strength is a difference in the structure
    rather than in the binning.

    Two properties to know before reading a number from this.

    It measures *distal* compartmentalization only. The observed-over-expected step
    divides out distance decay, so bringing together same-compartment beads that are
    already close along the chain shows up as nothing. That is the intended
    behaviour, and it means a term acting only at short range will score zero here
    however strongly it acts.

    It returns nan on a degenerate map. A structure compact enough that every pair
    is in contact gives an all-ones O/E and no usable cross-compartment block. Check
    that the contact map is not saturated before trusting a nan, or a strength of
    exactly 1.0.

    Returns strength plus the three raw enrichments, and `n_bins` actually used.
    """
    n = contacts.shape[0]
    t = np.asarray(track, dtype=np.float64)
    if t.shape[0] != n:
        raise ValueError(f"track length {t.shape[0]} != contact matrix size {n}")

    oe = observed_over_expected(contacts, min_sep_bins=1)
    usable = (t != 0.0) & np.isfinite(t) & (contacts.sum(axis=1) > 0)
    if int(usable.sum()) < 2 * n_quantiles:
        return {
            "strength": float("nan"),
            "aa": float("nan"),
            "bb": float("nan"),
            "ab": float("nan"),
            "n_bins": float(int(usable.sum())),
        }

    idx = np.flatnonzero(usable)
    order = idx[np.argsort(t[idx])]
    groups = np.array_split(order, n_quantiles)
    lo, hi = groups[0], groups[-1]  # most B, most A

    def mean_block(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        blk = oe[np.ix_(a, b)]
        vals = blk[blk > 0.0]
        return float(vals.mean()) if vals.size else float("nan")

    aa, bb, ab = mean_block(hi, hi), mean_block(lo, lo), mean_block(hi, lo)
    strength = (aa + bb) / (2.0 * ab) if ab and ab == ab and ab > 0 else float("nan")
    return {
        "strength": strength,
        "aa": aa,
        "bb": bb,
        "ab": ab,
        "n_bins": float(len(idx)),
    }
