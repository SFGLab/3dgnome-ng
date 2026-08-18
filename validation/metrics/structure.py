"""Resolution-independent structure-validation metrics.

Pure functions over reconstructed structures, either BeadOut lists or plain (N,3) arrays.
Each maps to a check in docs/validation.md.

  self-consistency    self_consistency        input contact strength vs output 3D distance
  scaling laws        distance_scaling        mean pairwise distance vs genomic separation, a power law
  scaling laws        contact_probability     P(contact) vs genomic separation, a power law
  ensemble diversity  dab_matrix / diversity  Szałaj-2016 mirror-insensitive inter-structure distance
  EV                  overlap_fraction        non-bonded bead interpenetrations, the EV and confinement test

There are no gnome3d imports here. It takes already-produced coordinates so it is trivially
unit-testable and reusable on any structure source. validate.py drives the pipeline and calls these.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gnome3d.types import F64Array, I64Array

# A bead is anything indexable as (start, end, x, y, z, ...). That is a gnome3d BeadOut,
# or the 6-tuples the integration harness uses. We read indices 0 (start) and 2, 3, 4 (xyz).
Bead = Sequence[Any]


def to_arrays(beads: Sequence[Bead]) -> tuple[F64Array, I64Array]:
    """(N,3) float64 coordinates and (N,) int64 genomic midpoints from a bead list.

    Midpoint is (start+end)//2. Beads are assumed genomic-sorted, which run_region output is.
    """
    coords = np.array([(b[2], b[3], b[4]) for b in beads], dtype=np.float64)
    mids = np.array([(int(b[0]) + int(b[1])) // 2 for b in beads], dtype=np.int64)
    return coords, mids


def radius_of_gyration(coords: F64Array) -> float:
    """Rg = sqrt(mean ||r_i - r_cm||^2)."""
    c = coords - coords.mean(axis=0)
    return float(np.sqrt((c * c).sum(axis=1).mean()))


def bond_lengths(coords: F64Array) -> F64Array:
    """Consecutive bead-bead distances along the chain (N-1,)."""
    d = np.diff(coords, axis=0)
    return np.sqrt((d * d).sum(axis=1))


def _pairwise(coords: F64Array) -> F64Array:
    """Full (N,N) Euclidean distance matrix via the gram-matrix form.

    Uses a BLAS matmul with only an N×N temporary. This avoids the (N,N,3) broadcast, which needs
    about 3× the memory and is far slower at N of a few thousand.
    """
    g = coords @ coords.T
    sq = np.einsum("ii->i", g)  # squared norms are the gram diagonal
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)  # clip tiny negatives from round-off
    return np.sqrt(d2)


# --------------------------------------------------------------------------- EV / confinement


def overlap_fraction(
    coords: F64Array, radius: float, skip_neighbors: int = 1
) -> tuple[float, int, int]:
    """Fraction of non-bonded bead pairs closer than radius.

    These are the physically impossible overlaps the 2016 paper admitted its model produced without
    excluded volume. Excluded volume and confinement should drive this down.

    Pairs with |i-j| <= skip_neighbors are bonded neighbours and excluded.
    Returns (fraction, n_overlapping, n_pairs_considered).
    """
    n = len(coords)
    if n < 2 or radius <= 0:
        return 0.0, 0, 0
    d = _pairwise(coords)
    iu, ju = np.triu_indices(n, k=skip_neighbors + 1)
    if iu.size == 0:
        return 0.0, 0, 0
    pair_d = d[iu, ju]
    n_over = int((pair_d < radius).sum())
    return n_over / pair_d.size, n_over, int(pair_d.size)


def overlap_fraction_binned(
    coords: F64Array,
    mids: I64Array,
    resolution_bp: int,
    radius_factor: float = 0.5,
    skip_neighbors: int = 1,
) -> tuple[float, int, int]:
    """Resolution-normalized overlap fraction.

    Coarse-grain the structure to fixed resolution_bp genomic bins, one centroid per occupied bin
    in genomic order, before counting overlaps. Structures at different bead resolutions then
    compare on the same footing, for example the reference near 4.5 kb versus dynamic-subanchor
    near 1 kb. Without this the finer structure has more, denser beads and its raw overlap_fraction
    is inflated purely by bead density rather than by worse physics.

    The contact radius is radius_factor times the median adjacent-centroid spacing, so it scales
    with the common resolution rather than the native bead size. Returns (fraction, n_over, n_pairs).
    """
    coords = np.asarray(coords)
    mids = np.asarray(mids)
    if len(coords) < 2:
        return 0.0, 0, 0
    binidx = (mids // resolution_bp).astype(np.int64)
    uniq, inv = np.unique(
        binidx, return_inverse=True
    )  # uniq is genomic-sorted, inv maps bead to bin
    if uniq.size < 2:
        return 0.0, 0, 0
    k = uniq.size
    cnt = np.bincount(inv, minlength=k).astype(
        np.float64
    )  # per-bin counts for the centroid average
    cent = np.stack(
        [np.bincount(inv, weights=coords[:, a], minlength=k) / cnt for a in range(3)], axis=1
    )
    step = np.linalg.norm(np.diff(cent, axis=0), axis=1)
    med = float(np.median(step[step > 0])) if np.any(step > 0) else 0.0
    if med <= 0:
        return 0.0, 0, 0
    return overlap_fraction(cent, radius_factor * med, skip_neighbors)


def max_extent(coords: F64Array) -> float:
    """Max distance of any bead from the centroid. Confinement should bound this."""
    c = coords - coords.mean(axis=0)
    return float(np.sqrt((c * c).sum(axis=1)).max())


# --------------------------------------------------------------------------- scaling laws


def _loglog_bins(sep: F64Array, val: F64Array, n_bins: int) -> tuple[F64Array, F64Array, float]:
    """Bin (sep, val) into n_bins log-spaced separation bins, take per-bin means, and fit a power
    law val ~ sep^slope on the log-log bin means. Returns (bin_sep, bin_mean_val, slope). NaNs and
    empties are dropped before the fit."""
    pos = sep > 0
    sep, val = sep[pos], val[pos]
    if sep.size < 2:
        return np.array([]), np.array([]), float("nan")
    edges = np.logspace(np.log10(sep.min()), np.log10(sep.max() + 1), n_bins + 1)
    idx = np.clip(np.digitize(sep, edges) - 1, 0, n_bins - 1)
    # per-bin means via bincount in a single C pass instead of an n_bins by N python loop. The
    # arrays here are the full ~N²/2 condensed pairs, so this is the post-MC hotspot.
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sep_sum = np.bincount(idx, weights=sep, minlength=n_bins)
    val_sum = np.bincount(idx, weights=val, minlength=n_bins)
    nz = cnt > 0
    bs, bv = sep_sum[nz] / cnt[nz], val_sum[nz] / cnt[nz]
    ok = (bs > 0) & (bv > 0)
    if ok.sum() < 2:
        return bs, bv, float("nan")
    slope = float(np.polyfit(np.log10(bs[ok]), np.log10(bv[ok]), 1)[0])
    return bs, bv, slope


def distance_scaling(
    coords: F64Array, mids: I64Array, n_bins: int = 20
) -> tuple[F64Array, F64Array, float]:
    """Scaling-law metric. Mean pairwise 3D distance vs genomic separation, as a power law.
    Returns (separation_bins, mean_distance, exponent). A sane polymer has a positive exponent, so
    distance grows with separation, typically around 0.2 to 0.5."""
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)
    d = _pairwise(coords)[iu, ju]
    sep = np.abs(mids[iu] - mids[ju]).astype(np.float64)
    return _loglog_bins(sep, d, n_bins)


def contact_probability(
    coords: F64Array, mids: I64Array, radius: float, n_bins: int = 20
) -> tuple[F64Array, F64Array, float]:
    """Scaling-law metric. Contact probability P(d < radius) vs genomic separation, as a power law.
    Returns (separation_bins, contact_prob, exponent). A sane polymer has a negative exponent, so
    contacts decay with separation, typically around -0.75 to -1.5."""
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)
    d = _pairwise(coords)[iu, ju]
    sep = np.abs(mids[iu] - mids[ju]).astype(np.float64)
    contact = (d < radius).astype(np.float64)
    return _loglog_bins(sep, contact, n_bins)


# --------------------------------------------------------------------------- genome-structure laws
#
# Polymer-physics scaling laws every chromatin model should reproduce. These come from the Hi-C
# literature, not from the 3dgnome papers. Lieberman-Aiden 2009 measured contact probability
# decaying as P(s) ~ s^-1 over roughly 500 kb to 7 Mb and proposed the fractal globule, whose
# size scales as R(s) ~ s^(1/3). An equilibrium or ideal chain instead gives 1/2.
#
# The 3dgnome 2016 paper does not validate these exponents. It uses power laws as model inputs.
# Methods, Subloop level, sets the preferred distance between consecutive subanchors to
# d = N^a for genomic span N, which is the genomic_dist_power setting. Methods, heat map
# normalization, converts frequency to distance as d = c·f^-α with α = 0.6 from a ranking analysis
# citing Rousseau 2011, which is the freq_dist_power setting. Neither is the exponent measured here.
# The paper's α and the contact-decay α below are different quantities that share a letter.
#
# We fit each law as a power law on log-log bin-means over the scaling window only. That excludes
# sub-resolution small separations, where a single CCD blob is flat, and the saturated tail. We
# report the exponent and the log-log R². A law holds when the fit is a power law, meaning high R²,
# with the exponent in the biological band. Canonical values appear over large, multi-IB or
# whole-chromosome ranges. A small single-IB region is one globule and legitimately reads flatter.

# (lo, hi, target) is the plausible band and canonical value for the power-law exponent.
LAW_BANDS: dict[str, tuple[float, float, float]] = {
    "dist_exp": (0.15, 0.60, 0.33),  # R(s) ~ s^β, fractal globule 1/3, ideal chain 1/2
    "contact_exp": (0.50, 1.60, 1.00),  # P(s) ~ s^-α, chromatin ≈ 1.0
}


def _loglog_fit(
    sep: F64Array, val: F64Array, n_bins: int, lo_bp: float, hi_frac: float
) -> tuple[float, float]:
    """Power-law slope and log-log R² of val vs sep over the scaling window
    [lo_bp, hi_frac·max_sep], on log-spaced bin means. Returns (slope, r2). NaN if too few bins."""
    pos = sep > 0
    sep, val = sep[pos], val[pos]
    if sep.size < 4:
        return float("nan"), float("nan")
    win = (sep >= lo_bp) & (sep <= hi_frac * sep.max())
    if win.sum() < 10:
        return float("nan"), float("nan")
    sep, val = sep[win], val[win]
    edges = np.logspace(np.log10(sep.min()), np.log10(sep.max() + 1), n_bins + 1)
    idx = np.clip(np.digitize(sep, edges) - 1, 0, n_bins - 1)
    bs, bv = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5 and val[m].mean() > 0:
            bs.append(float(sep[m].mean()))
            bv.append(float(val[m].mean()))
    if len(bs) < 3:
        return float("nan"), float("nan")
    lx, ly = np.log10(np.array(bs)), np.log10(np.array(bv))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    sstot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float(((ly - pred) ** 2).sum()) / sstot if sstot > 0 else float("nan")
    return float(slope), r2


def scaling_laws(
    coords: F64Array,
    mids: I64Array,
    radius: float,
    n_bins: int = 15,
    min_sep_bp: float = 30_000.0,
    hi_frac: float = 0.6,
    resolution_bp: int = 0,
    contact_radius_factor: float = 0.5,
) -> dict[str, float]:
    """Genome-structure scaling laws for one structure. Fits the mean-distance R(s)~s^β and
    contact-probability P(s)~s^-α power laws, each as an exponent and log-log R², plus Rg and the
    chain bond-length CV. contact_exp is reported as the positive α.

    resolution_bp > 0 resolution-normalizes the fit. It coarse-grains to one centroid per
    resolution_bp bin before fitting, so structures at different bead resolutions give comparable
    exponents, for example the reference near 4.5 kb versus dynamic-subanchor near 1 kb. P(s) and α in
    particular are otherwise bead-density-dependent, the same confound as raw overlap. The contact
    radius then scales with the centroid spacing, contact_radius_factor times the median step. Rg
    and bond CV are always measured on the native chain, since they describe the real structure and
    flag degeneracy."""
    bonds = bond_lengths(coords)
    bond_mean = float(np.mean(bonds)) if bonds.size else float("nan")
    out = {
        "dist_exp": float("nan"),
        "dist_r2": float("nan"),
        "contact_exp": float("nan"),
        "contact_r2": float("nan"),
        "rg": radius_of_gyration(coords),
        "bond_cv": float(np.std(bonds) / bond_mean)
        if bond_mean and bond_mean > 0
        else float("nan"),
    }
    fit_c, fit_m, fit_r = coords, np.asarray(mids), radius
    if resolution_bp > 0:  # coarse-grain to centroids at a common resolution
        binidx = (np.asarray(mids) // resolution_bp).astype(np.int64)
        uniq, inv = np.unique(binidx, return_inverse=True)
        if uniq.size < 3:
            return out
        k = uniq.size
        cnt = np.bincount(inv, minlength=k).astype(np.float64)
        fit_c = np.stack(
            [np.bincount(inv, weights=coords[:, a], minlength=k) / cnt for a in range(3)], axis=1
        )
        fit_m = np.bincount(inv, weights=np.asarray(mids, dtype=np.float64), minlength=k) / cnt
        step = np.linalg.norm(np.diff(fit_c, axis=0), axis=1)
        fit_r = (
            contact_radius_factor * float(np.median(step[step > 0])) if np.any(step > 0) else radius
        )
    n = len(fit_c)
    if n < 3:
        return out
    iu, ju = np.triu_indices(n, k=1)
    d = _pairwise(fit_c)[iu, ju]
    sep = np.abs(fit_m[iu] - fit_m[ju]).astype(np.float64)
    b_dist, r2_dist = _loglog_fit(sep, d, n_bins, min_sep_bp, hi_frac)
    b_con, r2_con = _loglog_fit(sep, (d < fit_r).astype(np.float64), n_bins, min_sep_bp, hi_frac)
    out["dist_exp"], out["dist_r2"] = b_dist, r2_dist
    out["contact_exp"], out["contact_r2"] = -b_con, r2_con  # negate so α is reported as positive
    return out


def ensemble_scaling_laws(
    ensemble: Sequence[F64Array], mids: I64Array, radius: float, resolution_bp: int = 0
) -> dict[str, float]:
    """scaling_laws averaged by nan-mean across an ensemble. resolution_bp > 0 normalizes the β/α
    fit to a common bp resolution. See scaling_laws."""
    rows = [scaling_laws(c, mids, radius, resolution_bp=resolution_bp) for c in ensemble]
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def check_law(name: str, exp: float, r2: float, r2_min: float = 0.80) -> tuple[bool, str]:
    """Does a fitted exponent satisfy its literature band and look like a power law with R² ≥ r2_min?
    Returns (ok, reason). A NaN exponent, from a region too small to have a scaling window, returns
    (False, 'n/a')."""
    if name not in LAW_BANDS or not np.isfinite(exp):
        return False, "n/a, no scaling window, region too small?"
    lo, hi, target = LAW_BANDS[name]
    in_band = lo <= exp <= hi
    powerlaw = np.isfinite(r2) and r2 >= r2_min
    reason = (
        f"exp={exp:.2f} (band {lo}-{hi}, ~{target}); R²={r2:.2f}{' <min' if not powerlaw else ''}"
    )
    return (in_band and powerlaw), reason


# --------------------------------------------------------------------------- self-consistency


def _spearman(a: F64Array, b: F64Array) -> float:
    """Spearman rank correlation, using scipy if available, else rank plus Pearson."""
    if a.size < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(a, b).statistic)
        return rho
    except ImportError:
        ar = np.argsort(np.argsort(a)).astype(np.float64)
        br = np.argsort(np.argsort(b)).astype(np.float64)
        ar -= ar.mean()
        br -= br.mean()
        denom = np.sqrt((ar * ar).sum() * (br * br).sum())
        return float((ar * br).sum() / denom) if denom > 0 else float("nan")


def self_consistency(
    coords: F64Array, mids: I64Array, contacts: Sequence[tuple[int, int, float]]
) -> tuple[float, int]:
    """Self-consistency metric. Does the structure reproduce the input contacts? For each input contact
    (genomic_pos_a, genomic_pos_b, score), map both endpoints to the nearest bead by midpoint and
    collect (score, 3D distance). Returns (Spearman rho, n_contacts).

    A good structure puts high-score contacts close, so rho should be clearly negative. This uses
    only public output and the user's own contact list, no internal heatmaps.
    """
    if len(coords) < 3 or not contacts:
        return float("nan"), 0
    order = np.argsort(mids)
    sorted_mids = mids[order]

    def nearest(pos: int) -> int:
        k = int(np.searchsorted(sorted_mids, pos))
        if k <= 0:
            return int(order[0])
        if k >= len(sorted_mids):
            return int(order[-1])
        lo, hi = sorted_mids[k - 1], sorted_mids[k]
        return int(order[k - 1] if (pos - lo) <= (hi - pos) else order[k])

    scores, dists = [], []
    for p1, p2, sc in contacts:
        i, j = nearest(int(p1)), nearest(int(p2))
        if i == j:
            continue
        scores.append(float(sc))
        dists.append(float(np.linalg.norm(coords[i] - coords[j])))
    if len(scores) < 3:
        return float("nan"), len(scores)
    return _spearman(np.array(scores), np.array(dists)), len(scores)


# --------------------------------------------------------------------------- ensemble diversity


def dab_matrix(
    ensemble: Sequence[F64Array], expected: F64Array | None = None, max_structures: int = 24
) -> F64Array:
    """Ensemble-diversity metric. Szałaj-2016 inter-structure distance matrix.

        d_AB = (1/M) * sum_{i,j} [ (D_A(i,j) - D_B(i,j)) / E(D(i,j)) ]^2

    where D_X is structure X's pairwise-distance matrix and E(D(i,j)) is the expected distance for
    that pair. It is insensitive to mirror symmetry because it uses distances rather than
    coordinates. It is deliberately not RMSD, as the paper specifies. When expected, the
    heat-map-derived expected-distance matrix, is not supplied, E is approximated by the
    ensemble-mean distance per pair, a data-driven normaliser. This is a documented divergence.

    Returns the (n_structures, n_structures) symmetric matrix with a zero diagonal.
    """
    from scipy.spatial.distance import pdist, squareform

    ens = list(ensemble)
    if not ens:
        return np.zeros((0, 0))
    # The diversity scalar, the median off-diagonal d_AB, is well estimated from a sample of
    # structures. We don't need all 100²/2 pairs times full N×N matrices, which is about 20 GB and
    # O(n²N²) at N near 5000 beads, the post-MC bottleneck. Subsample evenly to max_structures, and
    # use condensed upper-triangle distances in float32 via pdist. The (i<j) ordering is the same
    # across structures, so the subtraction is valid.
    if len(ens) > max_structures:
        idx = np.unique(np.linspace(0, len(ens) - 1, max_structures).round().astype(int))
        ens = [ens[i] for i in idx]
    mats = [pdist(c).astype(np.float32) for c in ens]
    n = len(mats)
    if expected is None:
        e = np.mean(np.stack(mats), axis=0)
    else:
        e = squareform(np.asarray(expected, dtype=np.float64), checks=False).astype(np.float32)
    e = np.where(np.abs(e) < 1e-9, np.float32(1.0), e)
    npairs = mats[0].size
    out = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            r = (mats[a] - mats[b]) / e
            val = float(np.dot(r, r) / npairs)
            out[a, b] = out[b, a] = val
    return out


def ensemble_diversity(dab: F64Array) -> float:
    """Median off-diagonal d_AB. Low means a reproducible, tight ensemble, high means a diverse one."""
    n = dab.shape[0]
    if n < 2:
        return float("nan")
    iu, ju = np.triu_indices(n, k=1)
    return float(np.median(dab[iu, ju]))


# --------------------------------------------------------------------------- reconstruction fidelity
# The two measures the 2016 paper uses to score a reconstructed structure against a reference,
# in Suppl. §III–IV and Fig. S19–S22. These are RMSD, which is mirror-insensitive, and the custom
# contact measure.


def contact_measure(
    coords_a: F64Array, coords_b: F64Array, expected: F64Array | None = None
) -> float:
    """Szałaj-2016 contact measure between two structures A and B, from the supplement.

        d_AB = (1/N_pairs) * sum_{i<j} [ (D_A(i,j) - D_B(i,j)) / ED(i,j) ]^2

    ED is the expected pairwise distance. For a synthetic ground-truth run it is the known
    reference, so pass the ground-truth structure's distances. When the truth is unknown the paper
    uses ED = (D_A + D_B)/2, which is the default here. It is insensitive to mirror and chirality
    because it compares distance matrices rather than coordinates. A and B must have the same bead
    count. Lower means a closer reconstruction.
    """
    from scipy.spatial.distance import pdist, squareform

    da = pdist(np.asarray(coords_a, dtype=np.float64))
    db = pdist(np.asarray(coords_b, dtype=np.float64))
    if expected is None:
        ed = 0.5 * (da + db)
    else:
        exp = np.asarray(expected, dtype=np.float64)
        ed = squareform(exp, checks=False) if exp.ndim == 2 else exp
    ed = np.where(np.abs(ed) < 1e-12, 1.0, ed)
    r = (da - db) / ed
    return float(np.dot(r, r) / r.size)


def _kabsch_rmsd(a: F64Array, b: F64Array) -> float:
    """Minimum RMSD superposing B onto A by an optimal proper rotation and translation, via Kabsch."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _s, vt = np.linalg.svd(bc.T @ ac)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    br = bc @ r.T
    return float(np.sqrt(((ac - br) ** 2).sum(axis=1).mean()))


def rmsd_superpose(
    coords_a: F64Array, coords_b: F64Array, mirror_insensitive: bool = True
) -> float:
    """Optimal-superposition RMSD via Kabsch. Mirror-insensitive by default per Szałaj-2016. A
    contact heatmap cannot distinguish chirality, so a structure and its mirror image are
    equivalent. Take the minimum RMSD over the reflection too, since a reflection composed with a
    proper rotation spans all improper rotations.
    """
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    rmsd = _kabsch_rmsd(a, b)
    if mirror_insensitive:
        rmsd = min(rmsd, _kabsch_rmsd(a, b * np.array([1.0, 1.0, -1.0])))
    return rmsd
