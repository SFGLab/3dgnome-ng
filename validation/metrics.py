"""Resolution-independent structure-validation metrics.

Pure functions over reconstructed structures (``BeadOut`` lists or plain (N,3) arrays).
Each maps to a check in ``docs/validation.md``:

  * V1  self_consistency        — input contact strength vs output 3D distance
  * V2  distance_scaling        — mean pairwise distance vs genomic separation (power law)
  * V2  contact_probability     — P(contact) vs genomic separation (power law)
  * V3  dab_matrix / diversity  — Szałaj-2016 mirror-insensitive inter-structure distance
  * EV  overlap_fraction        — non-bonded bead interpenetrations (the EV/confinement test)

No gnome3d imports here: takes already-produced coordinates so it is trivially unit-testable
and reusable on any structure source. ``validate.py`` drives the pipeline and calls these.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gnome3d.types import F64Array, I64Array

# A "bead" is anything indexable as (start, end, x, y, z, ...) — i.e. gnome3d BeadOut,
# or the 6-tuples the integration harness uses. We read indices 0 (start), 2,3,4 (xyz).
Bead = Sequence[Any]


def to_arrays(beads: Sequence[Bead]) -> tuple[F64Array, I64Array]:
    """(N,3) float64 coordinates and (N,) int64 genomic midpoints from a bead list.

    Midpoint = (start+end)//2; beads are assumed genomic-sorted (run_region output is).
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
    """Full (N,N) Euclidean distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


# --------------------------------------------------------------------------- EV / confinement


def overlap_fraction(
    coords: F64Array, radius: float, skip_neighbors: int = 1
) -> tuple[float, int, int]:
    """Fraction of non-bonded bead pairs closer than ``radius`` — the
    "physically impossible overlaps" the 2016 paper admitted its model produced
    (no excluded volume). Excluded-volume / confinement should drive this DOWN.

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


def max_extent(coords: F64Array) -> float:
    """Max distance of any bead from the centroid — confinement should bound this."""
    c = coords - coords.mean(axis=0)
    return float(np.sqrt((c * c).sum(axis=1)).max())


# --------------------------------------------------------------------------- V2 scaling laws


def _loglog_bins(sep: F64Array, val: F64Array, n_bins: int) -> tuple[F64Array, F64Array, float]:
    """Bin (sep, val) into ``n_bins`` log-spaced separation bins, take per-bin means,
    and fit a power law val ~ sep^slope on the log-log bin means. Returns
    (bin_sep, bin_mean_val, slope). NaNs/empties dropped before the fit."""
    pos = sep > 0
    sep, val = sep[pos], val[pos]
    if sep.size < 2:
        return np.array([]), np.array([]), float("nan")
    edges = np.logspace(np.log10(sep.min()), np.log10(sep.max() + 1), n_bins + 1)
    idx = np.clip(np.digitize(sep, edges) - 1, 0, n_bins - 1)
    bin_sep, bin_val = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        bin_sep.append(float(sep[m].mean()))
        bin_val.append(float(val[m].mean()))
    bs, bv = np.array(bin_sep), np.array(bin_val)
    ok = (bs > 0) & (bv > 0)
    if ok.sum() < 2:
        return bs, bv, float("nan")
    slope = float(np.polyfit(np.log10(bs[ok]), np.log10(bv[ok]), 1)[0])
    return bs, bv, slope


def distance_scaling(
    coords: F64Array, mids: I64Array, n_bins: int = 20
) -> tuple[F64Array, F64Array, float]:
    """V2: mean pairwise 3D distance vs genomic separation, as a power law.
    Returns (separation_bins, mean_distance, exponent). A sane polymer has a
    POSITIVE exponent (distance grows with separation), typically ~0.2-0.5."""
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)
    d = _pairwise(coords)[iu, ju]
    sep = np.abs(mids[iu] - mids[ju]).astype(np.float64)
    return _loglog_bins(sep, d, n_bins)


def contact_probability(
    coords: F64Array, mids: I64Array, radius: float, n_bins: int = 20
) -> tuple[F64Array, F64Array, float]:
    """V2: contact probability P(d < radius) vs genomic separation, as a power law.
    Returns (separation_bins, contact_prob, exponent). A sane polymer has a
    NEGATIVE exponent (contacts decay with separation), typically ~-0.75 to -1.5."""
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)
    d = _pairwise(coords)[iu, ju]
    sep = np.abs(mids[iu] - mids[ju]).astype(np.float64)
    contact = (d < radius).astype(np.float64)
    return _loglog_bins(sep, contact, n_bins)


# --------------------------------------------------------------------------- V1 self-consistency


def _spearman(a: F64Array, b: F64Array) -> float:
    """Spearman rank correlation (scipy if available, else rank+Pearson)."""
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
    """V1: does the structure reproduce the input contacts? For each input contact
    (genomic_pos_a, genomic_pos_b, score), map both endpoints to the nearest bead
    (by midpoint) and collect (score, 3D distance). Returns (Spearman rho, n_contacts).

    A good structure puts high-score contacts CLOSE → rho should be clearly NEGATIVE.
    Uses only public output + the user's own contact list (no internal heatmaps).
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


# --------------------------------------------------------------------------- V3 ensemble


def dab_matrix(ensemble: Sequence[F64Array], expected: F64Array | None = None) -> F64Array:
    """V3: Szałaj-2016 inter-structure distance matrix.

        d_AB = (1/M) * sum_{i,j} [ (D_A(i,j) - D_B(i,j)) / E(D(i,j)) ]^2

    where D_X is structure X's pairwise-distance matrix and E(D(i,j)) is the
    *expected* distance for that pair. Mirror-symmetry-insensitive (uses distances,
    not coordinates) — deliberately NOT RMSD, as the paper specifies. When ``expected``
    (the heat-map-derived expected-distance matrix) is not supplied, E is approximated by
    the ensemble-mean distance per pair, a data-driven normaliser (documented divergence).

    Returns the (n_structures, n_structures) symmetric matrix; diagonal 0.
    """
    mats = [_pairwise(c) for c in ensemble]
    n = len(mats)
    if n == 0:
        return np.zeros((0, 0))
    if expected is None:
        expected = np.mean(np.stack(mats), axis=0)
    e = np.where(np.abs(expected) < 1e-9, 1.0, expected)
    npairs = mats[0].size
    out = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            r = (mats[a] - mats[b]) / e
            val = float((r * r).sum() / npairs)
            out[a, b] = out[b, a] = val
    return out


def ensemble_diversity(dab: F64Array) -> float:
    """Median off-diagonal d_AB — low = reproducible/tight, high = diverse ensemble."""
    n = dab.shape[0]
    if n < 2:
        return float("nan")
    iu, ju = np.triu_indices(n, k=1)
    return float(np.median(dab[iu, ju]))
