"""
Heatmap matrix helpers: normalize a contact-frequency matrix and convert it to
an expected-distance matrix.

Pure numpy functions over an `(n, n)` matrix - no cluster graph, no MC.  Lifted
out of the former `CoarseModel` (where they were O(N^2) Python list-of-list
loops that blew up on large Hi-C matrices) so the coarse HEATMAP stages can build
their distance targets from these, and any caller can reuse them.

Each mirrors the Reference normalize*/createDistanceHeatmap routines; vectorized
exactly (validated bit/float-equal vs the original loops).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import ChrLevel, F64Array

# Accepts a freshly-built list-of-list heatmap or an ndarray; np.asarray unifies.
Matrix: TypeAlias = "F64Array | list[list[float]]"


def get_diagonal_size(h: Matrix, n: int) -> int:
    """Smallest superdiagonal offset holding a non-zero contact (the ignored
    diagonal band): min `j - i` over upper-triangle cells > 1e-6."""
    ha = np.asarray(h)
    iu = np.triu_indices(n)
    mask = ha[iu] > 1e-6
    if not mask.any():
        return 0
    return int((iu[1] - iu[0])[mask].min())


def normalize_heatmap(h: Matrix, n: int) -> F64Array:
    """Row-normalize so every row sums to the grand-average, then symmetrize.
    Mirrors Reference normalizeHeatmap()."""
    ha = np.asarray(h, dtype=np.float64)
    row_sums = ha.sum(axis=1)
    total = float(row_sums.sum())
    if total < 1e-10:
        return ha.copy()
    expected = total / n
    mn = np.where(row_sums > 1e-10, expected / np.where(row_sums > 1e-10, row_sums, 1.0), 1.0)
    out = ha * mn[:, None]
    return (out + out.T) / 2.0


def normalize_heatmap_diagonal_total(h: Matrix, n: int, val: float) -> F64Array:
    """Scale the whole matrix so the average of the first non-zero diagonal
    equals `val`.  Mirrors Reference normalizeHeatmapDiagonalTotal()."""
    ha = np.asarray(h, dtype=np.float64).copy()
    diag = get_diagonal_size(ha, n)
    count = n - diag
    if count <= 0:
        return ha
    avg = float(np.diagonal(ha, diag).mean())
    if avg < 1e-10:
        return ha
    ha *= val / avg
    return ha


def normalize_heatmap_inter(h: Matrix, n: int, current_level: ChrLevel, scale: float) -> F64Array:
    """Scale inter-chromosomal entries by ``scale``, leaving intra-chr blocks
    unchanged (multiply all by scale, divide intra-chr blocks back).  Mirrors
    Reference normalizeHeatmapInter()."""
    ha = np.asarray(h, dtype=np.float64)
    chrs_in_order = [c for c in current_level if current_level[c]]
    if len(chrs_in_order) <= 1:
        return ha.copy()
    block_starts = [0]
    for chr_ in chrs_in_order:
        block_starts.append(block_starts[-1] + len(current_level[chr_]))
    if block_starts[-1] != n:
        return ha.copy()  # shape mismatch - leave untouched
    out = ha * scale
    for b in range(len(chrs_in_order)):
        lo, hi = block_starts[b], block_starts[b + 1]
        out[lo:hi, lo:hi] /= scale
    return out


def create_distance_heatmap(
    settings: Settings, h: Matrix, n: int, inter: bool = False
) -> tuple[F64Array, float]:
    """Convert a normalized contact-frequency heatmap to an expected-distance
    heatmap.  Mirrors Reference createDistanceHeatmap(), per cell:
      - freq < 1e-6           -> 0   (no contact)
      - within diagonal band  -> -1  (ignored in scoring)
      - else                  -> scale * freq^power
    then clip distances above ``avg(>0) * heatmap_distance_stretching``.  Uses the
    upper triangle of `h` and mirrors it (matches the reference's j>=i loop)."""
    ha = np.asarray(h, dtype=np.float64)
    diag = get_diagonal_size(ha, n)
    scale, power = (
        (settings.freq_dist_scale_inter, settings.freq_dist_power_inter)
        if inter
        else (settings.freq_dist_scale, settings.freq_dist_power)
    )
    active = ha >= 1e-6
    ii, jj = np.indices((n, n))
    band = np.abs(ii - jj) < diag
    fd = scale * np.power(np.where(active, ha, 1.0), power)  # clamp inactive -> no 0**neg warn
    dist = np.where(active, np.where(band, -1.0, fd), 0.0)
    dist = np.triu(dist) + np.triu(dist, 1).T  # mirror upper -> lower

    pos = dist[dist > 0.0]
    avg = float(pos.mean()) if pos.size else 1.0
    max_d = avg * settings.heatmap_distance_stretching
    dist = np.where(dist > max_d, max_d, dist)
    return dist, avg
