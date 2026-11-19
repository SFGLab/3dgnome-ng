"""Normalises a contact frequency matrix and converts it to expected distances."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, ChrLevel, F64Array

Matrix: TypeAlias = "F64Array | list[list[float]]"


def get_diagonal_size(h: Matrix, n: int) -> int:
    """Returns the smallest offset `j - i` of an upper triangle cell above 1e-6."""
    ha = np.asarray(h)
    iu = np.triu_indices(n)
    mask = ha[iu] > 1e-6
    if not mask.any():
        return 0
    return int((iu[1] - iu[0])[mask].min())


def normalize_heatmap(h: Matrix, n: int) -> F64Array:
    """Normalises every row to the grand average and symmetrises."""
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
    """Scales the matrix to a mean of `val` on the first diagonal with a contact."""
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
    """Scales the inter chromosomal entries by `scale` and leaves the blocks within a
    chromosome unchanged.
    """
    ha = np.asarray(h, dtype=np.float64)
    chrs_in_order = [c for c in current_level if current_level[c]]
    if len(chrs_in_order) <= 1:
        return ha.copy()
    block_starts = [0]
    for chr_ in chrs_in_order:
        block_starts.append(block_starts[-1] + len(current_level[chr_]))
    if block_starts[-1] != n:
        return ha.copy()  # shape mismatch, returned unchanged
    out = ha * scale
    for b in range(len(chrs_in_order)):
        lo, hi = block_starts[b], block_starts[b + 1]
        out[lo:hi, lo:hi] /= scale
    return out


def _polymer_distances(
    law: PolymerLaw,
    ha: F64Array,
    use: BoolArray,
    separations_bp: F64Array | None,
    scale_bp: float | None,
) -> F64Array:
    """Returns the target of each cell under the polymer law.

    `use` marks the cells with a contact outside the diagonal band. With separations the expected
    contact is the mean over those cells in the same log separation bin, eight bins a decade.
    Without them every cell shares one expectation and the background is taken at `scale_bp`.
    """
    n = ha.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    if not use.any():
        return out
    if separations_bp is None:
        expected = float(ha[use].mean())
        sep_bp = int(scale_bp if scale_bp is not None else 1)
        for i, j in zip(*np.nonzero(use), strict=True):
            out[i, j] = law.heatmap_distance(float(ha[i, j]), expected, sep_bp)
        return out
    sep = np.abs(separations_bp[:, None] - separations_bp[None, :])
    ii, jj = np.nonzero(use)
    s_ij = sep[ii, jj]
    key = np.floor(np.log10(np.maximum(s_ij, 1.0)) * 8.0).astype(np.int64)
    f_ij = ha[ii, jj]
    # The cells are taken in the order of np.nonzero. The mean of a bin sums the same values in
    # the same order.
    s0 = max(int(law.s0_bp), 1)
    for k in np.unique(key):
        sel = key == k
        f = f_ij[sel]
        expected = float(f.mean())
        if expected <= 1e-12:
            continue
        sp_bp = np.maximum(np.abs(np.trunc(s_ij[sel])), 1.0)
        bg = np.maximum(1.0, (sp_bp / s0) ** law.nu)
        val = bg * (f / expected) ** (-1.0 / 3.0)
        val[f <= 1e-12] = 0.0
        out[ii[sel], jj[sel]] = val
    return out


def create_distance_heatmap(
    settings: Settings,
    h: Matrix,
    n: int,
    separations_bp: F64Array | None = None,
    scale_bp: float | None = None,
) -> tuple[F64Array, float]:
    """Converts a normalised contact frequency heatmap to expected distances under the polymer law.

    A cell with no contact gives 0 and the diagonal band gives -1, which scoring ignores. Every
    other cell is the background of its separation times observed over expected contact to the
    minus third. Without `separations_bp`, at the chromosome level, every cell shares one
    expectation and the background is taken at `scale_bp`. Distances are clipped at the mean
    positive distance times `heatmap_distance_stretching`. Uses the upper triangle and mirrors it.
    """
    ha = np.asarray(h, dtype=np.float64)
    diag = get_diagonal_size(ha, n)
    law = settings.polymer_law()
    active = ha >= 1e-6
    ii, jj = np.indices((n, n))
    band = np.abs(ii - jj) < diag
    fd = _polymer_distances(law, ha, active & ~band, separations_bp, scale_bp)
    dist = np.where(active, np.where(band, -1.0, fd), 0.0)
    dist = np.triu(dist) + np.triu(dist, 1).T

    pos = dist[dist > 0.0]
    avg = float(pos.mean()) if pos.size else 1.0
    max_d = avg * settings.heatmap_distance_stretching
    dist = np.where(dist > max_d, max_d, dist)
    return dist, avg
