"""
Singleton heatmap construction and the normalisation pipeline.

Device strategy:
- build_singleton_heatmap: always CPU (file I/O + Python loop, can't vectorise).
- normalize_heatmap / heatmap_to_expected_distances: run on whatever device the
  input tensor lives on; **fp32 throughout** (cudaMMC uses `float`, not `half`).

==============================================================================
cudaMMC reference (`LooperSolver.cpp`):

  1. normalizeHeatmap                    cpp:1709-1751
       expected_sum = mean_i(sum_j h[i][j])
       h[i][:]    *= expected_sum / row_sum_i
       symmetrise  h[i][j] = h[j][i] = (h[i][j] + h[j][i]) / 2
  2. normalizeHeatmapDiagonalTotal(h,1)  cpp:1857-1876
       diag       = heat.getDiagonalSize()        (data-driven)
       avg        = mean( h[i][i+diag] )          (first non-zero diagonal)
       h         *= 1.0 / avg                     (whole matrix)
  3. normalizeHeatmapInter(h, scale)     cpp:1817-1855   [multi-chrom only]
       h         *= scale
       intra-blocks /= scale                       (undoes scaling inside chrs)
  4. createDistanceHeatmap                cpp:1753-1796
       val<1e-6 →  0
       |i-j|<diag →  -1                            (REPULSION SENTINEL)
       else    → freqToDistanceHeatmap(val)        = scale * val^power
       clip positive entries at avg(heatmap_dist) * heatmapDistanceHeatmapStretching
       (the −1 sentinel is preserved because the clip is `v > max_dist`;
        cpp:1786-1791.)

All four are now mirrored in this module.
==============================================================================
"""

from typing import List, Tuple

import torch

from .data_structures import Anchor


# ── Heatmap construction ─────────────────────────────────────────────────────

def build_singleton_heatmap(singletons_path: str,
                             anchors: List[Anchor]) -> torch.Tensor:
    """
    Build a raw singleton count matrix.

    Always runs on CPU — file I/O and Python indexing loop can't be
    parallelised on GPU.  Returns a CPU float32 tensor of shape (N, N).
    Transfer to GPU before calling normalize_heatmap.
    """
    from .data_loading import _find_anchor
    N = len(anchors)
    mat = torch.zeros(N, N, dtype=torch.float32)   # CPU

    with open(singletons_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            s1, e1 = int(parts[1]), int(parts[2])
            s2, e2 = int(parts[4]), int(parts[5])
            ai = _find_anchor(anchors, (s1 + e1) // 2)
            bi = _find_anchor(anchors, (s2 + e2) // 2)
            if ai < 0 or bi < 0:
                continue
            mat[ai, bi] += 1
            if ai != bi:
                mat[bi, ai] += 1

    return mat   # CPU float32


def build_singleton_heatmap_bins(singletons_path: str,
                                  bins: List[Tuple[str, int, int]]
                                  ) -> torch.Tensor:
    """Bin-based singleton heatmap for arbitrary genomic intervals.

    Used by the segment-level cascade (cudaMMC ``createSingletonHeatmap(1)``
    at ``LooperSolver.cpp:252``): each row/column of the output matrix
    corresponds to one **segment bead** (level-2 cluster) whose
    ``(chrom, start, end)`` span defines a bin.  Singletons whose
    midpoint falls in bin ``i`` (left endpoint) and bin ``j`` (right
    endpoint) increment ``mat[i,j] += 1`` (symmetrised).

    ``bins`` must be sorted by ``(chrom, start)`` and non-overlapping;
    binary search is used for ``O(log N)`` lookup, mirroring the
    ``_find_anchor`` strategy used for anchor-level heatmaps.
    """
    N = len(bins)
    mat = torch.zeros(N, N, dtype=torch.float32)
    if N == 0:
        return mat

    # group bin indices by chromosome for a per-chrom binary search
    by_chr: dict = {}
    for i, (chrom, _start, _end) in enumerate(bins):
        by_chr.setdefault(chrom, []).append(i)

    def _find(chrom: str, pos: int) -> int:
        # binary-search bin indices on chrom; bins are sorted by start
        idxs = by_chr.get(chrom)
        if not idxs:
            return -1
        lo, hi = 0, len(idxs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            _, s, e = bins[idxs[mid]]
            if pos < s:
                hi = mid - 1
            elif pos > e:
                lo = mid + 1
            else:
                return idxs[mid]
        return -1

    with open(singletons_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            c1 = parts[0]
            s1, e1 = int(parts[1]), int(parts[2])
            c2 = parts[3]
            s2, e2 = int(parts[4]), int(parts[5])
            ai = _find(c1, (s1 + e1) // 2)
            bi = _find(c2, (s2 + e2) // 2)
            if ai < 0 or bi < 0:
                continue
            mat[ai, bi] += 1
            if ai != bi:
                mat[bi, ai] += 1
    return mat


# ── Diagonal-size helper (cudaMMC Heatmap.cpp:58-69) ─────────────────────────

def get_diagonal_size(mat: torch.Tensor, eps: float = 1e-6) -> int:
    """
    Mirror of ``Heatmap::getDiagonalSize`` (cudaMMC Heatmap.cpp:58-69):
    smallest width *w* such that some cell ``v[i][i+w]`` exceeds ``eps``.
    Returns 0 if the whole matrix is below ``eps``.

    Used by ``normalize_heatmap`` and by ``heatmap_to_expected_distances``
    so we never fall back to ``Settings.diagonal_size`` (AUDIT §B4, §C1).
    """
    n = mat.shape[0]
    for w in range(n):
        d = torch.diagonal(mat, offset=w)
        if d.numel() == 0:
            return 0
        if (d.abs() > eps).any():
            return w
    return 0


# ── Normalisation pipeline ────────────────────────────────────────────────────

def normalize_heatmap(raw: torch.Tensor,
                      anchors=None,
                      diagonal_size=None) -> Tuple[torch.Tensor, int]:
    """
    Full normalisation pipeline mirroring cudaMMC steps 1 + 2.

    Runs on whatever device ``raw`` lives on (pass a GPU tensor for GPU
    execution).  Returns ``(normalised matrix, diagonal_size)`` as a
    **fp32** tensor on the same device, shape ``(N, N)``.  The diagonal
    size is data-driven via :func:`get_diagonal_size`; the ``diagonal_size``
    argument is accepted only for back-compat with legacy callers and is
    *ignored* in favour of the heatmap-derived value (AUDIT §B4).

    ``anchors`` is accepted only for signature parity with the legacy
    bin-length-divided version (AUDIT §B1) — it is **unused** because
    cudaMMC does no bin-length normalisation.
    """
    del anchors, diagonal_size  # legacy signature, ignored per cudaMMC
    mat = raw.float()
    n = mat.shape[0]
    if n == 0:
        return mat, 0

    # ── Step 1 — normalizeHeatmap (cudaMMC LooperSolver.cpp:1709-1751) ─────
    # cpp:1722-1727: expected_sum = mean_i(row_sum_i)
    row_sums = mat.sum(dim=1)
    expected_sum = row_sums.mean()
    # cpp:1733-1742: mn = expected_sum / row_sum_i; row *= mn
    safe = row_sums.clamp(min=1e-12)
    mn = expected_sum / safe
    mat = mat * mn.unsqueeze(1)
    # cpp:1744-1748: symmetrise after row-scaling
    mat = 0.5 * (mat + mat.t())

    # ── Step 2 — normalizeHeatmapDiagonalTotal(h, 1.0)  (cpp:1857-1876) ────
    diag = get_diagonal_size(mat)
    if 0 <= diag < n:
        # cpp:1867-1869: avg = mean( h[i][i+diag] ) over i in [0, n-diag)
        diag_band = torch.diagonal(mat, offset=diag)
        avg = diag_band.mean()
        if avg.abs() > 1e-12:
            mat = mat / avg
    # cpp:1865 prints diag size; we return it so downstream uses the
    # heatmap-loaded value (AUDIT §C1) instead of Settings.diagonal_size.
    return mat, max(diag, 1)


def normalize_heatmap_inter(mat: torch.Tensor,
                            chrom_sizes: List[int],
                            scale: float) -> torch.Tensor:
    """
    Mirror of ``LooperSolver::normalizeHeatmapInter`` (cpp:1817-1855).

    Multi-chrom only.  Scales the whole matrix by ``scale`` then divides
    each intra-chromosomal block back by ``scale``, leaving only inter-chrom
    blocks scaled.  ``chrom_sizes`` gives the per-chromosome row count
    (``current_level[chr].size()``); their sum must equal ``mat.shape[0]``.
    """
    if len(chrom_sizes) <= 1:
        return mat   # cpp:1819-1820: skip when only one chromosome
    out = mat * scale
    start = 0
    for sz in chrom_sizes:
        end = start + sz
        out[start:end, start:end] /= scale
        start = end
    return out


# ── Distance heatmap (cudaMMC createDistanceHeatmap, cpp:1753-1796) ──────────

def heatmap_to_expected_distances(mat: torch.Tensor,
                                   scale: float = 100.0,
                                   power: float = -0.333,
                                   diagonal_size: int = 1,
                                   max_stretching: float = 2.0,
                                   eps: float = 1e-6) -> torch.Tensor:
    """
    Convert normalised frequency matrix → expected distance matrix (**fp32**).

    Mirror of ``LooperSolver::createDistanceHeatmap`` (LooperSolver.cpp:1753-1796):

      * ``val < eps``                → 0           (cpp:1764-1765)
      * ``val ≥ eps`` and ``|i-j| < diagonal_size`` → -1   (REPULSION SENTINEL, cpp:1768-1770)
      * otherwise                    → ``scale * val^power``  (freqToDistanceHeatmap,
                                                              cpp:2546-2549)
      * after the matrix is built, ``avg = mean( all entries )`` and every
        entry > ``avg * max_stretching`` is clipped to that bound (cpp:1780-1794);
        the sentinel ``-1`` passes through because the clip is ``v > max_dist``.

    ``diagonal_size`` should be the value returned by ``normalize_heatmap``
    (data-driven, cudaMMC Heatmap.cpp:58-69) — *not* ``Settings.diagonal_size``
    (AUDIT §B4, §C1).  Output is **fp32**, not fp16 (AUDIT §B8).
    """
    m = mat.float()
    n = m.shape[0]
    out = torch.zeros_like(m)

    if n == 0:
        return out

    # |i - j| via index broadcasting (small ops on the same device).
    idx = torch.arange(n, device=m.device)
    ij = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    has_data = m >= eps                                # cpp:1764-1765 gate
    in_band = ij < diagonal_size                       # cpp:1768

    # cpp:1768-1770: in-band & has-data → −1 sentinel (REPULSION)
    sentinel = has_data & in_band
    out[sentinel] = -1.0

    # cpp:1770-1771: otherwise (still has_data) → scale * val^power
    spring = has_data & ~in_band
    out[spring] = scale * m[spring].clamp(min=eps).pow(power)

    # cpp:1779-1791: avg = heatmap_dist.getAvg() (whole matrix, including
    # zeros and the −1 sentinels); cap every entry > avg*max_stretching.
    avg = out.mean().item()
    max_dist = avg * max_stretching
    if max_dist > 0:
        torch.clamp_(out, max=max_dist)
    return out   # fp32, same device as input
