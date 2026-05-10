"""
Singleton heatmap construction and the 5-step normalisation pipeline.

Mirrors cudaMMC HeatmapNormalization + LooperSolver heatmap building logic.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .data_structures import Anchor


# ── Heatmap construction ─────────────────────────────────────────────────────

def build_singleton_heatmap(singletons_path: str,
                             anchors: List[Anchor],
                             device: str = "cpu") -> torch.Tensor:
    """
    Build a raw singleton count matrix from a BEDPE singletons file.

    Returns a float32 tensor of shape (N, N) where N = len(anchors).
    """
    from .data_loading import _find_anchor
    N = len(anchors)
    mat = torch.zeros(N, N, dtype=torch.float32)

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

    return mat.to(device)


# ── Normalisation pipeline ───────────────────────────────────────────────────

def normalize_heatmap(raw: torch.Tensor,
                      anchors: List[Anchor],
                      diagonal_size: int = 3
                      ) -> torch.Tensor:
    """
    Full 5-step normalisation:
      1. Bin-length normalisation
      2. Row normalisation
      3. Diagonal normalisation
      4. (Inter-chr scaling — N/A for single-chr heatmap; included for completeness)
      5. Frequency → distance conversion handled externally via distances.py

    Returns a normalised float32 tensor of shape (N, N).
    """
    mat = raw.clone().float()
    N = mat.shape[0]

    # Step 1 – bin-length normalisation
    lengths = torch.tensor([a.length for a in anchors],
                           dtype=torch.float32, device=mat.device)
    # divide each cell by geometric mean of the two anchor lengths
    len_outer = torch.outer(lengths, lengths)  # (N, N)
    geom_mean = torch.sqrt(len_outer)
    mat = mat / geom_mean.clamp(min=1.0)

    # Step 2 – row normalisation (divide each row by its sum)
    row_sums = mat.sum(dim=1, keepdim=True).clamp(min=1e-9)
    mat = mat / row_sums

    # Step 3 – diagonal normalisation
    # For each diagonal d, divide by the mean of that diagonal
    for d in range(diagonal_size, N):
        diag = torch.diagonal(mat, offset=d)
        diag_mean = diag.mean()
        if diag_mean > 1e-9:
            diag_val = diag / diag_mean
            # write back (torch.diagonal returns a view for square tensors)
            mat.diagonal(offset=d).copy_(diag_val)
            mat.diagonal(offset=-d).copy_(diag_val)

    return mat


def heatmap_to_expected_distances(mat: torch.Tensor,
                                   scale: float = 100.0,
                                   power: float = -0.333,
                                   min_freq: float = 1e-9) -> torch.Tensor:
    """Convert normalised frequency matrix to expected distance matrix."""
    safe = mat.clamp(min=min_freq)
    return scale * (safe ** power)
