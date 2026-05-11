"""
Singleton heatmap construction and the 5-step normalisation pipeline.

Memory strategy for large N:
- Build and normalise entirely on CPU (avoids GPU OOM during N×N ops).
- Never create an N×N outer-product intermediate; use broadcast row/col divides.
- Return float16 so the GPU copy is ~half the size (~1.4 GB for N=26603).
"""

import math
from typing import List, Optional

import torch

from .data_structures import Anchor


# ── Heatmap construction ─────────────────────────────────────────────────────

def build_singleton_heatmap(singletons_path: str,
                             anchors: List[Anchor]) -> torch.Tensor:
    """
    Build a raw singleton count matrix on CPU.

    Returns a float32 CPU tensor of shape (N, N).
    Caller should normalise on CPU, then move the float16 result to GPU.
    """
    from .data_loading import _find_anchor
    N = len(anchors)
    mat = torch.zeros(N, N, dtype=torch.float32)   # CPU, no device arg

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


# ── Normalisation pipeline ───────────────────────────────────────────────────

def normalize_heatmap(raw: torch.Tensor,
                      anchors: List[Anchor],
                      diagonal_size: int = 3) -> torch.Tensor:
    """
    Full normalisation pipeline – operates entirely on CPU in float32.

    Avoids any N×N intermediate allocation (uses broadcast row/col ops).
    Returns a CPU float32 tensor of shape (N, N).
    """
    mat = raw.float().cpu()
    N = mat.shape[0]

    # Step 1 – bin-length normalisation in Mb (matching original cudaMMC)
    len_mb = torch.tensor([max(a.length, 1) / 1_000_000.0 for a in anchors],
                           dtype=torch.float32)
    mat = mat / len_mb[:, None]
    mat = mat / len_mb[None, :]

    # Step 2 – row normalisation
    row_sums = mat.sum(dim=1, keepdim=True).clamp(min=1e-9)
    mat /= row_sums

    # Step 3 – single global scale so first non-zero diagonal averages 1.0
    first_diag = mat.diagonal(offset=diagonal_size)
    diag_mean = first_diag[first_diag > 1e-9].mean().item()
    if diag_mean > 1e-9:
        mat /= diag_mean

    return mat   # CPU float32


def heatmap_to_expected_distances(mat: torch.Tensor,
                                   scale: float = 100.0,
                                   power: float = -0.333,
                                   min_freq: float = 1e-9) -> torch.Tensor:
    """Convert normalised frequency matrix to expected distance matrix (float16)."""
    safe = mat.float().clamp(min=min_freq)
    return (scale * (safe ** power)).half()   # float16 halves GPU footprint
