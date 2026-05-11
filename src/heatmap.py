"""
Singleton heatmap construction and the normalisation pipeline.

Device strategy:
- build_singleton_heatmap: always CPU (file I/O + Python loop, can't vectorise).
- normalize_heatmap / heatmap_to_expected_distances: run on whatever device the
  input tensor lives on.

NOTE original cudaMMC normalises on CPU then copies to GPU; we match the
  normalisation maths but run it on GPU for speed.
"""

from typing import List

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


# ── Normalisation pipeline ────────────────────────────────────────────────────

def normalize_heatmap(raw: torch.Tensor,
                      anchors: List[Anchor],
                      diagonal_size: int = 3) -> torch.Tensor:
    """
    Full normalisation pipeline.  Runs on whatever device `raw` lives on
    (pass a GPU tensor for GPU execution).

    Returns a float32 tensor on the same device as `raw`, shape (N, N).
    """
    mat = raw.float()   # keep on raw's device; no forced .cpu()
    device = mat.device
    N = mat.shape[0]

    # Step 1 – bin-length normalisation in Mb (matching original cudaMMC)
    len_mb = torch.tensor([max(a.length, 1) / 1_000_000.0 for a in anchors],
                           dtype=torch.float32, device=device)
    mat = mat / len_mb[:, None]
    mat = mat / len_mb[None, :]

    # Step 2 – row normalisation
    row_sums = mat.sum(dim=1, keepdim=True).clamp(min=1e-9)
    mat = mat / row_sums

    # Step 3 – single global scale so first non-zero diagonal averages 1.0
    first_diag = mat.diagonal(offset=diagonal_size)
    diag_mean = first_diag[first_diag > 1e-9].mean().item()
    if diag_mean > 1e-9:
        mat = mat / diag_mean

    return mat   # float32, same device as input


def heatmap_to_expected_distances(mat: torch.Tensor,
                                   scale: float = 100.0,
                                   power: float = -0.333,
                                   min_freq: float = 1e-9) -> torch.Tensor:
    """
    Convert normalised frequency matrix → expected distance matrix (float16).
    Runs on whatever device `mat` lives on.
    Returns float16 on the same device (~half the memory of float32).
    """
    # Only convert entries that have actual contact data (freq > 0).
    # Zero-count pairs keep expected = 0 so they are excluded from scoring
    # (score functions filter by exp > 1e-3).  Clamping zeros to min_freq and
    # then converting would produce ~100,000 → Inf in float16, corrupting scoring.
    mat_f = mat.float()
    has_data = mat_f > 0
    result = torch.zeros_like(mat_f)
    if has_data.any():
        result[has_data] = scale * (mat_f[has_data].clamp(min=min_freq) ** power)
    return result.half()
