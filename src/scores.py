"""
Energy / score functions - vectorised PyTorch.

All functions return a scalar float tensor (the lower the better).
Position tensors have shape (N, 3) and dtype float32.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Utility ──────────────────────────────────────────────────────────────────

def pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Return (N, N) matrix of Euclidean distances."""
    # Using the squared-norm trick for numerical stability
    sq = (pos ** 2).sum(dim=1)
    dot = pos @ pos.t()
    sq_dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot
    return sq_dist.clamp(min=0.0).sqrt()


def single_bead_distances(pos: torch.Tensor, idx: int) -> torch.Tensor:
    """Return (N,) distances from bead `idx` to all other beads."""
    diff = pos - pos[idx].unsqueeze(0)  # (N, 3)
    return (diff ** 2).sum(dim=1).clamp(min=0.0).sqrt()


# ── Heatmap score ─────────────────────────────────────────────────────────────

def score_heatmap(pos: torch.Tensor,
                  expected: torch.Tensor,
                  diagonal_size: int = 3,
                  same_chr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Heatmap score for ALL beads:
        sum_{i<j, |i-j|>=diag} ((dist_ij / expected_ij) - 1)^2
    where expected_ij > 1e-3 and (same chromosome).

    expected: (N, N) tensor with expected distances (<=0 entries are ignored).
    same_chr_mask: optional (N, N) bool mask; if None all pairs are included.
    """
    N = pos.shape[0]
    dist = pairwise_distances(pos)  # (N, N)

    # build diagonal mask (|i-j| >= diagonal_size)
    idx = torch.arange(N, device=pos.device)
    diff_idx = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    diag_mask = diff_idx >= diagonal_size

    valid = (expected > 1e-3) & diag_mask
    if same_chr_mask is not None:
        valid = valid & same_chr_mask

    ratio = dist[valid] / expected[valid] - 1.0
    return (ratio ** 2).sum()


def score_heatmap_single(pos: torch.Tensor,
                          idx: int,
                          expected: torch.Tensor,
                          diagonal_size: int = 3,
                          same_chr_mask: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
    """
    Heatmap score contribution of a single moved bead `idx`.
    O(N) memory – safe for large N.  Handles float16 expected.
    """
    N = pos.shape[0]
    dist = single_bead_distances(pos, idx)  # (N,)

    j = torch.arange(N, device=pos.device)
    diff_idx = (j - idx).abs()
    diag_mask = diff_idx >= diagonal_size

    exp_row = expected[idx].float()  # cast fp16→fp32 if needed; (N,)
    valid = (exp_row > 1e-3) & diag_mask
    if same_chr_mask is not None:
        valid = valid & same_chr_mask[idx]

    ratio = dist[valid] / exp_row[valid] - 1.0
    return (ratio ** 2).sum()


def score_heatmap_chunked(pos: torch.Tensor,
                           expected: torch.Tensor,
                           diagonal_size: int = 3,
                           same_chr_mask: Optional[torch.Tensor] = None,
                           chunk_size: int = 512) -> torch.Tensor:
    """
    Full heatmap score without ever allocating an N×N tensor.

    Processes `chunk_size` rows at a time using the squared-norm dot-product
    trick so the largest intermediate is (chunk_size, N) ~ 50 MB for chunk=512.
    Counts only i < j pairs (upper triangle).
    """
    N = pos.shape[0]
    device = pos.device
    total = torch.zeros(1, device=device)
    pos_sq = (pos ** 2).sum(dim=1)   # (N,) – precomputed once

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk = pos[i_start:i_end]                        # (C, 3)
        chunk_sq = (chunk ** 2).sum(dim=1)                # (C,)

        # squared distances via matmul (no (C,N,3) intermediate)
        dot = chunk @ pos.t()                             # (C, N)
        sq_dist = (chunk_sq[:, None] + pos_sq[None, :] - 2.0 * dot).clamp(min=0)
        dist = sq_dist.sqrt()                             # (C, N)

        exp_chunk = expected[i_start:i_end].float()       # (C, N) fp32

        i_idx = torch.arange(i_start, i_end, device=device)[:, None]   # (C, 1)
        j_idx = torch.arange(N, device=device)[None, :]                 # (1, N)

        valid = ((j_idx > i_idx)
                 & ((i_idx - j_idx).abs() >= diagonal_size)
                 & (exp_chunk > 1e-3))
        if same_chr_mask is not None:
            valid = valid & same_chr_mask[i_start:i_end]

        if valid.any():
            ratio = dist[valid] / exp_chunk[valid] - 1.0
            total = total + (ratio ** 2).sum()

    return total.squeeze()


# ── Arc / distance score ──────────────────────────────────────────────────────

def score_arcs(pos: torch.Tensor,
               arc_starts: torch.Tensor,
               arc_ends: torch.Tensor,
               arc_expected: torch.Tensor,
               k: float = 1.0,
               k_repulsion: float = 1.0) -> torch.Tensor:
    """
    Spring score for all arcs.

    arc_expected[i] = expected distance for arc i
        > 0 : harmonic spring  k * ((d - e) / e)^2   (asymmetric: stretch/squeeze)
        < 0 : repulsion        k_repulsion / d

    arc_starts, arc_ends: (M,) long tensors of cluster indices.
    """
    d = (pos[arc_starts] - pos[arc_ends]).norm(dim=1)  # (M,)

    spring_mask = arc_expected > 0
    repulse_mask = arc_expected < 0

    score = torch.tensor(0.0, device=pos.device)

    if spring_mask.any():
        e = arc_expected[spring_mask]
        di = d[spring_mask]
        score = score + k * (((di - e) / e) ** 2).sum()

    if repulse_mask.any():
        di = d[repulse_mask].clamp(min=1e-6)
        score = score + (k_repulsion / di).sum()

    return score


def score_arcs_single(pos: torch.Tensor,
                       idx: int,
                       arc_starts: torch.Tensor,
                       arc_ends: torch.Tensor,
                       arc_expected: torch.Tensor,
                       k: float = 1.0,
                       k_repulsion: float = 1.0) -> torch.Tensor:
    """Arc score contribution of bead `idx` (all arcs touching idx)."""
    mask = (arc_starts == idx) | (arc_ends == idx)
    if not mask.any():
        return torch.tensor(0.0, device=pos.device)

    s = arc_starts[mask]
    e = arc_ends[mask]
    exp = arc_expected[mask]
    d = (pos[s] - pos[e]).norm(dim=1)

    spring_mask = exp > 0
    repulse_mask = exp < 0
    score = torch.tensor(0.0, device=pos.device)

    if spring_mask.any():
        ei = exp[spring_mask]
        di = d[spring_mask]
        score = score + k * (((di - ei) / ei) ** 2).sum()
    if repulse_mask.any():
        di = d[repulse_mask].clamp(min=1e-6)
        score = score + (k_repulsion / di).sum()

    return score


# ── Structural / smooth score ─────────────────────────────────────────────────

def score_structure_smooth(pos: torch.Tensor,
                            chain_lengths: torch.Tensor,
                            k_chain: float = 1.0,
                            angular_k: float = 0.1) -> torch.Tensor:
    """
    Chain-level structural score:
        sum_i  k_chain * ((|p_{i+1} - p_i| - L_i) / L_i)^2
      + sum_i  angular_k * angle_i^3

    chain_lengths: (N-1,) expected distances between consecutive beads.
    """
    if pos.shape[0] < 2:
        return torch.tensor(0.0, device=pos.device)

    # linker spring
    diffs = pos[1:] - pos[:-1]             # (N-1, 3)
    dist = diffs.norm(dim=1)               # (N-1,)
    L = chain_lengths.clamp(min=1e-6)
    chain_score = k_chain * (((dist - L) / L) ** 2).sum()

    # angular penalty
    if pos.shape[0] < 3:
        return chain_score

    v1 = diffs[:-1]  # (N-2, 3)
    v2 = diffs[1:]   # (N-2, 3)
    cos_angle = F.cosine_similarity(v1, v2, dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)  # (N-2,)
    angular_score = angular_k * (angle ** 3).sum()

    return chain_score + angular_score


# ── CTCF orientation score ────────────────────────────────────────────────────

def _angle_between_orientations(o1: str, o2: str, flip: bool = False) -> float:
    """
    Compute the penalty angle between two orientation vectors.
    Orientation: 'R' → +x, 'L' → -x, 'N' → no constraint.
    Returns 0.0 if either orientation is 'N'.
    """
    if o1 == "N" or o2 == "N":
        return 0.0
    v1 = 1.0 if o1 == "R" else -1.0
    v2 = 1.0 if o2 == "R" else -1.0
    if flip:
        v2 = -v2
    # angle normalised to [0, π]
    dot = v1 * v2
    return math.acos(max(-1.0, min(1.0, dot)))


def score_orientation(pos: torch.Tensor,
                      orientations: List[str],
                      arc_starts: torch.Tensor,
                      arc_ends: torch.Tensor,
                      weight: float = 1.0) -> torch.Tensor:
    """
    CTCF orientation penalty for all arcs:
        sum  weight * min(angle(o_i, o_j), angle(o_i, -o_j))^2

    For convergent CTCF (→ ←) the penalty is zero; divergent (← →) is maximal.
    """
    score = 0.0
    for k in range(arc_starts.shape[0]):
        si = arc_starts[k].item()
        ei = arc_ends[k].item()
        o1 = orientations[si]
        o2 = orientations[ei]
        if o1 == "N" or o2 == "N":
            continue
        a1 = _angle_between_orientations(o1, o2, flip=False)
        a2 = _angle_between_orientations(o1, o2, flip=True)
        a = min(a1, a2)
        score += weight * a * a
    return torch.tensor(score, dtype=torch.float32, device=pos.device)


def score_orientation_single(pos: torch.Tensor,
                              idx: int,
                              orientations: List[str],
                              arc_starts: torch.Tensor,
                              arc_ends: torch.Tensor,
                              weight: float = 1.0) -> torch.Tensor:
    """Orientation score contribution of bead `idx`."""
    mask = (arc_starts == idx) | (arc_ends == idx)
    s_idx = arc_starts[mask].tolist()
    e_idx = arc_ends[mask].tolist()
    score = 0.0
    for si, ei in zip(s_idx, e_idx):
        o1 = orientations[si]
        o2 = orientations[ei]
        if o1 == "N" or o2 == "N":
            continue
        a1 = _angle_between_orientations(o1, o2, flip=False)
        a2 = _angle_between_orientations(o1, o2, flip=True)
        a = min(a1, a2)
        score += weight * a * a
    return torch.tensor(score, dtype=torch.float32, device=pos.device)
