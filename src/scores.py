"""
Energy / score functions — vectorised PyTorch.

All functions return a scalar float tensor (the lower the better).
Position tensors have shape (N, 3) and dtype float32.

Every algorithmic line carries a # cudaMMC: annotation pointing to the exact
C++ source line it replicates.  Source files:
  cudammc/src/LooperSolver.cpp   – all score functions
  cudammc/thirdparty/common.cpp  – angle() / angle_norm() helpers
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


# ── Utility ──────────────────────────────────────────────────────────────────

def pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Return (N, N) matrix of Euclidean distances."""
    # cudaMMC computes inline: v = pos[i]-pos[j]; d = v.length()  (LooperSolver.cpp:1928,2219)
    # We use the squared-norm dot-product trick to avoid an (N,N,3) intermediate.
    sq = (pos ** 2).sum(dim=1)
    dot = pos @ pos.t()
    sq_dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot
    return sq_dist.clamp(min=0.0).sqrt()


def single_bead_distances(pos: torch.Tensor, idx: int) -> torch.Tensor:
    """Return (N,) distances from bead `idx` to all other beads."""
    # cudaMMC LooperSolver.cpp:2219-2221:
    #   d = (clusters[ar[i]].pos - clusters[ar[moved]].pos).length()
    diff = pos - pos[idx].unsqueeze(0)  # (N, 3)
    return (diff ** 2).sum(dim=1).clamp(min=0.0).sqrt()


# ── Heatmap score ─────────────────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:2195-2228  calcScoreHeatmapActiveRegion(int moved)
# calcScoreHeatmapActiveRegion(-1) calls calcScoreHeatmapActiveRegion(i) for each i,
# double-counting every pair (i→j and j→i).  mc.py compensates with 2× factor.

def score_heatmap(pos: torch.Tensor,
                  expected: torch.Tensor,
                  diagonal_size: int = 3,
                  same_chr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Upper-triangle heatmap score (= cudaMMC full score / 2):
        sum_{i<j, |i-j|>=diag} ((dist_ij / expected_ij) - 1)^2
    where expected_ij > 1e-3 and (same chromosome).

    cudaMMC computes the same formula over all i≠j (double-counting).
    """
    N = pos.shape[0]
    dist = pairwise_distances(pos)  # (N, N)

    # cudaMMC LooperSolver.cpp:2214: if (abs(i - moved) >= heatmap_dist.diagonal_size)
    idx = torch.arange(N, device=pos.device)
    diff_idx = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    diag_mask = diff_idx >= diagonal_size

    # cudaMMC LooperSolver.cpp:2216: if (heatmap_dist.v[i][moved] < 1e-6) continue
    # We use > 1e-3; equivalent since values are 0 or ~100+ (heatmap_to_expected_distances)
    valid = (expected > 1e-3) & diag_mask
    if same_chr_mask is not None:
        # cudaMMC LooperSolver.cpp:2210-2211: getChromosomeHeatmapBoundary(moved, st, end)
        valid = valid & same_chr_mask

    # cudaMMC LooperSolver.cpp:2222-2223: cerr = (d-exp)/exp; err += cerr*cerr
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
    O(N) memory — safe for large N.  Handles float16 expected.

    cudaMMC source: LooperSolver.cpp:2207-2228  calcScoreHeatmapActiveRegion(int moved)
    Sums over ALL j≠idx in the chromosome range (not just upper triangle).
    """
    N = pos.shape[0]
    # cudaMMC LooperSolver.cpp:2219-2221: d = (pos[i] - pos[moved]).length()
    dist = single_bead_distances(pos, idx)  # (N,)

    j = torch.arange(N, device=pos.device)
    # cudaMMC LooperSolver.cpp:2214: abs(i - moved) >= heatmap_dist.diagonal_size
    diff_idx = (j - idx).abs()
    diag_mask = diff_idx >= diagonal_size

    exp_row = expected[idx].float()  # cast fp16→fp32 if needed; (N,)
    # cudaMMC LooperSolver.cpp:2216: if (heatmap_dist.v[i][moved] < 1e-6) continue
    valid = (exp_row > 1e-3) & diag_mask
    if same_chr_mask is not None:
        # cudaMMC LooperSolver.cpp:2210-2211: getChromosomeHeatmapBoundary
        valid = valid & same_chr_mask[idx]

    # cudaMMC LooperSolver.cpp:2222-2223: cerr = (d - exp) / exp; err += cerr*cerr
    ratio = dist[valid] / exp_row[valid] - 1.0
    return (ratio ** 2).sum()


def score_heatmap_chunked(pos: torch.Tensor,
                           expected: torch.Tensor,
                           diagonal_size: int = 3,
                           same_chr_mask: Optional[torch.Tensor] = None,
                           chunk_size: int = 512) -> torch.Tensor:
    """
    Full heatmap score without ever allocating an N×N tensor.

    Equivalent to calcScoreHeatmapActiveRegion(-1)/2 (upper-triangle only).
    mc.py compensates with the 2× factor when initialising total_score.

    Processes `chunk_size` rows at a time using the squared-norm dot-product
    trick so the largest intermediate is (chunk_size, N) ~ 50 MB for chunk=512.
    Counts only i < j pairs (upper triangle).
    """
    N = pos.shape[0]
    device = pos.device
    total = torch.zeros(1, device=device)
    pos_sq = (pos ** 2).sum(dim=1)   # (N,) — precomputed once

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

        # cudaMMC LooperSolver.cpp:2214: abs(i-moved) >= diagonal_size
        # cudaMMC LooperSolver.cpp:2216: < 1e-6 → skip
        valid = ((j_idx > i_idx)
                 & ((i_idx - j_idx).abs() >= diagonal_size)
                 & (exp_chunk > 1e-3))
        if same_chr_mask is not None:
            valid = valid & same_chr_mask[i_start:i_end]

        # cudaMMC LooperSolver.cpp:2222-2223: cerr = (d-exp)/exp; err += cerr*cerr
        if valid.any():
            ratio = dist[valid] / exp_chunk[valid] - 1.0
            total = total + (ratio ** 2).sum()

    return total.squeeze()


# ── Arc / distance score ──────────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:1919-1984  calcScoreDistancesActiveRegion()
#
# Spring constants: springConstantStretchArcs=1.0, springConstantSqueezeArcs=1.0
# (Settings.cpp:254-255).  Symmetric defaults → single k=1.0 matches both.
#
# Repulsion note: the full-score version (cpp:1932-1934) adds 1/d for negative
# entries.  The single-bead version (cpp:1966-1969) has this commented out.
# Python keeps repulsion in both for internal consistency.  In practice, negative-
# entry "unknown" pairs are not present in the arc list for typical inputs.

def score_arcs(pos: torch.Tensor,
               arc_starts: torch.Tensor,
               arc_ends: torch.Tensor,
               arc_expected: torch.Tensor,
               k: float = 1.0,
               k_repulsion: float = 1.0) -> torch.Tensor:
    """
    Spring score for all arcs.

    arc_expected[i] = expected distance for arc i
        > 0 : harmonic spring  k * ((d - e) / e)^2
        < 0 : repulsion        k_repulsion / d

    arc_starts, arc_ends: (M,) long tensors of cluster indices.

    cudaMMC source: LooperSolver.cpp:1919-1950  calcScoreDistancesActiveRegion()
    """
    # cudaMMC LooperSolver.cpp:1928: v = clusters[ar[i]].pos - clusters[ar[j]].pos
    d = (pos[arc_starts] - pos[arc_ends]).norm(dim=1)  # (M,)

    # cudaMMC LooperSolver.cpp:1932: heatmap_exp_dist_anchor.v[i][j] < 0.0f → repulsion
    spring_mask = arc_expected > 0
    repulse_mask = arc_expected < 0

    score = torch.tensor(0.0, device=pos.device)

    if spring_mask.any():
        e = arc_expected[spring_mask]
        di = d[spring_mask]
        # cudaMMC LooperSolver.cpp:1940-1944:
        #   diff = (v.length() - exp) / exp
        #   sc += diff*diff * (diff>=0 ? springConstantStretchArcs : springConstantSqueezeArcs)
        score = score + k * (((di - e) / e) ** 2).sum()

    if repulse_mask.any():
        # cudaMMC LooperSolver.cpp:1933: sc += 1.0f / v.length()
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
    """
    Arc score contribution of bead `idx` (all arcs touching idx).

    cudaMMC source: LooperSolver.cpp:1954-1984  calcScoreDistancesActiveRegion(int cluster_moved)

    Note: cudaMMC single-bead version has the repulsion term commented out
    (cpp:1967-1969 // sc += 1.0f / v.length()).  Python keeps repulsion in both
    full and single-bead for self-consistency.
    """
    mask = (arc_starts == idx) | (arc_ends == idx)
    if not mask.any():
        return torch.tensor(0.0, device=pos.device)

    s = arc_starts[mask]
    e = arc_ends[mask]
    exp = arc_expected[mask]
    # cudaMMC LooperSolver.cpp:1964: v = clusters[st].pos - clusters[ar[i]].pos
    d = (pos[s] - pos[e]).norm(dim=1)

    spring_mask = exp > 0
    repulse_mask = exp < 0
    score = torch.tensor(0.0, device=pos.device)

    if spring_mask.any():
        ei = exp[spring_mask]
        di = d[spring_mask]
        # cudaMMC LooperSolver.cpp:1975-1982:
        #   diff = (v.length() - exp) / exp
        #   sc += diff*diff * springConstantStretchArcs / SqueezeArcs
        score = score + k * (((di - ei) / ei) ** 2).sum()
    if repulse_mask.any():
        # cudaMMC LooperSolver.cpp:1967-1969: commented out in single-bead version
        di = d[repulse_mask].clamp(min=1e-6)
        score = score + (k_repulsion / di).sum()

    return score


# ── Structural / smooth score ─────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:2026-2107  calcScoreStructureSmooth()
#
# Spring constants: springConstantStretch=springConstantSqueeze=0.1 (Settings.cpp:250-251)
# and springAngularConstant=0.1 (Settings.cpp:252). Symmetric stretch/squeeze →
# single k_chain=0.1 matches both.
#
# Weights: weightDistSmooth=weightAngleSmooth=1.0 (Settings.cpp:240-241).
# Full score returns sca*weightDist + scb*weightAngle (cpp:2062) = sca+scb at defaults.
# Single-bead returns sca+scb with no weight multipliers (cpp:2106). Equivalent.
#
# Angular formula: cudaMMC common.cpp:44: angle()=(1-dot)/2  NOT acos.
#   range [0,1] → cubed range [0,1] with 0.1 coefficient.
#   Do NOT use acos (range [0,π]≈3.14, cubed ≈31 — 31× too large).

def score_structure_smooth(pos: torch.Tensor,
                            chain_lengths: torch.Tensor,
                            k_chain: float = 1.0,
                            angular_k: float = 0.1) -> torch.Tensor:
    """
    Chain-level structural score:
        sum_i  k_chain * ((|p_{i+1} - p_i| - L_i) / L_i)^2   [length spring]
      + sum_i  angular_k * angle_i^3                           [angular penalty]

    where angle_i = (1 - cos_theta_i) / 2  per cudaMMC common.cpp:44 angle()

    chain_lengths: (N-1,) expected distances between consecutive beads.

    cudaMMC source: LooperSolver.cpp:2026-2063  calcScoreStructureSmooth(bool,bool)
    """
    if pos.shape[0] < 2:
        return torch.tensor(0.0, device=pos.device)

    # cudaMMC LooperSolver.cpp:2034: v = clusters[ar[i]].pos - clusters[ar[i+1]].pos
    diffs = pos[1:] - pos[:-1]             # (N-1, 3)  direction reversed vs cudaMMC but length same
    dist = diffs.norm(dim=1)               # (N-1,)

    # linker spring — skip zero-length links (overlapping anchors have no spring)
    valid_link = chain_lengths >= 1e-9
    if valid_link.any():
        # cudaMMC LooperSolver.cpp:2037-2038: dtn < 1e-6 → dtn = 1e-6
        L = chain_lengths[valid_link].clamp(min=1e-6)
        # cudaMMC LooperSolver.cpp:2040-2043:
        #   diff = (v.length() - dtn) / dtn
        #   sca += diff*diff * (diff>=0 ? springConstantStretch : springConstantSqueeze)
        chain_score = k_chain * (((dist[valid_link] - L) / L) ** 2).sum()
    else:
        chain_score = torch.tensor(0.0, device=pos.device)

    # angular penalty
    if pos.shape[0] < 3:
        return chain_score

    v1 = diffs[:-1]  # (N-2, 3)  bond vectors p[i+1]-p[i]  (cudaMMC: p[i]-p[i+1], same angle)
    v2 = diffs[1:]   # (N-2, 3)  bond vectors p[i+2]-p[i+1]
    cos_angle = F.cosine_similarity(v1, v2, dim=1).clamp(-1.0, 1.0)
    # cudaMMC common.cpp:42-44: angle(v1,v2) { normalize; dot = DotProduct; return 1-(dot+1)/2 }
    # = (1 - dot) / 2  — NOT acos.  Range [0,1] for anti-parallel → [0,0.1] when cubed with k=0.1.
    angle = (1.0 - cos_angle) / 2.0                   # cudaMMC common.cpp:44: 1-(dot+1)/2
    # cudaMMC LooperSolver.cpp:2048: scb += ang*ang*ang * springAngularConstant
    angular_score = angular_k * (angle ** 3).sum()
    # cudaMMC LooperSolver.cpp:2062: return sca*weightDistSmooth + scb*weightAngleSmooth
    # weightDistSmooth=weightAngleSmooth=1.0 (Settings.cpp:240-241) — applied directly here
    return chain_score + angular_score


def score_chain_single(pos: torch.Tensor,
                       idx: int,
                       chain_lengths: torch.Tensor,
                       k_chain: float = 1.0,
                       angular_k: float = 0.1) -> torch.Tensor:
    """
    Chain + angular score contribution of bead `idx` only.
    O(1) — only evaluates springs and angles touching bead idx.
    Moving idx affects springs (idx-1,idx) and (idx,idx+1),
    and angles at idx-1, idx, and idx+1.

    cudaMMC source: LooperSolver.cpp:2065-2107  calcScoreStructureSmooth(int cluster_moved,bool,bool)
    Returns sca+scb without weight multipliers (cpp:2106); equivalent to weights=1.0.
    """
    N = pos.shape[0]
    device = pos.device
    score = torch.zeros(1, device=device)

    # cudaMMC LooperSolver.cpp:2072: for (int i = cluster_moved-2; i < cluster_moved+2; i++)
    # Chain springs — only i==cluster_moved-1 or i==cluster_moved (cpp:2080)
    # Skip zero-length links (overlapping anchors)
    if idx > 0 and chain_lengths[idx - 1] >= 1e-9:
        # cudaMMC LooperSolver.cpp:2082-2083: dtn < 1e-6 → dtn = 1e-6
        L = chain_lengths[idx - 1].clamp(min=1e-6)
        d = (pos[idx] - pos[idx - 1]).norm()
        # cudaMMC LooperSolver.cpp:2084-2087: diff=(v.length()-dtn)/dtn; sca+=diff*diff*k
        score = score + k_chain * ((d - L) / L) ** 2
    if idx < N - 1 and chain_lengths[idx] >= 1e-9:
        L = chain_lengths[idx].clamp(min=1e-6)
        d = (pos[idx + 1] - pos[idx]).norm()
        score = score + k_chain * ((d - L) / L) ** 2

    # cudaMMC LooperSolver.cpp:2091: angles at three junctions touching idx
    # for i in {cluster_moved-2..cluster_moved+1}, angles when i > cluster_moved-2 and i > 0
    # → junctions at idx-1, idx, idx+1
    for j in (idx - 1, idx, idx + 1):
        if 0 < j < N - 1:
            v1 = pos[j] - pos[j - 1]
            v2 = pos[j + 1] - pos[j]
            cos_a = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).clamp(-1.0, 1.0)
            # cudaMMC LooperSolver.cpp:2092: ang = angle(v, v2)
            # common.cpp:44: angle() = 1-(dot+1)/2 = (1-dot)/2   NOT acos
            angle = (1.0 - cos_a) / 2.0                # cudaMMC common.cpp:44
            # cudaMMC LooperSolver.cpp:2093: scb += ang*ang*ang * springAngularConstant
            score = score + angular_k * (angle ** 3)

    # cudaMMC LooperSolver.cpp:2106: return sca + scb  (no weight multipliers)
    return score.squeeze()


# ── CTCF orientation score ────────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:2109-2162  calcScoreOrientation()
#
# IMPORTANT SIMPLIFICATION: cudaMMC computes 3D orientation vectors via
# calcOrientation(cind) = normalize( p[cind+1] - p[cind-1] ) flipped by 'L'/'R' label,
# then scores using angle_norm(o1, o2) = (1-dot)/2 between those 3D vectors.
# These orientation vectors are position-dependent (updated every bead move in MC).
#
# Python uses static string labels ('L','R','N') from input — the score is
# position-independent and does not change during MC.  This is a known simplification:
# the orientation term does not drive bead movement in the Python reimplementation.
#
# angle_norm (common.cpp:48-52): (1-dot)/2 with pre-normalised vectors.
# Python uses 1D ±1 dot products as a proxy for the 3D angle_norm formula.

def _angle_between_orientations(o1: str, o2: str, flip: bool = False) -> float:
    """
    Scalar orientation dissimilarity between two 1D orientation labels.
    Orientation: 'R' → +1, 'L' → -1, 'N' → no constraint.
    Returns 0.0 if either orientation is 'N'.

    Approximates cudaMMC common.cpp:48-52 angle_norm(o1, o2) = (1-dot)/2
    using 1D ±1 vectors as a proxy for 3D orientation vectors.
    """
    if o1 == "N" or o2 == "N":
        return 0.0
    v1 = 1.0 if o1 == "R" else -1.0
    v2 = 1.0 if o2 == "R" else -1.0
    if flip:
        v2 = -v2
    # cudaMMC common.cpp:51: return 1.0 - (dot+1.0)/2.0 = (1-dot)/2
    dot = v1 * v2
    return (1.0 - dot) / 2.0


def score_orientation(pos: torch.Tensor,
                      orientations: List[str],
                      arc_starts: torch.Tensor,
                      arc_ends: torch.Tensor,
                      weight: float = 1.0) -> torch.Tensor:
    """
    CTCF orientation penalty for all arcs:
        sum  weight * min(dissim(o_i, o_j), dissim(o_i, -o_j))^3

    dissim = (1-dot)/2  per cudaMMC common.cpp angle_norm formula.
    For convergent CTCF (→ ←) the penalty is zero; divergent (← →) is maximal.

    cudaMMC source: LooperSolver.cpp:2109-2131  calcScoreOrientation(orientation)
    Uses motifOrientationWeight=1.0 (Settings.cpp:160), motifsSymmetric=true (cpp:159).

    NOTE: This is a static approximation. cudaMMC computes position-dependent
    3D orientation vectors (calcOrientation: normalize(p[i+1]-p[i-1]) ± L/R flip),
    so its orientation score changes with every bead move.  Python uses fixed
    string labels — the score is constant throughout MC and does not drive movement.
    """
    score = 0.0
    for k in range(arc_starts.shape[0]):
        si = arc_starts[k].item()
        ei = arc_ends[k].item()
        o1 = orientations[si]
        o2 = orientations[ei]
        if o1 == "N" or o2 == "N":
            continue
        # cudaMMC LooperSolver.cpp:2114-2116:
        #   ang = angle_norm(orientation[el.first],
        #                    (motifsSymmetric ? 1.0 : -1.0) * orientation[el.second[i]])
        # motifsSymmetric=true → multiply neighbor by +1 (no flip).
        # We try both flip=False and flip=True and take min (convergent CTCF = 0 penalty).
        a1 = _angle_between_orientations(o1, o2, flip=False)
        a2 = _angle_between_orientations(o1, o2, flip=True)
        a = min(a1, a2)
        # cudaMMC LooperSolver.cpp:2117: err += ang*ang * weight[el.first][i]
        # cudaMMC LooperSolver.cpp:2131: return err * motifOrientationWeight (=1.0)
        score += weight * a * a
    return torch.tensor(score, dtype=torch.float32, device=pos.device)


def score_orientation_single(pos: torch.Tensor,
                              idx: int,
                              orientations: List[str],
                              arc_starts: torch.Tensor,
                              arc_ends: torch.Tensor,
                              weight: float = 1.0) -> torch.Tensor:
    """
    Orientation score contribution of bead `idx`.

    cudaMMC source: LooperSolver.cpp:2134-2162  calcScoreOrientation(orientation, anchor_index)
    Sums over active_anchors_neighbors[anchor_index] (arcs touching idx).

    NOTE: Same static-label simplification as score_orientation above.
    """
    mask = (arc_starts == idx) | (arc_ends == idx)
    s_idx = arc_starts[mask].tolist()
    e_idx = arc_ends[mask].tolist()
    score = 0.0
    for si, ei in zip(s_idx, e_idx):
        o1 = orientations[si]
        o2 = orientations[ei]
        if o1 == "N" or o2 == "N":
            continue
        # cudaMMC LooperSolver.cpp:2148-2151:
        #   ang = angle_norm(orientation[anchor_index],
        #                    (motifsSymmetric ? 1.0 : -1.0) * orientation[val])
        #   err += ang * ang
        a1 = _angle_between_orientations(o1, o2, flip=False)
        a2 = _angle_between_orientations(o1, o2, flip=True)
        a = min(a1, a2)
        score += weight * a * a
    return torch.tensor(score, dtype=torch.float32, device=pos.device)
