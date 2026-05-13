"""
Energy / score functions — vectorised PyTorch.

All functions return a scalar float tensor (the lower the better).  Position
tensors have shape (N, 3) and dtype float32.

cudaMMC source files mirrored here:
  cudaMMC/src/LooperSolver.cpp   — all score functions
  cudaMMC/thirdparty/common.cpp  — angle() / angle_norm() helpers

Every algorithmic line carries a ``# cudaMMC LINE`` annotation pointing to the
exact upstream source.  Numerical equivalence rules (AGENTS.md prime directive):

  * `expected < 1e-6 → continue`  (cudaMMC cpp:1939 + cpp:2216)
  * `expected < 0   → 1/d`        (REPULSION sentinel, cpp:1932-1934 — full score only;
                                   the single-bead version comments it out,
                                   cpp:1966-1969)
  * spring branch: `diff = (d - exp)/exp`; coefficient picked by SIGN of diff
    (stretch when ``diff >= 0``, squeeze when ``diff < 0``); cpp:1940-1944.
  * orientation:  ``angle_norm(o1, o2) = (1 - dot)/2`` between **unit 3-vectors**
    that are re-derived per call from current positions:
    ``orn[i] = normalize(p[i+1] - p[i-1])`` flipped by 'L' label; cpp:3437-3454.
"""

import math  # noqa: F401  (kept for downstream tooling)
from typing import List, Optional

import torch
import torch.nn.functional as F


# ── Utility ──────────────────────────────────────────────────────────────────

def pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """(N, N) Euclidean distance matrix."""
    # cudaMMC LooperSolver.cpp:1928,2219: v = pos[i]-pos[j]; v.length()
    sq = (pos ** 2).sum(dim=1)
    dot = pos @ pos.t()
    sq_dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot
    return sq_dist.clamp(min=0.0).sqrt()


def single_bead_distances(pos: torch.Tensor, idx: int) -> torch.Tensor:
    """(N,) distances from bead ``idx`` to all other beads."""
    # cudaMMC LooperSolver.cpp:2219-2221:
    #   d = (clusters[ar[i]].pos - clusters[ar[moved]].pos).length()
    diff = pos - pos[idx].unsqueeze(0)
    return (diff ** 2).sum(dim=1).clamp(min=0.0).sqrt()


# =============================================================================
# Heatmap score — calcScoreHeatmapActiveRegion  (cudaMMC LooperSolver.cpp:2195-2228)
# =============================================================================
# cpp:2216:  if (heatmap_dist.v[i][moved] < 1e-6) continue;
#   →  -1 sentinels are SKIPPED here (-1 < 1e-6 is true).  Repulsion happens
#      in the *arc/distance* score, not the heatmap score (cpp:1932-1934).
#
# cpp:2222-2223:  cerr = (d - exp)/exp;  err += cerr*cerr
# cpp:2210-2211:  getChromosomeHeatmapBoundary restricts to moved-bead's chrom
#                 (multi-chrom only).  Passed in via `same_chr_mask`.
# `diagonal_size` MUST come from the heatmap object (AUDIT §C1), not Settings.

def score_heatmap(pos: torch.Tensor,
                  expected: torch.Tensor,
                  diagonal_size: int = 1,
                  same_chr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Upper-triangle heatmap score (= cudaMMC full score / 2).

    cudaMMC iterates all i≠j (cpp:2207-2208 calls ``(i)`` for every i, double
    counting); mc.py compensates with the 2× factor when initialising the
    running total.

    `expected` is the dense (N,N) heatmap-derived expected-distance matrix
    from :func:`heatmap.heatmap_to_expected_distances` (fp32, may contain
    the −1 sentinel — which is naturally skipped by the ``< 1e-6`` test).
    """
    N = pos.shape[0]
    dist = pairwise_distances(pos)

    idx = torch.arange(N, device=pos.device)
    diff_idx = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    diag_mask = diff_idx >= diagonal_size                # cpp:2214

    # cpp:2216: < 1e-6 → continue.  Use the matching threshold so the −1
    # sentinel and true zeros are skipped together (AUDIT §C2).
    valid = (expected.float() >= 1e-6) & diag_mask
    # Upper triangle only — full score is recovered by mc.py with the 2× factor.
    valid = valid & (idx.unsqueeze(0) > idx.unsqueeze(1))
    if same_chr_mask is not None:
        valid = valid & same_chr_mask                    # cpp:2210-2211

    ratio = dist[valid] / expected[valid].float() - 1.0  # cpp:2222
    return (ratio ** 2).sum()                            # cpp:2223


def score_heatmap_single(pos: torch.Tensor,
                          idx: int,
                          expected: torch.Tensor,
                          diagonal_size: int = 1,
                          same_chr_mask: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
    """Heatmap score contribution of one moved bead — cpp:2207-2228."""
    N = pos.shape[0]
    dist = single_bead_distances(pos, idx)               # cpp:2219-2221

    j = torch.arange(N, device=pos.device)
    diag_mask = (j - idx).abs() >= diagonal_size         # cpp:2214

    exp_row = expected[idx].float()
    valid = (exp_row >= 1e-6) & diag_mask                # cpp:2216
    if same_chr_mask is not None:
        valid = valid & same_chr_mask[idx]               # cpp:2210-2211

    ratio = dist[valid] / exp_row[valid] - 1.0           # cpp:2222
    return (ratio ** 2).sum()                            # cpp:2223


def score_heatmap_chunked(pos: torch.Tensor,
                           expected: torch.Tensor,
                           diagonal_size: int = 1,
                           same_chr_mask: Optional[torch.Tensor] = None,
                           chunk_size: int = 512) -> torch.Tensor:
    """Memory-bounded variant of :func:`score_heatmap`.

    Equivalent to ``calcScoreHeatmapActiveRegion(-1) / 2`` — upper triangle
    only, mc.py applies the 2× factor.  Never allocates an (N, N) matrix.
    """
    N = pos.shape[0]
    device = pos.device
    total = torch.zeros(1, device=device)
    pos_sq = (pos ** 2).sum(dim=1)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk = pos[i_start:i_end]
        chunk_sq = (chunk ** 2).sum(dim=1)
        dot = chunk @ pos.t()
        sq_dist = (chunk_sq[:, None] + pos_sq[None, :] - 2.0 * dot).clamp(min=0)
        dist = sq_dist.sqrt()
        exp_chunk = expected[i_start:i_end].float()

        i_idx = torch.arange(i_start, i_end, device=device)[:, None]
        j_idx = torch.arange(N, device=device)[None, :]

        valid = ((j_idx > i_idx)
                 & ((i_idx - j_idx).abs() >= diagonal_size)   # cpp:2214
                 & (exp_chunk >= 1e-6))                       # cpp:2216
        if same_chr_mask is not None:
            valid = valid & same_chr_mask[i_start:i_end]

        if valid.any():
            ratio = dist[valid] / exp_chunk[valid] - 1.0      # cpp:2222
            total = total + (ratio ** 2).sum()                # cpp:2223

    return total.squeeze()


# =============================================================================
# Distance / arc score
# =============================================================================
# Two distinct entry points map to the SAME cudaMMC function
# `calcScoreDistancesActiveRegion` (cpp:1919-1984):
#
#   - score_distances_active_region / _single   ←  DENSE N×N expected matrix,
#       iterates every pair, with the −1 repulsion branch.  This is what
#       cudaMMC actually calls in MonteCarloArcs (cpp:3086).  Used by the new
#       Phase-5 solver path (item 17).
#   - score_arcs / score_arcs_single            ←  SPARSE arc list (legacy).
#       Kept for the current solver until Phase-5 lands; same per-pair formula.

def _spring_term(d: torch.Tensor, exp: torch.Tensor,
                 k_stretch: float, k_squeeze: float) -> torch.Tensor:
    """cudaMMC cpp:1940-1944: diff = (d-exp)/exp; sc += diff² * (stretch | squeeze)."""
    diff = (d - exp) / exp
    coef = torch.where(diff >= 0,
                       torch.full_like(diff, k_stretch),
                       torch.full_like(diff, k_squeeze))
    return (diff * diff * coef).sum()


def score_distances_active_region(pos: torch.Tensor,
                                  expected: torch.Tensor,
                                  k_stretch: float = 1.0,
                                  k_squeeze: float = 1.0,
                                  k_repulsion: float = 1.0) -> torch.Tensor:
    """Full score over the **dense** per-IB expected-distance matrix.

    Mirror of ``calcScoreDistancesActiveRegion()`` (cpp:1919-1950):

      * ``expected[i,j] < 0``      → ``sc += 1 / d``      (cpp:1932-1934)
      * ``expected[i,j] < 1e-6``   → skip                  (cpp:1939)
      * else                        → spring term (cpp:1940-1944)

    Returns the upper-triangle sum (cpp loops ``i<j``).
    ``expected`` must be (N, N) fp32 with ``-1`` for repulsion pairs and
    arc expected-distances for spring pairs (zero elsewhere).
    """
    N = pos.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=pos.device)
    dist = pairwise_distances(pos)                          # cpp:1928
    exp = expected.float()

    iu, ju = torch.triu_indices(N, N, offset=1, device=pos.device)
    e = exp[iu, ju]
    d = dist[iu, ju]

    repulse = e < 0                                          # cpp:1932
    skip = (~repulse) & (e < 1e-6)                           # cpp:1939
    spring = ~repulse & ~skip

    score = torch.zeros((), device=pos.device)
    if repulse.any():
        score = score + (k_repulsion / d[repulse].clamp(min=1e-6)).sum()
    if spring.any():
        score = score + _spring_term(d[spring], e[spring], k_stretch, k_squeeze)
    return score


def score_distances_active_region_single(pos: torch.Tensor,
                                         idx: int,
                                         expected: torch.Tensor,
                                         k_stretch: float = 1.0,
                                         k_squeeze: float = 1.0,
                                         k_repulsion: float = 1.0,
                                         include_repulsion: bool = False
                                         ) -> torch.Tensor:
    """Single-bead variant of :func:`score_distances_active_region`.

    Mirror of ``calcScoreDistancesActiveRegion(int cluster_moved)``
    (cpp:1954-1984).  cudaMMC's single-bead variant **comments out** the
    repulsion contribution (cpp:1967-1969); set ``include_repulsion=True``
    only if you intentionally want to deviate.  The spring branch is
    identical to the full version.
    """
    N = pos.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=pos.device)
    j = torch.arange(N, device=pos.device)
    mask = j != idx
    d = single_bead_distances(pos, idx)[mask]                # cpp:1964
    e = expected[idx].float()[mask]
    # cudaMMC indexes heatmap_exp_dist_anchor.v[i][cluster_moved] (cpp:1965),
    # which is symmetric (built that way at cpp:3902-3903), so either row or
    # column lookup is identical.

    repulse = e < 0                                          # cpp:1965-1969
    skip = (~repulse) & (e < 1e-6)                           # cpp:1971
    spring = ~repulse & ~skip

    score = torch.zeros((), device=pos.device)
    if include_repulsion and repulse.any():
        score = score + (k_repulsion / d[repulse].clamp(min=1e-6)).sum()
    if spring.any():
        score = score + _spring_term(d[spring], e[spring], k_stretch, k_squeeze)
    return score


# ── Legacy sparse-arc variants (still used by the Phase-4-pending mc.py) ────

def score_arcs(pos: torch.Tensor,
               arc_starts: torch.Tensor,
               arc_ends: torch.Tensor,
               arc_expected: torch.Tensor,
               k: float = 1.0,
               k_repulsion: float = 1.0) -> torch.Tensor:
    """Spring score over an explicit arc list (legacy sparse API).

    Same per-pair formula as :func:`score_distances_active_region` but only
    visits the supplied arc pairs — *does not* see the −1 sentinel for
    non-arc pairs.  Phase 5 swaps callers to the dense variant.
    """
    if arc_starts.numel() == 0:
        return torch.zeros((), device=pos.device)
    d = (pos[arc_starts] - pos[arc_ends]).norm(dim=1)
    e = arc_expected.float()
    spring = e > 1e-6
    repulse = e < 0
    score = torch.zeros((), device=pos.device)
    if spring.any():
        score = score + _spring_term(d[spring], e[spring], k, k)
    if repulse.any():
        score = score + (k_repulsion / d[repulse].clamp(min=1e-6)).sum()
    return score


def score_arcs_single(pos: torch.Tensor,
                       idx: int,
                       arc_starts: torch.Tensor,
                       arc_ends: torch.Tensor,
                       arc_expected: torch.Tensor,
                       k: float = 1.0,
                       k_repulsion: float = 1.0) -> torch.Tensor:
    """Single-bead variant of :func:`score_arcs` (legacy sparse API)."""
    mask = (arc_starts == idx) | (arc_ends == idx)
    if not mask.any():
        return torch.zeros((), device=pos.device)
    s = arc_starts[mask]
    e_idx = arc_ends[mask]
    exp = arc_expected[mask].float()
    d = (pos[s] - pos[e_idx]).norm(dim=1)
    spring = exp > 1e-6
    repulse = exp < 0
    score = torch.zeros((), device=pos.device)
    if spring.any():
        score = score + _spring_term(d[spring], exp[spring], k, k)
    if repulse.any():
        # cudaMMC single-bead repulsion is commented out (cpp:1967-1969); we
        # mirror that by *not* adding it here unless the caller opts in.
        pass
    return score


# =============================================================================
# Structural / smooth score — calcScoreStructureSmooth  (cpp:2026-2107)
# =============================================================================
# weightDistSmooth / weightAngleSmooth from Settings.cpp:240-241; loadFromINI
# at cpp:594-597 reads them with the keys SWAPPED (preserved in Settings).
# Single-bead variant returns sca+scb with no weights (cpp:2106).
#
# Angular formula: cudaMMC common.cpp:42-52  angle() = (1 - dot)/2   NOT acos.

def score_structure_smooth(pos: torch.Tensor,
                            chain_lengths: torch.Tensor,
                            k_stretch: float = 0.1,
                            k_squeeze: float = 0.1,
                            angular_k: float = 0.1,
                            weight_dist: float = 1.0,
                            weight_angle: float = 1.0,
                            *,
                            k_chain: Optional[float] = None) -> torch.Tensor:
    """Chain + angular smooth score (cpp:2026-2063).

    ``k_chain`` is a back-compat alias used by the legacy mc.py (it sets both
    stretch and squeeze to the same value, matching cudaMMC defaults of
    0.1/0.1).  New callers should pass ``k_stretch`` / ``k_squeeze`` from
    ``Settings.spring_constant_stretch`` / ``_squeeze``.
    """
    if k_chain is not None:
        k_stretch = k_squeeze = k_chain
    if pos.shape[0] < 2:
        return torch.zeros((), device=pos.device)

    diffs = pos[1:] - pos[:-1]                              # cpp:2034
    dist = diffs.norm(dim=1)

    sca = torch.zeros((), device=pos.device)
    valid_link = chain_lengths >= 1e-9
    if valid_link.any():
        L = chain_lengths[valid_link].clamp(min=1e-6)       # cpp:2037-2038
        sca = _spring_term(dist[valid_link], L, k_stretch, k_squeeze)  # cpp:2040-2043

    scb = torch.zeros((), device=pos.device)
    if pos.shape[0] >= 3:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        cos_a = F.cosine_similarity(v1, v2, dim=1).clamp(-1.0, 1.0)
        ang = (1.0 - cos_a) / 2.0                           # common.cpp:44
        scb = angular_k * (ang ** 3).sum()                  # cpp:2048

    # cpp:2062: return sca * weightDistSmooth + scb * weightAngleSmooth
    return sca * weight_dist + scb * weight_angle


def score_chain_single(pos: torch.Tensor,
                       idx: int,
                       chain_lengths: torch.Tensor,
                       k_stretch: float = 0.1,
                       k_squeeze: float = 0.1,
                       angular_k: float = 0.1,
                       *,
                       k_chain: Optional[float] = None) -> torch.Tensor:
    """Single-bead chain + angular contribution (cpp:2065-2107).

    cpp:2106 returns ``sca + scb`` with **no** weight multipliers.  Moving
    bead ``idx`` only changes the two links touching it and the three
    junction angles at ``idx-1, idx, idx+1``.
    """
    if k_chain is not None:
        k_stretch = k_squeeze = k_chain
    N = pos.shape[0]
    device = pos.device
    score = torch.zeros((), device=device)

    # cpp:2072-2074: i ∈ [moved-2, moved+1]; valid links at i=moved-1, moved
    for i in (idx - 1, idx):                                # cpp:2080
        if 0 <= i < N - 1 and chain_lengths[i] >= 1e-9:
            L = chain_lengths[i].clamp(min=1e-6)            # cpp:2082-2083
            d = (pos[i + 1] - pos[i]).norm()
            diff = (d - L) / L                              # cpp:2084
            coef = k_stretch if diff.item() >= 0 else k_squeeze
            score = score + coef * diff * diff              # cpp:2085-2087

    # cpp:2091: junctions touching idx — angles at idx-1, idx, idx+1
    for j in (idx - 1, idx, idx + 1):
        if 0 < j < N - 1:
            v1 = pos[j] - pos[j - 1]
            v2 = pos[j + 1] - pos[j]
            cos_a = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).clamp(-1.0, 1.0)
            ang = (1.0 - cos_a) / 2.0                       # common.cpp:44
            score = score + angular_k * (ang ** 3)          # cpp:2092-2093

    # cpp:2106: return sca + scb  (no weights)
    return score.squeeze()


# =============================================================================
# Orientation score — calcScoreOrientation  (cpp:2109-2162)
# =============================================================================
# cudaMMC stores `orientation[i]` as a unit 3-vector that is **recomputed
# every bead move** from the current positions:
#     orn[i] = normalize(p[i+1] - p[i-1])   (boundaries: forward/backward diff)
#     orn[i] *= -1   if  label[i] == 'L'    (calcOrientation, cpp:3437-3454)
#
# Then `angle_norm(o1, o2) = (1 - dot)/2`  on those unit vectors
# (common.cpp:48-52).  Gated by `Settings::useCTCFMotifOrientation` via the
# caller — when disabled, score is 0 regardless.

def _calc_orientation_vectors(pos: torch.Tensor,
                              labels: List[str]) -> torch.Tensor:
    """Recompute per-bead unit orientation vectors from current positions.

    Mirror of cudaMMC ``calcOrientation(int cind)`` (LooperSolver.cpp:3437-3454)
    applied to every active-region bead.  Returns ``(N, 3)`` unit fp32 tensor.
    A label of ``'N'`` keeps the geometric tangent unchanged; ``'L'`` flips it.
    """
    N = pos.shape[0]
    if N == 0:
        return torch.zeros(0, 3, device=pos.device)
    orn = torch.zeros_like(pos)
    if N == 1:
        return orn
    # Interior beads: orn[i] = p[i+1] - p[i-1]                  (cpp:3445-3447)
    orn[1:-1] = pos[2:] - pos[:-2]
    # Endpoints: forward / backward differences                  (cpp:3439-3443)
    orn[0] = pos[1] - pos[0]
    orn[-1] = pos[-1] - pos[-2]
    # Flip 'L' anchors                                            (cpp:3449-3450)
    if labels:
        flip = torch.tensor([1.0 if l != 'L' else -1.0 for l in labels],
                            dtype=pos.dtype, device=pos.device).unsqueeze(1)
        orn = orn * flip
    # Normalise                                                   (cpp:3452)
    norm = orn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return orn / norm


def score_orientation(pos: torch.Tensor,
                      orientations: List[str],
                      arc_starts: torch.Tensor,
                      arc_ends: torch.Tensor,
                      weight: float = 1.0,
                      motifs_symmetric: bool = True,
                      use_ctcf: bool = True) -> torch.Tensor:
    """CTCF orientation penalty over every arc (cpp:2109-2131).

    Mirrors the production formula::

        ang = angle_norm( orn[i],  (symmetric ? +1 : -1) * orn[j] )
        err += ang² * weight_ij
        return err * motifOrientationWeight

    ``orientations`` is the per-bead BED-strand label list ('R' / 'L' / 'N');
    geometric unit vectors are derived from the current ``pos``.

    Returns 0 when ``use_ctcf`` is False (gated by
    ``Settings.use_ctcf_motif_orientation``).
    """
    if not use_ctcf or arc_starts.numel() == 0:
        return torch.zeros((), device=pos.device)
    orn = _calc_orientation_vectors(pos, orientations)       # cpp:3437-3454
    # cpp:2114-2116: ang = angle_norm(o1, sign * o2)
    o1 = orn[arc_starts]
    o2 = orn[arc_ends]
    if not motifs_symmetric:
        o2 = -o2
    # common.cpp:48-52  angle_norm(a, b) = 1 - (dot + 1)/2 = (1 - dot)/2
    dot = (o1 * o2).sum(dim=1).clamp(-1.0, 1.0)
    ang = (1.0 - dot) / 2.0
    # cpp:2117: err += ang² * weight_ij  (per-arc weight defaults to 1)
    err = (ang * ang).sum()
    return weight * err                                       # cpp:2131


def score_orientation_single(pos: torch.Tensor,
                              idx: int,
                              orientations: List[str],
                              arc_starts: torch.Tensor,
                              arc_ends: torch.Tensor,
                              weight: float = 1.0,
                              motifs_symmetric: bool = True,
                              use_ctcf: bool = True) -> torch.Tensor:
    """Single-bead orientation contribution (cpp:2134-2162).

    cudaMMC iterates ``active_anchors_neighbors[idx]`` (the arcs touching
    bead ``idx``).  We mirror that exactly.
    """
    if not use_ctcf:
        return torch.zeros((), device=pos.device)
    mask = (arc_starts == idx) | (arc_ends == idx)
    if not mask.any():
        return torch.zeros((), device=pos.device)
    # Re-derive geometric orientation from current pos.  Moving bead `idx`
    # changes orn[idx-1], orn[idx], orn[idx+1] — but cudaMMC just recomputes
    # the whole vector and queries the affected indices, so we do too.
    orn = _calc_orientation_vectors(pos, orientations)
    s = arc_starts[mask]
    e = arc_ends[mask]
    o1 = orn[s]
    o2 = orn[e]
    if not motifs_symmetric:
        o2 = -o2
    dot = (o1 * o2).sum(dim=1).clamp(-1.0, 1.0)
    ang = (1.0 - dot) / 2.0
    err = (ang * ang).sum()
    return weight * err                                       # cpp:2161


# =============================================================================
# Subanchor heatmap score — stub gated by Settings.use_subanchor_heatmap
# =============================================================================
# cudaMMC `calcScoreHeatmapSubanchor` (cpp:2230-2269) is a separate heatmap
# evaluated only on subanchor beads inserted by ``densifyActiveRegion``.
# Without the densification step (Phase 5 item 19) there is nothing to score
# against, so we expose a stub that raises if the flag is enabled — keeping
# the "do X or fail loudly" guarantee (AUDIT §E3, §F7, §G6).

def score_subanchor_heatmap(pos: torch.Tensor,
                            expected: torch.Tensor,
                            diagonal_size: int = 1,
                            *,
                            enabled: bool = False) -> torch.Tensor:
    """Stub for ``calcScoreHeatmapSubanchor`` (cpp:2230-2269).

    Phase-5 densification (`densifyActiveRegion`, cpp:2645) must land before
    this score can produce a meaningful value.  Until then we explicitly
    refuse to silently return 0 when the user enables the upstream flag.
    """
    if not enabled:
        return torch.zeros((), device=pos.device)
    raise NotImplementedError(
        "Settings.use_subanchor_heatmap is set but the subanchor "
        "densification pipeline (cudaMMC LooperSolver.cpp:2645 "
        "densifyActiveRegion) is not implemented yet — Phase 5 work."
    )
