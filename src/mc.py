"""
Monte Carlo optimisation phases.

Phase 1 - GPU heatmap MC  (ParallelMonteCarloHeatmap equivalent)
Phase 2 - CPU arcs MC     (MonteCarloArcs equivalent)
Phase 3 - CPU smooth MC   (MonteCarloArcsSmooth equivalent)
"""

import math
import random
from typing import List, Optional

import torch

from .scores import (
    score_heatmap_chunked,
    score_arcs,
    score_arcs_single,
    score_chain_single,
    score_structure_smooth,
    score_orientation,
    score_orientation_single,
)
from .settings import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_displacement(step_size: float, use_2d: bool,
                          device: torch.device) -> torch.Tensor:
    v = torch.randn(3, device=device)
    if use_2d:
        v[2] = 0.0
    return v * (step_size / v.norm().clamp(min=1e-9))


def _with_chance(jump_scale: float, temp: float, delta: float,
                  rng: random.Random) -> bool:
    prob = jump_scale * math.exp(-delta / max(temp, 1e-10))
    return rng.random() < prob


# ── Per-bead score delta (vectorised over all beads simultaneously) ───────────

def _bead_score_delta(
    pos: torch.Tensor,       # (N, 3) current positions
    new_pos: torch.Tensor,   # (N, 3) proposed positions
    expected: torch.Tensor,  # (N, N) fp16 or fp32
    diagonal_size: int,
    same_chr_mask: Optional[torch.Tensor],
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Return (N,) tensor: delta[i] = S_new[i] - S_old[i]

    S[i] = sum_{j: j≠i, |i-j|>=diag, exp>1e-3} (dist(pos_i, pos_j)/exp[i,j] - 1)^2

    Jacobi: S_new[i] uses new_pos[i] vs the CURRENT pos[j] for all j.
    Uses the squared-norm matmul trick: largest intermediate is (chunk, N) ~ 50 MB.
    """
    N = pos.shape[0]
    device = pos.device
    delta = torch.zeros(N, device=device)

    pos_sq = (pos ** 2).sum(dim=1)       # (N,) precomputed
    new_sq = (new_pos ** 2).sum(dim=1)   # (N,) precomputed

    for i0 in range(0, N, chunk_size):
        i1 = min(i0 + chunk_size, N)

        p_old = pos[i0:i1]             # (C, 3)
        p_new = new_pos[i0:i1]         # (C, 3)
        sq_o = pos_sq[i0:i1]           # (C,)
        sq_n = new_sq[i0:i1]           # (C,)

        dot_o = p_old @ pos.t()        # (C, N)
        d_old = (sq_o[:, None] + pos_sq[None, :] - 2 * dot_o).clamp(0).sqrt()

        dot_n = p_new @ pos.t()        # (C, N)
        d_new = (sq_n[:, None] + pos_sq[None, :] - 2 * dot_n).clamp(0).sqrt()

        exp_c = expected[i0:i1].float()   # (C, N)

        i_idx = torch.arange(i0, i1, device=device)[:, None]   # (C, 1)
        j_idx = torch.arange(N, device=device)[None, :]         # (1, N)

        valid = ((i_idx != j_idx)
                 & ((i_idx - j_idx).abs() >= diagonal_size)
                 & (exp_c > 1e-3))
        if same_chr_mask is not None:
            valid = valid & same_chr_mask[i0:i1]

        safe_e = exp_c.clamp(min=1e-9)
        r_old = torch.where(valid, d_old / safe_e - 1.0, torch.zeros_like(d_old))
        r_new = torch.where(valid, d_new / safe_e - 1.0, torch.zeros_like(d_new))

        delta[i0:i1] = (r_new ** 2 - r_old ** 2).sum(dim=1)

    return delta


# ── Phase 1: vectorised GPU heatmap MC ───────────────────────────────────────

def monte_carlo_heatmap(
    pos: torch.Tensor,          # (N, 3) float32, on GPU
    expected: torch.Tensor,     # (N, N) expected distances (float16 ok), on GPU
    fixed_mask: torch.Tensor,   # (N,) bool
    settings: Settings,
    same_chr_mask: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Vectorised Jacobi heatmap MC – mirrors the CUDA warp-parallel kernel.

    Each inner step proposes a displacement for ALL beads simultaneously,
    computes per-bead score deltas in one chunked GPU pass (no Python loop
    over beads, no N×N allocation), and accepts/rejects per-bead with a
    vectorised Metropolis criterion.

    Outer step = mc_inner_steps Jacobi rounds, followed by temperature
    cooling and a milestone check.
    """
    s = settings
    device = pos.device
    N = pos.shape[0]

    T = s.max_temp_heatmap
    step = s.step_size_heatmap
    free = ~fixed_mask                  # (N,) mask of movable beads

    total_score = score_heatmap_chunked(pos, expected,
                                         s.diagonal_size, same_chr_mask).item()
    best_score = total_score
    milestone_fails = 0
    outer_step = 0

    # effective_inner = min(s.mc_inner_steps, max(1, (512 * 512) // max(N, 1)))
    # if verbose and effective_inner < s.mc_inner_steps:
    #     print(f"  [heatmap MC] auto-scaling mc_inner_steps "
    #           f"{s.mc_inner_steps} → {effective_inner} (N={N})")
    effective_inner = s.mc_inner_steps

    # Don't check milestones until T drops to ≤10% of initial — at high T,
    # score fluctuates randomly and milestone_fails would trigger after ~3 steps.
    milestone_temp_threshold = T * 0.1

    if verbose:
        print(f"  [heatmap MC] N={N}  T_initial={T:.1f}  "
              f"milestone starts at T={milestone_temp_threshold:.2f}")

    while milestone_fails < s.milestone_fails_threshold:
        outer_step += 1

        for _ in range(effective_inner):
            # propose one displacement for every free bead
            disp = torch.zeros(N, 3, device=device)
            if s.use_2d:
                disp[free, :2] = torch.randn(free.sum(), 2, device=device) * step
            else:
                raw = torch.randn(free.sum(), 3, device=device)
                disp[free] = raw * (step / raw.norm(dim=1, keepdim=True).clamp(min=1e-9))

            new_pos = pos + disp   # (N, 3); zeros for fixed beads → no change

            # per-bead score delta, chunked to avoid N×N
            delta = _bead_score_delta(pos, new_pos, expected,
                                       s.diagonal_size, same_chr_mask)

            # vectorised Metropolis
            T_eff = max(T * s.temp_jump_coef_heatmap, 1e-10)
            log_thresh = (math.log(max(s.temp_jump_scale_heatmap, 1e-30))
                          - delta / T_eff)
            rand_log = torch.rand(N, device=device).log()
            accept = free & ((delta <= 0) | (rand_log < log_thresh))

            # in-place update (accepted beads only)
            pos[accept] = new_pos[accept]
            total_score += delta[accept].sum().item()

        T *= s.dt_temp_heatmap
        step *= s.step_size_decay_heatmap

        # Recompute true score — Jacobi delta accumulation drifts from reality
        # because deltas are computed against stale neighbour positions.
        total_score = score_heatmap_chunked(pos, expected,
                                             s.diagonal_size, same_chr_mask).item()

        annealing = T <= milestone_temp_threshold
        log_interval = 10 if annealing else 100
        if verbose and outer_step % log_interval == 0:
            phase = "anneal" if annealing else "explore"
            print(f"  [heatmap MC] step={outer_step:5d}  T={T:.4f}  "
                  f"score={total_score:.4f}  fails={milestone_fails}  [{phase}]")

        # Only track milestones once temperature is low enough for convergence
        # to be meaningful — at high T, score fluctuates randomly.
        if T <= milestone_temp_threshold:
            if total_score < best_score - 1e-4:
                best_score = total_score
                milestone_fails = 0
            else:
                milestone_fails += 1
        else:
            best_score = min(best_score, total_score)

        if total_score < 1e-4:
            break

    return pos


# ── Phase 2: arcs MC (GPU tensors, CPU control loop) ─────────────────────────
# Original cudaMMC: fully CPU.  We keep the same sequential Gauss-Seidel design
# but tensor math runs on pos.device (GPU).  The per-bead Python loop and
# .item() syncs are unavoidable without a full GPU kernel rewrite.

def monte_carlo_arcs(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,
    fixed_mask: torch.Tensor,
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    """Incremental single-bead MC for arc spring energy + chain springs."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    T = s.max_temp_arcs
    step = s.step_size_arcs

    total_score = (score_arcs(pos, arc_starts, arc_ends, arc_expected,
                               s.k_spring, s.k_spring_repulsion)
                   + score_structure_smooth(pos, chain_lengths, s.k_chain, s.angular_k)).item()
    prev_milestone = total_score
    outer_step = 0

    while True:
        outer_step += 1
        successes = 0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            local_prev = (score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                            arc_expected, s.k_spring, s.k_spring_repulsion)
                          + score_chain_single(pos, bead_idx, chain_lengths,
                                               s.k_chain, s.angular_k)).item()

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            local_curr = (score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                            arc_expected, s.k_spring, s.k_spring_repulsion)
                          + score_chain_single(pos, bead_idx, chain_lengths,
                                               s.k_chain, s.angular_k)).item()
            delta = local_curr - local_prev
            if delta < -1e-7:
                total_score += delta
                successes += 1
            elif delta <= 0:
                total_score += delta
            elif _with_chance(s.temp_jump_scale_arcs,
                              T * s.temp_jump_coef_arcs, delta, rng):
                total_score += delta
            else:
                pos[bead_idx] -= disp

        T *= s.dt_temp_arcs
        step *= s.step_size_decay_arcs

        milestone_improvement = prev_milestone - total_score
        prev_milestone = total_score

        if verbose and outer_step % 200 == 0:
            print(f"  [arcs MC] step={outer_step:6d}  T={T:.5f}  "
                  f"score={total_score:.6f}  imp={milestone_improvement:.6f}")

        if (milestone_improvement < s.improvement_threshold_arcs
                and successes < s.min_successes_arcs):
            break

    return pos


# ── Phase 3: smooth MC (GPU tensors, CPU control loop — same as Phase 2) ─────

def monte_carlo_arcs_smooth(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,
    orientations: List[str],
    fixed_mask: torch.Tensor,
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    """Combined smooth MC: structure (chain+angle) + orientation + arc springs."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    T = s.max_temp_smooth
    step = s.step_size_smooth

    def total_score():
        return (score_structure_smooth(pos, chain_lengths, s.k_chain, s.angular_k)
                + score_arcs(pos, arc_starts, arc_ends, arc_expected,
                              s.k_spring, s.k_spring_repulsion)
                + score_orientation(pos, orientations, arc_starts, arc_ends,
                                    s.k_orient)).item()

    ts = total_score()
    prev_milestone = ts
    outer_step = 0

    while True:
        outer_step += 1
        successes = 0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            struct_b = score_chain_single(pos, bead_idx, chain_lengths,
                                          s.k_chain, s.angular_k).item()
            arc_b = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                      arc_expected, s.k_spring,
                                      s.k_spring_repulsion).item()
            ori_b = score_orientation_single(pos, bead_idx, orientations,
                                             arc_starts, arc_ends,
                                             s.k_orient).item()

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            struct_a = score_chain_single(pos, bead_idx, chain_lengths,
                                          s.k_chain, s.angular_k).item()
            arc_a = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                      arc_expected, s.k_spring,
                                      s.k_spring_repulsion).item()
            ori_a = score_orientation_single(pos, bead_idx, orientations,
                                             arc_starts, arc_ends,
                                             s.k_orient).item()

            # pairwise terms counted from both endpoints → factor 2
            delta = ((struct_a - struct_b)
                     + 2.0 * (arc_a - arc_b)
                     + 2.0 * (ori_a - ori_b))

            if delta < -1e-7:
                ts += delta
                successes += 1
            elif delta <= 0:
                ts += delta
            elif _with_chance(s.temp_jump_scale_smooth,
                              T * s.temp_jump_coef_smooth, delta, rng):
                ts += delta
            else:
                pos[bead_idx] -= disp

        T *= s.dt_temp_smooth
        step *= s.step_size_decay_smooth

        milestone_improvement = prev_milestone - ts
        prev_milestone = ts

        if verbose and outer_step % 200 == 0:
            print(f"  [smooth MC] step={outer_step:6d}  T={T:.5f}  "
                  f"score={ts:.6f}  imp={milestone_improvement:.6f}")

        if (milestone_improvement < s.improvement_threshold_smooth
                and successes < s.min_successes_smooth):
            break

    return pos
