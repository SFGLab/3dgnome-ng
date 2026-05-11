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
    # cudaMMC common.cpp random_vector: uniform[-step, step] per component
    v = (torch.rand(3, device=device) * 2.0 - 1.0) * step_size
    if use_2d:
        v[2] = 0.0
    return v


def _with_chance_ratio(jump_scale: float, jump_coef: float,
                        score_curr: float, score_prev: float,
                        T: float, rng: random.Random) -> bool:
    # cudaMMC LooperSolver.cpp line 3114:
    # tp = tempJumpScale * exp(-tempJumpCoef * (score_curr/score_prev) / T)
    prob = jump_scale * math.exp(
        -jump_coef * (score_curr / max(score_prev, 1e-30)) / max(T, 1e-30))
    return rng.random() < prob


# ── Per-bead score delta (vectorised over all beads simultaneously) ───────────

def _bead_score_delta(
    pos: torch.Tensor,       # (N, 3) current positions
    new_pos: torch.Tensor,   # (N, 3) proposed positions
    expected: torch.Tensor,  # (N, N) fp16 or fp32
    diagonal_size: int,
    same_chr_mask: Optional[torch.Tensor],
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (delta, score_old) each (N,):
      delta[i]     = S_new[i] - S_old[i]
      score_old[i] = S_old[i]  (per-bead score before move)

    S[i] = sum_{j: j≠i, |i-j|>=diag, exp>1e-3} (dist(pos_i, pos_j)/exp[i,j] - 1)^2

    Jacobi: S_new[i] uses new_pos[i] vs the CURRENT pos[j] for all j.
    Both outputs are needed for the ratio-based Metropolis criterion.
    """
    N = pos.shape[0]
    device = pos.device
    delta = torch.zeros(N, device=device)
    score_old = torch.zeros(N, device=device)

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

        score_old[i0:i1] = (r_old ** 2).sum(dim=1)
        delta[i0:i1] = (r_new ** 2 - r_old ** 2).sum(dim=1)

    return delta, score_old


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
            # cudaMMC: uniform[-step, step] per component (common.cpp random_vector)
            disp = torch.zeros(N, 3, device=device)
            disp[free] = (torch.rand(free.sum(), 3, device=device) * 2.0 - 1.0) * step
            if s.use_2d:
                disp[:, 2] = 0.0

            new_pos = pos + disp   # (N, 3); zeros for fixed beads → no change

            # per-bead score delta and per-bead old score, chunked to avoid N×N
            delta, score_old = _bead_score_delta(pos, new_pos, expected,
                                                  s.diagonal_size, same_chr_mask)

            # ratio-based Metropolis (cudaMMC ParallelMonteCarloHeatmap.cu):
            # withChance(scale * expf(-coef * score_new/score_old / T))
            score_new = score_old + delta                            # (N,) per-bead
            T_safe = max(T, 1e-30)
            log_thresh = (math.log(max(s.temp_jump_scale_heatmap, 1e-30))
                          - s.temp_jump_coef_heatmap
                            * (score_new / score_old.clamp(min=1e-30)) / T_safe)
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

        # Milestone: ratio-based criterion matching cudaMMC
        # (LooperSolver.cpp line 3143: score_curr > 0.995*milestone_score)
        if T <= milestone_temp_threshold:
            if total_score < s.milestone_improvement_ratio * best_score:
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

    # cudaMMC MonteCarloArcs: arc springs ONLY (LooperSolver.cpp:3086)
    # calcScoreDistancesActiveRegion() — no chain springs in arcs phase
    total_score = score_arcs(pos, arc_starts, arc_ends, arc_expected,
                              s.k_spring, s.k_spring_repulsion).item()
    milestone_score = total_score
    outer_step = 0
    milestone_success = 0
    individual_steps = 0

    while True:
        outer_step += 1
        successes = 0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            # cudaMMC: calcScoreDistancesActiveRegion(p) — arc springs only
            local_prev = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                           arc_expected, s.k_spring,
                                           s.k_spring_repulsion).item()
            score_prev = total_score

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            local_curr = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                           arc_expected, s.k_spring,
                                           s.k_spring_repulsion).item()

            # cudaMMC: score_curr = score_curr - local_prev + local_curr
            delta = local_curr - local_prev
            score_curr = total_score + delta

            # ratio-based Metropolis (LooperSolver.cpp line 3111–3118)
            if score_curr <= score_prev:
                total_score = score_curr
                successes += 1
                milestone_success += 1
            elif _with_chance_ratio(s.temp_jump_scale_arcs, s.temp_jump_coef_arcs,
                                    score_curr, score_prev, T, rng):
                total_score = score_curr
                milestone_success += 1
            else:
                pos[bead_idx] -= disp
                # total_score unchanged

            # cudaMMC: T *= dt every individual bead move (line 3130)
            T *= s.dt_temp_arcs
            individual_steps += 1

            # milestone check every MCstopConditionSteps individual moves
            if individual_steps % s.milestone_steps_arcs == 0:
                ratio = total_score / max(milestone_score, 1e-30)
                if verbose:
                    print(f"  [arcs MC] step={individual_steps:7d}  T={T:.5f}  "
                          f"score={total_score:.6f}  ratio={ratio:.5f}  "
                          f"ms_succ={milestone_success}")
                if ((total_score > s.milestone_improvement_ratio * milestone_score
                        and milestone_success < s.min_successes_arcs)
                        or total_score < 1e-5
                        or ratio > 0.9999):
                    return pos
                milestone_score = total_score
                milestone_success = 0

        step *= s.step_size_decay_arcs

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
    # cudaMMC MonteCarloArcsSmooth: chain+angular+orientation (NO arc springs)
    # score = calcScoreStructureSmooth + calcScoreOrientation
    """Combined smooth MC: structure (chain+angle) + orientation."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    T = s.max_temp_smooth
    step = s.step_size_smooth

    # cudaMMC: curr_score_structure + curr_score_orientation (LooperSolver.cpp:3249)
    ts = (score_structure_smooth(pos, chain_lengths, s.k_chain, s.angular_k)
          + score_orientation(pos, orientations, arc_starts, arc_ends,
                              s.k_orient)).item()
    milestone_score = ts
    outer_step = 0
    milestone_success = 0
    individual_steps = 0

    while True:
        outer_step += 1
        successes = 0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            # cudaMMC: calcScoreStructureSmooth(p) + calcScoreOrientation(orn, idx)
            struct_b = score_chain_single(pos, bead_idx, chain_lengths,
                                          s.k_chain, s.angular_k).item()
            ori_b = score_orientation_single(pos, bead_idx, orientations,
                                             arc_starts, arc_ends,
                                             s.k_orient).item()
            score_prev = ts

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            struct_a = score_chain_single(pos, bead_idx, chain_lengths,
                                          s.k_chain, s.angular_k).item()
            ori_a = score_orientation_single(pos, bead_idx, orientations,
                                             arc_starts, arc_ends,
                                             s.k_orient).item()

            # cudaMMC: structure delta factor 1, orientation factor 2 (pairwise)
            # LooperSolver.cpp line 3322–3313: curr_structure += (new-old);
            #   curr_orientation += 2*(new_orient - old_orient)
            delta = (struct_a - struct_b) + 2.0 * (ori_a - ori_b)
            score_curr = ts + delta

            # ratio-based Metropolis (LooperSolver.cpp line 3332–3334)
            if score_curr < score_prev:
                ts = score_curr
                successes += 1
                milestone_success += 1
            elif _with_chance_ratio(s.temp_jump_scale_smooth, s.temp_jump_coef_smooth,
                                    score_curr, score_prev, T, rng):
                ts = score_curr
                milestone_success += 1
            else:
                pos[bead_idx] -= disp

            # cudaMMC: T *= dt every individual bead move (line 3382)
            T *= s.dt_temp_smooth
            individual_steps += 1

            # milestone check every MCstopConditionStepsSmooth individual moves
            if individual_steps % s.milestone_steps_smooth == 0:
                ratio = ts / max(milestone_score, 1e-30)
                if verbose:
                    print(f"  [smooth MC] step={individual_steps:7d}  T={T:.5f}  "
                          f"score={ts:.6f}  ratio={ratio:.5f}  "
                          f"ms_succ={milestone_success}")
                if ((ts > s.milestone_improvement_ratio * milestone_score
                        and milestone_success < s.min_successes_smooth)
                        or ts < 1e-6
                        or ratio > 0.9999):
                    return pos
                milestone_score = ts
                milestone_success = 0

        step *= s.step_size_decay_smooth

    return pos
