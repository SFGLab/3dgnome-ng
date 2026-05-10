"""
Monte Carlo optimisation phases.

Phase 1 - GPU heatmap MC  (ParallelMonteCarloHeatmap equivalent)
Phase 2 - CPU arcs MC     (MonteCarloArcs equivalent)
Phase 3 - CPU smooth MC   (MonteCarloArcsSmooth equivalent)
"""

import math
import random
from typing import List, Optional, Tuple

import torch

from .scores import (
    score_heatmap,
    score_heatmap_single,
    score_arcs,
    score_arcs_single,
    score_structure_smooth,
    score_orientation,
    score_orientation_single,
)
from .settings import Settings


# ── Random displacement helpers ───────────────────────────────────────────────

def _random_displacement(step_size: float, use_2d: bool,
                          device: torch.device) -> torch.Tensor:
    """Sample a random unit vector scaled by step_size."""
    v = torch.randn(3, device=device)
    if use_2d:
        v[2] = 0.0
    norm = v.norm().clamp(min=1e-9)
    return (v / norm) * step_size


def _with_chance(jump_scale: float, temp: float, delta: float,
                  rng: random.Random) -> bool:
    """Metropolis acceptance: accept worse moves with probability jump_scale * exp(-delta/T)."""
    prob = jump_scale * math.exp(-delta / max(temp, 1e-10))
    return rng.random() < prob


# ── Phase 1: GPU heatmap MC ───────────────────────────────────────────────────

def monte_carlo_heatmap(
    pos: torch.Tensor,          # (N, 3) float32, on GPU
    expected: torch.Tensor,     # (N, N) expected distances, on GPU
    fixed_mask: torch.Tensor,   # (N,) bool, True = don't move
    settings: Settings,
    same_chr_mask: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    GPU-accelerated heatmap Monte Carlo.

    Mimics the CUDA kernel: each bead is moved in turn (serial over beads,
    but position storage stays on GPU for vectorised score computation).

    Returns updated pos tensor.
    """
    s = settings
    device = pos.device

    T = s.max_temp_heatmap
    step = s.step_size_heatmap
    N = pos.shape[0]
    rng = random.Random()

    milestone_fails = 0
    best_score = score_heatmap(pos, expected, s.diagonal_size, same_chr_mask).item()

    outer_step = 0
    while milestone_fails < s.milestone_fails_threshold:
        outer_step += 1

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            # N inner steps per bead (mirrors CUDA N=512 inner loop)
            score_prev = score_heatmap_single(pos, bead_idx, expected,
                                              s.diagonal_size, same_chr_mask).item()

            for _ in range(s.mc_inner_steps):
                disp = _random_displacement(step, s.use_2d, device)
                pos[bead_idx] += disp

                score_curr = score_heatmap_single(pos, bead_idx, expected,
                                                   s.diagonal_size, same_chr_mask).item()
                delta = score_curr - score_prev
                if delta <= 0:
                    score_prev = score_curr
                elif not _with_chance(s.temp_jump_scale_heatmap,
                                      T * s.temp_jump_coef_heatmap,
                                      delta, rng):
                    pos[bead_idx] -= disp
                else:
                    score_prev = score_curr

        # milestone check
        T *= s.dt_temp_heatmap
        step *= s.step_size_decay_heatmap

        total = score_heatmap(pos, expected, s.diagonal_size, same_chr_mask).item()
        if verbose and outer_step % 100 == 0:
            print(f"  [heatmap MC] step={outer_step:6d}  T={T:.4f}  "
                  f"score={total:.6f}  milestone_fails={milestone_fails}")

        if total < best_score - 1e-4:
            best_score = total
            milestone_fails = 0
        else:
            milestone_fails += 1

        if total < 1e-4:
            break

    return pos


# ── Phase 2: CPU arcs MC ──────────────────────────────────────────────────────

def monte_carlo_arcs(
    pos: torch.Tensor,           # (N, 3) float32
    arc_starts: torch.Tensor,    # (M,) long
    arc_ends: torch.Tensor,      # (M,) long
    arc_expected: torch.Tensor,  # (M,) float  (>0 spring, <0 repulsion)
    fixed_mask: torch.Tensor,    # (N,) bool
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Incremental single-bead MC for arc spring energy.
    Uses Metropolis criterion.  Stops when improvement per outer step
    drops below threshold AND successes < min_successes.
    """
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    T = s.max_temp_arcs
    step = s.step_size_arcs

    total_score = score_arcs(pos, arc_starts, arc_ends, arc_expected,
                              s.k_spring, s.k_spring_repulsion).item()
    prev_milestone = total_score
    outer_step = 0

    while True:
        outer_step += 1
        successes = 0
        improvement = 0.0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            local_prev = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                           arc_expected, s.k_spring,
                                           s.k_spring_repulsion).item()

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            local_curr = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                           arc_expected, s.k_spring,
                                           s.k_spring_repulsion).item()
            delta = local_curr - local_prev
            if delta <= 0:
                total_score += delta
                successes += 1
                improvement -= delta
            elif _with_chance(s.temp_jump_scale_arcs,
                              T * s.temp_jump_coef_arcs,
                              delta, rng):
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

        if milestone_improvement < s.improvement_threshold_arcs and \
                successes < s.min_successes_arcs:
            break

    return pos


# ── Phase 3: CPU smooth MC ────────────────────────────────────────────────────

def monte_carlo_arcs_smooth(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,    # (N-1,) expected linker lengths
    orientations: List[str],
    fixed_mask: torch.Tensor,
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Combined smooth MC: structure (chain+angle) + orientation + heatmap-style arcs.

    Mimics MonteCarloArcsSmooth.  The pairwise arc and orientation terms
    contribute with a factor-2 scale in the incremental update (matching the
    C++ implementation that counts pairs from both endpoints).
    """
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    T = s.max_temp_smooth
    step = s.step_size_smooth

    def total_score():
        sc = score_structure_smooth(pos, chain_lengths,
                                    s.k_chain, s.angular_k)
        sc = sc + score_arcs(pos, arc_starts, arc_ends, arc_expected,
                              s.k_spring, s.k_spring_repulsion)
        sc = sc + score_orientation(pos, orientations, arc_starts, arc_ends,
                                    s.k_orient)
        return sc.item()

    ts = total_score()
    prev_milestone = ts
    outer_step = 0

    while True:
        outer_step += 1
        successes = 0
        improvement = 0.0

        for bead_idx in range(N):
            if fixed_mask[bead_idx]:
                continue

            # structural score is computed globally (chain terms span neighbours)
            struct_before = score_structure_smooth(pos, chain_lengths,
                                                   s.k_chain, s.angular_k).item()
            arc_before = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                           arc_expected, s.k_spring,
                                           s.k_spring_repulsion).item()
            ori_before = score_orientation_single(pos, bead_idx, orientations,
                                                  arc_starts, arc_ends,
                                                  s.k_orient).item()

            disp = _random_displacement(step, s.use_2d, device)
            pos[bead_idx] += disp

            struct_after = score_structure_smooth(pos, chain_lengths,
                                                  s.k_chain, s.angular_k).item()
            arc_after = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                          arc_expected, s.k_spring,
                                          s.k_spring_repulsion).item()
            ori_after = score_orientation_single(pos, bead_idx, orientations,
                                                 arc_starts, arc_ends,
                                                 s.k_orient).item()

            # pairwise terms are counted from both endpoints → factor 2
            delta = ((struct_after - struct_before)
                     + 2.0 * (arc_after - arc_before)
                     + 2.0 * (ori_after - ori_before))

            if delta <= 0:
                ts += delta
                successes += 1
                improvement -= delta
            elif _with_chance(s.temp_jump_scale_smooth,
                              T * s.temp_jump_coef_smooth,
                              delta, rng):
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

        if milestone_improvement < s.improvement_threshold_smooth and \
                successes < s.min_successes_smooth:
            break

    return pos
