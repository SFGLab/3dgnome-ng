"""
src/mc.py  —  Monte Carlo simulation loops for 3dgnome-torch.

Mirrors C++ LooperSolver::MonteCarloHeatmap() and MonteCarloArcs().

Both loops use:
  - Sequential Metropolis steps (cannot be parallelised within a chain)
  - Uniform cube displacement: random_vector_np(step)
  - Temperature cooling: T *= dt each step
  - Milestone-based convergence: check every mc_stop_steps steps whether
    the score improved by at least (1 - threshold) and there were enough
    successful moves

Acceptance criterion (both loops):
    ok = (score_curr <= score_prev) OR rand() < jump_scale * exp(-jump_coef * score_curr/score_prev / T)
"""

import math
import random

import numpy as np

from .energy import (
    local_score_heatmap_np,
    global_score_heatmap_np,
    local_score_arcs_np,
    global_score_arcs_np,
    random_vector_np,
)


def mc_heatmap(
    pos: np.ndarray,           # (N, 3) — positions of active_region beads (modified in place)
    exp_dist: np.ndarray,      # (N, N) — expected pairwise distances
    diag_size: int,
    step_size: float,
    settings,                  # Settings object
) -> float:
    """
    MonteCarloHeatmap: simulated annealing using heatmap distance energy.

    The global score is double-counted (Σ_moved Σ_i err), so the update rule is:
        score_curr += 2 * (local_curr - local_prev)

    Mirrors C++ LooperSolver::MonteCarloHeatmap().
    Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    T = settings.max_temp_heatmap
    dt = settings.dt_temp_heatmap
    jump_scale = settings.jump_scale_heatmap
    jump_coef = settings.jump_coef_heatmap
    stop_steps = settings.mc_stop_steps_heatmap
    stop_improvement = settings.mc_stop_improvement_heatmap
    stop_successes = settings.mc_stop_successes_heatmap

    score_curr = global_score_heatmap_np(pos, exp_dist, diag_size)
    score_prev = score_curr
    milestone_score = score_curr
    milestone_success = 0
    step_i = 1

    while True:
        p = random.randrange(n)
        disp = random_vector_np(step_size)

        local_prev = local_score_heatmap_np(pos, exp_dist, diag_size, p)
        pos[p] += disp
        local_curr = local_score_heatmap_np(pos, exp_dist, diag_size, p)

        score_curr = score_curr + 2.0 * (local_curr - local_prev)

        ok = score_curr <= score_prev
        if not ok and T > 0.0:
            tp = jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)
            ok = random.random() < tp

        if ok:
            milestone_success += 1
        else:
            pos[p] -= disp
            score_curr = score_prev

        T *= dt

        if step_i % stop_steps == 0:
            if (
                (score_curr > stop_improvement * milestone_score
                 and milestone_success < stop_successes)
                or score_curr < 1e-6
            ):
                break
            milestone_score = score_curr
            milestone_success = 0

        score_prev = score_curr
        step_i += 1

    return score_curr


def mc_arcs(
    pos: np.ndarray,          # (N, 3) — positions of active_region beads (modified in place)
    exp_dist_mat: np.ndarray, # (N, N) — -1 = repulsion, 0 = no arc, >0 = expected distance
    step_size: float,
    settings,
) -> float:
    """
    MonteCarloArcs: simulated annealing using arc spring energy.

    Unlike heatmap, the global score is NOT double-counted (i < j pairs only).
    The local score for bead p sums over ALL other beads (not just i < p), so:
        score_curr = score_curr - local_prev + local_curr   (no factor 2)

    Mirrors C++ LooperSolver::MonteCarloArcs().
    Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    T = settings.max_temp
    dt = settings.dt_temp
    jump_scale = settings.jump_scale
    jump_coef = settings.jump_coef
    stop_steps = settings.mc_stop_steps
    stop_improvement = settings.mc_stop_improvement
    stop_successes = settings.mc_stop_successes
    stretch_k = settings.spring_stretch_arcs
    squeeze_k = settings.spring_squeeze_arcs

    score_curr = global_score_arcs_np(pos, exp_dist_mat, stretch_k, squeeze_k)
    score_prev = score_curr
    milestone_score = score_curr
    milestone_success = 0
    step_i = 1

    while True:
        p = random.randrange(n)
        disp = random_vector_np(step_size)

        local_prev = local_score_arcs_np(pos, exp_dist_mat, p, stretch_k, squeeze_k)
        pos[p] += disp
        local_curr = local_score_arcs_np(pos, exp_dist_mat, p, stretch_k, squeeze_k)

        score_curr = score_curr - local_prev + local_curr

        ok = score_curr <= score_prev
        if not ok:
            if score_prev > 0:
                tp = jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)
                ok = random.random() < tp

        if ok:
            milestone_success += 1
        else:
            pos[p] -= disp
            score_curr = score_prev

        T *= dt

        if step_i % stop_steps == 0:
            ratio = score_curr / milestone_score if milestone_score > 0 else 1.0
            if (
                (score_curr > stop_improvement * milestone_score
                 and milestone_success < stop_successes)
                or score_curr < 1e-5
                or ratio > 0.9999
            ):
                break
            milestone_score = score_curr
            milestone_success = 0

        score_prev = score_curr
        step_i += 1

    return score_curr
