"""
src/mc.py  —  Monte Carlo simulation loops for 3dgnome-torch.

Mirrors C++ LooperSolver::MonteCarloHeatmap() and MonteCarloArcs().

The sequential MC inner loop runs on CPU with vectorized NumPy operations.
GPU dispatch overhead (~100–200 µs/kernel) dominates for the small N typical
in this algorithm (50–200 anchor beads per IB), making MPS/CUDA slower than
CPU for the per-step O(N) local score computation.  NumPy is 50–200× faster
here in practice.

The O(N²) initial global score is still computed with a single vectorized
NumPy call (fast regardless of device).

Acceptance criterion (both loops):
    ok = (score_curr <= score_prev)
      or rand() < jump_scale * exp(-jump_coef * score_curr/score_prev / T)
"""

import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized local score helpers (NumPy, no Python loop over beads)

def _local_heatmap(pos, exp_safe, skip_col, p):
    """Local heatmap score for bead p.  skip_col: (N,) bool for column p."""
    diff = pos - pos[p]                          # (N, 3)
    d = np.sqrt((diff * diff).sum(axis=1))       # (N,)
    e = np.where(skip_col, 1.0, exp_safe[:, p])
    err = (d - e) / e
    err[skip_col] = 0.0
    return float(np.dot(err, err))


def _local_arcs(pos, exp, p, stretch_k, squeeze_k):
    """Local arc score for bead p.  exp[i,j]=-1 repulsion, 0 none, >0 spring."""
    diff = pos - pos[p]
    d = np.sqrt((diff * diff).sum(axis=1))   # (N,)
    e = exp[:, p]                             # (N,)

    rep = e < 0.0
    spr = e >= 1e-6
    sc = 0.0
    if rep.any():
        sc += (1.0 / np.maximum(d[rep], 1e-10)).sum()
    if spr.any():
        es, ds = e[spr], d[spr]
        rel = (ds - es) / es
        st = rel >= 0.0
        sc += float((rel[st] * rel[st]).sum()) * stretch_k
        sc += float((rel[~st] * rel[~st]).sum()) * squeeze_k
    return sc


# ---------------------------------------------------------------------------
# MC loops

def mc_heatmap(
    pos: np.ndarray,           # (N, 3) float32 — modified in place
    exp_dist: np.ndarray,      # (N, N) — expected pairwise distances
    diag_size: int,
    step_size: float,
    settings,
    label: str = "",
) -> float:
    """
    MonteCarloHeatmap: simulated annealing using heatmap distance energy.

    Global score is double-counted, so the MC update rule is:
        score_curr += 2 * (local_curr - local_prev)

    Mirrors C++ LooperSolver::MonteCarloHeatmap().  Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    # Precompute static skip mask once — never changes during the run
    idx = np.arange(n)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size  # (N, N)
    skip = diag_mask | (exp_dist < 1e-6)                          # (N, N) bool
    exp_safe = np.where(skip, 1.0, exp_dist)                       # safe denominator

    T = settings.max_temp_heatmap
    dt = settings.dt_temp_heatmap
    jump_scale = settings.jump_scale_heatmap
    jump_coef = settings.jump_coef_heatmap
    stop_steps = settings.mc_stop_steps_heatmap
    stop_improvement = settings.mc_stop_improvement_heatmap
    stop_successes = settings.mc_stop_successes_heatmap

    # Initial global score: vectorized O(N²) — done once
    diff0 = pos[:, None, :] - pos[None, :, :]       # (N, N, 3)
    d0 = np.sqrt((diff0 * diff0).sum(axis=2))        # (N, N)
    cerr0 = (d0 - exp_safe) / exp_safe
    score_curr = float(np.where(skip, 0.0, cerr0 * cerr0).sum())

    score_prev = score_curr
    milestone_score = score_curr
    milestone_success = 0
    step_i = 1
    prefix = f"    [{label}] " if label else "    "

    while True:
        p = random.randrange(n)
        disp = np.array([
            random.uniform(-step_size, step_size),
            random.uniform(-step_size, step_size),
            random.uniform(-step_size, step_size),
        ], dtype=pos.dtype)

        local_prev = _local_heatmap(pos, exp_safe, skip[:, p], p)
        pos[p] += disp
        local_curr = _local_heatmap(pos, exp_safe, skip[:, p], p)

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
            ratio = score_curr / milestone_score if milestone_score > 0 else 1.0
            converged = (
                (score_curr > stop_improvement * milestone_score
                 and milestone_success < stop_successes)
                or score_curr < 1e-6
                or ratio > 0.9999
            )
            print(
                f"{prefix}step {step_i:>7,}  score={score_curr:.4f}"
                f"  ratio={ratio:.4f}  ok={milestone_success}/{stop_steps}"
                + ("  [done]" if converged else ""),
                flush=True,
            )
            if converged:
                break
            milestone_score = score_curr
            milestone_success = 0

        score_prev = score_curr
        step_i += 1

    return score_curr


def mc_arcs(
    pos: np.ndarray,           # (N, 3) float32 — modified in place
    exp_dist_mat: np.ndarray,  # (N, N) — -1=repulsion, 0=none, >0=spring distance
    step_size: float,
    settings,
    label: str = "",
) -> float:
    """
    MonteCarloArcs: simulated annealing using arc spring energy.

    Global score counts i < j pairs once.  Local score sums ALL other beads,
    so the MC update rule is:
        score_curr = score_curr - local_prev + local_curr   (no factor 2)

    Mirrors C++ LooperSolver::MonteCarloArcs().  Returns final score.
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

    # Initial global score: i < j pairs, vectorized O(N²) — done once
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff0 = pos[i_idx] - pos[j_idx]                   # (M, 3)
    d0 = np.sqrt((diff0 * diff0).sum(axis=1))          # (M,)
    e0 = exp_dist_mat[i_idx, j_idx]                    # (M,)
    rep0, spr0 = e0 < 0.0, e0 >= 1e-6
    score_curr = 0.0
    if rep0.any():
        score_curr += (1.0 / np.maximum(d0[rep0], 1e-10)).sum()
    if spr0.any():
        es0, ds0 = e0[spr0], d0[spr0]
        rel0 = (ds0 - es0) / es0
        st0 = rel0 >= 0.0
        score_curr += float((rel0[st0] * rel0[st0]).sum()) * stretch_k
        score_curr += float((rel0[~st0] * rel0[~st0]).sum()) * squeeze_k

    score_prev = score_curr
    milestone_score = score_curr
    milestone_success = 0
    step_i = 1
    prefix = f"    [{label}] " if label else "    "

    while True:
        p = random.randrange(n)
        disp = np.array([
            random.uniform(-step_size, step_size),
            random.uniform(-step_size, step_size),
            random.uniform(-step_size, step_size),
        ], dtype=pos.dtype)

        local_prev = _local_arcs(pos, exp_dist_mat, p, stretch_k, squeeze_k)
        pos[p] += disp
        local_curr = _local_arcs(pos, exp_dist_mat, p, stretch_k, squeeze_k)

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
            converged = (
                (score_curr > stop_improvement * milestone_score
                 and milestone_success < stop_successes)
                or score_curr < 1e-5
                or ratio > 0.9999
            )
            print(
                f"{prefix}step {step_i:>7,}  score={score_curr:.4f}"
                f"  ratio={ratio:.4f}  ok={milestone_success}/{stop_steps}"
                + ("  [done]" if converged else ""),
                flush=True,
            )
            if converged:
                break
            milestone_score = score_curr
            milestone_success = 0

        score_prev = score_curr
        step_i += 1

    return score_curr
