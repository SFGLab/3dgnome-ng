"""
src/mc.py  —  Monte Carlo simulation loops for 3dgnome-torch.

Mirrors C++ LooperSolver::MonteCarloHeatmap() and MonteCarloArcs().

Both loops use:
  - Sequential Metropolis steps (cannot be parallelised within a chain)
  - Uniform cube displacement: each component in [-step, step]
  - Temperature cooling: T *= dt each step
  - Milestone-based convergence: check every mc_stop_steps steps whether
    the score improved by at least (1 - threshold) and there were enough
    successful moves

Acceptance criterion (both loops):
    ok = (score_curr <= score_prev) OR rand() < jump_scale * exp(-jump_coef * score_curr/score_prev / T)

Local scores are computed with vectorized torch tensor ops (no Python loop
over beads), enabling GPU/MPS acceleration.  The accept/reject decision and
temperature schedule remain on CPU (scalar operations).
"""

import math
import random

import numpy as np
import torch

from .energy import get_device


# ---------------------------------------------------------------------------
# Vectorized local score helpers
# Each computes the score contribution of one bead relative to all others
# using a single set of tensor operations (O(N) kernel, no Python loop).

def _local_heatmap(
    pos: torch.Tensor,       # (N, 3)
    exp: torch.Tensor,       # (N, N)
    skip: torch.Tensor,      # (N, N) bool — True = exclude
    p: int,
) -> torch.Tensor:
    """Local heatmap score for bead p — column-p slice of the global score."""
    skip_p = skip[:, p]                          # (N,)
    d = (pos - pos[p]).norm(dim=1)               # (N,)
    e = exp[:, p].masked_fill(skip_p, 1.0)       # safe denominator
    err = (d - e) / e
    return (err * err).masked_fill(skip_p, 0.0).sum()


def _local_arcs(
    pos: torch.Tensor,       # (N, 3)
    exp: torch.Tensor,       # (N, N)  -1=repulsion, 0=none, >0=spring
    p: int,
    stretch_k: float,
    squeeze_k: float,
) -> torch.Tensor:
    """Local arc score for bead p — all other beads, no Python loop."""
    d = (pos - pos[p]).norm(dim=1)   # (N,)
    e = exp[:, p]                    # (N,)

    sc = pos.new_zeros(())

    rep = e < 0.0
    if rep.any():
        sc = sc + (1.0 / d[rep].clamp(min=1e-10)).sum()

    spr = e >= 1e-6
    if spr.any():
        es, ds = e[spr], d[spr]
        rel = (ds - es) / es
        st = rel >= 0.0
        sc = sc + (rel[st] ** 2).sum() * stretch_k
        sc = sc + (rel[~st] ** 2).sum() * squeeze_k

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

    Global score is double-counted (Σ_moved Σ_i err), so the update rule is:
        score_curr += 2 * (local_curr - local_prev)

    Uses vectorized torch ops on the best available device (CUDA > MPS > CPU).
    Mirrors C++ LooperSolver::MonteCarloHeatmap().
    Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    device = get_device()
    dtype = torch.float32 if device.type == "mps" else torch.float64

    pos_t = torch.tensor(pos, dtype=dtype, device=device)
    exp_t = torch.tensor(exp_dist, dtype=dtype, device=device)

    # Precompute skip mask once — valid for entire run (exp_t never changes)
    idx = torch.arange(n, device=device, dtype=torch.long)
    diag_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() < diag_size
    skip = diag_mask | (exp_t < 1e-6)   # (N, N)

    T = settings.max_temp_heatmap
    dt = settings.dt_temp_heatmap
    jump_scale = settings.jump_scale_heatmap
    jump_coef = settings.jump_coef_heatmap
    stop_steps = settings.mc_stop_steps_heatmap
    stop_improvement = settings.mc_stop_improvement_heatmap
    stop_successes = settings.mc_stop_successes_heatmap

    with torch.no_grad():
        # Initial global score: fully vectorized O(N²)
        e_safe = exp_t.masked_fill(skip, 1.0)
        diff = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)   # (N, N, 3)
        d_mat = diff.norm(dim=2)                           # (N, N)
        cerr = (d_mat - e_safe) / e_safe
        score_curr = (cerr * cerr).masked_fill(skip, 0.0).sum().item()

        score_prev = score_curr
        milestone_score = score_curr
        milestone_success = 0
        step_i = 1
        prefix = f"    [{label}] " if label else "    "

        while True:
            p = random.randrange(n)
            disp = (torch.rand(3, device=device, dtype=dtype) * 2.0 - 1.0) * step_size

            local_prev = _local_heatmap(pos_t, exp_t, skip, p).item()
            pos_t[p] += disp
            local_curr = _local_heatmap(pos_t, exp_t, skip, p).item()

            score_curr = score_curr + 2.0 * (local_curr - local_prev)

            ok = score_curr <= score_prev
            if not ok and T > 0.0:
                tp = jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)
                ok = random.random() < tp

            if ok:
                milestone_success += 1
            else:
                pos_t[p] -= disp
                score_curr = score_prev

            T *= dt

            if step_i % stop_steps == 0:
                ratio = score_curr / milestone_score if milestone_score > 0 else 1.0
                converged = (
                    (score_curr > stop_improvement * milestone_score
                     and milestone_success < stop_successes)
                    or score_curr < 1e-6
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

    pos[:] = pos_t.cpu().to(torch.float32).numpy()
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

    Global score counts i < j pairs once.  Local score for bead p sums ALL
    other beads (not just i < p), so:
        score_curr = score_curr - local_prev + local_curr   (no factor 2)

    Uses vectorized torch ops on the best available device (CUDA > MPS > CPU).
    Mirrors C++ LooperSolver::MonteCarloArcs().
    Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    device = get_device()
    dtype = torch.float32 if device.type == "mps" else torch.float64

    pos_t = torch.tensor(pos, dtype=dtype, device=device)
    exp_t = torch.tensor(exp_dist_mat, dtype=dtype, device=device)

    T = settings.max_temp
    dt = settings.dt_temp
    jump_scale = settings.jump_scale
    jump_coef = settings.jump_coef
    stop_steps = settings.mc_stop_steps
    stop_improvement = settings.mc_stop_improvement
    stop_successes = settings.mc_stop_successes
    stretch_k = settings.spring_stretch_arcs
    squeeze_k = settings.spring_squeeze_arcs

    with torch.no_grad():
        # Initial global score: i < j pairs, fully vectorized
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
        d_ij = (pos_t[i_idx] - pos_t[j_idx]).norm(dim=1)
        e_ij = exp_t[i_idx, j_idx]

        rep = e_ij < 0.0
        spr = e_ij >= 1e-6
        score_curr = 0.0
        if rep.any():
            score_curr += (1.0 / d_ij[rep].clamp(min=1e-10)).sum().item()
        if spr.any():
            es, ds = e_ij[spr], d_ij[spr]
            rel = (ds - es) / es
            st = rel >= 0.0
            score_curr += ((rel[st] ** 2).sum() * stretch_k
                           + (rel[~st] ** 2).sum() * squeeze_k).item()

        score_prev = score_curr
        milestone_score = score_curr
        milestone_success = 0
        step_i = 1
        prefix = f"    [{label}] " if label else "    "

        while True:
            p = random.randrange(n)
            disp = (torch.rand(3, device=device, dtype=dtype) * 2.0 - 1.0) * step_size

            local_prev = _local_arcs(pos_t, exp_t, p, stretch_k, squeeze_k).item()
            pos_t[p] += disp
            local_curr = _local_arcs(pos_t, exp_t, p, stretch_k, squeeze_k).item()

            score_curr = score_curr - local_prev + local_curr

            ok = score_curr <= score_prev
            if not ok:
                if score_prev > 0:
                    tp = jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)
                    ok = random.random() < tp

            if ok:
                milestone_success += 1
            else:
                pos_t[p] -= disp
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

    pos[:] = pos_t.cpu().to(torch.float32).numpy()
    return score_curr
