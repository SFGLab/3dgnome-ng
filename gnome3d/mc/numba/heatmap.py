"""Heatmap-energy MC (numba) - a fully self-contained kernel.

Unlike arcs/smooth/ib (which share the unified `_batch_mc_nb` + `terms` math),
the heatmap kernel is deliberately standalone: it carries its OWN copies of the
two energy terms it needs (heatmap distance-to-expected + excluded volume), its
own MC inner loop (`_batch_heatmap_nb`), and its own convergence driver
(`_run_heatmap_loop`).  The duplication is intentional - heatmap is the simplest
energy (double-counted structure, optional EV, non-strict acceptance) and keeping
it apart means the shared kernel never has to carry a heatmap branch.

`mc_heatmap_numba` is the public entry.  Single-chain runs wire in excluded
volume; multi-chain runs (`mc_heatmap_chains > 1`) run the same inner kernel with
EV off across K prange-parallel chains and keep the best.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.types import BoolArray, F64Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")

# Typed wrapper around numba.njit so pyright sees decorated functions with their
# original signatures.  At runtime this is just numba.njit.  (Duplicated from
# `terms` on purpose - this module shares no code with the unified kernel.)
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


def _as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


# ----- energy terms (heatmap distance-to-expected + excluded volume) -----


@njit(cache=True, fastmath=True, nogil=True)
def _local_heatmap_nb(pos: F64Array, exp_safe: F64Array, skip_col: BoolArray, p: int) -> float:
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if skip_col[i]:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        e = exp_safe[i, p]
        err = (d - e) / e
        sc += err * err
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _init_heatmap_nb(pos: F64Array, exp_safe: F64Array, skip: BoolArray) -> float:
    """O(N^2) init - row-at-a-time so the sum order is stable."""
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        row_sc = 0.0
        for j in range(n):
            if skip[i, j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            e = exp_safe[i, j]
            err = (d - e) / e
            row_sc += err * err
        sc += row_sc
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _excl_pair_nb(d: float, r0: float, weight: float) -> float:
    if d >= r0:
        return 0.0
    rel = (r0 - d) / r0
    return weight * rel * rel


@njit(cache=True, fastmath=True, nogil=True)
def _local_excl_nb(pos: F64Array, p: int, r0: float, weight: float, skip: int) -> float:
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        diff = i - p
        if diff < 0:
            diff = -diff
        if diff <= skip:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        err += _excl_pair_nb(d, r0, weight)
    return err


@njit(cache=True, fastmath=True, nogil=True)
def _init_excl_nb(pos: F64Array, r0: float, weight: float, skip: int) -> float:
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        row_err = 0.0
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            if diff <= skip:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            row_err += _excl_pair_nb(d, r0, weight)
        err += row_err
    return err


# ----- MC inner loop + convergence driver -----
#
# Heatmap acceptance is NON-strict (score_new <= score); structure is
# double-counted (delta factor 2), and so is excluded volume.  All beads move.


@njit(cache=True, fastmath=True, nogil=True)
def _batch_heatmap_nb(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    score_struct: float,
    score_excl: float,
) -> tuple[float, float, float, int]:
    """One batch of `n_steps` heatmap MC steps for a single chain (heatmap term +
    optional excluded volume).  Returns (T, score_struct, score_excl, n_ok)."""
    n = pos.shape[0]
    n_ok = 0
    score = score_struct + score_excl

    for _ in range(n_steps):
        p: int = int(np.random.randint(0, n))  # pyright: ignore[reportUnknownArgumentType]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_struct_new = score_struct + 2.0 * (loc_struct_curr - loc_struct_prev)

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_new = score_struct_new + score_excl_new

        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_excl = score_excl_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score_struct, score_excl, n_ok


def _run_heatmap_loop(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    stop_steps: int,
    stop_improvement: float,
    stop_successes: int,
    score_eps: float,
    stop_when_ratio_above: float,
    score_struct: float,
    score_excl: float,
) -> float:
    """Drive `_batch_heatmap_nb` to convergence; return the final total score.
    Same C++-style stop condition as the shared driver (plateau / score_eps /
    ratio guard), specialised to the heatmap term set."""
    score = score_struct + score_excl
    ms_score = score
    step_i = 0
    while True:
        T, score_struct, score_excl, n_ok = _batch_heatmap_nb(
            pos,
            exp_safe,
            skip,
            use_excl,
            excl_r0,
            excl_weight,
            excl_skip,
            float(step_size),
            T,
            dt,
            jump_scale,
            jump_coef,
            stop_steps,
            score_struct,
            score_excl,
        )
        score = score_struct + score_excl
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < score_eps
            or ratio > stop_when_ratio_above
        )
        LOG.debug(
            "heatmap step %7s  score=%.4f  ratio=%.4f  ok=%d/%d%s",
            f"{step_i:,}",
            score,
            ratio,
            n_ok,
            stop_steps,
            "  [done]" if converged else "",
        )
        if converged:
            return score
        ms_score = score


# ----- multi-chain (prange best-of-K restarts; EV off, as the reference) -----


@njit(cache=True, parallel=True, nogil=True)
def _mc_heatmap_kchains_nb(
    pos_k: F64Array,  # (K, N, 3)
    exp_safe: F64Array,  # (N, N)
    skip: BoolArray,  # (N, N)
    max_temp: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    stop_steps: int,
    stop_improvement: float,
    stop_successes: int,
    step_size: float,
    final_scores: F64Array,  # (K,) output
) -> None:
    """Run K independent heatmap MC chains in parallel.  `for k in prange(K)`
    gives each chain a thread-local execution context with its own RNG state -
    the cudaMMC-style "K parallel chains, take the best" pattern in pure numba.
    EV is off here (matches the reference multi-chain path)."""
    K = pos_k.shape[0]
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]  # view into the (k, :, :) slice
        T = max_temp
        score = _init_heatmap_nb(pos, exp_safe, skip)
        ms_score = score
        # Outer convergence loop entirely inside the kernel.
        while True:
            T, score, _se, n_ok = _batch_heatmap_nb(
                pos,
                exp_safe,
                skip,
                False,  # use_excl
                1.0,  # excl_r0 (unused)
                0.0,  # excl_weight (unused)
                0,  # excl_skip (unused)
                step_size,
                T,
                dt,
                jump_scale,
                jump_coef,
                stop_steps,
                score,
                0.0,  # score_excl
            )
            converged = (
                score > stop_improvement * ms_score and n_ok < stop_successes
            ) or score < 1e-6
            if converged:
                break
            ms_score = score
        final_scores[k] = score


def _mc_heatmap_multichain(
    pos: np.ndarray[Any, Any],
    exp_dist: np.ndarray[Any, Any],
    diag_size: int,
    step_size: float,
    settings: Settings,
) -> float:
    """Run K independent MC chains via `@njit(parallel=True)` + prange, then
    pick the best.  All chains live in a single kernel launch, so per-thread
    RNG state is independent (no contention) and Python/GIL is out of the loop
    once the kernel starts.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    K = max(1, int(settings.mc_heatmap_chains))

    idx: I64Array = np.arange(n, dtype=np.int64)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip_np = diag_mask | (exp_dist < 1e-6)
    exp_safe_np = np.where(skip_np, 1.0, exp_dist)

    pos_k: F64Array = np.ascontiguousarray(
        np.broadcast_to(pos.astype(np.float64), (K, n, 3)).copy()
    )
    exp_safe = np.ascontiguousarray(exp_safe_np.astype(np.float64))
    skip = np.ascontiguousarray(skip_np.astype(np.bool_))
    final_scores: F64Array = np.zeros(K, dtype=np.float64)

    LOG.debug("K=%d N=%d (numba prange parallel)", K, n)

    _mc_heatmap_kchains_nb(
        pos_k,
        exp_safe,
        skip,
        float(settings.max_temp_heatmap),
        float(settings.dt_temp_heatmap),
        float(settings.jump_scale_heatmap),
        float(settings.jump_coef_heatmap),
        int(settings.mc_stop_steps_heatmap),
        float(settings.mc_stop_improvement_heatmap),
        int(settings.mc_stop_successes_heatmap),
        float(step_size),
        final_scores,
    )

    best_k: int = int(np.argmin(final_scores))
    pos[:] = pos_k[best_k].astype(pos.dtype)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "scores: %s  -> picked ch%d",
            ", ".join(f"{s:.2f}" for s in final_scores),
            best_k,
        )
    return float(final_scores[best_k])


def mc_heatmap_numba(
    pos: np.ndarray[Any, Any],  # (N, 3) float32 - modified in place
    exp_dist: np.ndarray[Any, Any],  # (N, N) - expected pairwise distances
    diag_size: int,
    step_size: float,
    settings: Settings,
) -> float:
    """Numba simulated-annealing implementation for heatmap-energy MC.
    Double-counted structure (delta factor 2). Mirrors Reference
    LooperSolver::MonteCarloHeatmap().  Called by `gnome3d.mc.mc_heatmap`
    when `settings.mc_backend != "jax"`.

    When `settings.mc_heatmap_chains > 1`, runs that many independent MC
    chains in parallel via numba threading and keeps the one with the best
    final score - an embarrassingly-parallel restart strategy.
    """
    n = pos.shape[0]

    if int(settings.mc_heatmap_chains) > 1:
        return _mc_heatmap_multichain(pos, exp_dist, diag_size, step_size, settings)

    if n <= 1:
        return 0.0

    idx: I64Array = np.arange(n, dtype=np.int64)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip = diag_mask | (exp_dist < 1e-6)
    exp_safe = np.where(skip, 1.0, exp_dist)

    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_heatmap)
    excl_r0 = float(settings.exclusion_radius_heatmap)
    if use_excl and excl_r0 <= 0.0:
        active = np.asarray(exp_dist)[~skip]
        factor = float(settings.exclusion_auto_factor_heatmap)
        excl_r0 = factor * float(active.mean()) if active.size > 0 else 1.0

    pw = _as_f64(pos)
    es64 = _as_f64(exp_safe)
    skip_b: BoolArray = np.ascontiguousarray(skip, dtype=np.bool_)
    score_struct = float(_init_heatmap_nb(pw, es64, skip_b))
    score_excl = (
        float(
            _init_excl_nb(
                pw,
                excl_r0,
                float(settings.exclusion_weight),
                int(settings.exclusion_skip_neighbors),
            )
        )
        if use_excl
        else 0.0
    )

    score = _run_heatmap_loop(
        pos=pw,
        exp_safe=es64,
        skip=skip_b,
        use_excl=use_excl,
        excl_r0=excl_r0,
        excl_weight=float(settings.exclusion_weight),
        excl_skip=int(settings.exclusion_skip_neighbors),
        step_size=step_size,
        T=float(settings.max_temp_heatmap),
        dt=float(settings.dt_temp_heatmap),
        jump_scale=float(settings.jump_scale_heatmap),
        jump_coef=float(settings.jump_coef_heatmap),
        stop_steps=int(settings.mc_stop_steps_heatmap),
        stop_improvement=float(settings.mc_stop_improvement_heatmap),
        stop_successes=int(settings.mc_stop_successes_heatmap),
        score_eps=1e-6,
        stop_when_ratio_above=0.9999,
        score_struct=score_struct,
        score_excl=score_excl,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
