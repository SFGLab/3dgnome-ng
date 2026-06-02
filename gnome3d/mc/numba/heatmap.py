"""Heatmap-energy MC (numba).

`mc_heatmap_numba` is the public entry.  Single-chain runs go through the shared
`common._run_outer_loop` (so excluded volume can be wired in); multi-chain runs
(`mc_heatmap_chains > 1`) use the stripped-down prange-parallel K-chain kernel
here (`_batch_heatmap_chain_nb` -> `_mc_heatmap_kchains_nb`) and keep the best.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.common import _as_f64, _dummy_f64, _dummy_i32, _run_outer_loop
from gnome3d.mc.numba.terms import (
    STRUCT_HEATMAP,
    _init_excl_nb,
    _init_heatmap_nb,
    _local_heatmap_nb,
    njit,
)
from gnome3d.types import BoolArray, F64Array, I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")


@njit(cache=True, fastmath=True, nogil=True)
def _batch_heatmap_chain_nb(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    score_hm: float,
) -> tuple[float, float, int]:
    """One batch of heatmap MC steps for a single chain.  Stripped-down vs
    `_batch_mc_nb` (no EV, no other terms) so it can be called from a parallel
    K-chain kernel without lugging 40+ args around.
    """
    n = pos.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p: int = int(np.random.randint(0, n))  # pyright: ignore[reportUnknownArgumentType]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz
        loc_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_new = score_hm + 2.0 * (loc_curr - loc_prev)

        ok = score_new <= score_hm
        if not ok and T > 0.0 and score_hm > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score_hm) / T)

        if ok:
            n_ok += 1
            score_hm = score_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score_hm, n_ok


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
    gives each chain a thread-local execution context with its own RNG state,
    so true parallelism is achievable - this is the cudaMMC-style "K parallel
    chains, take the best" pattern expressed in pure numba.
    """
    K = pos_k.shape[0]
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]  # view into the (k, :, :) slice
        T = max_temp
        score = _init_heatmap_nb(pos, exp_safe, skip)
        ms_score = score
        # Outer convergence loop entirely inside the kernel.
        while True:
            T, score, n_ok = _batch_heatmap_chain_nb(
                pos,
                exp_safe,
                skip,
                step_size,
                T,
                dt,
                jump_scale,
                jump_coef,
                stop_steps,
                score,
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
    final score — an embarrassingly-parallel restart strategy.
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
    movable: I64Array = np.arange(n, dtype=np.int64)
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

    score = _run_outer_loop(
        pw=pw,
        movable=movable,
        struct_type=STRUCT_HEATMAP,
        exp_mat=es64,
        dtn=_dummy_f64((1,)),
        skip_mat=skip_b,
        stretch_k=1.0,
        squeeze_k=1.0,
        ang_k=0.0,
        dist_w=1.0,
        ang_w=1.0,
        struct_delta_factor=2.0,
        use_heat=False,
        heat_dist=_dummy_f64(),
        heat_weight=0.0,
        use_orn=False,
        orn_is_L=np.zeros(1, dtype=np.bool_),
        anchor_ar=_dummy_i32(),
        nbr_offsets=np.zeros(2, dtype=np.int32),
        nbr_indices=_dummy_i32(),
        nbr_weights=np.zeros(1, dtype=np.float64),
        anchor_orn=np.zeros((1, 3), dtype=np.float64),
        bead_to_anchor_k=cast(I32Array, np.full(n, -1, dtype=np.int32)),
        motif_weight=0.0,
        motifs_symmetric=True,
        use_excl=use_excl,
        excl_r0=excl_r0,
        excl_weight=float(settings.exclusion_weight),
        excl_skip=int(settings.exclusion_skip_neighbors),
        use_conf=False,
        conf_cx=0.0,
        conf_cy=0.0,
        conf_cz=0.0,
        conf_R=1.0,
        conf_weight=0.0,
        step_size=step_size,
        T=float(settings.max_temp_heatmap),
        dt=float(settings.dt_temp_heatmap),
        jump_scale=float(settings.jump_scale_heatmap),
        jump_coef=float(settings.jump_coef_heatmap),
        stop_steps=int(settings.mc_stop_steps_heatmap),
        stop_improvement=float(settings.mc_stop_improvement_heatmap),
        stop_successes=int(settings.mc_stop_successes_heatmap),
        strict_better=False,
        score_eps=1e-6,
        stop_when_ratio_above=0.9999,
        score_struct=score_struct,
        score_heat=0.0,
        score_orn=0.0,
        score_excl=score_excl,
        score_conf=0.0,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
