"""The heatmap MC on numba."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.terms import init_affinity_nb, local_affinity_nb
from gnome3d.types import BoolArray, F64Array, I8Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")

# A typed wrapper of numba.njit. Pyright sees the signatures of decorated functions.
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


def _as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


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
    """Returns the full heatmap score, summed row by row in a fixed order."""
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


class CoarseTerms(NamedTuple):
    """The arguments of the optional compartment term at the coarse level. `off()` builds a
    disabled instance.
    """

    use_comp: bool
    comp_cls: I8Array
    comp_r0: float
    comp_weight: float
    comp_ea: float
    comp_eb: float

    @staticmethod
    def off(n: int = 1) -> CoarseTerms:
        return CoarseTerms(
            use_comp=False,
            comp_cls=np.zeros(n, dtype=np.int8),
            comp_r0=1.0,
            comp_weight=0.0,
            comp_ea=0.0,
            comp_eb=0.0,
        )

    @property
    def any_on(self) -> bool:
        return self.use_comp


def build_coarse_terms(
    settings: Settings,
    pos: F64Array,
    bond_scale: float,
    compartment: np.ndarray[Any, Any] | None,
) -> CoarseTerms:
    """Resolves the optional compartment term of one heatmap MC call. The term is on only when
    its flag and its track are present.
    """
    from gnome3d.mc.numba.common import affinity_params

    n = pos.shape[0]
    off = CoarseTerms.off(n)
    aff = affinity_params(settings, "heatmap", bond_scale, compartment)

    return CoarseTerms(
        use_comp=aff.use_comp,
        comp_cls=(
            np.ascontiguousarray(compartment, dtype=np.int8)
            if aff.use_comp and compartment is not None
            else off.comp_cls
        ),
        comp_r0=aff.comp_r0,
        comp_weight=aff.comp_weight,
        comp_ea=aff.comp_ea,
        comp_eb=aff.comp_eb,
    )


def init_coarse_scores(pos: F64Array, ct: CoarseTerms) -> float:
    """Returns the compartment score of the starting positions."""
    comp = 0.0
    if ct.use_comp:
        comp = init_affinity_nb(
            pos,
            ct.use_comp,
            ct.comp_cls,
            ct.comp_r0,
            ct.comp_weight,
            ct.comp_ea,
            ct.comp_eb,
        )
    return float(comp)


# Acceptance is `score_new <= score`. The structure term and the excluded volume are double counted.
# Every bead moves.


@njit(cache=True, fastmath=True, nogil=True)
def _batch_heatmap_nb(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    score_struct: float,
    score_excl: float,
    score_comp: float,
) -> tuple[float, float, float, float, int]:
    """Runs one batch of `n_steps` heatmap MC steps of one chain. Returns (T, score_struct,
    score_excl, score_comp, n_ok).
    """
    n = pos.shape[0]
    n_ok = 0
    score = score_struct + score_excl + score_comp

    for _ in range(n_steps):
        p: int = int(np.random.randint(0, n))  # pyright: ignore[reportUnknownArgumentType]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        loc_comp_prev = 0.0
        if use_comp:
            loc_comp_prev = local_affinity_nb(
                pos,
                p,
                use_comp,
                comp_cls,
                comp_r0,
                comp_weight,
                comp_ea,
                comp_eb,
            )

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_struct_new = score_struct + 2.0 * (loc_struct_curr - loc_struct_prev)

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_comp_new = score_comp
        if use_comp:
            loc_comp_curr = local_affinity_nb(
                pos,
                p,
                use_comp,
                comp_cls,
                comp_r0,
                comp_weight,
                comp_ea,
                comp_eb,
            )
            score_comp_new = score_comp + 2.0 * (loc_comp_curr - loc_comp_prev)

        score_new = score_struct_new + score_excl_new + score_comp_new

        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_excl = score_excl_new
            score_comp = score_comp_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return (
        T,
        score_struct,
        score_excl,
        score_comp,
        n_ok,
    )


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
    ct: CoarseTerms | None = None,
) -> float:
    """Runs `_batch_heatmap_nb` until convergence and returns the final total score."""
    if ct is None:
        ct = CoarseTerms.off(pos.shape[0])
    score_comp = init_coarse_scores(pos, ct)
    score = score_struct + score_excl + score_comp
    ms_score = score
    step_i = 0
    while True:
        (
            T,
            score_struct,
            score_excl,
            score_comp,
            n_ok,
        ) = _batch_heatmap_nb(
            pos,
            exp_safe,
            skip,
            use_excl,
            excl_r0,
            excl_weight,
            excl_skip,
            ct.use_comp,
            ct.comp_cls,
            ct.comp_r0,
            ct.comp_weight,
            ct.comp_ea,
            ct.comp_eb,
            float(step_size),
            T,
            dt,
            jump_scale,
            jump_coef,
            stop_steps,
            score_struct,
            score_excl,
            score_comp,
        )
        score = score_struct + score_excl + score_comp
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
    """Runs K independent heatmap MC chains in parallel, without excluded volume. Each `prange`
    iteration has its own thread local RNG.
    """
    K = pos_k.shape[0]
    # A placeholder for the compartment term, which this path does not use.
    no_i8 = np.zeros(1, dtype=np.int8)
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]
        T = max_temp
        score = _init_heatmap_nb(pos, exp_safe, skip)
        ms_score = score
        while True:
            T, score, _se, _sc, n_ok = _batch_heatmap_nb(
                pos,
                exp_safe,
                skip,
                False,  # use_excl
                1.0,  # excl_r0 (unused)
                0.0,  # excl_weight (unused)
                0,  # excl_skip (unused)
                False,  # use_comp
                no_i8,
                1.0,
                0.0,
                0.0,
                0.0,
                step_size,
                T,
                dt,
                jump_scale,
                jump_coef,
                stop_steps,
                score,
                0.0,  # score_excl
                0.0,  # score_comp
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
    """Runs K chains in one kernel launch and keeps the best."""
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
    pos: np.ndarray[Any, Any],  # (N, 3), modified in place
    exp_dist: np.ndarray[Any, Any],  # (N, N) expected distances
    diag_size: int,
    step_size: float,
    settings: Settings,
    compartment: np.ndarray[Any, Any] | None = None,
) -> float:
    """Anneals the beads in place and returns the final score.

    With `mc_heatmap_chains` above 1 that many chains run in parallel and the best is kept.
    """
    n = pos.shape[0]

    if int(settings.mc_heatmap_chains) > 1 and compartment is None:
        # The K chain kernel has the heatmap term alone and runs only without the compartment term.
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

    ct = build_coarse_terms(
        settings,
        pw,
        float(np.asarray(exp_dist)[~skip].mean()) if (~skip).any() else 1.0,
        compartment,
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
        ct=ct,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
