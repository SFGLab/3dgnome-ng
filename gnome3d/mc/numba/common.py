"""Shared numba driver + array helpers.

`_run_outer_loop` is the Python-side convergence loop every public entry uses:
it runs `terms._batch_mc_nb` for `stop_steps` at a time and checks the C++-style
stop condition.  `_prepare_orientation` builds the CSR orientation arrays the
kernel needs from the Python neighbour dicts.  `_as_f64` / `_dummy_*` produce the
contiguous / placeholder arrays the kernel's fixed signature expects.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from gnome3d import log
from gnome3d.mc.numba.terms import _batch_mc_nb, _score_orientation_full_nb
from gnome3d.types import BoolArray, F64Array, I32Array, I64Array

LOG = log.get("mc.numba")


def _as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _dummy_f64(shape: tuple[int, ...] = (1, 1)) -> F64Array:
    return np.zeros(shape, dtype=np.float64)


def _dummy_bool(shape: tuple[int, ...] = (1, 1)) -> BoolArray:
    return np.zeros(shape, dtype=np.bool_)


def _dummy_i32(shape: tuple[int, ...] = (1,)) -> I32Array:
    return np.zeros(shape, dtype=np.int32)


def _prepare_orientation(
    pw: F64Array,
    fixed: np.ndarray[Any, Any],
    char_orientations: np.ndarray[Any, Any],
    anchor_neighbors: dict[int, list[int]],
    anchor_neighbor_weights: dict[int, list[float]],
    motif_weight: float,
    motifs_symmetric: bool,
) -> tuple[I32Array, I32Array, I32Array, F64Array, BoolArray, I32Array, F64Array, float]:
    """Build numba-friendly orientation arrays from the Python dicts.

    Returns (anchor_ar, nbr_offsets, nbr_indices, nbr_weights, orn_is_L,
             bead_to_anchor_k, anchor_orn, score_orn).
    """
    from gnome3d.util import calc_orientation as _calc_orn

    n = pw.shape[0]
    anchor_ar: I32Array = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
    n_anchors = len(anchor_ar)
    nbr_offsets: I32Array = np.zeros(n_anchors + 1, dtype=np.int32)
    for k in range(n_anchors):
        nbr_offsets[k + 1] = nbr_offsets[k] + len(anchor_neighbors.get(k, []))
    total = int(nbr_offsets[n_anchors])
    nbr_indices: I32Array = np.empty(total, dtype=np.int32)
    nbr_weights: F64Array = np.empty(total, dtype=np.float64)
    for k in range(n_anchors):
        for ki, (j, w) in enumerate(
            zip(anchor_neighbors.get(k, []), anchor_neighbor_weights.get(k, []), strict=True)
        ):
            off = nbr_offsets[k] + ki
            nbr_indices[off] = j
            nbr_weights[off] = w
    orn_is_L: BoolArray = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
    bead_to_anchor_k: I32Array = cast(I32Array, np.full(n, -1, dtype=np.int32))
    for k in range(n_anchors):
        ar = int(anchor_ar[k])
        if ar > 0:
            bead_to_anchor_k[ar - 1] = k
        if ar + 1 < n:
            bead_to_anchor_k[ar + 1] = k
    anchor_orn: F64Array = np.zeros((n_anchors, 3), dtype=np.float64)
    for k in range(n_anchors):
        ar = int(anchor_ar[k])
        anchor_orn[k] = _calc_orn(pw, ar, n, char_orientations[ar])
    score_orn = float(
        _score_orientation_full_nb(
            anchor_orn, nbr_offsets, nbr_indices, nbr_weights, motif_weight, motifs_symmetric
        )
    )
    return (
        anchor_ar,
        nbr_offsets,
        nbr_indices,
        nbr_weights,
        orn_is_L,
        bead_to_anchor_k,
        anchor_orn,
        score_orn,
    )


def _run_outer_loop(
    pw: F64Array,
    movable: I64Array,
    struct_type: int,
    exp_mat: F64Array,
    dtn: F64Array,
    skip_mat: BoolArray,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
    struct_delta_factor: float,
    use_heat: bool,
    heat_dist: F64Array,
    heat_weight: float,
    use_orn: bool,
    orn_is_L: BoolArray,
    anchor_ar: I32Array,
    nbr_offsets: I32Array,
    nbr_indices: I32Array,
    nbr_weights: F64Array,
    anchor_orn: F64Array,
    bead_to_anchor_k: I32Array,
    motif_weight: float,
    motifs_symmetric: bool,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    use_conf: bool,
    conf_cx: float,
    conf_cy: float,
    conf_cz: float,
    conf_R: float,
    conf_weight: float,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    stop_steps: int,
    stop_improvement: float,
    stop_successes: int,
    strict_better: bool,
    score_eps: float,
    stop_when_ratio_above: float,
    score_struct: float,
    score_heat: float,
    score_orn: float,
    score_excl: float,
    score_conf: float,
) -> float:
    """Drive the unified kernel until convergence; return the final total score."""
    score = score_struct + score_heat + score_orn + score_excl + score_conf
    ms_score = score
    step_i = 0
    while True:
        (T, score_struct, score_heat, score_orn, score_excl, score_conf, n_ok) = _batch_mc_nb(
            pw,
            movable,
            struct_type,
            exp_mat,
            dtn,
            skip_mat,
            stretch_k,
            squeeze_k,
            ang_k,
            dist_w,
            ang_w,
            struct_delta_factor,
            use_heat,
            heat_dist,
            heat_weight,
            use_orn,
            orn_is_L,
            anchor_ar,
            nbr_offsets,
            nbr_indices,
            nbr_weights,
            anchor_orn,
            bead_to_anchor_k,
            motif_weight,
            motifs_symmetric,
            use_excl,
            excl_r0,
            excl_weight,
            excl_skip,
            use_conf,
            conf_cx,
            conf_cy,
            conf_cz,
            conf_R,
            conf_weight,
            float(step_size),
            T,
            dt,
            jump_scale,
            jump_coef,
            stop_steps,
            strict_better,
            score_struct,
            score_heat,
            score_orn,
            score_excl,
            score_conf,
        )
        score = score_struct + score_heat + score_orn + score_excl + score_conf
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < score_eps
            or ratio > stop_when_ratio_above
        )
        LOG.debug(
            "step %7s  score=%.4f  ratio=%.4f  ok=%d/%d%s",
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
