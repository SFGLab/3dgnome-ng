"""Smooth-MC (numba): chain bonds + angles, optional CTCF orientation / heat.

`mc_smooth_numba` is the public entry.  The full config (orientation, excluded
volume, confinement) runs through the shared `common._run_outer_loop`; the simple
config (chain + optional heat only) with `mc_smooth_chains > 1` dispatches to the
prange-parallel K-chain kernel here and keeps the best.  `mc_ib_numba` (ib.py) is
a peer that reuses the same chain term math.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.common import (
    affinity_params,
    as_f64,
    dummy_bool,
    dummy_f64,
    dummy_i32,
    init_affinity_scores,
    prepare_orientation,
    run_outer_loop,
)
from gnome3d.mc.numba.terms import (
    STRUCT_CHAIN,
    init_confine_nb,
    init_excl_nb,
    init_heat_nb,
    init_smooth_nb,
    local_heat_nb,
    local_smooth_nb,
    njit,
)
from gnome3d.types import F64Array, I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")


@njit(cache=True, fastmath=True, nogil=True)
def _batch_smooth_chain_nb(
    pos: F64Array,
    dtn: F64Array,
    movable: I64Array,
    use_heat: bool,
    heat_dist: F64Array,
    heat_weight: float,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
    score_struct: float,
    score_heat: float,
) -> tuple[float, float, float, int]:
    """One batch of smooth MC steps for a single chain - simplified form (chain
    bonds + angles + optional heat; no orientation, EV, confinement).  Used
    only inside the parallel K-chain kernel `_mc_smooth_kchains_nb`."""
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    score = score_struct + score_heat

    for _ in range(n_steps):
        p: int = int(movable[np.random.randint(0, n_mov)])
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = local_smooth_nb(
            pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
        )
        loc_heat_prev = 0.0
        if use_heat:
            loc_heat_prev = local_heat_nb(pos, heat_dist, p, heat_weight)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = local_smooth_nb(
            pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
        )
        score_struct_new = score_struct - loc_struct_prev + loc_struct_curr

        score_heat_new = score_heat
        if use_heat:
            loc_heat_curr = local_heat_nb(pos, heat_dist, p, heat_weight)
            score_heat_new = score_heat + 2.0 * (loc_heat_curr - loc_heat_prev)

        score_new = score_struct_new + score_heat_new

        ok = score_new < score  # smooth uses strict less-than
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_heat = score_heat_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score_struct, score_heat, n_ok


@njit(cache=True, parallel=True, nogil=True)
def _mc_smooth_kchains_nb(
    pos_k: F64Array,
    dtn: F64Array,
    movable: I64Array,
    use_heat: bool,
    heat_dist: F64Array,
    heat_weight: float,
    max_temp: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    stop_steps: int,
    stop_improvement: float,
    stop_successes: int,
    step_size: float,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
    final_scores: F64Array,
) -> None:
    """Run K independent smooth MC chains in parallel.  Same pattern as
    `_mc_heatmap_kchains_nb`: `for k in prange(K)` gives each chain its own
    thread-local execution context with independent RNG state.
    """
    K = pos_k.shape[0]
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]
        T = max_temp
        score_struct = init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        score_heat = init_heat_nb(pos, heat_dist, heat_weight) if use_heat else 0.0
        score = score_struct + score_heat
        ms_score = score
        while True:
            T, score_struct, score_heat, n_ok = _batch_smooth_chain_nb(
                pos,
                dtn,
                movable,
                use_heat,
                heat_dist,
                heat_weight,
                step_size,
                T,
                dt,
                jump_scale,
                jump_coef,
                stop_steps,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                score_struct,
                score_heat,
            )
            score = score_struct + score_heat
            converged = (
                score > stop_improvement * ms_score and n_ok < stop_successes
            ) or score < 1e-6
            if converged:
                break
            ms_score = score
        final_scores[k] = score


def _mc_smooth_multichain(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    heat_dist: np.ndarray[Any, Any] | None,
) -> float:
    """K-chain prange-parallel smooth MC.  Only supports the simple config
    (chain bonds + optional heat; no orientation/EV/confinement).  Callers
    must verify those terms are off before dispatching here."""
    n = pos.shape[0]
    K = max(1, int(settings.mc_smooth_chains))

    movable: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable) == 0:
        return 0.0

    pos_k: F64Array = np.ascontiguousarray(
        np.broadcast_to(pos.astype(np.float64), (K, n, 3)).copy()
    )
    dtn64 = as_f64(dtn)
    use_heat = heat_dist is not None
    if use_heat:
        assert heat_dist is not None
        heat64 = as_f64(heat_dist)
    else:
        heat64 = np.zeros((1, 1), dtype=np.float64)

    final_scores: F64Array = np.zeros(K, dtype=np.float64)

    LOG.debug("smooth K=%d N=%d (numba prange parallel)", K, n)

    _mc_smooth_kchains_nb(
        pos_k,
        dtn64,
        movable,
        use_heat,
        heat64,
        float(settings.subanchor_heatmap_dist_weight),
        float(settings.max_temp_smooth),
        float(settings.dt_temp_smooth),
        float(settings.jump_scale_smooth),
        float(settings.jump_coef_smooth),
        int(settings.mc_stop_steps_smooth),
        float(settings.mc_stop_improvement_smooth),
        int(settings.mc_stop_successes_smooth),
        float(step_size),
        float(settings.spring_stretch),
        float(settings.spring_squeeze),
        float(settings.spring_angular),
        float(settings.smooth_dist_weight),
        float(settings.smooth_angle_weight),
        final_scores,
    )

    best_k: int = int(np.argmin(final_scores))
    pos[:] = pos_k[best_k].astype(pos.dtype)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "smooth scores: %s  -> picked ch%d",
            ", ".join(f"{s:.2f}" for s in final_scores),
            best_k,
        )
    return float(final_scores[best_k])


def mc_smooth_numba(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    compartment: np.ndarray[Any, Any] | None = None,
    accessibility: np.ndarray[Any, Any] | None = None,
) -> float:
    """Chain connectivity + angle MC.  Optionally adds CTCF orientation and/or
    subanchor heat. Anchor beads (fixed=True) never move. Single-counted
    structure (delta factor 1). Mirrors Reference MonteCarloArcsSmooth.

    When `settings.mc_smooth_chains > 1` AND the call uses the simple config
    (no orientation, no EV, no confinement), dispatches to a prange-parallel
    K-chain kernel and keeps the best score.  Complex configs fall back to
    single-chain.

    Called by `gnome3d.mc.mc_smooth` when `settings.mc_backend != "jax"`.
    """
    n = pos.shape[0]
    if n <= 2:
        return 0.0

    # Multi-chain dispatch (simple-config path only).
    if int(settings.mc_smooth_chains) > 1:
        simple_config = (
            char_orientations is None
            and not (
                bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
            )
            and not (bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth))
            and compartment is None
            and accessibility is None
        )
        if simple_config:
            return _mc_smooth_multichain(pos, dtn, fixed, step_size, settings, heat_dist)

    movable: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable) == 0:
        return 0.0

    pw = as_f64(pos)
    dtn64 = as_f64(dtn)

    stretch_k = float(settings.spring_stretch)
    squeeze_k = float(settings.spring_squeeze)
    ang_k = float(settings.spring_angular)
    dist_w = float(settings.smooth_dist_weight)
    ang_w = float(settings.smooth_angle_weight)

    use_heat = heat_dist is not None
    use_orn = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
        and bool(settings.use_ctcf_motif)
    )
    motif_weight = float(settings.motif_weight)
    motifs_symmetric = bool(settings.motifs_symmetric)
    heat_weight = float(settings.subanchor_heatmap_dist_weight)

    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    excl_r0 = float(settings.exclusion_radius_smooth)
    if use_excl and excl_r0 <= 0.0:
        factor = float(settings.exclusion_auto_factor_smooth)
        excl_r0 = factor * float(dtn64.mean()) if dtn64.size > 0 else 1.0

    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    conf_cx = conf_cy = conf_cz = 0.0
    conf_R = 1.0
    if use_conf:
        conf_cx = float(pw[:, 0].mean())
        conf_cy = float(pw[:, 1].mean())
        conf_cz = float(pw[:, 2].mean())
        conf_R = float(settings.confinement_radius_smooth)
        if conf_R <= 0.0:
            avg_bond = float(dtn64.mean()) if dtn64.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))

    aff = affinity_params(
        settings,
        "smooth",
        float(dtn64.mean()) if dtn64.size > 0 else 1.0,
        compartment,
        accessibility,
    )
    score_comp, score_brdg = init_affinity_scores(pw, aff)

    if use_heat:
        assert heat_dist is not None
        heat64 = as_f64(heat_dist)
        score_heat = float(init_heat_nb(pw, heat64, heat_weight))
    else:
        heat64 = dummy_f64()
        score_heat = 0.0

    if use_orn:
        assert char_orientations is not None
        assert anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        (
            anchor_ar,
            nbr_offsets,
            nbr_indices,
            nbr_weights,
            orn_is_L,
            bead_to_anchor_k,
            anchor_orn,
            score_orn,
        ) = prepare_orientation(
            pw,
            fixed,
            char_orientations,
            anchor_neighbors,
            anchor_neighbor_weights,
            motif_weight,
            motifs_symmetric,
        )
    else:
        anchor_ar = dummy_i32()
        nbr_offsets = np.zeros(2, dtype=np.int32)
        nbr_indices = dummy_i32()
        nbr_weights = np.zeros(1, dtype=np.float64)
        orn_is_L = np.zeros(1, dtype=np.bool_)
        bead_to_anchor_k = cast(I32Array, np.full(n, -1, dtype=np.int32))
        anchor_orn = np.zeros((1, 3), dtype=np.float64)
        score_orn = 0.0

    score_struct = float(init_smooth_nb(pw, dtn64, stretch_k, squeeze_k, ang_k, dist_w, ang_w))
    score_excl = (
        float(
            init_excl_nb(
                pw,
                excl_r0,
                float(settings.exclusion_weight),
                int(settings.exclusion_skip_neighbors),
            )
        )
        if use_excl
        else 0.0
    )
    score_conf = (
        float(
            init_confine_nb(
                pw, conf_cx, conf_cy, conf_cz, conf_R, float(settings.confinement_weight)
            )
        )
        if use_conf
        else 0.0
    )

    score = run_outer_loop(
        pw=pw,
        movable=movable,
        struct_type=STRUCT_CHAIN,
        exp_mat=dummy_f64(),
        dtn=dtn64,
        skip_mat=dummy_bool(),
        stretch_k=stretch_k,
        squeeze_k=squeeze_k,
        ang_k=ang_k,
        dist_w=dist_w,
        ang_w=ang_w,
        struct_delta_factor=1.0,
        use_heat=use_heat,
        heat_dist=heat64,
        heat_weight=heat_weight,
        use_orn=use_orn,
        orn_is_L=orn_is_L,
        anchor_ar=anchor_ar,
        nbr_offsets=nbr_offsets,
        nbr_indices=nbr_indices,
        nbr_weights=nbr_weights,
        anchor_orn=anchor_orn,
        bead_to_anchor_k=bead_to_anchor_k,
        motif_weight=motif_weight,
        motifs_symmetric=motifs_symmetric,
        use_excl=use_excl,
        excl_r0=excl_r0,
        excl_weight=float(settings.exclusion_weight),
        excl_skip=int(settings.exclusion_skip_neighbors),
        use_conf=use_conf,
        conf_cx=conf_cx,
        conf_cy=conf_cy,
        conf_cz=conf_cz,
        conf_R=conf_R,
        conf_weight=float(settings.confinement_weight),
        step_size=step_size,
        T=float(settings.max_temp_smooth),
        dt=float(settings.dt_temp_smooth),
        jump_scale=float(settings.jump_scale_smooth),
        jump_coef=float(settings.jump_coef_smooth),
        stop_steps=int(settings.mc_stop_steps_smooth),
        stop_improvement=float(settings.mc_stop_improvement_smooth),
        stop_successes=int(settings.mc_stop_successes_smooth),
        strict_better=True,
        score_eps=1e-6,
        stop_when_ratio_above=2.0,
        score_struct=score_struct,
        score_heat=score_heat,
        score_orn=score_orn,
        score_excl=score_excl,
        score_conf=score_conf,
        use_comp=aff.use_comp,
        comp_cls=aff.comp_cls,
        comp_r0=aff.comp_r0,
        comp_weight=aff.comp_weight,
        comp_ea=aff.comp_ea,
        comp_eb=aff.comp_eb,
        use_brdg=aff.use_brdg,
        brdg_a=aff.brdg_a,
        brdg_r0=aff.brdg_r0,
        brdg_weight=aff.brdg_weight,
        score_comp=score_comp,
        score_brdg=score_brdg,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
