"""The smooth MC on numba: chain bonds and angles, with optional orientation, heat, excluded
volume and confinement.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.cells import BUF, build_grid, grid_shape
from gnome3d.mc.numba.common import (
    NO_F64,
    NO_F64_3,
    NO_F64_N3,
    NO_I32,
    NO_I64_3,
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
    init_excl_cells_nb,
    init_excl_nb,
    init_heat_nb,
    init_smooth_nb,
    local_heat_nb,
    local_smooth_nb,
    njit,
)
from gnome3d.types import BoolArray, F64Array, I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")

# The chain length from which the excluded volume term runs on the cell grid. The grid gives the
# same sum.
GRID_MIN_BEADS = 128


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
    """Runs one batch of smooth MC steps of one chain with chain bonds, angles and optional heat."""
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

        ok = score_new < score  # strict less than
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
    """Runs K independent smooth MC chains in parallel. Each `prange` iteration has its own
    thread local RNG.
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
    """Runs K chains in parallel and keeps the best. Supports chain bonds and optional heat
    only. The caller checks that the other terms are off.
    """
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
    on_round: Callable[[F64Array], None] | None = None,
    centre: BoolArray | None = None,
) -> float:
    """Anneals the chain in place and returns the final score.

    Fixed beads do not move. With `mc_smooth_chains` above 1 and only chain bonds and heat on, K
    chains run in parallel and the best is kept. A fixed bead of `centre` keeps the centre of
    mass of itself, the bead before it and the bead after it: it moves with its two neighbours
    taking minus half of its move each, and a neighbour moves with the other neighbour taking
    minus its move. Two beads of `centre` are three beads or more apart and no bead beside one
    is fixed.
    """
    n = pos.shape[0]
    if n <= 2:
        return 0.0

    if int(settings.mc_smooth_chains) > 1:
        simple_config = (
            not bool(getattr(settings, "smooth_hard_wall", False))
            and float(getattr(settings, "smooth_anchor_cap", 0.0)) == 0.0
            and char_orientations is None
            and not (
                bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
            )
            and not (bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth))
            and compartment is None
            and centre is None
            and float(settings.motif_bond_limit) <= 0.0
        )
        if simple_config:
            return _mc_smooth_multichain(pos, dtn, fixed, step_size, settings, heat_dist)

    # Under the cap the anchors move too, each within `cap` mean bonds of its place.
    cap_frac = float(getattr(settings, "smooth_anchor_cap", 0.0))
    use_cap = cap_frac > 0.0
    use_wall = bool(getattr(settings, "smooth_hard_wall", False))
    if use_cap:
        cap_r: F64Array = np.where(fixed, cap_frac * float(np.mean(dtn)), 0.0).astype(np.float64)
        cap_home: F64Array = np.ascontiguousarray(pos, dtype=np.float64).copy()
        movable: I64Array = np.arange(pos.shape[0], dtype=np.int64)
    else:
        cap_r = NO_F64
        cap_home = NO_F64_N3
        free = ~fixed if centre is None else (~fixed | centre)
        movable = np.ascontiguousarray(np.where(free)[0], dtype=np.int64)
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
    )
    score_comp = init_affinity_scores(pw, aff)

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
    # The cell grid sums the same pairs in the same order as the full scan.
    excl_w = float(settings.exclusion_weight)
    excl_skip_n = int(settings.exclusion_skip_neighbors)
    use_cells = (
        use_excl and bool(getattr(settings, "mc_neighbour_grid", True)) and n >= GRID_MIN_BEADS
    )
    cell_lo = NO_F64_3
    cell_dim = NO_I64_3
    cell_size = 1.0
    cell_head = NO_I32
    cell_next = NO_I32
    cell_where = NO_I32
    cell_buf = NO_I32
    score_excl = 0.0
    if use_cells:
        cell_lo, cell_dim, cell_size = grid_shape(pw, excl_r0)
        cell_head, cell_next, cell_where = build_grid(pw, cell_lo, cell_dim, cell_size)
        cell_buf = np.empty(BUF, dtype=np.int32)
        score_excl = float(
            init_excl_cells_nb(
                pw,
                excl_r0,
                excl_w,
                excl_skip_n,
                cell_lo,
                cell_dim,
                cell_size,
                cell_head,
                cell_next,
                cell_buf,
            )
        )
        if score_excl < 0.0:  # a bead had more neighbours than the buffer holds
            use_cells = False
    if use_excl and not use_cells:
        score_excl = float(init_excl_nb(pw, excl_r0, excl_w, excl_skip_n))
    score_conf = (
        float(
            init_confine_nb(
                pw, conf_cx, conf_cy, conf_cz, conf_R, float(settings.confinement_weight)
            )
        )
        if use_conf
        else 0.0
    )

    use_centre = centre is not None and not use_cap
    centre_of: I32Array = NO_I32
    site_of: I32Array = NO_I32
    if centre is not None and use_centre:
        owner = cast(I32Array, np.full(n, -1, dtype=np.int32))
        for c in np.flatnonzero(centre):
            owner[c - 1 : c + 2] = c
        centre_of = owner
        sites = cast(I32Array, np.full(n, -1, dtype=np.int32))
        if use_orn:
            sites[anchor_ar] = np.arange(anchor_ar.shape[0], dtype=np.int32)
        site_of = sites

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
        score_comp=score_comp,
        use_cells=use_cells,
        cell_lo=cell_lo,
        cell_dim=cell_dim,
        cell_size=cell_size,
        cell_head=cell_head,
        cell_next=cell_next,
        cell_where=cell_where,
        cell_buf=cell_buf,
        use_wall=use_wall,
        use_cap=use_cap,
        cap_home=cap_home,
        cap_r=cap_r,
        on_round=on_round,
        use_centre=use_centre,
        centre_of=centre_of,
        site_of=site_of,
        orn_limit=float(settings.motif_bond_limit) if use_orn else 0.0,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
