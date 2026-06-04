"""Arc-MC (numba): anchor springs to expected distances.

`mc_arcs_numba` is the only entry here.  It drives the shared unified kernel
(`common._run_outer_loop` with `STRUCT_ARCS`) with single-counted structure
(delta factor 1) plus optional excluded volume / confinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from gnome3d.mc.numba.common import (
    as_f64,
    dummy_bool,
    dummy_f64,
    dummy_i32,
    run_outer_loop,
)
from gnome3d.mc.numba.terms import (
    STRUCT_ARCS,
    init_arcs_nb,
    init_confine_nb,
    init_excl_nb,
)
from gnome3d.types import I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings


def mc_arcs_numba(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
) -> float:
    """Numba simulated-annealing implementation for arc-MC.  Single-counted
    structure (delta factor 1). Mirrors Reference LooperSolver::MonteCarloArcs().
    Called by `gnome3d.mc.mc_arcs` when `settings.mc_backend != "jax"`.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    pw = as_f64(pos)
    exp64 = as_f64(exp_dist_mat)

    stretch_k = float(settings.spring_stretch_arcs)
    squeeze_k = float(settings.spring_squeeze_arcs)

    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    excl_r0 = float(settings.exclusion_radius_arcs)
    if use_excl and excl_r0 <= 0.0:
        pos_mask = exp64 > 1e-6
        factor = float(settings.exclusion_auto_factor_arcs)
        excl_r0 = factor * float(exp64[pos_mask].mean()) if pos_mask.any() else 1.0

    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    conf_cx = conf_cy = conf_cz = 0.0
    conf_R = 1.0
    if use_conf:
        conf_cx = float(pw[:, 0].mean())
        conf_cy = float(pw[:, 1].mean())
        conf_cz = float(pw[:, 2].mean())
        conf_R = float(settings.confinement_radius_arcs)
        if conf_R <= 0.0:
            pos_mask = exp64 > 1e-6
            avg_bond = float(exp64[pos_mask].mean()) if pos_mask.any() else 1.0
            pf = float(settings.confinement_packing_factor_arcs)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))

    movable: I64Array = np.arange(n, dtype=np.int64)
    score_struct = float(init_arcs_nb(pw, exp64, stretch_k, squeeze_k))
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
        struct_type=STRUCT_ARCS,
        exp_mat=exp64,
        dtn=dummy_f64((1,)),
        skip_mat=dummy_bool(),
        stretch_k=stretch_k,
        squeeze_k=squeeze_k,
        ang_k=0.0,
        dist_w=1.0,
        ang_w=1.0,
        struct_delta_factor=1.0,
        use_heat=False,
        heat_dist=dummy_f64(),
        heat_weight=0.0,
        use_orn=False,
        orn_is_L=np.zeros(1, dtype=np.bool_),
        anchor_ar=dummy_i32(),
        nbr_offsets=np.zeros(2, dtype=np.int32),
        nbr_indices=dummy_i32(),
        nbr_weights=np.zeros(1, dtype=np.float64),
        anchor_orn=np.zeros((1, 3), dtype=np.float64),
        bead_to_anchor_k=cast(I32Array, np.full(n, -1, dtype=np.int32)),
        motif_weight=0.0,
        motifs_symmetric=True,
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
        T=float(settings.max_temp),
        dt=float(settings.dt_temp),
        jump_scale=float(settings.jump_scale),
        jump_coef=float(settings.jump_coef),
        stop_steps=int(settings.mc_stop_steps),
        stop_improvement=float(settings.mc_stop_improvement),
        stop_successes=int(settings.mc_stop_successes),
        strict_better=False,
        score_eps=1e-5,
        stop_when_ratio_above=0.9999,
        score_struct=score_struct,
        score_heat=0.0,
        score_orn=0.0,
        score_excl=score_excl,
        score_conf=score_conf,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
