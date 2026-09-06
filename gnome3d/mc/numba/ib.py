"""IB-centroid chain MC (numba).

`mc_ib_numba` is a peer of `mc_smooth_numba` (not a sub-mode): chain bonds only
(no angle / orientation / heat) plus optional IB-scale excluded volume and
confinement, driving the shared unified kernel (`STRUCT_CHAIN`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from gnome3d.mc.numba.common import (
    affinity_params,
    as_f64,
    dummy_bool,
    dummy_f64,
    dummy_i32,
    init_affinity_scores,
    run_outer_loop,
)
from gnome3d.mc.numba.terms import (
    STRUCT_CHAIN,
    init_confine_nb,
    init_excl_nb,
    init_smooth_nb,
)
from gnome3d.types import I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings


def mc_ib_numba(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    compartment: np.ndarray[Any, Any] | None = None,
    accessibility: np.ndarray[Any, Any] | None = None,
) -> float:
    """Numba simulated-annealing implementation for IB-centroid chain MC.
    Peer to mc_smooth (not a sub-mode of it).  Called by `gnome3d.mc.mc_ib`.

    Energy: chain bonds (no angle term, no orientation) + optional IB-scale excluded volume +
    optional IB-scale confinement.  All IBs move
    (no fixed set). Reads only its own settings: `spring_*_ib`, `dist_weight_ib`,
    `max_temp_ib`/`dt_temp_ib`/`jump_*_ib`/`mc_stop_*_ib` under [simulation_ib],
    plus the `*_ib` knobs under [excluded_volume] and [confinement].
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    pw = as_f64(pos)
    dtn64 = as_f64(dtn)
    movable: I64Array = np.arange(n, dtype=np.int64)

    stretch_k = float(settings.spring_stretch_ib)
    squeeze_k = float(settings.spring_squeeze_ib)
    dist_w = float(settings.dist_weight_ib)
    # IB chain has no angle term: too few IBs for stable angle statistics,
    # and the chain is meant to be flexible (loops curl back on themselves).
    ang_k = 0.0
    ang_w = 0.0

    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_ib)
    excl_r0 = float(settings.exclusion_radius_ib)
    if use_excl and excl_r0 <= 0.0:
        factor = float(settings.exclusion_auto_factor_ib)
        excl_r0 = factor * float(dtn64.mean()) if dtn64.size > 0 else 1.0

    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_ib)
    conf_cx = conf_cy = conf_cz = 0.0
    conf_R = 1.0
    if use_conf:
        conf_cx = float(pw[:, 0].mean())
        conf_cy = float(pw[:, 1].mean())
        conf_cz = float(pw[:, 2].mean())
        conf_R = float(settings.confinement_radius_ib)
        if conf_R <= 0.0:
            avg_bond = float(dtn64.mean()) if dtn64.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_ib)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))

    aff = affinity_params(
        settings,
        "ib",
        float(dtn64.mean()) if dtn64.size > 0 else 1.0,
        compartment,
        accessibility,
    )
    score_comp, score_brdg = init_affinity_scores(pw, aff)

    score_struct = float(init_smooth_nb(pw, dtn64, stretch_k, squeeze_k, ang_k, dist_w, ang_w))
    score_excl = (
        float(init_excl_nb(pw, excl_r0, float(settings.exclusion_weight), 1)) if use_excl else 0.0
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
        # IB chain: only skip the immediate neighbor (the bond itself) so
        # non-neighbor IBs still repel each other.
        excl_skip=1,
        use_conf=use_conf,
        conf_cx=conf_cx,
        conf_cy=conf_cy,
        conf_cz=conf_cz,
        conf_R=conf_R,
        conf_weight=float(settings.confinement_weight),
        step_size=step_size,
        T=float(settings.max_temp_ib),
        dt=float(settings.dt_temp_ib),
        jump_scale=float(settings.jump_scale_ib),
        jump_coef=float(settings.jump_coef_ib),
        stop_steps=int(settings.mc_stop_steps_ib),
        stop_improvement=float(settings.mc_stop_improvement_ib),
        stop_successes=int(settings.mc_stop_successes_ib),
        strict_better=True,
        score_eps=1e-6,
        stop_when_ratio_above=2.0,
        score_struct=score_struct,
        score_heat=0.0,
        score_orn=0.0,
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
