"""Shared numba driver + array helpers.

`_run_outer_loop` is the Python-side convergence loop every public entry uses:
it runs `terms._batch_mc_nb` for `stop_steps` at a time and checks the reference-style
stop condition.  `_prepare_orientation` builds the CSR orientation arrays the
kernel needs from the Python neighbour dicts.  `_as_f64` / `_dummy_*` produce the
contiguous / placeholder arrays the kernel's fixed signature expects.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np

from gnome3d import log
from gnome3d.mc.numba.terms import (
    batch_mc_nb,
    decayed_step_nb,
    init_affinity_nb,
    score_orientation_full_nb,
)
from gnome3d.types import BoolArray, F64Array, I8Array, I32Array, I64Array

LOG = log.get("mc.numba")


def as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


def dummy_f64(shape: tuple[int, ...] = (1, 1)) -> F64Array:
    return np.zeros(shape, dtype=np.float64)


def dummy_bool(shape: tuple[int, ...] = (1, 1)) -> BoolArray:
    return np.zeros(shape, dtype=np.bool_)


def dummy_i32(shape: tuple[int, ...] = (1,)) -> I32Array:
    return np.zeros(shape, dtype=np.int32)


# Placeholders for disabled optional-term arrays.  A disabled term is never
# indexed, so these exist only to give numba a concrete type.  Never written to.
NO_I8: I8Array = np.zeros(1, dtype=np.int8)
NO_F64: F64Array = np.zeros(1, dtype=np.float64)
NO_MAT: F64Array = np.zeros((1, 1), dtype=np.float64)
NO_F64_3: F64Array = np.zeros(3, dtype=np.float64)
NO_I64_3: I64Array = np.ones(3, dtype=np.int64)
NO_I32: I32Array = np.zeros(1, dtype=np.int32)


class AffinityParams(NamedTuple):
    """Resolved compartment + bridging kernel arguments for one MC level."""

    use_comp: bool
    comp_cls: I8Array
    comp_r0: float
    comp_weight: float
    comp_ea: float
    comp_eb: float
    use_brdg: bool
    brdg_a: F64Array
    brdg_r0: float
    brdg_weight: float

    @property
    def any_on(self) -> bool:
        return self.use_comp or self.use_brdg


def affinity_params(
    settings: Any,
    level: str,
    bond_scale: float,
    compartment: np.ndarray[Any, Any] | None,
    accessibility: np.ndarray[Any, Any] | None,
) -> AffinityParams:
    """Resolve the affinity terms for one MC level.

    A term is on only when its master flag, its per-level apply flag and its
    track are all present, so a missing track silently leaves it off rather than
    scoring against zeros.  A radius of 0 auto-derives as
    `auto_factor * bond_scale`, matching the excluded-volume convention.

    Parameters
    ----------
    level : str
        One of "smooth", "ib", "heatmap".  Selects the per-level settings.
    bond_scale : float
        This level's mean bead-bead target distance, the auto-radius base.
    """
    have_c = compartment is not None and compartment.size > 0
    use_comp = (
        bool(settings.use_compartments)
        and bool(getattr(settings, f"compartment_apply_to_{level}"))
        and have_c
    )
    comp_r0 = float(getattr(settings, f"compartment_radius_{level}"))
    if use_comp and comp_r0 <= 0.0:
        comp_r0 = float(getattr(settings, f"compartment_auto_factor_{level}")) * bond_scale
    comp_r0 = comp_r0 if comp_r0 > 0.0 else 1.0

    have_a = accessibility is not None and accessibility.size > 0
    use_brdg = (
        bool(settings.use_bridging)
        and bool(getattr(settings, f"bridging_apply_to_{level}"))
        and have_a
    )
    brdg_r0 = float(getattr(settings, f"bridging_radius_{level}"))
    if use_brdg and brdg_r0 <= 0.0:
        brdg_r0 = float(getattr(settings, f"bridging_auto_factor_{level}")) * bond_scale
    brdg_r0 = brdg_r0 if brdg_r0 > 0.0 else 1.0

    return AffinityParams(
        use_comp=use_comp,
        comp_cls=(
            np.ascontiguousarray(compartment, dtype=np.int8)
            if use_comp and compartment is not None
            else NO_I8
        ),
        comp_r0=comp_r0,
        comp_weight=float(settings.compartment_weight),
        comp_ea=float(settings.compartment_energy_a),
        comp_eb=float(settings.compartment_energy_b),
        use_brdg=use_brdg,
        brdg_a=(as_f64(accessibility) if use_brdg and accessibility is not None else NO_F64),
        brdg_r0=brdg_r0,
        brdg_weight=float(settings.bridging_weight),
    )


def init_affinity_scores(pw: F64Array, aff: AffinityParams) -> tuple[float, float]:
    """Full compartment and bridging scores for the starting positions."""
    if not aff.any_on:
        return 0.0, 0.0
    c, b = init_affinity_nb(
        pw,
        aff.use_comp,
        aff.comp_cls,
        aff.comp_r0,
        aff.comp_weight,
        aff.comp_ea,
        aff.comp_eb,
        aff.use_brdg,
        aff.brdg_a,
        aff.brdg_r0,
        aff.brdg_weight,
    )
    return float(c), float(b)


def prepare_orientation(
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
        score_orientation_full_nb(
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


def run_outer_loop(
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
    rep_inv_cutoff: float = 0.0,
    # Affinity terms default to off so a stage that never uses them (arcs) needs
    # no extra arguments.  The dummy arrays are only there to fix numba's types.
    use_comp: bool = False,
    comp_cls: I8Array = NO_I8,
    comp_r0: float = 1.0,
    comp_weight: float = 0.0,
    comp_ea: float = 0.0,
    comp_eb: float = 0.0,
    use_brdg: bool = False,
    brdg_a: F64Array = NO_F64,
    brdg_r0: float = 1.0,
    brdg_weight: float = 0.0,
    score_comp: float = 0.0,
    score_brdg: float = 0.0,
    use_excl_mat: bool = False,
    excl_r0_mat: F64Array = NO_MAT,
    use_cells: bool = False,
    cell_lo: F64Array = NO_F64_3,
    cell_dim: I64Array = NO_I64_3,
    cell_size: float = 1.0,
    cell_head: I32Array = NO_I32,
    cell_next: I32Array = NO_I32,
    cell_where: I32Array = NO_I32,
    cell_buf: I32Array = NO_I32,
    step_decay: float = 1.0,
    step_decay_floor: float = 0.1,
) -> float:
    """Drive the unified kernel until convergence; return the final total score."""
    score = (
        score_struct + score_heat + score_orn + score_excl + score_conf + score_comp + score_brdg
    )
    ms_score = score
    step_i = 0
    round_i = 0
    step0 = float(step_size)
    while True:
        step_size = decayed_step_nb(step0, step_decay, step_decay_floor, round_i)
        (
            T,
            score_struct,
            score_heat,
            score_orn,
            score_excl,
            score_conf,
            score_comp,
            score_brdg,
            n_ok,
        ) = batch_mc_nb(
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
            use_comp,
            comp_cls,
            comp_r0,
            comp_weight,
            comp_ea,
            comp_eb,
            use_brdg,
            brdg_a,
            brdg_r0,
            brdg_weight,
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
            score_comp,
            score_brdg,
            rep_inv_cutoff,
            use_excl_mat,
            excl_r0_mat,
            use_cells,
            cell_lo,
            cell_dim,
            cell_size,
            cell_head,
            cell_next,
            cell_where,
            cell_buf,
        )
        score = (
            score_struct
            + score_heat
            + score_orn
            + score_excl
            + score_conf
            + score_comp
            + score_brdg
        )
        step_i += stop_steps
        round_i += 1
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
