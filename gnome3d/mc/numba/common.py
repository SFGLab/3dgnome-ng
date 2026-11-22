"""The driver of the numba kernels and array helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from gnome3d import log
from gnome3d.mc.numba.terms import (
    NO_I32_N1,
    NO_W,
    ORIENTATION_STRETCH,
    batch_mc_nb,
    calc_orientation_stretch_nb,
    init_affinity_nb,
    score_orientation_full_nb,
)
from gnome3d.types import BoolArray, F32Array, F64Array, I8Array, I32Array, I64Array

LOG = log.get("mc.numba")

# A round of the smooth stage that ends above this many times the score before it ends the run,
# in both backends.
SMOOTH_STOP_RATIO = 2.0


def as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


def dummy_f64(shape: tuple[int, ...] = (1, 1)) -> F64Array:
    return np.zeros(shape, dtype=np.float64)


def dummy_bool(shape: tuple[int, ...] = (1, 1)) -> BoolArray:
    return np.zeros(shape, dtype=np.bool_)


def dummy_i32(shape: tuple[int, ...] = (1,)) -> I32Array:
    return np.zeros(shape, dtype=np.int32)


# Placeholders for the arrays of terms that are off. They give numba a concrete type and are not
# read.
NO_I8: I8Array = np.zeros(1, dtype=np.int8)
NO_F64: F64Array = np.zeros(1, dtype=np.float64)
NO_MAT: F64Array = np.zeros((1, 1), dtype=np.float64)
NO_F64_3: F64Array = np.zeros(3, dtype=np.float64)
NO_F64_N3: F64Array = np.zeros((1, 3), dtype=np.float64)
NO_I64_3: I64Array = np.ones(3, dtype=np.int64)
NO_I32: I32Array = np.zeros(1, dtype=np.int32)


class AffinityParams(NamedTuple):
    """The compartment kernel arguments of one MC level."""

    use_comp: bool
    comp_cls: I8Array
    comp_r0: float
    comp_weight: float
    comp_ea: float
    comp_eb: float

    @property
    def any_on(self) -> bool:
        return self.use_comp


def affinity_params(
    settings: Any,
    level: str,
    bond_scale: float,
    compartment: np.ndarray[Any, Any] | None,
) -> AffinityParams:
    """Resolves the compartment term of one MC level.

    The term is on only when its master flag, its apply flag of the level and its track are
    present. A radius of 0 becomes `auto_factor * bond_scale`.

    Parameters
    ----------
    level
        "smooth", "ib" or "heatmap".
    bond_scale
        The mean target distance between beads at this level.
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
    )


def init_affinity_scores(pw: F64Array, aff: AffinityParams) -> float:
    """Returns the compartment score of the starting positions."""
    if not aff.any_on:
        return 0.0
    c = init_affinity_nb(
        pw,
        aff.use_comp,
        aff.comp_cls,
        aff.comp_r0,
        aff.comp_weight,
        aff.comp_ea,
        aff.comp_eb,
    )
    return float(c)


def orientation_weight(settings: Any) -> float:
    """Returns the weight of the orientation term: the energy of the bonds of the stretch of a
    site, `2 * ORIENTATION_STRETCH` of them, each one target too long. A site at its worst
    misfit has that energy, design/orientation-term.md.
    """
    return (
        2.0
        * ORIENTATION_STRETCH
        * float(settings.spring_stretch)
        * float(settings.smooth_dist_weight)
    )


def loop_shares(nbr_offsets: I32Array, nbr_indices: I32Array, nbr_weights: F64Array) -> F64Array:
    """Returns the weight of each loop over the larger of the sums of the weights at its two
    anchors. The loops of an anchor then sum to one at most and a loop has one weight seen from
    either anchor.
    """
    count = int(nbr_offsets.shape[0]) - 1
    total: F64Array = np.array(
        [float(nbr_weights[nbr_offsets[k] : nbr_offsets[k + 1]].sum()) for k in range(count)],
        dtype=np.float64,
    )
    out: F64Array = np.zeros_like(nbr_weights)
    for k in range(count):
        for off in range(int(nbr_offsets[k]), int(nbr_offsets[k + 1])):
            largest = max(float(total[k]), float(total[int(nbr_indices[off])]))
            out[off] = float(nbr_weights[off]) / largest if largest > 0.0 else 0.0
    return out


def sites_of_moves(
    n: int,
    anchor_ar: I32Array,
    nbr_offsets: I32Array,
    centre_of: I32Array | None,
    stretch: int = ORIENTATION_STRETCH,
) -> I32Array:
    """Returns for each bead the sites that its move reads, a row of site indices ended by -1.

    A bead moves alone, or with the two other beads of its node when `centre_of` gives one. A
    site reads a move when one of the moved beads is within `stretch` beads of it. A site
    without a loop is left out.
    """
    rows: list[list[int]] = []
    live = [k for k in range(len(anchor_ar)) if nbr_offsets[k + 1] > nbr_offsets[k]]
    at: I32Array = np.array([int(anchor_ar[k]) for k in live], dtype=np.int32)
    for p in range(n):
        c = -1 if centre_of is None else int(centre_of[p])
        lo, hi = (p, p) if c < 0 else (c - 1, c + 1)
        first = int(np.searchsorted(at, lo - stretch, side="left"))
        last = int(np.searchsorted(at, hi + stretch, side="right"))
        rows.append([live[i] for i in range(first, last)])
    width = max((len(r) for r in rows), default=0) + 1
    table: I32Array = np.full((n, width), -1, dtype=np.int32)
    for p, r in enumerate(rows):
        table[p, : len(r)] = r
    return table


def prepare_orientation(
    pw: F64Array,
    fixed: np.ndarray[Any, Any],
    char_orientations: np.ndarray[Any, Any],
    anchor_neighbors: dict[int, list[int]],
    anchor_neighbor_weights: dict[int, list[float]],
    motif_weight: float,
    motifs_symmetric: bool,
) -> tuple[I32Array, I32Array, I32Array, F64Array, BoolArray, F64Array, float]:
    """Builds the orientation arrays of the kernel from the Python dicts.

    The vector of an anchor is over `ORIENTATION_STRETCH` beads to either side. The weight of
    a loop is its weight over the larger of the sums of the weights at its two anchors.

    Returns (anchor_ar, nbr_offsets, nbr_indices, nbr_weights, orn_is_L, anchor_orn, score_orn).
    """
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
    nbr_weights = loop_shares(nbr_offsets, nbr_indices, nbr_weights)
    orn_is_L: BoolArray = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
    anchor_orn: F64Array = np.zeros((n_anchors, 3), dtype=np.float64)
    for k in range(n_anchors):
        ar = int(anchor_ar[k])
        anchor_orn[k] = calc_orientation_stretch_nb(
            pw, ar, n, bool(orn_is_L[ar]), ORIENTATION_STRETCH
        )
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
    bg_weight: float = 0.0,
    use_comp: bool = False,
    comp_cls: I8Array = NO_I8,
    comp_r0: float = 1.0,
    comp_weight: float = 0.0,
    comp_ea: float = 0.0,
    comp_eb: float = 0.0,
    score_comp: float = 0.0,
    use_cells: bool = False,
    cell_lo: F64Array = NO_F64_3,
    cell_dim: I64Array = NO_I64_3,
    cell_size: float = 1.0,
    cell_head: I32Array = NO_I32,
    cell_next: I32Array = NO_I32,
    cell_where: I32Array = NO_I32,
    cell_buf: I32Array = NO_I32,
    use_wall: bool = False,
    use_cap: bool = False,
    cap_home: F64Array = NO_F64_N3,
    cap_r: F64Array = NO_F64,
    on_round: Callable[[F64Array], None] | None = None,
    use_arc_w: bool = False,
    arc_w: F32Array = NO_W,
    use_centre: bool = False,
    centre_of: I32Array = NO_I32,
    move_sites: I32Array = NO_I32_N1,
) -> float:
    """Runs the kernel until convergence and returns the final total score."""
    score = score_struct + score_heat + score_orn + score_excl + score_conf + score_comp
    ms_score = score
    step_i = 0
    round_i = 0
    while True:
        (
            T,
            score_struct,
            score_heat,
            score_orn,
            score_excl,
            score_conf,
            score_comp,
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
            rep_inv_cutoff,
            bg_weight,
            use_cells,
            cell_lo,
            cell_dim,
            cell_size,
            cell_head,
            cell_next,
            cell_where,
            cell_buf,
            use_wall,
            use_cap,
            cap_home,
            cap_r,
            use_arc_w,
            arc_w,
            use_centre,
            centre_of,
            move_sites,
        )
        score = score_struct + score_heat + score_orn + score_excl + score_conf + score_comp
        step_i += stop_steps
        round_i += 1
        if on_round is not None:
            on_round(pw)
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
