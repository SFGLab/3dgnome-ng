"""Numba term math + the unified MC kernel.

Every energy term (chain bonds, excluded volume, confinement, orientation,
subanchor heat, arc springs, heatmap) has a `_local_*_nb` (per-bead delta) and
`_init_*_nb` (full score) njit function here.  `_batch_mc_nb` is the single
inner kernel all four public entries run: it switches on `struct_type`
(STRUCT_ARCS / STRUCT_CHAIN / STRUCT_HEATMAP) and toggles the optional terms via
`use_*` flags.

These are the shared building blocks.  The per-kernel public entries live in
`arcs.py`, `smooth.py`, `heatmap.py`, `ib.py`; they drive this kernel through
`common._run_outer_loop`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]

from gnome3d.types import BoolArray, F64Array, I32Array, I64Array

# Typed wrapper around numba.njit so pyright sees decorated functions
# with their original signatures.  At runtime this is just numba.njit.
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


# Structure-term selector.  Kept as plain ints (numba-friendly).
STRUCT_ARCS = 0
STRUCT_CHAIN = 1
STRUCT_HEATMAP = 2


@njit(cache=True, nogil=True)
def seed_numba(seed: int) -> None:
    """Seed numba's per-thread RNG so the MC kernels are reproducible from a
    given seed.  numba keeps its own RNG state, separate from numpy's global
    and Python's `random`; this is the only way to make `np.random.*` calls
    inside @njit deterministic.  Used by the pipeline's per-node seeding."""
    np.random.seed(seed)


# Smooth MC helpers

@njit(cache=True, fastmath=True, nogil=True)
def _smooth_len_nb(
    pos: F64Array,
    dtn: F64Array,
    i: int,
    stretch_k: float,
    squeeze_k: float,
    dist_w: float,
) -> float:
    dx = pos[i, 0] - pos[i + 1, 0]
    dy = pos[i, 1] - pos[i + 1, 1]
    dz = pos[i, 2] - pos[i + 1, 2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    e = dtn[i]
    if e < 1e-6:
        e = 1e-6
    rel = (d - e) / e
    k = stretch_k if rel >= 0.0 else squeeze_k
    return rel * rel * k * dist_w


@njit(cache=True, fastmath=True, nogil=True)
def _smooth_ang_nb(pos: F64Array, i: int, ang_k: float, ang_w: float) -> float:
    v1x = pos[i, 0] - pos[i + 1, 0]
    v1y = pos[i, 1] - pos[i + 1, 1]
    v1z = pos[i, 2] - pos[i + 1, 2]
    v2x = pos[i + 1, 0] - pos[i + 2, 0]
    v2y = pos[i + 1, 1] - pos[i + 2, 1]
    v2z = pos[i + 1, 2] - pos[i + 2, 2]
    n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
    if cos_a > 1.0:
        cos_a = 1.0
    if cos_a < -1.0:
        cos_a = -1.0
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return ang * ang * ang * ang_k * ang_w


@njit(cache=True, fastmath=True, nogil=True)
def _local_smooth_nb(
    pos: F64Array,
    dtn: F64Array,
    p: int,
    n: int,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
) -> float:
    sc = 0.0
    i = p - 1
    if 0 <= i < n - 1:
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    if 0 <= p < n - 1:
        sc += _smooth_len_nb(pos, dtn, p, stretch_k, squeeze_k, dist_w)
    for off in range(-2, 1):
        i = p + off
        if 0 <= i < n - 2:
            sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _init_smooth_nb(
    pos: F64Array,
    dtn: F64Array,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
) -> float:
    n = pos.shape[0]
    sc = 0.0
    for i in range(n - 1):
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    for i in range(n - 2):
        sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


# Confinement helpers (soft spherical envelope around a center)
#
#   E(p) = weight * ((|r_p - c| - R) / R)^2   if |r_p - c| > R
#        = 0                                  otherwise
#
# Per-bead (not per-pair), single-counted globally. Delta is (curr - prev),
# no factor of 2.


@njit(cache=True, fastmath=True, nogil=True)
def _local_confine_nb(
    pos: F64Array, p: int, cx: float, cy: float, cz: float, R: float, weight: float
) -> float:
    dx = pos[p, 0] - cx
    dy = pos[p, 1] - cy
    dz = pos[p, 2] - cz
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r <= R:
        return 0.0
    rel = (r - R) / R
    return weight * rel * rel


@njit(cache=True, fastmath=True, nogil=True)
def _init_confine_nb(
    pos: F64Array, cx: float, cy: float, cz: float, R: float, weight: float
) -> float:
    n = pos.shape[0]
    err = 0.0
    for p in range(n):
        err += _local_confine_nb(pos, p, cx, cy, cz, R, weight)
    return err


# Excluded-volume helpers (harmonic soft repulsion, cutoff at r0)
#
#   E_pair(d) = weight * ((r0 - d) / r0)^2   if d < r0
#             = 0                            otherwise
#
# Normalized by r0 so `weight` is dimensionally comparable to spring constants.
# Global score double-counts pairs (matches the heat-energy convention):
# sum_{i != j, |i-j| > skip} E_pair(d_ij). Delta is 2 * (local_curr - local_prev).


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


# Orientation MC helpers (smooth-only)


@njit(cache=True, fastmath=True, nogil=True)
def _calc_orientation_nb(
    pos: F64Array, cind: int, n: int, is_L: bool
) -> tuple[float, float, float]:
    """Returns (ox, oy, oz) normalized orientation vector for anchor at cind."""
    if cind == 0:
        ox = pos[cind + 1, 0] - pos[cind, 0]
        oy = pos[cind + 1, 1] - pos[cind, 1]
        oz = pos[cind + 1, 2] - pos[cind, 2]
    elif cind == n - 1:
        ox = pos[cind, 0] - pos[cind - 1, 0]
        oy = pos[cind, 1] - pos[cind - 1, 1]
        oz = pos[cind, 2] - pos[cind - 1, 2]
    else:
        ox = pos[cind + 1, 0] - pos[cind - 1, 0]
        oy = pos[cind + 1, 1] - pos[cind - 1, 1]
        oz = pos[cind + 1, 2] - pos[cind - 1, 2]
    if is_L:
        ox = -ox
        oy = -oy
        oz = -oz
    nm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if nm > 1e-12:
        ox /= nm
        oy /= nm
        oz /= nm
    return ox, oy, oz


@njit(cache=True, fastmath=True, nogil=True)
def _score_orientation_full_nb(
    anchor_orn: F64Array,
    nbr_offsets: I32Array,
    nbr_indices: I32Array,
    nbr_weights: F64Array,
    motif_weight: float,
    symmetric: bool,
) -> float:
    """Global orientation score with arc weights; used for initialisation only."""
    n_anchors = anchor_orn.shape[0]
    err = 0.0
    for i in range(n_anchors):
        for ki in range(nbr_offsets[i], nbr_offsets[i + 1]):
            j = nbr_indices[ki]
            w = nbr_weights[ki]
            ax = anchor_orn[i, 0]
            ay = anchor_orn[i, 1]
            az = anchor_orn[i, 2]
            bx = anchor_orn[j, 0]
            by = anchor_orn[j, 1]
            bz = anchor_orn[j, 2]
            if not symmetric:
                bx = -bx
                by = -by
                bz = -bz
            dot = ax * bx + ay * by + az * bz
            ang = 1.0 - (dot + 1.0) * 0.5
            err += ang * ang * w
    return err * motif_weight


@njit(cache=True, fastmath=True, nogil=True)
def _local_score_orientation_nb(
    anchor_orn: F64Array,
    k: int,
    nbr_offsets: I32Array,
    nbr_indices: I32Array,
    nbr_weights: F64Array,
    motif_weight: float,
    symmetric: bool,
) -> float:
    """Local orientation score for anchor k, weighted by per-arc weights.
    Used for the incremental update: score_orn += 2*(local_curr - local_prev).
    The weights make this delta exact w.r.t. _score_orientation_full_nb - no drift.
    Diverges from Reference calcScoreOrientation(orn, anchor_index), which is unweighted
    and therefore drifts.
    """
    err = 0.0
    for ki in range(nbr_offsets[k], nbr_offsets[k + 1]):
        j = nbr_indices[ki]
        w = nbr_weights[ki]
        ax = anchor_orn[k, 0]
        ay = anchor_orn[k, 1]
        az = anchor_orn[k, 2]
        bx = anchor_orn[j, 0]
        by = anchor_orn[j, 1]
        bz = anchor_orn[j, 2]
        if not symmetric:
            bx = -bx
            by = -by
            bz = -bz
        dot = ax * bx + ay * by + az * bz
        ang = 1.0 - (dot + 1.0) * 0.5
        err += ang * ang * w
    return err * motif_weight


# Heat MC helpers (smooth-only, subanchor heatmap)


@njit(cache=True, fastmath=True, nogil=True)
def _local_heat_nb(pos: F64Array, heat_dist: F64Array, p: int, heat_weight: float) -> float:
    """Local heat score for bead p vs all others.
    Mirrors Reference calcScoreSubanchorHeatmap(int moved) - sums all i != p.
    """
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        if i == p:
            continue
        exp_d = heat_dist[i, p]
        if exp_d < 1e-6:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (d - exp_d) / exp_d
        err += rel * rel
    return err * heat_weight


@njit(cache=True, fastmath=True, nogil=True)
def _init_heat_nb(pos: F64Array, heat_dist: F64Array, heat_weight: float) -> float:
    """Global heat score (double-counts pairs, matching Reference calcScoreSubanchorHeatmap())."""
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        row_err = 0.0
        for j in range(n):
            if i == j:
                continue
            exp_d = heat_dist[i, j]
            if exp_d < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            rel = (d - exp_d) / exp_d
            row_err += rel * rel
        err += row_err
    return err * heat_weight


# Arcs MC helpers


@njit(cache=True, fastmath=True, nogil=True)
def _local_arcs_nb(
    pos: F64Array, exp: F64Array, p: int, stretch_k: float, squeeze_k: float
) -> float:
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if i == p:
            continue
        e = exp[i, p]
        dx = pos[p, 0] - pos[i, 0]
        dy = pos[p, 1] - pos[i, 1]
        dz = pos[p, 2] - pos[i, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if e < 0.0:
            sc += 1.0 / (d if d > 1e-10 else 1e-10)
        elif e >= 1e-6:
            rel = (d - e) / e
            sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _init_arcs_nb(pos: F64Array, exp: F64Array, stretch_k: float, squeeze_k: float) -> float:
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        row_sc = 0.0
        for j in range(i + 1, n):
            e = exp[i, j]
            if -1e-10 < e < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if e < 0.0:
                row_sc += 1.0 / (d if d > 1e-10 else 1e-10)
            else:
                rel = (d - e) / e
                row_sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
        sc += row_sc
    return sc


# Heatmap MC helpers


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
    """O(N^2) init - parallelised over rows; sum reduction is auto-handled."""
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


# Unified MC kernel
#
# One numba kernel handles all four stages.  Structure-term variant is
# selected by `struct_type` (STRUCT_ARCS / STRUCT_CHAIN / STRUCT_HEATMAP).
# Optional energy terms (heat / orn / excl / conf) are toggled by their use_*
# flags; their data arrays must still be valid-typed (any shape) since
# disabled-term arrays are not indexed.
#
# Delta conventions:
#   * structure  : score += struct_delta_factor * (local_curr - local_prev)
#                  (1.0 for arcs/chain, 2.0 for heatmap)
#   * heat       : score += 2 * (curr - prev)
#   * orientation: score += 2 * (curr - prev)
#   * excluded   : score += 2 * (curr - prev)
#   * confinement: score += 1 * (curr - prev)
#
# Acceptance: ok = (score_new < score) if strict_better else (score_new <= score)
# Smooth uses strict (preserves prior behaviour); arcs/heatmap use non-strict.


@njit(cache=True, fastmath=True, nogil=True)
def _batch_mc_nb(
    pos: F64Array,
    movable: I64Array,
    # ---- Structure term ----
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
    # ---- Heat term ----
    use_heat: bool,
    heat_dist: F64Array,
    heat_weight: float,
    # ---- Orientation term ----
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
    # ---- Excluded volume term ----
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    # ---- Confinement term ----
    use_conf: bool,
    conf_cx: float,
    conf_cy: float,
    conf_cz: float,
    conf_R: float,
    conf_weight: float,
    # ---- MC schedule ----
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    strict_better: bool,
    # ---- Initial scores ----
    score_struct: float,
    score_heat: float,
    score_orn: float,
    score_excl: float,
    score_conf: float,
) -> tuple[float, float, float, float, float, float, int]:
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    score = score_struct + score_heat + score_orn + score_excl + score_conf

    for _ in range(n_steps):
        p: int = int(movable[np.random.randint(0, n_mov)])
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        # --- prev local scores ---
        if struct_type == STRUCT_ARCS:
            loc_struct_prev = _local_arcs_nb(pos, exp_mat, p, stretch_k, squeeze_k)
        elif struct_type == STRUCT_CHAIN:
            loc_struct_prev = _local_smooth_nb(
                pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
        else:  # STRUCT_HEATMAP
            loc_struct_prev = _local_heatmap_nb(pos, exp_mat, skip_mat[:, p], p)

        loc_heat_prev = 0.0
        if use_heat:
            loc_heat_prev = _local_heat_nb(pos, heat_dist, p, heat_weight)

        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        loc_conf_prev = 0.0
        if use_conf:
            loc_conf_prev = _local_confine_nb(
                pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight
            )

        orn_k: int = -1
        prev_ox = 0.0
        prev_oy = 0.0
        prev_oz = 0.0
        loc_orn_prev = 0.0
        if use_orn:
            orn_k = int(bead_to_anchor_k[p])
            if orn_k >= 0:
                prev_ox = anchor_orn[orn_k, 0]
                prev_oy = anchor_orn[orn_k, 1]
                prev_oz = anchor_orn[orn_k, 2]
                loc_orn_prev = _local_score_orientation_nb(
                    anchor_orn,
                    orn_k,
                    nbr_offsets,
                    nbr_indices,
                    nbr_weights,
                    motif_weight,
                    motifs_symmetric,
                )

        # --- trial move ---
        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        # --- new local scores ---
        if struct_type == STRUCT_ARCS:
            loc_struct_curr = _local_arcs_nb(pos, exp_mat, p, stretch_k, squeeze_k)
        elif struct_type == STRUCT_CHAIN:
            loc_struct_curr = _local_smooth_nb(
                pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
        else:  # STRUCT_HEATMAP
            loc_struct_curr = _local_heatmap_nb(pos, exp_mat, skip_mat[:, p], p)
        score_struct_new = score_struct + struct_delta_factor * (loc_struct_curr - loc_struct_prev)

        score_heat_new = score_heat
        if use_heat:
            loc_heat_curr = _local_heat_nb(pos, heat_dist, p, heat_weight)
            score_heat_new = score_heat + 2.0 * (loc_heat_curr - loc_heat_prev)

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_conf_new = score_conf
        if use_conf:
            loc_conf_curr = _local_confine_nb(
                pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight
            )
            score_conf_new = score_conf + (loc_conf_curr - loc_conf_prev)

        score_orn_new = score_orn
        if use_orn and orn_k >= 0:
            ar: int = int(anchor_ar[orn_k])
            ox, oy, oz = _calc_orientation_nb(pos, ar, n, bool(orn_is_L[ar]))
            anchor_orn[orn_k, 0] = ox
            anchor_orn[orn_k, 1] = oy
            anchor_orn[orn_k, 2] = oz
            loc_orn_curr = _local_score_orientation_nb(
                anchor_orn,
                orn_k,
                nbr_offsets,
                nbr_indices,
                nbr_weights,
                motif_weight,
                motifs_symmetric,
            )
            score_orn_new = score_orn + 2.0 * (loc_orn_curr - loc_orn_prev)

        score_new = (
            score_struct_new + score_heat_new + score_orn_new + score_excl_new + score_conf_new
        )

        if strict_better:
            ok = score_new < score
        else:
            ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_heat = score_heat_new
            score_orn = score_orn_new
            score_excl = score_excl_new
            score_conf = score_conf_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
            if use_orn and orn_k >= 0:
                anchor_orn[orn_k, 0] = prev_ox
                anchor_orn[orn_k, 1] = prev_oy
                anchor_orn[orn_k, 2] = prev_oz
        T *= dt
    return T, score_struct, score_heat, score_orn, score_excl, score_conf, n_ok
