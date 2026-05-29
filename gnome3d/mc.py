"""
Monte Carlo simulation loops for 3dgnome-ng.

Mirrors Reference LooperSolver::MonteCarloHeatmap(), MonteCarloArcs(), and
MonteCarloArcsSmooth(). The four public entry points - mc_heatmap, mc_arcs,
mc_smooth, mc_ib - all share a single numba kernel `_batch_mc_nb`; they
differ only in (a) which structure-energy variant to use, and (b) which
optional energy terms (heat / orientation / excluded volume / confinement)
are wired up. Each public function reads its own level's settings; the
kernel itself is settings-free.

On first import the JIT functions compile (~10-30 s); subsequent runs load
from cache.

Acceptance criterion (all loops):
    ok = (score_new <= score_curr)   (or <  for smooth - strict_better=True)
      or rand() < jump_scale * exp(-jump_coef * score_new/score_curr / T)
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]
from numba import prange  # type: ignore[reportMissingTypeStubs]

from .types import BoolArray, F64Array, I32Array, I64Array

if TYPE_CHECKING:
    from .settings import Settings


# MC call profiler: append-only CSV of (level, N, K, n_steps_per_batch,
# wall_s, score, label) per top-level MC call.  Enabled by setting the env
# var GNOME3D_MC_PROFILE to a target path.  Used to characterise the N/K
# distribution of real workloads (e.g. to decide whether a GPU backend is
# worthwhile for a given MC level).  No-op when the env var is unset.
_MC_PROFILE_PATH = os.environ.get("GNOME3D_MC_PROFILE")


def _log_mc_call(
    level: str, n: int, k: int, n_steps: int, wall_s: float, score: float, label: str
) -> None:
    if not _MC_PROFILE_PATH:
        return
    new_file = (not os.path.exists(_MC_PROFILE_PATH)) or os.path.getsize(_MC_PROFILE_PATH) == 0
    with open(_MC_PROFILE_PATH, "a") as f:
        if new_file:
            f.write("level,N,K,n_steps_per_batch,wall_s,score,label\n")
        safe_label = label.replace('"', "'")
        f.write(f'{level},{n},{k},{n_steps},{wall_s:.6f},{score:.6f},"{safe_label}"\n')

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


# ----- shared helpers -----


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
    from .util import calc_orientation as _calc_orn

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
    label: str,
    verbose: bool,
) -> float:
    """Drive the unified kernel until convergence; return the final total score."""
    prefix = f"    [{label}] " if label else "    "
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
        if verbose:
            print(
                f"{prefix}step {step_i:>7,}  score={score:.4f}"
                f"  ratio={ratio:.4f}  ok={n_ok}/{stop_steps}" + ("  [done]" if converged else ""),
                flush=True,
            )
        if converged:
            return score
        ms_score = score


# Public MC entry points
#
# All four delegate to `_batch_mc_nb` via `_run_outer_loop`.  Each reads its
# own level's settings and prepares the term data the kernel needs.


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
    label: str,
    verbose: bool,
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

    if verbose:
        prefix = f"    [{label}] " if label else "    "
        print(f"{prefix}K={K} N={n} (numba prange parallel)", flush=True)

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
    if verbose:
        prefix = f"    [{label}] " if label else "    "
        print(
            f"{prefix}scores: "
            + ", ".join(f"{s:.2f}" for s in final_scores)
            + f"  -> picked ch{best_k}",
            flush=True,
        )
    return float(final_scores[best_k])


# Smooth-MC multi-chain (chain bonds + optional heat term)


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

        loc_struct_prev = _local_smooth_nb(
            pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
        )
        loc_heat_prev = 0.0
        if use_heat:
            loc_heat_prev = _local_heat_nb(pos, heat_dist, p, heat_weight)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = _local_smooth_nb(
            pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
        )
        score_struct_new = score_struct - loc_struct_prev + loc_struct_curr

        score_heat_new = score_heat
        if use_heat:
            loc_heat_curr = _local_heat_nb(pos, heat_dist, p, heat_weight)
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
        score_struct = _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        score_heat = _init_heat_nb(pos, heat_dist, heat_weight) if use_heat else 0.0
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
    label: str,
    verbose: bool,
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
    dtn64 = _as_f64(dtn)
    use_heat = heat_dist is not None
    if use_heat:
        assert heat_dist is not None
        heat64 = _as_f64(heat_dist)
    else:
        heat64 = np.zeros((1, 1), dtype=np.float64)

    final_scores: F64Array = np.zeros(K, dtype=np.float64)

    if verbose:
        prefix = f"    [{label}] " if label else "    "
        print(f"{prefix}smooth K={K} N={n} (numba prange parallel)", flush=True)

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
    if verbose:
        prefix = f"    [{label}] " if label else "    "
        print(
            f"{prefix}smooth scores: "
            + ", ".join(f"{s:.2f}" for s in final_scores)
            + f"  -> picked ch{best_k}",
            flush=True,
        )
    return float(final_scores[best_k])


def mc_heatmap(
    pos: np.ndarray[Any, Any],  # (N, 3) float32 - modified in place
    exp_dist: np.ndarray[Any, Any],  # (N, N) - expected pairwise distances
    diag_size: int,
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Simulated annealing using heatmap distance energy.  Double-counted
    structure (delta factor 2). Mirrors Reference LooperSolver::MonteCarloHeatmap().

    When `settings.mc_heatmap_chains > 1`, runs that many independent MC
    chains in parallel via numba threading and keeps the one with the best
    final score - essentially an embarrassingly-parallel restart strategy.
    """
    n = pos.shape[0]
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    if int(settings.mc_heatmap_chains) > 1:
        score = _mc_heatmap_multichain(pos, exp_dist, diag_size, step_size, settings, label, verbose)
        if _MC_PROFILE_PATH:
            _log_mc_call(
                "heatmap", n, int(settings.mc_heatmap_chains),
                int(settings.mc_stop_steps_heatmap),
                time.perf_counter() - _t0, score, label,
            )
        return score

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
        stop_when_ratio_above=2.0,
        score_struct=score_struct,
        score_heat=0.0,
        score_orn=0.0,
        score_excl=score_excl,
        score_conf=0.0,
        label=label,
        verbose=verbose,
    )
    pos[:] = pw.astype(pos.dtype)
    if _MC_PROFILE_PATH:
        _log_mc_call(
            "heatmap", n, int(settings.mc_heatmap_chains),
            int(settings.mc_stop_steps_heatmap),
            time.perf_counter() - _t0, score, label,
        )
    return score


def mc_arcs(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Simulated annealing with arc spring energy (pair springs from `exp_dist_mat`).
    Single-counted structure (delta factor 1). Mirrors Reference LooperSolver::MonteCarloArcs().
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    pw = _as_f64(pos)
    exp64 = _as_f64(exp_dist_mat)

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
    score_struct = float(_init_arcs_nb(pw, exp64, stretch_k, squeeze_k))
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
    score_conf = (
        float(
            _init_confine_nb(
                pw, conf_cx, conf_cy, conf_cz, conf_R, float(settings.confinement_weight)
            )
        )
        if use_conf
        else 0.0
    )

    score = _run_outer_loop(
        pw=pw,
        movable=movable,
        struct_type=STRUCT_ARCS,
        exp_mat=exp64,
        dtn=_dummy_f64((1,)),
        skip_mat=_dummy_bool(),
        stretch_k=stretch_k,
        squeeze_k=squeeze_k,
        ang_k=0.0,
        dist_w=1.0,
        ang_w=1.0,
        struct_delta_factor=1.0,
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
        label=label,
        verbose=verbose,
    )
    pos[:] = pw.astype(pos.dtype)
    if _MC_PROFILE_PATH:
        _log_mc_call(
            "arcs", n, 1, int(settings.mc_stop_steps),
            time.perf_counter() - _t0, score, label,
        )
    return score


def mc_smooth(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Chain connectivity + angle MC.  Optionally adds CTCF orientation and/or
    subanchor heat. Anchor beads (fixed=True) never move. Single-counted
    structure (delta factor 1). Mirrors Reference MonteCarloArcsSmooth.

    When `settings.mc_smooth_chains > 1` AND the call uses the simple config
    (no orientation, no EV, no confinement), dispatches to a prange-parallel
    K-chain kernel and keeps the best score.  Complex configs fall back to
    single-chain.
    """
    n = pos.shape[0]
    if n <= 2:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    # JAX backend dispatch (opt-in via settings.mc_backend='jax').  Supports the
    # chain+EV path — no orientation, no confinement, no heat term yet.  When
    # the call's config matches, route to gnome3d.mc_jax and return.  If JAX
    # is requested but not installed, mc_jax raises a clear error.
    if str(settings.mc_backend).lower() == "jax":
        jax_compatible = (
            char_orientations is None
            and heat_dist is None
            and not (bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth))
        )
        if jax_compatible:
            from . import mc_jax

            score = mc_jax.mc_smooth_jax(
                pos, dtn, fixed, step_size, settings, label=label, verbose=verbose
            )
            if _MC_PROFILE_PATH:
                _log_mc_call(
                    "smooth", n, int(settings.mc_smooth_chains),
                    int(settings.mc_stop_steps_smooth),
                    time.perf_counter() - _t0, score, label,
                )
            return score

    # Multi-chain dispatch (simple-config path only).
    if int(settings.mc_smooth_chains) > 1:
        simple_config = (
            char_orientations is None
            and not (
                bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
            )
            and not (bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth))
        )
        if simple_config:
            score = _mc_smooth_multichain(
                pos, dtn, fixed, step_size, settings, heat_dist, label, verbose
            )
            if _MC_PROFILE_PATH:
                _log_mc_call(
                    "smooth", n, int(settings.mc_smooth_chains),
                    int(settings.mc_stop_steps_smooth),
                    time.perf_counter() - _t0, score, label,
                )
            return score

    movable: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable) == 0:
        return 0.0

    pw = _as_f64(pos)
    dtn64 = _as_f64(dtn)

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

    if use_heat:
        assert heat_dist is not None
        heat64 = _as_f64(heat_dist)
        score_heat = float(_init_heat_nb(pw, heat64, heat_weight))
    else:
        heat64 = _dummy_f64()
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
        ) = _prepare_orientation(
            pw,
            fixed,
            char_orientations,
            anchor_neighbors,
            anchor_neighbor_weights,
            motif_weight,
            motifs_symmetric,
        )
    else:
        anchor_ar = _dummy_i32()
        nbr_offsets = np.zeros(2, dtype=np.int32)
        nbr_indices = _dummy_i32()
        nbr_weights = np.zeros(1, dtype=np.float64)
        orn_is_L = np.zeros(1, dtype=np.bool_)
        bead_to_anchor_k = cast(I32Array, np.full(n, -1, dtype=np.int32))
        anchor_orn = np.zeros((1, 3), dtype=np.float64)
        score_orn = 0.0

    score_struct = float(_init_smooth_nb(pw, dtn64, stretch_k, squeeze_k, ang_k, dist_w, ang_w))
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
    score_conf = (
        float(
            _init_confine_nb(
                pw, conf_cx, conf_cy, conf_cz, conf_R, float(settings.confinement_weight)
            )
        )
        if use_conf
        else 0.0
    )

    score = _run_outer_loop(
        pw=pw,
        movable=movable,
        struct_type=STRUCT_CHAIN,
        exp_mat=_dummy_f64(),
        dtn=dtn64,
        skip_mat=_dummy_bool(),
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
        label=label,
        verbose=verbose,
    )
    pos[:] = pw.astype(pos.dtype)
    if _MC_PROFILE_PATH:
        _log_mc_call(
            "smooth", n, 1, int(settings.mc_stop_steps_smooth),
            time.perf_counter() - _t0, score, label,
        )
    return score


def mc_ib(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """IB-centroid chain MC (peer to mc_smooth, not a sub-mode of it).

    Energy: chain bonds (no angle term, no orientation, no heat) + optional
    IB-scale excluded volume + optional IB-scale confinement.  All IBs move
    (no fixed set). Reads only its own settings: `spring_*_ib`, `dist_weight_ib`,
    `max_temp_ib`/`dt_temp_ib`/`jump_*_ib`/`mc_stop_*_ib` under [simulation_ib],
    plus the `*_ib` knobs under [excluded_volume] and [confinement].
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    pw = _as_f64(pos)
    dtn64 = _as_f64(dtn)
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

    score_struct = float(_init_smooth_nb(pw, dtn64, stretch_k, squeeze_k, ang_k, dist_w, ang_w))
    score_excl = (
        float(_init_excl_nb(pw, excl_r0, float(settings.exclusion_weight), 1)) if use_excl else 0.0
    )
    score_conf = (
        float(
            _init_confine_nb(
                pw, conf_cx, conf_cy, conf_cz, conf_R, float(settings.confinement_weight)
            )
        )
        if use_conf
        else 0.0
    )

    score = _run_outer_loop(
        pw=pw,
        movable=movable,
        struct_type=STRUCT_CHAIN,
        exp_mat=_dummy_f64(),
        dtn=dtn64,
        skip_mat=_dummy_bool(),
        stretch_k=stretch_k,
        squeeze_k=squeeze_k,
        ang_k=ang_k,
        dist_w=dist_w,
        ang_w=ang_w,
        struct_delta_factor=1.0,
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
        label=label,
        verbose=verbose,
    )
    pos[:] = pw.astype(pos.dtype)
    if _MC_PROFILE_PATH:
        _log_mc_call(
            "ib", n, 1, int(settings.mc_stop_steps_ib),
            time.perf_counter() - _t0, score, label,
        )
    return score
