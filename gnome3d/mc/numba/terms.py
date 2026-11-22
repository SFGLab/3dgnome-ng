"""The energy terms and the inner MC kernel of the numba backend.

Each term has a local function, the score of one bead and an init function, the full score.
`batch_mc_nb` is the one inner kernel of arcs, smooth and ib. `struct_type` selects the
structure term and the `use_*` flags the optional terms.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]

from gnome3d.types import BoolArray, F32Array, F64Array, I8Array, I32Array, I64Array

# A typed wrapper of numba.njit. Pyright sees the signatures of decorated functions.
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


STRUCT_ARCS = 0
STRUCT_CHAIN = 1
STRUCT_HEATMAP = 2


@njit(cache=True, nogil=True)
def seed_numba(seed: int) -> None:
    """Seeds the RNG of numba for the calling thread. numba keeps its own RNG state, apart from
    numpy and `random`.
    """
    np.random.seed(seed)


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
def local_smooth_nb(
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
def init_smooth_nb(
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


# Confinement, a soft sphere around a centre.
#
# E(p) = weight * ((|r_p - c| - R) / R)^2 if |r_p - c| > R, else 0
#
# Per bead and single counted. The delta is (curr - prev).


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
def init_confine_nb(
    pos: F64Array, cx: float, cy: float, cz: float, R: float, weight: float
) -> float:
    n = pos.shape[0]
    err = 0.0
    for p in range(n):
        err += _local_confine_nb(pos, p, cx, cy, cz, R, weight)
    return err


# Excluded volume, a soft harmonic repulsion under r0.
#
# E_pair(d) = weight * ((r0 - d) / r0)^2 if d < r0, else 0
#
# The score counts each pair twice, the sum over i != j with |i - j| > skip. The delta is 2 * (curr
# - prev).


NO_MAT: F64Array = np.zeros((1, 1), dtype=np.float64)
# Stands in for the loop weight matrix when every pair is at one.
NO_W: F32Array = np.zeros((1, 1), dtype=np.float32)
NO_F64_3: F64Array = np.zeros(3, dtype=np.float64)
NO_I64_3: I64Array = np.ones(3, dtype=np.int64)
NO_I32: I32Array = np.zeros(1, dtype=np.int32)
NO_F64: F64Array = np.zeros(1, dtype=np.float64)
NO_F64_N3: F64Array = np.zeros((1, 3), dtype=np.float64)
NO_I32_N1: I32Array = np.full((1, 1), -1, dtype=np.int32)


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
def _wall_local_nb(pos: F64Array, p: int, r0: float, skip: int) -> tuple[int, float]:
    """Returns the number of non neighbour beads under `r0` from `p` and their total depth."""
    n = pos.shape[0]
    cnt = 0
    depth = 0.0
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
        if d < r0:
            cnt += 1
            depth += r0 - d
    return cnt, depth


@njit(cache=True, fastmath=True, nogil=True)
def _wall_local_cells_nb(
    pos: F64Array,
    p: int,
    r0: float,
    skip: int,
    lo: F64Array,
    dim: I64Array,
    c: float,
    head: I32Array,
    nxt: I32Array,
) -> tuple[int, float]:
    """`_wall_local_nb` over the twenty seven cells around `p`. The count is the same and the
    depth agrees to rounding.
    """
    px = pos[p, 0]
    py = pos[p, 1]
    pz = pos[p, 2]
    r2 = r0 * r0
    nx = int(dim[0])
    ny = int(dim[1])
    nz = int(dim[2])
    # Clamped as `cell_of_nb` clamps. A bead beyond the box searches the edge cells, which hold
    # every other bead beyond it.
    ix = min(max(int((px - lo[0]) / c), 0), nx - 1)
    iy = min(max(int((py - lo[1]) / c), 0), ny - 1)
    iz = min(max(int((pz - lo[2]) / c), 0), nz - 1)
    cnt = 0
    depth = 0.0
    for jz in range(max(iz - 1, 0), min(iz + 2, nz)):
        for jy in range(max(iy - 1, 0), min(iy + 2, ny)):
            base = nx * (jy + ny * jz)
            for jx in range(max(ix - 1, 0), min(ix + 2, nx)):
                i = int(head[jx + base])
                while i != -1:
                    diff = i - p
                    if diff < 0:
                        diff = -diff
                    if diff > skip:
                        dx = pos[i, 0] - px
                        dy = pos[i, 1] - py
                        dz = pos[i, 2] - pz
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 < r2:
                            cnt += 1
                            depth += r0 - math.sqrt(d2)
                    i = int(nxt[i])
    return cnt, depth


@njit(cache=True, fastmath=True, nogil=True)
def init_excl_nb(pos: F64Array, r0: float, weight: float, skip: int) -> float:
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


# Affinity, the A/B compartment term.
#
# E_pair(d) = weight * g * (1 - exp(-d^2 / (2 r0^2)))
#
# g is Ea when both beads are A, Eb when both are B and 0 otherwise. The energy is 0 at contact and
# `weight * g` far apart. It must not be negative. The Metropolis rule divides by the running
# score. Every pair takes part. The score counts each pair twice and the delta is 2 * (curr - prev).


@njit(cache=True, fastmath=True, nogil=True)
def _comp_strength_nb(ci: int, cj: int, ea: float, eb: float) -> float:
    """Returns the pair strength. Positive codes are A, negative codes are B."""
    if ci > 0 and cj > 0:
        return ea
    if ci < 0 and cj < 0:
        return eb
    return 0.0


@njit(cache=True, fastmath=True, nogil=True)
def local_affinity_nb(
    pos: F64Array,
    p: int,
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
) -> float:
    """Returns the compartment energy of bead `p` against every other bead."""
    n = pos.shape[0]
    e_comp = 0.0
    # Divided by N - 1 partners. The strength of the term does not grow with the region.
    inv_n = 1.0 / (n - 1) if n > 1 else 1.0
    comp_den = 2.0 * comp_r0 * comp_r0
    ci = int(comp_cls[p]) if use_comp else 0
    for i in range(n):
        if i == p:
            continue
        g_comp = 0.0
        if use_comp:
            g_comp = _comp_strength_nb(ci, int(comp_cls[i]), comp_ea, comp_eb)
        if g_comp == 0.0:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d2 = dx * dx + dy * dy + dz * dz
        e_comp += comp_weight * g_comp * (1.0 - math.exp(-d2 / comp_den)) * inv_n
    return e_comp


@njit(cache=True, fastmath=True, nogil=True)
def init_affinity_nb(
    pos: F64Array,
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
) -> float:
    """Returns the compartment energy over every ordered pair."""
    n = pos.shape[0]
    e_comp = 0.0
    for p in range(n):
        e_comp += local_affinity_nb(
            pos,
            p,
            use_comp,
            comp_cls,
            comp_r0,
            comp_weight,
            comp_ea,
            comp_eb,
        )
    return e_comp


@njit(cache=True, fastmath=True, nogil=True)
def _calc_orientation_nb(
    pos: F64Array, cind: int, n: int, is_L: bool
) -> tuple[float, float, float]:
    """Returns the unit orientation vector of the anchor at `cind`."""
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
def calc_orientation_stretch_nb(
    pos: F64Array, cind: int, n: int, is_L: bool, stretch: int
) -> tuple[float, float, float]:
    """Returns the unit orientation vector of the anchor at `cind` over a stretch of chain: from
    the centroid of the `stretch` beads before it to the centroid of the `stretch` beads after
    it. A side with fewer beads has the centroid of those there are and a side with none has
    the anchor bead.
    """
    lo = cind - stretch
    if lo < 0:
        lo = 0
    hi = cind + stretch
    if hi > n - 1:
        hi = n - 1
    bx = pos[cind, 0]
    by = pos[cind, 1]
    bz = pos[cind, 2]
    if cind > lo:
        bx = 0.0
        by = 0.0
        bz = 0.0
        for i in range(lo, cind):
            bx += pos[i, 0]
            by += pos[i, 1]
            bz += pos[i, 2]
        bx /= cind - lo
        by /= cind - lo
        bz /= cind - lo
    ax = pos[cind, 0]
    ay = pos[cind, 1]
    az = pos[cind, 2]
    if hi > cind:
        ax = 0.0
        ay = 0.0
        az = 0.0
        for i in range(cind + 1, hi + 1):
            ax += pos[i, 0]
            ay += pos[i, 1]
            az += pos[i, 2]
        ax /= hi - cind
        ay /= hi - cind
        az /= hi - cind
    ox = ax - bx
    oy = ay - by
    oz = az - bz
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
def rescore_sites_nb(
    pos: F64Array,
    n: int,
    sites: I32Array,
    site_old: F64Array,
    anchor_orn: F64Array,
    anchor_ar: I32Array,
    orn_is_L: BoolArray,
    nbr_offsets: I32Array,
    nbr_indices: I32Array,
    nbr_weights: F64Array,
    motif_weight: float,
    motifs_symmetric: bool,
    stretch: int,
) -> float:
    """Gives the sites of `sites` their vectors at the current positions, one after another,
    and returns the change of the orientation score. `site_old` receives the vectors before, for
    `restore_sites_nb`. `sites` ends at its first -1.
    """
    change = 0.0
    for m in range(sites.shape[0]):
        k = sites[m]
        if k < 0:
            break
        site_old[m, 0] = anchor_orn[k, 0]
        site_old[m, 1] = anchor_orn[k, 1]
        site_old[m, 2] = anchor_orn[k, 2]
        before = _local_score_orientation_nb(
            anchor_orn, k, nbr_offsets, nbr_indices, nbr_weights, motif_weight, motifs_symmetric
        )
        ar = int(anchor_ar[k])
        ox, oy, oz = calc_orientation_stretch_nb(pos, ar, n, bool(orn_is_L[ar]), stretch)
        anchor_orn[k, 0] = ox
        anchor_orn[k, 1] = oy
        anchor_orn[k, 2] = oz
        after = _local_score_orientation_nb(
            anchor_orn, k, nbr_offsets, nbr_indices, nbr_weights, motif_weight, motifs_symmetric
        )
        change += 2.0 * (after - before)
    return change


@njit(cache=True, fastmath=True, nogil=True)
def restore_sites_nb(sites: I32Array, site_old: F64Array, anchor_orn: F64Array) -> None:
    """Puts back the vectors that `rescore_sites_nb` replaced."""
    for m in range(sites.shape[0]):
        k = sites[m]
        if k < 0:
            break
        anchor_orn[k, 0] = site_old[m, 0]
        anchor_orn[k, 1] = site_old[m, 1]
        anchor_orn[k, 2] = site_old[m, 2]


@njit(cache=True, fastmath=True, nogil=True)
def score_orientation_full_nb(
    anchor_orn: F64Array,
    nbr_offsets: I32Array,
    nbr_indices: I32Array,
    nbr_weights: F64Array,
    motif_weight: float,
    symmetric: bool,
) -> float:
    """Returns the orientation score with arc weights, for initialisation."""
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
    """Returns the orientation score of anchor k with arc weights.

    The incremental update is `score_orn += 2 * (curr - prev)`. With the weights it is exact
    against `score_orientation_full_nb`.
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


@njit(cache=True, fastmath=True, nogil=True)
def local_heat_nb(pos: F64Array, heat_dist: F64Array, p: int, heat_weight: float) -> float:
    """Returns the heat score of bead p against every other bead."""
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
def init_heat_nb(pos: F64Array, heat_dist: F64Array, heat_weight: float) -> float:
    """Returns the heat score over every ordered pair."""
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


@njit(cache=True, fastmath=True, nogil=True)
def _local_arcs_nb(
    pos: F64Array,
    exp: F64Array,
    p: int,
    stretch_k: float,
    squeeze_k: float,
    rep_inv_cutoff: float = 0.0,
    bg_weight: float = 0.0,
    use_w: bool = False,
    w: F32Array = NO_W,
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
        if e <= -0.75:
            # An arcless pair inside the short range holds minus its background. It is a weak spring
            # at the background, symmetric in log distance. A background is at least one bead, so
            # -0.5 below is not one.
            bg = -e
            rel = (d - bg) / (d if d < bg and d > 1e-10 else bg)
            sc += bg_weight * rel * rel
        elif e < 0.0:
            sc += max(0.0, 1.0 / (d if d > 1e-10 else 1e-10) - rep_inv_cutoff)
        elif e >= 1e-6:
            rel = (d - e) / e
            k = stretch_k if rel >= 0.0 else squeeze_k
            if use_w:
                k *= w[i, p]
            sc += rel * rel * k
    return sc


# fastmath as in `_local_arcs_nb`. The two feed one incremental total and must round alike.
@njit(cache=True, fastmath=True, nogil=True)
def init_arcs_nb(
    pos: F64Array,
    exp: F64Array,
    stretch_k: float,
    squeeze_k: float,
    rep_inv_cutoff: float = 0.0,
    bg_weight: float = 0.0,
    use_w: bool = False,
    w: F32Array = NO_W,
) -> float:
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
            if e <= -0.75:
                bg = -e
                rel = (d - bg) / (d if d < bg and d > 1e-10 else bg)
                row_sc += bg_weight * rel * rel
            elif e < 0.0:
                row_sc += max(0.0, 1.0 / (d if d > 1e-10 else 1e-10) - rep_inv_cutoff)
            else:
                rel = (d - e) / e
                k = stretch_k if rel >= 0.0 else squeeze_k
                if use_w:
                    k *= w[i, j]
                row_sc += rel * rel * k
        sc += row_sc
    return sc


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
def init_heatmap_nb(pos: F64Array, exp_safe: F64Array, skip: BoolArray) -> float:
    """Returns the full heatmap score, in parallel over rows."""
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


# The inner kernel.
#
# The arrays of a term that is off must have a valid type and are not read.
#
# Structure adds `struct_delta_factor * (curr - prev)`, 1 for arcs and chain and 2 for heatmap.
# Heat, orientation, excluded volume and compartment add 2 * (curr - prev). Confinement adds (curr -
# prev).
#
# Under `use_centre` a node keeps the centre of mass of itself, the bead before it and the bead
# after it. `centre_of` gives that node for each of the three beads, -1 for any other bead.
# `site_of` gives the orientation site at a bead, -1 for none.
#
# A move is kept when `score_new < score` under `strict_better` and when `score_new <= score`
# otherwise. Smooth is strict.


# The grid is built in cells.py. These functions run in the step loop and are here. Cells.py
# imports this module.


@njit(cache=True, nogil=True)
def cell_of_nb(x: float, y: float, z: float, lo: F64Array, dim: I64Array, c: float) -> int:
    """Returns the cell of a point, clamped to the grid."""
    ix = int((x - lo[0]) / c)
    iy = int((y - lo[1]) / c)
    iz = int((z - lo[2]) / c)
    if ix < 0:
        ix = 0
    elif ix >= dim[0]:
        ix = dim[0] - 1
    if iy < 0:
        iy = 0
    elif iy >= dim[1]:
        iy = dim[1] - 1
    if iz < 0:
        iz = 0
    elif iz >= dim[2]:
        iz = dim[2] - 1
    return int(ix + dim[0] * (iy + dim[1] * iz))


@njit(cache=True, nogil=True)
def relink_nb(head: I32Array, nxt: I32Array, where: I32Array, i: int, k_new: int) -> None:
    """Moves bead `i` to cell `k_new`."""
    k_old = int(where[i])
    if k_old == k_new:
        return
    j = int(head[k_old])
    if j == i:
        head[k_old] = nxt[i]
    else:
        while j != -1 and nxt[j] != i:
            j = int(nxt[j])
        if j != -1:
            nxt[j] = nxt[i]
    nxt[i] = head[k_new]
    head[k_new] = i
    where[i] = k_new


@njit(cache=True, fastmath=True, nogil=True)
def local_excl_cells_nb(
    pos: F64Array,
    p: int,
    r0: float,
    weight: float,
    skip: int,
    lo: F64Array,
    dim: I64Array,
    c: float,
    head: I32Array,
    nxt: I32Array,
    buf: I32Array,
) -> float:
    """Returns the excluded volume score of bead `p` over the twenty seven cells around it,
    summed in ascending index order. The score equals `_local_excl_nb` to rounding, and to the
    bit on ARM.

    Returns a negative number when more beads are inside the radius than the buffer holds. The
    caller then uses the full scan for this bead.
    """
    px = pos[p, 0]
    py = pos[p, 1]
    pz = pos[p, 2]
    r2 = r0 * r0
    nx = int(dim[0])
    ny = int(dim[1])
    nz = int(dim[2])
    # Clamped as `cell_of_nb` clamps.
    ix = min(max(int((px - lo[0]) / c), 0), nx - 1)
    iy = min(max(int((py - lo[1]) / c), 0), ny - 1)
    iz = min(max(int((pz - lo[2]) / c), 0), nz - 1)
    m = 0
    for jz in range(max(iz - 1, 0), min(iz + 2, nz)):
        for jy in range(max(iy - 1, 0), min(iy + 2, ny)):
            base = nx * (jy + ny * jz)
            for jx in range(max(ix - 1, 0), min(ix + 2, nx)):
                i = int(head[jx + base])
                while i != -1:
                    diff = i - p
                    if diff < 0:
                        diff = -diff
                    if diff > skip:
                        dx = pos[i, 0] - px
                        dy = pos[i, 1] - py
                        dz = pos[i, 2] - pz
                        if dx * dx + dy * dy + dz * dz < r2:
                            if m >= buf.shape[0]:
                                return -1.0
                            buf[m] = i
                            m += 1
                    i = int(nxt[i])
    # Insertion sort over the few dozen beads inside the radius.
    for a in range(1, m):
        v = buf[a]
        b = a - 1
        while b >= 0 and buf[b] > v:
            buf[b + 1] = buf[b]
            b -= 1
        buf[b + 1] = v
    err = 0.0
    for t in range(m):
        i = int(buf[t])
        dx = pos[i, 0] - px
        dy = pos[i, 1] - py
        dz = pos[i, 2] - pz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        err += _excl_pair_nb(d, r0, weight)
    return err


@njit(cache=True, fastmath=True, nogil=True)
def init_excl_cells_nb(
    pos: F64Array,
    r0: float,
    weight: float,
    skip: int,
    lo: F64Array,
    dim: I64Array,
    c: float,
    head: I32Array,
    nxt: I32Array,
    buf: I32Array,
) -> float:
    """Returns the whole excluded volume score through the grid, the sum of the local scores in
    bead order, equal to `init_excl_nb`.

    Returns a negative number when a bead overflowed the buffer.
    """
    err = 0.0
    for i in range(pos.shape[0]):
        row = local_excl_cells_nb(pos, i, r0, weight, skip, lo, dim, c, head, nxt, buf)
        if row < 0.0:
            return -1.0
        err += row
    return err


@njit(cache=True, fastmath=True, nogil=True)
def step_delta_nb(
    pos: F64Array,
    p: int,
    dx: float,
    dy: float,
    dz: float,
    dtn: F64Array,
    stretch_k: float,
    squeeze_k: float,
    ang_k: float,
    dist_w: float,
    ang_w: float,
    use_heat: bool,
    heat_dist: F64Array,
    heat_weight: float,
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
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
    use_cells: bool,
    cell_lo: F64Array,
    cell_dim: I64Array,
    cell_size: float,
    cell_head: I32Array,
    cell_next: I32Array,
    cell_where: I32Array,
    cell_buf: I32Array,
    use_wall: bool,
) -> tuple[float, float, float, float, float, int, float]:
    """Moves bead p of a chain by (dx, dy, dz) in place and returns the changes that the move
    makes: of the structure, heat, excluded volume, confinement and compartment scores, of the
    count of pairs under the wall radius and of their summed depth.

    The changes of several beads moved one after another sum to the change of the whole move.
    The cell grid follows the bead.
    """
    n = pos.shape[0]
    s0 = local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
    h0 = 0.0
    if use_heat:
        h0 = local_heat_nb(pos, heat_dist, p, heat_weight)
    e0 = 0.0
    if use_excl:
        if use_cells:
            e0 = local_excl_cells_nb(
                pos,
                p,
                excl_r0,
                excl_weight,
                excl_skip,
                cell_lo,
                cell_dim,
                cell_size,
                cell_head,
                cell_next,
                cell_buf,
            )
            if e0 < 0.0:
                e0 = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
        else:
            e0 = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
    cnt0 = 0
    dep0 = 0.0
    if use_wall:
        if use_cells:
            cnt0, dep0 = _wall_local_cells_nb(
                pos, p, excl_r0, excl_skip, cell_lo, cell_dim, cell_size, cell_head, cell_next
            )
        else:
            cnt0, dep0 = _wall_local_nb(pos, p, excl_r0, excl_skip)
    c0 = 0.0
    if use_conf:
        c0 = _local_confine_nb(pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight)
    a0 = 0.0
    if use_comp:
        a0 = local_affinity_nb(pos, p, use_comp, comp_cls, comp_r0, comp_weight, comp_ea, comp_eb)

    pos[p, 0] += dx
    pos[p, 1] += dy
    pos[p, 2] += dz
    if use_cells:
        relink_nb(
            cell_head,
            cell_next,
            cell_where,
            p,
            cell_of_nb(pos[p, 0], pos[p, 1], pos[p, 2], cell_lo, cell_dim, cell_size),
        )

    s1 = local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
    h1 = 0.0
    if use_heat:
        h1 = local_heat_nb(pos, heat_dist, p, heat_weight)
    e1 = 0.0
    if use_excl:
        if use_cells:
            e1 = local_excl_cells_nb(
                pos,
                p,
                excl_r0,
                excl_weight,
                excl_skip,
                cell_lo,
                cell_dim,
                cell_size,
                cell_head,
                cell_next,
                cell_buf,
            )
            if e1 < 0.0:
                e1 = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
        else:
            e1 = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
    cnt1 = 0
    dep1 = 0.0
    if use_wall:
        if use_cells:
            cnt1, dep1 = _wall_local_cells_nb(
                pos, p, excl_r0, excl_skip, cell_lo, cell_dim, cell_size, cell_head, cell_next
            )
        else:
            cnt1, dep1 = _wall_local_nb(pos, p, excl_r0, excl_skip)
    c1 = 0.0
    if use_conf:
        c1 = _local_confine_nb(pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight)
    a1 = 0.0
    if use_comp:
        a1 = local_affinity_nb(pos, p, use_comp, comp_cls, comp_r0, comp_weight, comp_ea, comp_eb)
    return (
        s1 - s0,
        2.0 * (h1 - h0),
        2.0 * (e1 - e0),
        c1 - c0,
        2.0 * (a1 - a0),
        cnt1 - cnt0,
        dep1 - dep0,
    )


@njit(cache=True, fastmath=True, nogil=True)
def batch_mc_nb(
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
    # ---- Affinity term (compartment) ----
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
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
    score_comp: float,
    rep_inv_cutoff: float = 0.0,
    bg_weight: float = 0.0,
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
    use_arc_w: bool = False,
    arc_w: F32Array = NO_W,
    use_centre: bool = False,
    centre_of: I32Array = NO_I32,
    site_of: I32Array = NO_I32,
    orn_stretch: int = 0,
    move_sites: I32Array = NO_I32_N1,
) -> tuple[float, float, float, float, float, float, float, int]:
    n = pos.shape[0]
    moves = np.zeros(3)
    site_k = np.full(3, -1, dtype=np.int32)
    site_old = np.zeros((3, 3))
    # Under `orn_stretch` a move reads the sites of its row of `move_sites`.
    stretch_on = use_orn and orn_stretch > 0
    moved_old = np.zeros((move_sites.shape[1], 3))
    n_mov = movable.shape[0]
    n_ok = 0
    wall_cnt0 = 0
    wall_depth0 = 0.0
    score = score_struct + score_heat + score_orn + score_excl + score_conf + score_comp

    for _ in range(n_steps):
        p: int = int(movable[np.random.randint(0, n_mov)])
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        if use_centre and centre_of[p] >= 0:
            # The bead belongs to a node that keeps the centre of mass of itself, the bead before
            # it and the bead after it. The node moves by the draw and its two neighbours by
            # minus half of it. A neighbour moves by the draw and the other neighbour by minus
            # it. The single moves of the three beads sum to the change of the whole move. The
            # sites at the node and two beads from it read a moved bead.
            c = int(centre_of[p])
            d_struct = 0.0
            d_heat = 0.0
            d_excl = 0.0
            d_conf = 0.0
            d_comp = 0.0
            d_cnt = 0
            d_dep = 0.0
            for part in range(3):
                q = c - 1 + part
                if p == c:
                    f = 1.0 if part == 1 else -0.5
                elif q == p:
                    f = 1.0
                elif q == c:
                    f = 0.0
                else:
                    f = -1.0
                moves[part] = f
                if f == 0.0:
                    continue
                ds, dh, de, dc, da, dn, dd = step_delta_nb(
                    pos,
                    q,
                    f * dx,
                    f * dy,
                    f * dz,
                    dtn,
                    stretch_k,
                    squeeze_k,
                    ang_k,
                    dist_w,
                    ang_w,
                    use_heat,
                    heat_dist,
                    heat_weight,
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
                    use_cells,
                    cell_lo,
                    cell_dim,
                    cell_size,
                    cell_head,
                    cell_next,
                    cell_where,
                    cell_buf,
                    use_wall,
                )
                d_struct += ds
                d_heat += dh
                d_excl += de
                d_conf += dc
                d_comp += da
                d_cnt += dn
                d_dep += dd
            d_orn = 0.0
            if stretch_on:
                d_orn = rescore_sites_nb(
                    pos, n, move_sites[p], moved_old, anchor_orn, anchor_ar, orn_is_L,
                    nbr_offsets, nbr_indices, nbr_weights, motif_weight, motifs_symmetric,
                    orn_stretch,
                )  # fmt: skip
            for part in range(3):
                site_k[part] = -1
                bead = c - 2 + 2 * part
                if use_orn and not stretch_on and 0 <= bead < n:
                    site_k[part] = site_of[bead]
                k = site_k[part]
                if k < 0:
                    continue
                site_old[part, 0] = anchor_orn[k, 0]
                site_old[part, 1] = anchor_orn[k, 1]
                site_old[part, 2] = anchor_orn[k, 2]
                before = _local_score_orientation_nb(
                    anchor_orn, k, nbr_offsets, nbr_indices, nbr_weights, motif_weight,
                    motifs_symmetric,
                )  # fmt: skip
                ar_k = int(anchor_ar[k])
                ox, oy, oz = _calc_orientation_nb(pos, ar_k, n, bool(orn_is_L[ar_k]))
                anchor_orn[k, 0] = ox
                anchor_orn[k, 1] = oy
                anchor_orn[k, 2] = oz
                after = _local_score_orientation_nb(
                    anchor_orn, k, nbr_offsets, nbr_indices, nbr_weights, motif_weight,
                    motifs_symmetric,
                )  # fmt: skip
                d_orn += 2.0 * (after - before)

            score_new = score + d_struct + d_heat + d_excl + d_conf + d_comp + d_orn
            if use_wall and (d_cnt > 0 or (d_cnt == 0 and d_dep > 0.0)):
                ok = False
            else:
                ok = score_new < score if strict_better else score_new <= score
                if not ok and T > 0.0 and score > 0.0:
                    ok = np.random.random() < jump_scale * math.exp(
                        -jump_coef * (score_new / score) / T
                    )
            if ok:
                n_ok += 1
                score = score_new
                score_struct += d_struct
                score_heat += d_heat
                score_excl += d_excl
                score_conf += d_conf
                score_comp += d_comp
                score_orn += d_orn
            else:
                for part in range(3):
                    f = moves[part]
                    if f == 0.0:
                        continue
                    q = c - 1 + part
                    pos[q, 0] -= f * dx
                    pos[q, 1] -= f * dy
                    pos[q, 2] -= f * dz
                    if use_cells:
                        relink_nb(
                            cell_head,
                            cell_next,
                            cell_where,
                            q,
                            cell_of_nb(
                                pos[q, 0], pos[q, 1], pos[q, 2], cell_lo, cell_dim, cell_size
                            ),
                        )
                for part in range(3):
                    k = site_k[part]
                    if k >= 0:
                        anchor_orn[k, 0] = site_old[part, 0]
                        anchor_orn[k, 1] = site_old[part, 1]
                        anchor_orn[k, 2] = site_old[part, 2]
                if stretch_on:
                    restore_sites_nb(move_sites[p], moved_old, anchor_orn)
            T *= dt
            continue

        if struct_type == STRUCT_ARCS:
            loc_struct_prev = _local_arcs_nb(
                pos, exp_mat, p, stretch_k, squeeze_k, rep_inv_cutoff, bg_weight, use_arc_w, arc_w
            )
        elif struct_type == STRUCT_CHAIN:
            loc_struct_prev = local_smooth_nb(
                pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
        else:  # STRUCT_HEATMAP
            loc_struct_prev = _local_heatmap_nb(pos, exp_mat, skip_mat[:, p], p)

        loc_heat_prev = 0.0
        if use_heat:
            loc_heat_prev = local_heat_nb(pos, heat_dist, p, heat_weight)

        loc_excl_prev = 0.0
        if use_excl:
            if use_cells:
                loc_excl_prev = local_excl_cells_nb(
                    pos,
                    p,
                    excl_r0,
                    excl_weight,
                    excl_skip,
                    cell_lo,
                    cell_dim,
                    cell_size,
                    cell_head,
                    cell_next,
                    cell_buf,
                )
                if loc_excl_prev < 0.0:
                    loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            else:
                loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        if use_wall:
            if use_cells:
                wall_cnt0, wall_depth0 = _wall_local_cells_nb(
                    pos, p, excl_r0, excl_skip, cell_lo, cell_dim, cell_size, cell_head, cell_next
                )
            else:
                wall_cnt0, wall_depth0 = _wall_local_nb(pos, p, excl_r0, excl_skip)
        loc_conf_prev = 0.0
        if use_conf:
            loc_conf_prev = _local_confine_nb(
                pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight
            )

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

        orn_k: int = -1
        prev_ox = 0.0
        prev_oy = 0.0
        prev_oz = 0.0
        loc_orn_prev = 0.0
        if use_orn and not stretch_on:
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

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz
        # A capped bead may not leave its sphere. Under the wall a move may not add a pair under the
        # radius or deepen one. Neither rule draws a random number.
        hard_reject = False
        if use_cap and cap_r[p] > 0.0:
            hx = pos[p, 0] - cap_home[p, 0]
            hy = pos[p, 1] - cap_home[p, 1]
            hz = pos[p, 2] - cap_home[p, 2]
            if hx * hx + hy * hy + hz * hz > cap_r[p] * cap_r[p]:
                hard_reject = True
        if use_wall and not hard_reject:
            if use_cells:
                wall_cnt1, wall_depth1 = _wall_local_cells_nb(
                    pos, p, excl_r0, excl_skip, cell_lo, cell_dim, cell_size, cell_head, cell_next
                )
            else:
                wall_cnt1, wall_depth1 = _wall_local_nb(pos, p, excl_r0, excl_skip)
            if wall_cnt1 > wall_cnt0 or (
                wall_cnt1 == wall_cnt0 and wall_cnt1 > 0 and wall_depth1 > wall_depth0
            ):
                hard_reject = True

        if struct_type == STRUCT_ARCS:
            loc_struct_curr = _local_arcs_nb(
                pos, exp_mat, p, stretch_k, squeeze_k, rep_inv_cutoff, bg_weight, use_arc_w, arc_w
            )
        elif struct_type == STRUCT_CHAIN:
            loc_struct_curr = local_smooth_nb(
                pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
        else:  # STRUCT_HEATMAP
            loc_struct_curr = _local_heatmap_nb(pos, exp_mat, skip_mat[:, p], p)
        score_struct_new = score_struct + struct_delta_factor * (loc_struct_curr - loc_struct_prev)

        score_heat_new = score_heat
        if use_heat:
            loc_heat_curr = local_heat_nb(pos, heat_dist, p, heat_weight)
            score_heat_new = score_heat + 2.0 * (loc_heat_curr - loc_heat_prev)

        score_excl_new = score_excl
        if use_excl:
            if use_cells:
                loc_excl_curr = local_excl_cells_nb(
                    pos,
                    p,
                    excl_r0,
                    excl_weight,
                    excl_skip,
                    cell_lo,
                    cell_dim,
                    cell_size,
                    cell_head,
                    cell_next,
                    cell_buf,
                )
                if loc_excl_curr < 0.0:
                    loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            else:
                loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_conf_new = score_conf
        if use_conf:
            loc_conf_curr = _local_confine_nb(
                pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight
            )
            score_conf_new = score_conf + (loc_conf_curr - loc_conf_prev)

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
        if stretch_on:
            score_orn_new = score_orn + rescore_sites_nb(
                pos, n, move_sites[p], moved_old, anchor_orn, anchor_ar, orn_is_L, nbr_offsets,
                nbr_indices, nbr_weights, motif_weight, motifs_symmetric, orn_stretch,
            )  # fmt: skip

        score_new = (
            score_struct_new
            + score_heat_new
            + score_orn_new
            + score_excl_new
            + score_conf_new
            + score_comp_new
        )

        if hard_reject:
            ok = False
        else:
            if strict_better:
                ok = score_new < score
            else:
                ok = score_new <= score
            if not ok and T > 0.0 and score > 0.0:
                ok = np.random.random() < jump_scale * math.exp(
                    -jump_coef * (score_new / score) / T
                )

        if ok:
            n_ok += 1
            if use_cells:
                relink_nb(
                    cell_head,
                    cell_next,
                    cell_where,
                    p,
                    cell_of_nb(pos[p, 0], pos[p, 1], pos[p, 2], cell_lo, cell_dim, cell_size),
                )
            score = score_new
            score_struct = score_struct_new
            score_heat = score_heat_new
            score_orn = score_orn_new
            score_excl = score_excl_new
            score_conf = score_conf_new
            score_comp = score_comp_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
            if use_orn and orn_k >= 0:
                anchor_orn[orn_k, 0] = prev_ox
                anchor_orn[orn_k, 1] = prev_oy
                anchor_orn[orn_k, 2] = prev_oz
            if stretch_on:
                restore_sites_nb(move_sites[p], moved_old, anchor_orn)
        T *= dt
    return (
        T,
        score_struct,
        score_heat,
        score_orn,
        score_excl,
        score_conf,
        score_comp,
        n_ok,
    )
