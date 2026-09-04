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

from gnome3d.types import BoolArray, F64Array, I8Array, I32Array, I64Array

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
def init_confine_nb(
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


NO_MAT: F64Array = np.zeros((1, 1), dtype=np.float64)
NO_F64_3: F64Array = np.zeros(3, dtype=np.float64)
NO_I64_3: I64Array = np.ones(3, dtype=np.int64)
NO_I32: I32Array = np.zeros(1, dtype=np.int32)


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
def local_excl_mat_nb(pos: F64Array, p: int, r0: F64Array, weight: float, skip: int) -> float:
    """Excluded volume local score with one radius per pair. A radius of zero skips the pair,
    which is how arc pairs and the diagonal opt out of the genomic floor."""
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        diff = i - p
        if diff < 0:
            diff = -diff
        if diff <= skip:
            continue
        r = r0[i, p]
        if r <= 0.0:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        err += _excl_pair_nb(d, r, weight)
    return err


@njit(cache=True, fastmath=True, nogil=True)
def init_excl_mat_nb(pos: F64Array, r0: F64Array, weight: float, skip: int) -> float:
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
            r = r0[i, j]
            if r <= 0.0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            row_err += _excl_pair_nb(d, r, weight)
        err += row_err
    return err


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


# Affinity helpers (A/B compartment segregation + accessibility bridging)
#
#   E_pair(d) = weight * g * (1 - exp(-d^2 / (2 r0^2)))
#
#     compartment : g = Ea when both beads are A, Eb when both are B, else 0
#     bridging    : g = a_i * a_j
#
# Both are attractions: the energy is 0 at contact and rises to `weight * g` far
# apart.  MultiMM and HiP-HoP write the same well as a negative energy, which
# 3dgnome cannot use - its Metropolis rule divides by the running score and is
# guarded on `score > 0`, so a negative-definite term would silently disable the
# temperature branch.  Shifting by the well depth changes only an additive
# constant, not the minimum or the gradient.
#
# Every pair participates, matching MultiMM's CustomNonbondedForce, which sets up
# no bonded exclusions.  A bonded pair sits near the bottom of the well and so
# contributes almost no gradient.
#
# Both terms double-count pairs like excluded volume, so the delta is
# 2 * (local_curr - local_prev).  They share one distance loop because they are
# usually enabled together and the loop is the cost.


@njit(cache=True, fastmath=True, nogil=True)
def _comp_strength_nb(ci: int, cj: int, ea: float, eb: float) -> float:
    """Compartment pair strength.  Positive codes are A, negative are B."""
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
    use_brdg: bool,
    brdg_a: F64Array,
    brdg_r0: float,
    brdg_weight: float,
) -> tuple[float, float]:
    """Both affinity energies for bead `p` against every other bead."""
    n = pos.shape[0]
    e_comp = 0.0
    e_brdg = 0.0
    # Per-partner normalisation.  Without it the term sums over N partners while
    # the chain springs act per bond, so its relative strength would grow with
    # the region and a weight tuned on a small region would collapse a large one.
    inv_n = 1.0 / (n - 1) if n > 1 else 1.0
    comp_den = 2.0 * comp_r0 * comp_r0
    brdg_den = 2.0 * brdg_r0 * brdg_r0
    ci = int(comp_cls[p]) if use_comp else 0
    ap = brdg_a[p] if use_brdg else 0.0
    for i in range(n):
        if i == p:
            continue
        g_comp = 0.0
        if use_comp:
            g_comp = _comp_strength_nb(ci, int(comp_cls[i]), comp_ea, comp_eb)
        g_brdg = ap * brdg_a[i] if use_brdg else 0.0
        if g_comp == 0.0 and g_brdg == 0.0:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d2 = dx * dx + dy * dy + dz * dz
        if g_comp != 0.0:
            e_comp += comp_weight * g_comp * (1.0 - math.exp(-d2 / comp_den)) * inv_n
        if g_brdg != 0.0:
            e_brdg += brdg_weight * g_brdg * (1.0 - math.exp(-d2 / brdg_den)) * inv_n
    return e_comp, e_brdg


@njit(cache=True, fastmath=True, nogil=True)
def init_affinity_nb(
    pos: F64Array,
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
    use_brdg: bool,
    brdg_a: F64Array,
    brdg_r0: float,
    brdg_weight: float,
) -> tuple[float, float]:
    """Full double-counted affinity energies over every ordered pair."""
    n = pos.shape[0]
    e_comp = 0.0
    e_brdg = 0.0
    for p in range(n):
        c, b = local_affinity_nb(
            pos,
            p,
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
        )
        e_comp += c
        e_brdg += b
    return e_comp, e_brdg


# Nuclear-frame helpers (lamina shell + nucleolar pull), coarse levels only
#
#   lamina (B-compartment beads only, MultiMM add_Blamina_interaction):
#     E(p) = weight * (1 - sin^8(pi * (r - R1) / (R2 - R1)))   for R1 <= r <= R2
#          = weight                                            outside the shell
#
#   central (MultiMM add_central_force, harmonic mode):
#     E(p) = weight * w_chr(p) * (r - R1)^2
#
# `r` is the distance from the nuclear center.  Both are per bead and single
# counted, so the delta is (curr - prev) with no factor of 2, like confinement.
# The lamina form is MultiMM's shifted to be non-negative; see the affinity note
# above for why that shift is required.
#
# These need a nuclear frame shared across the whole active region, so they only
# run where one MC call spans the nucleus.  That is the segment-level heatmap MC.


@njit(cache=True, fastmath=True, nogil=True)
def local_nuclear_nb(
    pos: F64Array,
    p: int,
    comp_cls: I8Array,
    chrom_w: F64Array,
    use_lam: bool,
    lam_weight: float,
    use_cen: bool,
    cen_weight: float,
    cx: float,
    cy: float,
    cz: float,
    R1: float,
    R2: float,
) -> tuple[float, float]:
    """Lamina and central energies for bead `p`."""
    dx = pos[p, 0] - cx
    dy = pos[p, 1] - cy
    dz = pos[p, 2] - cz
    r = math.sqrt(dx * dx + dy * dy + dz * dz)

    e_lam = 0.0
    if use_lam and comp_cls[p] < 0:
        if r < R1 or r > R2 or R2 <= R1:
            e_lam = lam_weight
        else:
            sn = math.sin(math.pi * (r - R1) / (R2 - R1))
            s2 = sn * sn
            s8 = s2 * s2 * s2 * s2
            e_lam = lam_weight * (1.0 - s8)

    e_cen = 0.0
    if use_cen:
        rel = r - R1
        e_cen = cen_weight * chrom_w[p] * rel * rel

    return e_lam, e_cen


@njit(cache=True, fastmath=True, nogil=True)
def init_nuclear_nb(
    pos: F64Array,
    comp_cls: I8Array,
    chrom_w: F64Array,
    use_lam: bool,
    lam_weight: float,
    use_cen: bool,
    cen_weight: float,
    cx: float,
    cy: float,
    cz: float,
    R1: float,
    R2: float,
) -> tuple[float, float]:
    n = pos.shape[0]
    e_lam = 0.0
    e_cen = 0.0
    for p in range(n):
        a, b = local_nuclear_nb(
            pos, p, comp_cls, chrom_w, use_lam, lam_weight, use_cen, cen_weight, cx, cy, cz, R1, R2
        )
        e_lam += a
        e_cen += b
    return e_lam, e_cen


# Chromosomal-block helper (same-chromosome self-attraction, territories)
#
#   E_pair(d) = weight * (kc * d^4 - d^3 + d^2)   when i and j share a chromosome
#
# MultiMM add_chromosomal_blocks, polynomial mode, taken unchanged.  The
# polynomial is d^2 (kc d^2 - d + 1), which has no real root for the default
# kc = 0.3, so the term is already non-negative and needs no shift.  Pairwise and
# double counted, so the delta is 2 * (curr - prev).


@njit(cache=True, fastmath=True, nogil=True)
def local_chrom_block_nb(
    pos: F64Array, p: int, chrom_id: I32Array, kc: float, weight: float
) -> float:
    n = pos.shape[0]
    cp = chrom_id[p]
    err = 0.0
    for i in range(n):
        if i == p or chrom_id[i] != cp:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        d2 = d * d
        err += weight * (kc * d2 * d2 - d2 * d + d2)
    return err


@njit(cache=True, fastmath=True, nogil=True)
def init_chrom_block_nb(pos: F64Array, chrom_id: I32Array, kc: float, weight: float) -> float:
    n = pos.shape[0]
    err = 0.0
    for p in range(n):
        err += local_chrom_block_nb(pos, p, chrom_id, kc, weight)
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
def score_orientation_full_nb(
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
def local_heat_nb(pos: F64Array, heat_dist: F64Array, p: int, heat_weight: float) -> float:
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
def init_heat_nb(pos: F64Array, heat_dist: F64Array, heat_weight: float) -> float:
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
    pos: F64Array,
    exp: F64Array,
    p: int,
    stretch_k: float,
    squeeze_k: float,
    rep_inv_cutoff: float = 0.0,
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
            sc += max(0.0, 1.0 / (d if d > 1e-10 else 1e-10) - rep_inv_cutoff)
        elif e >= 1e-6:
            rel = (d - e) / e
            sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def init_arcs_nb(
    pos: F64Array, exp: F64Array, stretch_k: float, squeeze_k: float, rep_inv_cutoff: float = 0.0
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
            if e < 0.0:
                row_sc += max(0.0, 1.0 / (d if d > 1e-10 else 1e-10) - rep_inv_cutoff)
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
def init_heatmap_nb(pos: F64Array, exp_safe: F64Array, skip: BoolArray) -> float:
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
#   * compartment: score += 2 * (curr - prev)
#   * bridging   : score += 2 * (curr - prev)
#
# Acceptance: ok = (score_new < score) if strict_better else (score_new <= score)
# Smooth uses strict (preserves prior behaviour); arcs/heatmap use non-strict.


# --- cell grid primitives for the excluded volume term ------------------------------------
# The grid itself is built by gnome3d/mc/numba/cells.py; these three run inside the step loop,
# so they live here to keep that module's import one way.


@njit(cache=True, nogil=True)
def cell_of_nb(x: float, y: float, z: float, lo: F64Array, dim: I64Array, c: float) -> int:
    """The cell a point falls in, clamped so a bead outside the original extent still lands
    somewhere valid."""
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
    """Move bead `i` to cell `k_new`. Unlinking walks the old chain, which holds a handful of
    beads at this cell size."""
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
    """The excluded volume local score for bead `p` over the grid's twenty seven cell
    neighbourhood, summed in ascending index order so the result matches `_local_excl_nb` bit
    for bit. Returns a negative number when more beads lie inside the radius than the buffer
    holds, which tells the caller to fall back to the full scan for this bead."""
    px = pos[p, 0]
    py = pos[p, 1]
    pz = pos[p, 2]
    r2 = r0 * r0
    nx = int(dim[0])
    ny = int(dim[1])
    nz = int(dim[2])
    ix = int((px - lo[0]) / c)
    iy = int((py - lo[1]) / c)
    iz = int((pz - lo[2]) / c)
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
    # Insertion sort: only the beads inside the radius reach here, a few dozen of them.
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
    """The whole excluded volume score through the grid. `init_excl_nb` is a full pair scan,
    quadratic in the structure and a second or two on a chromosome, and it runs once per call.
    Summing each bead's local score over the grid gives the same number, since a row here is
    that function's inner loop, and the rows are added in the same order.

    Returns a negative number if any bead overflowed the buffer, so the caller can fall back."""
    err = 0.0
    for i in range(pos.shape[0]):
        row = local_excl_cells_nb(pos, i, r0, weight, skip, lo, dim, c, head, nxt, buf)
        if row < 0.0:
            return -1.0
        err += row
    return err


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
    # ---- Affinity terms (compartment + bridging) ----
    use_comp: bool,
    comp_cls: I8Array,
    comp_r0: float,
    comp_weight: float,
    comp_ea: float,
    comp_eb: float,
    use_brdg: bool,
    brdg_a: F64Array,
    brdg_r0: float,
    brdg_weight: float,
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
    score_brdg: float,
    rep_inv_cutoff: float = 0.0,
    # The genomic floor gives the excluded volume term one radius per pair. Off by default so
    # every stage but arcs passes nothing new.
    use_excl_mat: bool = False,
    excl_r0_mat: F64Array = NO_MAT,
    # Cell grid for the excluded volume term. Off by default so every caller that does not
    # build one passes nothing new. See gnome3d/mc/numba/cells.py.
    use_cells: bool = False,
    cell_lo: F64Array = NO_F64_3,
    cell_dim: I64Array = NO_I64_3,
    cell_size: float = 1.0,
    cell_head: I32Array = NO_I32,
    cell_next: I32Array = NO_I32,
    cell_where: I32Array = NO_I32,
    cell_buf: I32Array = NO_I32,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    use_aff = use_comp or use_brdg
    score = (
        score_struct + score_heat + score_orn + score_excl + score_conf + score_comp + score_brdg
    )

    for _ in range(n_steps):
        p: int = int(movable[np.random.randint(0, n_mov)])
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        # --- prev local scores ---
        if struct_type == STRUCT_ARCS:
            loc_struct_prev = _local_arcs_nb(pos, exp_mat, p, stretch_k, squeeze_k, rep_inv_cutoff)
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
            if use_excl_mat:
                loc_excl_prev = local_excl_mat_nb(pos, p, excl_r0_mat, excl_weight, excl_skip)
            elif use_cells:
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

        loc_conf_prev = 0.0
        if use_conf:
            loc_conf_prev = _local_confine_nb(
                pos, p, conf_cx, conf_cy, conf_cz, conf_R, conf_weight
            )

        loc_comp_prev = 0.0
        loc_brdg_prev = 0.0
        if use_aff:
            loc_comp_prev, loc_brdg_prev = local_affinity_nb(
                pos,
                p,
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
            loc_struct_curr = _local_arcs_nb(pos, exp_mat, p, stretch_k, squeeze_k, rep_inv_cutoff)
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
            if use_excl_mat:
                loc_excl_curr = local_excl_mat_nb(pos, p, excl_r0_mat, excl_weight, excl_skip)
            elif use_cells:
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
        score_brdg_new = score_brdg
        if use_aff:
            loc_comp_curr, loc_brdg_curr = local_affinity_nb(
                pos,
                p,
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
            )
            score_comp_new = score_comp + 2.0 * (loc_comp_curr - loc_comp_prev)
            score_brdg_new = score_brdg + 2.0 * (loc_brdg_curr - loc_brdg_prev)

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
            score_struct_new
            + score_heat_new
            + score_orn_new
            + score_excl_new
            + score_conf_new
            + score_comp_new
            + score_brdg_new
        )

        if strict_better:
            ok = score_new < score
        else:
            ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

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
            score_brdg = score_brdg_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
            if use_orn and orn_k >= 0:
                anchor_orn[orn_k, 0] = prev_ox
                anchor_orn[orn_k, 1] = prev_oy
                anchor_orn[orn_k, 2] = prev_oz
        T *= dt
    return (
        T,
        score_struct,
        score_heat,
        score_orn,
        score_excl,
        score_conf,
        score_comp,
        score_brdg,
        n_ok,
    )
