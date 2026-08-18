"""Heatmap-energy MC (numba) - a fully self-contained kernel.

Unlike arcs/smooth/ib (which share the unified `_batch_mc_nb` + `terms` math),
the heatmap kernel is deliberately standalone: it carries its OWN copies of the
two energy terms it needs (heatmap distance-to-expected + excluded volume), its
own MC inner loop (`_batch_heatmap_nb`), and its own convergence driver
(`_run_heatmap_loop`).  The duplication is intentional - heatmap is the simplest
energy (double-counted structure, optional EV, non-strict acceptance) and keeping
it apart means the shared kernel never has to carry a heatmap branch.

`mc_heatmap_numba` is the public entry.  Single-chain runs wire in excluded
volume; multi-chain runs (`mc_heatmap_chains > 1`) run the same inner kernel with
EV off across K prange-parallel chains and keep the best.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

import numpy as np
from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.terms import (
    init_affinity_nb,
    init_chrom_block_nb,
    init_nuclear_nb,
    local_affinity_nb,
    local_chrom_block_nb,
    local_nuclear_nb,
)
from gnome3d.types import BoolArray, F64Array, I8Array, I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.numba")

# Typed wrapper around numba.njit so pyright sees decorated functions with their
# original signatures.  At runtime this is just numba.njit.  (Duplicated from
# `terms` on purpose - this module shares no code with the unified kernel.)
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


def _as_f64(arr: np.ndarray[Any, Any]) -> F64Array:
    return np.ascontiguousarray(arr, dtype=np.float64)


# ----- energy terms (heatmap distance-to-expected + excluded volume) -----


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
    """O(N^2) init - row-at-a-time so the sum order is stable."""
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


class CoarseTerms(NamedTuple):
    """Resolved arguments for the optional coarse-level terms.

    Bundled because the heatmap kernel takes all of them positionally and the
    driver would otherwise carry two dozen extra parameters.  `off()` builds an
    all-disabled instance whose placeholder arrays exist only to fix numba's
    types; a disabled term never indexes them.
    """

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
    use_lam: bool
    lam_weight: float
    use_cen: bool
    cen_weight: float
    chrom_w: F64Array
    nuc_cx: float
    nuc_cy: float
    nuc_cz: float
    nuc_R1: float
    nuc_R2: float
    use_chb: bool
    chrom_id: I32Array
    chb_kc: float
    chb_weight: float

    @staticmethod
    def off(n: int = 1) -> CoarseTerms:
        return CoarseTerms(
            use_comp=False,
            comp_cls=np.zeros(n, dtype=np.int8),
            comp_r0=1.0,
            comp_weight=0.0,
            comp_ea=0.0,
            comp_eb=0.0,
            use_brdg=False,
            brdg_a=np.zeros(n, dtype=np.float64),
            brdg_r0=1.0,
            brdg_weight=0.0,
            use_lam=False,
            lam_weight=0.0,
            use_cen=False,
            cen_weight=0.0,
            chrom_w=np.zeros(n, dtype=np.float64),
            nuc_cx=0.0,
            nuc_cy=0.0,
            nuc_cz=0.0,
            nuc_R1=1.0,
            nuc_R2=2.0,
            use_chb=False,
            chrom_id=np.zeros(n, dtype=np.int32),
            chb_kc=0.0,
            chb_weight=0.0,
        )

    @property
    def any_on(self) -> bool:
        return self.use_comp or self.use_brdg or self.use_lam or self.use_cen or self.use_chb


def build_coarse_terms(
    settings: Settings,
    pos: F64Array,
    bond_scale: float,
    compartment: np.ndarray[Any, Any] | None,
    accessibility: np.ndarray[Any, Any] | None,
    chrom_id: np.ndarray[Any, Any] | None,
    chrom_weight: np.ndarray[Any, Any] | None,
) -> CoarseTerms:
    """Resolve the optional coarse-level terms for one heatmap MC call.

    The nuclear frame is derived here because this call is the only one that
    spans the whole active region.  `R2` follows MultiMM's constant-density rule
    `packing * mean(bond) * N^(1/3)` and `R1 = R2 * inner_fraction^(1/3)`; the
    center is the centroid of the starting positions.

    Each term switches on only when its flag and its data are both present, so a
    missing track leaves the term off instead of scoring against zeros.
    """
    from gnome3d.mc.numba.common import affinity_params

    n = pos.shape[0]
    off = CoarseTerms.off(n)
    aff = affinity_params(settings, "heatmap", bond_scale, compartment, accessibility)

    have_c = compartment is not None and compartment.size == n
    use_lam = bool(settings.use_lamina) and have_c
    use_cen = bool(settings.use_central_force) and chrom_weight is not None
    use_chb = bool(settings.use_chromosomal_blocks) and chrom_id is not None

    nuc_cx = nuc_cy = nuc_cz = 0.0
    nuc_R1, nuc_R2 = off.nuc_R1, off.nuc_R2
    if use_lam or use_cen:
        nuc_cx = float(pos[:, 0].mean())
        nuc_cy = float(pos[:, 1].mean())
        nuc_cz = float(pos[:, 2].mean())
        nuc_R2 = float(settings.nucleus_radius)
        if nuc_R2 <= 0.0:
            nuc_R2 = float(settings.nucleus_packing_factor) * bond_scale * (n ** (1.0 / 3.0))
        nuc_R1 = nuc_R2 * float(settings.nucleus_inner_fraction) ** (1.0 / 3.0)

    return CoarseTerms(
        use_comp=aff.use_comp,
        # The lamina term reads the same compartment array, so it must be present
        # whenever either of the two is on.
        comp_cls=(
            np.ascontiguousarray(compartment, dtype=np.int8)
            if (aff.use_comp or use_lam) and compartment is not None
            else off.comp_cls
        ),
        comp_r0=aff.comp_r0,
        comp_weight=aff.comp_weight,
        comp_ea=aff.comp_ea,
        comp_eb=aff.comp_eb,
        use_brdg=aff.use_brdg,
        brdg_a=aff.brdg_a if aff.use_brdg else off.brdg_a,
        brdg_r0=aff.brdg_r0,
        brdg_weight=aff.brdg_weight,
        use_lam=use_lam,
        lam_weight=float(settings.lamina_weight),
        use_cen=use_cen,
        cen_weight=float(settings.central_weight),
        chrom_w=(
            np.ascontiguousarray(chrom_weight, dtype=np.float64)
            if use_cen and chrom_weight is not None
            else off.chrom_w
        ),
        nuc_cx=nuc_cx,
        nuc_cy=nuc_cy,
        nuc_cz=nuc_cz,
        nuc_R1=nuc_R1,
        nuc_R2=nuc_R2,
        use_chb=use_chb,
        chrom_id=(
            np.ascontiguousarray(chrom_id, dtype=np.int32)
            if use_chb and chrom_id is not None
            else off.chrom_id
        ),
        chb_kc=float(settings.chrom_block_kc),
        chb_weight=float(settings.chrom_block_weight),
    )


def init_coarse_scores(pos: F64Array, ct: CoarseTerms) -> tuple[float, float, float, float, float]:
    """Full scores for the optional coarse terms at the starting positions.

    Returns (compartment, bridging, lamina, central, chromosomal-block).
    """
    comp = brdg = lam = cen = chb = 0.0
    if ct.use_comp or ct.use_brdg:
        comp, brdg = init_affinity_nb(
            pos,
            ct.use_comp,
            ct.comp_cls,
            ct.comp_r0,
            ct.comp_weight,
            ct.comp_ea,
            ct.comp_eb,
            ct.use_brdg,
            ct.brdg_a,
            ct.brdg_r0,
            ct.brdg_weight,
        )
    if ct.use_lam or ct.use_cen:
        lam, cen = init_nuclear_nb(
            pos,
            ct.comp_cls,
            ct.chrom_w,
            ct.use_lam,
            ct.lam_weight,
            ct.use_cen,
            ct.cen_weight,
            ct.nuc_cx,
            ct.nuc_cy,
            ct.nuc_cz,
            ct.nuc_R1,
            ct.nuc_R2,
        )
    if ct.use_chb:
        chb = init_chrom_block_nb(pos, ct.chrom_id, ct.chb_kc, ct.chb_weight)
    return float(comp), float(brdg), float(lam), float(cen), float(chb)


# ----- MC inner loop + convergence driver -----
#
# Heatmap acceptance is NON-strict (score_new <= score); structure is
# double-counted (delta factor 2), and so is excluded volume.  All beads move.


@njit(cache=True, fastmath=True, nogil=True)
def _batch_heatmap_nb(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
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
    use_lam: bool,
    lam_weight: float,
    use_cen: bool,
    cen_weight: float,
    chrom_w: F64Array,
    nuc_cx: float,
    nuc_cy: float,
    nuc_cz: float,
    nuc_R1: float,
    nuc_R2: float,
    use_chb: bool,
    chrom_id: I32Array,
    chb_kc: float,
    chb_weight: float,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    n_steps: int,
    score_struct: float,
    score_excl: float,
    score_comp: float,
    score_brdg: float,
    score_lam: float,
    score_cen: float,
    score_chb: float,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    """One batch of `n_steps` heatmap MC steps for a single chain.

    Carries the heatmap distance term plus the optional excluded-volume,
    compartment, bridging, lamina, central and chromosomal-block terms.  Returns
    (T, score_struct, score_excl, score_comp, score_brdg, score_lam, score_cen,
    score_chb, n_ok)."""
    n = pos.shape[0]
    n_ok = 0
    use_aff = use_comp or use_brdg
    use_nuc = use_lam or use_cen
    score = score_struct + score_excl + score_comp + score_brdg + score_lam + score_cen + score_chb

    for _ in range(n_steps):
        p: int = int(np.random.randint(0, n))  # pyright: ignore[reportUnknownArgumentType]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

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

        loc_lam_prev = 0.0
        loc_cen_prev = 0.0
        if use_nuc:
            loc_lam_prev, loc_cen_prev = local_nuclear_nb(
                pos,
                p,
                comp_cls,
                chrom_w,
                use_lam,
                lam_weight,
                use_cen,
                cen_weight,
                nuc_cx,
                nuc_cy,
                nuc_cz,
                nuc_R1,
                nuc_R2,
            )

        loc_chb_prev = 0.0
        if use_chb:
            loc_chb_prev = local_chrom_block_nb(pos, p, chrom_id, chb_kc, chb_weight)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_struct_new = score_struct + 2.0 * (loc_struct_curr - loc_struct_prev)

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

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

        score_lam_new = score_lam
        score_cen_new = score_cen
        if use_nuc:
            loc_lam_curr, loc_cen_curr = local_nuclear_nb(
                pos,
                p,
                comp_cls,
                chrom_w,
                use_lam,
                lam_weight,
                use_cen,
                cen_weight,
                nuc_cx,
                nuc_cy,
                nuc_cz,
                nuc_R1,
                nuc_R2,
            )
            score_lam_new = score_lam + (loc_lam_curr - loc_lam_prev)
            score_cen_new = score_cen + (loc_cen_curr - loc_cen_prev)

        score_chb_new = score_chb
        if use_chb:
            loc_chb_curr = local_chrom_block_nb(pos, p, chrom_id, chb_kc, chb_weight)
            score_chb_new = score_chb + 2.0 * (loc_chb_curr - loc_chb_prev)

        score_new = (
            score_struct_new
            + score_excl_new
            + score_comp_new
            + score_brdg_new
            + score_lam_new
            + score_cen_new
            + score_chb_new
        )

        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T)

        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_excl = score_excl_new
            score_comp = score_comp_new
            score_brdg = score_brdg_new
            score_lam = score_lam_new
            score_cen = score_cen_new
            score_chb = score_chb_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return (
        T,
        score_struct,
        score_excl,
        score_comp,
        score_brdg,
        score_lam,
        score_cen,
        score_chb,
        n_ok,
    )


def _run_heatmap_loop(
    pos: F64Array,
    exp_safe: F64Array,
    skip: BoolArray,
    use_excl: bool,
    excl_r0: float,
    excl_weight: float,
    excl_skip: int,
    step_size: float,
    T: float,
    dt: float,
    jump_scale: float,
    jump_coef: float,
    stop_steps: int,
    stop_improvement: float,
    stop_successes: int,
    score_eps: float,
    stop_when_ratio_above: float,
    score_struct: float,
    score_excl: float,
    ct: CoarseTerms | None = None,
) -> float:
    """Drive `_batch_heatmap_nb` to convergence; return the final total score.
    Same reference-style stop condition as the shared driver (plateau / score_eps /
    ratio guard), specialised to the heatmap term set."""
    if ct is None:
        ct = CoarseTerms.off(pos.shape[0])
    score_comp, score_brdg, score_lam, score_cen, score_chb = init_coarse_scores(pos, ct)
    score = score_struct + score_excl + score_comp + score_brdg + score_lam + score_cen + score_chb
    ms_score = score
    step_i = 0
    while True:
        (
            T,
            score_struct,
            score_excl,
            score_comp,
            score_brdg,
            score_lam,
            score_cen,
            score_chb,
            n_ok,
        ) = _batch_heatmap_nb(
            pos,
            exp_safe,
            skip,
            use_excl,
            excl_r0,
            excl_weight,
            excl_skip,
            ct.use_comp,
            ct.comp_cls,
            ct.comp_r0,
            ct.comp_weight,
            ct.comp_ea,
            ct.comp_eb,
            ct.use_brdg,
            ct.brdg_a,
            ct.brdg_r0,
            ct.brdg_weight,
            ct.use_lam,
            ct.lam_weight,
            ct.use_cen,
            ct.cen_weight,
            ct.chrom_w,
            ct.nuc_cx,
            ct.nuc_cy,
            ct.nuc_cz,
            ct.nuc_R1,
            ct.nuc_R2,
            ct.use_chb,
            ct.chrom_id,
            ct.chb_kc,
            ct.chb_weight,
            float(step_size),
            T,
            dt,
            jump_scale,
            jump_coef,
            stop_steps,
            score_struct,
            score_excl,
            score_comp,
            score_brdg,
            score_lam,
            score_cen,
            score_chb,
        )
        score = (
            score_struct + score_excl + score_comp + score_brdg + score_lam + score_cen + score_chb
        )
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < score_eps
            or ratio > stop_when_ratio_above
        )
        LOG.debug(
            "heatmap step %7s  score=%.4f  ratio=%.4f  ok=%d/%d%s",
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


# ----- multi-chain (prange best-of-K restarts; EV off, as the reference) -----


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
    gives each chain a thread-local execution context with its own RNG state -
    the cudaMMC-style "K parallel chains, take the best" pattern in pure numba.
    EV is off here (matches the reference multi-chain path)."""
    K = pos_k.shape[0]
    # Placeholders for the optional terms, which this path never enables.  They
    # exist only to satisfy the kernel signature; nothing indexes them.
    no_i8 = np.zeros(1, dtype=np.int8)
    no_f64 = np.zeros(1, dtype=np.float64)
    no_i32 = np.zeros(1, dtype=np.int32)
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]  # view into the (k, :, :) slice
        T = max_temp
        score = _init_heatmap_nb(pos, exp_safe, skip)
        ms_score = score
        # Outer convergence loop entirely inside the kernel.
        while True:
            T, score, _se, _sc, _sb, _sl, _scn, _sch, n_ok = _batch_heatmap_nb(
                pos,
                exp_safe,
                skip,
                False,  # use_excl
                1.0,  # excl_r0 (unused)
                0.0,  # excl_weight (unused)
                0,  # excl_skip (unused)
                False,  # use_comp
                no_i8,
                1.0,
                0.0,
                0.0,
                0.0,
                False,  # use_brdg
                no_f64,
                1.0,
                0.0,
                False,  # use_lam
                0.0,
                False,  # use_cen
                0.0,
                no_f64,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                False,  # use_chb
                no_i32,
                0.0,
                0.0,
                step_size,
                T,
                dt,
                jump_scale,
                jump_coef,
                stop_steps,
                score,
                0.0,  # score_excl
                0.0,  # score_comp
                0.0,  # score_brdg
                0.0,  # score_lam
                0.0,  # score_cen
                0.0,  # score_chb
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

    LOG.debug("K=%d N=%d (numba prange parallel)", K, n)

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
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "scores: %s  -> picked ch%d",
            ", ".join(f"{s:.2f}" for s in final_scores),
            best_k,
        )
    return float(final_scores[best_k])


def mc_heatmap_numba(
    pos: np.ndarray[Any, Any],  # (N, 3) float32 - modified in place
    exp_dist: np.ndarray[Any, Any],  # (N, N) - expected pairwise distances
    diag_size: int,
    step_size: float,
    settings: Settings,
    compartment: np.ndarray[Any, Any] | None = None,
    accessibility: np.ndarray[Any, Any] | None = None,
    chrom_id: np.ndarray[Any, Any] | None = None,
    chrom_weight: np.ndarray[Any, Any] | None = None,
) -> float:
    """Numba simulated-annealing implementation for heatmap-energy MC.
    Double-counted structure (delta factor 2). Mirrors Reference
    LooperSolver::MonteCarloHeatmap().  Called by `gnome3d.mc.mc_heatmap`
    when `settings.mc_backend != "jax"`.

    When `settings.mc_heatmap_chains > 1`, runs that many independent MC
    chains in parallel via numba threading and keeps the one with the best
    final score - an embarrassingly-parallel restart strategy.
    """
    n = pos.shape[0]

    if int(settings.mc_heatmap_chains) > 1 and compartment is None and chrom_id is None:
        # The K-chain kernel carries only the heatmap term, so it can only be
        # used when no coarse term is active.
        return _mc_heatmap_multichain(pos, exp_dist, diag_size, step_size, settings)

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

    ct = build_coarse_terms(
        settings,
        pw,
        float(np.asarray(exp_dist)[~skip].mean()) if (~skip).any() else 1.0,
        compartment,
        accessibility,
        chrom_id,
        chrom_weight,
    )

    score = _run_heatmap_loop(
        pos=pw,
        exp_safe=es64,
        skip=skip_b,
        use_excl=use_excl,
        excl_r0=excl_r0,
        excl_weight=float(settings.exclusion_weight),
        excl_skip=int(settings.exclusion_skip_neighbors),
        step_size=step_size,
        T=float(settings.max_temp_heatmap),
        dt=float(settings.dt_temp_heatmap),
        jump_scale=float(settings.jump_scale_heatmap),
        jump_coef=float(settings.jump_coef_heatmap),
        stop_steps=int(settings.mc_stop_steps_heatmap),
        stop_improvement=float(settings.mc_stop_improvement_heatmap),
        stop_successes=int(settings.mc_stop_successes_heatmap),
        score_eps=1e-6,
        stop_when_ratio_above=0.9999,
        score_struct=score_struct,
        score_excl=score_excl,
        ct=ct,
    )
    pos[:] = pw.astype(pos.dtype)
    return score
