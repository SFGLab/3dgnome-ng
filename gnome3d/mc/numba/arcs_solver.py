"""Solves the arcs stage by L-BFGS-B.

The energy is the one `mc_arcs_numba` scores, with the repulsion cutoff and the confinement
centre and radius derived as that driver derives them. The annealer stays available as
`solver = mc`. See design/algorithm-improvements.md.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.terms import NO_W, njit
from gnome3d.types import F32Array, F64Array, I32Array, I64Array
from gnome3d.util import blas_threads

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.settings import Settings

LOG = log.get("arcs.solver")


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def arcs_energy_grad(
    x: F64Array,
    exp: F64Array,
    stretch_k: float,
    squeeze_k: float,
    rep_inv_cutoff: float,
    bg_weight: float,
    cx: float,
    cy: float,
    cz: float,
    conf_r: float,
    conf_w: float,
    excl_r0: float,
    excl_w: float,
    excl_skip: int,
    use_w: bool = False,
    arc_w: F32Array = NO_W,
) -> tuple[float, F64Array]:
    """Returns the arcs energy of a structure and its gradient, for a flattened `(3N,)` vector.

    With `use_w` the constant of each arc spring is scaled by `arc_w[j, i]`.

    The arc term sums unordered pairs. Each pair is visited from both ends with half its energy
    and its whole gradient. The excluded volume sums ordered pairs, with a whole energy and a
    doubled gradient. Confinement is per anchor.

    The energies per anchor are summed after the loop. A scalar accumulator beside the array
    writes breaks the parallel reduction of numba.
    """
    n = exp.shape[0]
    pos = x.reshape(n, 3)
    g = np.zeros((n, 3))
    ener = np.zeros(n)
    for i in prange(n):
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        ei = 0.0
        for j in range(n):
            if j == i:
                continue
            e = exp[j, i]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            dd = d if d > 1e-10 else 1e-10
            if e <= -0.75:
                bg = -e
                if d < bg:
                    rel = (dd - bg) / dd
                    ei += 0.5 * rel * rel * bg_weight
                    w = 2.0 * bg_weight * rel * bg / (dd * dd * dd)
                else:
                    rel = (d - bg) / bg
                    ei += 0.5 * rel * rel * bg_weight
                    w = 2.0 * bg_weight * rel / (bg * dd)
                gi0 += w * dx
                gi1 += w * dy
                gi2 += w * dz
            elif e < 0.0:
                v = 1.0 / dd - rep_inv_cutoff
                if v > 0.0:
                    ei += 0.5 * v
                    w = -1.0 / (dd * dd * dd)
                    gi0 += w * dx
                    gi1 += w * dy
                    gi2 += w * dz
            elif e >= 1e-6:
                rel = (d - e) / e
                k = stretch_k if rel >= 0.0 else squeeze_k
                if use_w:
                    k *= arc_w[j, i]
                ei += 0.5 * rel * rel * k
                w = 2.0 * k * rel / (e * dd)
                gi0 += w * dx
                gi1 += w * dy
                gi2 += w * dz
            if excl_w > 0.0 and d < excl_r0:
                sep = i - j
                if sep < 0:
                    sep = -sep
                if sep > excl_skip:
                    u = (excl_r0 - d) / excl_r0
                    ei += excl_w * u * u
                    w = -4.0 * excl_w * u / (excl_r0 * dd)
                    gi0 += w * dx
                    gi1 += w * dy
                    gi2 += w * dz
        rx = pos[i, 0] - cx
        ry = pos[i, 1] - cy
        rz = pos[i, 2] - cz
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        if r > conf_r:
            u = (r - conf_r) / conf_r
            ei += conf_w * u * u
            w = 2.0 * conf_w * u / (conf_r * (r if r > 1e-10 else 1e-10))
            gi0 += w * rx
            gi1 += w * ry
            gi2 += w * rz
        g[i, 0] = gi0
        g[i, 1] = gi1
        g[i, 2] = gi2
        ener[i] = ei
    return ener.sum(), g.reshape(-1)


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def factory_energy_grad(x: F64Array, act: F64Array, w: float, r: float) -> tuple[float, F64Array]:
    """Returns the factory term of a structure and its gradient, for a flattened `(3N,)` vector.

    An anchor with `act` above zero gains from the active anchors near it through
    `S_i = sum_j act_j exp(-d_ij / r)`. Its energy is `w act_i (log(1 + A) - log(1 + S_i))`,
    with `A` the total activity, which is at least zero. `S` is built in a first pass over the
    pairs and read in the second.
    """
    n = act.shape[0]
    pos = x.reshape(n, 3)
    sacc = np.zeros(n)
    for i in prange(n):
        if act[i] <= 0.0:
            continue
        si = 0.0
        for j in range(n):
            if j == i or act[j] <= 0.0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            si += act[j] * np.exp(-np.sqrt(dx * dx + dy * dy + dz * dz) / r)
        sacc[i] = si
    total = 0.0
    for i in range(n):
        total += act[i]
    log_a = np.log1p(total)
    ener = np.zeros(n)
    g = np.zeros((n, 3))
    for i in prange(n):
        if act[i] <= 0.0:
            continue
        ener[i] = w * act[i] * (log_a - np.log1p(sacc[i]))
        inv_i = 1.0 / (1.0 + sacc[i])
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        for j in range(n):
            if j == i or act[j] <= 0.0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            dd = d if d > 1e-10 else 1e-10
            coef = w * act[i] * act[j] * np.exp(-d / r) / (r * dd) * (inv_i + 1.0 / (1.0 + sacc[j]))
            gi0 += coef * dx
            gi1 += coef * dy
            gi2 += coef * dz
        g[i, 0] = gi0
        g[i, 1] = gi1
        g[i, 2] = gi2
    return ener.sum(), g.reshape(-1)


def gyration_windows(
    mids: I64Array, law: PolymerLaw, smallest_bp: int
) -> tuple[Any, F64Array, F64Array]:
    """Returns the windows of the gyration term and their target radii.

    Windows tile the span of the anchors at every scale from `smallest_bp` doubling up to the
    span, twice at each scale, the second tiling shifted by half a window. A window with fewer
    than four anchors is left out. The target is the radius of gyration of the law for the span
    of its anchors.

    Returns
    -------
    member, count, target
        A sparse window by anchor matrix of ones, the anchor count of each window and its target.
    """
    import scipy.sparse as sp  # noqa: PLC0415

    m = np.asarray(mids, dtype=np.int64)
    lo, hi = int(m.min()), int(m.max())
    rows: list[I64Array] = []
    targets: list[float] = []
    w = max(int(smallest_bp), 1)
    while w <= max(hi - lo, 1):
        for off in (0, w // 2):
            lab = (m - lo + off) // w
            order = np.argsort(lab, kind="stable")
            cuts = np.flatnonzero(np.diff(lab[order])) + 1
            for grp in np.split(order, cuts):
                if grp.size < 4:
                    continue
                rows.append(grp)
                targets.append(law.radius_of_gyration(int(m[grp].max() - m[grp].min())))
        w *= 2
    if not rows:
        return sp.csr_matrix((0, len(m))), np.zeros(0), np.zeros(0)
    ri = np.concatenate([np.full(len(g), k) for k, g in enumerate(rows)])
    ci = np.concatenate(rows)
    member = sp.csr_matrix((np.ones(len(ci)), (ri, ci)), shape=(len(rows), len(m)))
    return member, np.array([len(g) for g in rows], dtype=np.float64), np.array(targets)


def gyration_energy_grad(
    x: F64Array, member: Any, count: F64Array, target: F64Array, weight: float
) -> tuple[float, F64Array]:
    """Returns `weight` times the sum over windows of the anchor count times the squared relative
    misfit of the radius of gyration to its target and the gradient.
    """
    n = member.shape[1]
    pos = x.reshape(n, 3)
    cen = (member @ pos) / count[:, None]
    msq = (member @ (pos * pos).sum(axis=1)) / count
    rg = np.sqrt(np.maximum(msq - (cen * cen).sum(axis=1), 1e-12))
    rel = rg / target - 1.0
    e = weight * float((count * rel * rel).sum())
    coef = 2.0 * weight * rel / (target * rg)
    g = (member.T @ coef)[:, None] * pos - member.T @ (coef[:, None] * cen)
    return e, g.reshape(-1)


@njit(cache=True, fastmath=True, nogil=True)
def reach_energy_grad(
    x: F64Array, pair_i: I32Array, pair_j: I32Array, length: F64Array, weight: float
) -> tuple[float, F64Array]:
    """Returns the reach energy and its gradient.

    Pair k of nodes has `length[k]` of chain between its two beads. It has no energy up to that
    distance and `weight * ((d - length) / length)^2` beyond, the form of a loop spring.
    """
    n = x.shape[0] // 3
    pos = x.reshape(n, 3)
    g = np.zeros((n, 3))
    energy = 0.0
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dz = pos[i, 2] - pos[j, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        reach = length[k]
        if d <= reach:
            continue
        rel = (d - reach) / reach
        energy += weight * rel * rel
        w = 2.0 * weight * rel / (reach * d)
        g[i, 0] += w * dx
        g[i, 1] += w * dy
        g[i, 2] += w * dz
        g[j, 0] -= w * dx
        g[j, 1] -= w * dy
        g[j, 2] -= w * dz
    return energy, g.reshape(-1)


@njit(cache=True, fastmath=True, nogil=True)
def capped_exclusion_energy_grad(
    x: F64Array, pair_i: I32Array, pair_j: I32Array, radius: F64Array, full: float, weight: float
) -> tuple[float, F64Array]:
    """Returns the change of the excluded volume energy for the pairs with a radius of their own,
    and its gradient.

    `arcs_energy_grad` scores every pair at the radius `full` of the stage. Pair k is to be scored
    at `radius[k]`, which is under `full`. The change is the energy at `radius[k]` minus the energy
    at `full`, in the form of `arcs_energy_grad`, `2 * weight * ((r - d) / r)^2` per pair closer
    than r. The sum of the two is the excluded volume at the radius of each pair and is not
    negative.
    """
    n = x.shape[0] // 3
    pos = x.reshape(n, 3)
    g = np.zeros((n, 3))
    energy = 0.0
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dz = pos[i, 2] - pos[j, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d >= full:
            continue
        dd = d if d > 1e-10 else 1e-10
        u = (full - d) / full
        e = -2.0 * weight * u * u
        w = 4.0 * weight * u / (full * dd)
        r = radius[k]
        if d < r:
            u = (r - d) / r
            e += 2.0 * weight * u * u
            w -= 4.0 * weight * u / (r * dd)
        energy += e
        g[i, 0] += w * dx
        g[i, 1] += w * dy
        g[i, 2] += w * dz
        g[j, 0] -= w * dx
        g[j, 1] -= w * dy
        g[j, 2] -= w * dz
    return energy, g.reshape(-1)


def capped_pairs(
    exp: F64Array,
    full: float,
    skip: int,
) -> tuple[I32Array, I32Array, F64Array]:
    """Returns the pairs whose excluded volume radius is under the radius `full` of the stage, and
    the radius of each, for `[excluded_volume] arcs_cap = target`.

    The limit of a pair is the distance its spring holds it at, the loop target or the
    background. A pair without a spring has no limit. Pairs within `skip` of each other in the
    order of the matrix are left out, as in `arcs_energy_grad`.
    """
    n = int(exp.shape[0])
    held = (exp >= 1e-6) | (exp <= -0.75)
    limit: F64Array = np.where(held, np.abs(exp), np.inf)
    idx = np.arange(n)
    apart = (idx[None, :] - idx[:, None]) > skip
    ci, cj = np.nonzero(apart & (limit < full) & (limit > 0.0))
    radius: F64Array = np.ascontiguousarray(limit[ci, cj], dtype=np.float64)
    return ci.astype(np.int32), cj.astype(np.int32), radius


def solve_arcs(
    pos: F32Array,
    exp_dist: F64Array,
    s: Settings,
    iters: int | None = None,
    backend: str = "numba",
    arc_w: F32Array | None = None,
    activity: F32Array | None = None,
    mids: I64Array | None = None,
    reach: tuple[I32Array, I32Array, F64Array] | None = None,
) -> tuple[float, F32Array]:
    """Minimises the arcs energy from `pos`. Returns `(energy, positions)`.

    `backend` is the place of the energy evaluation, `numba` on the CPU or `jax` on the device.
    The minimiser runs on the host. `arc_w` is the spring weight of each arc pair, None for one.
    `activity` is the activity of each anchor for the factory term, None for no term. `mids` are
    the genomic midpoints of the anchors, which the gyration term needs. `reach` is the pairs of
    nodes of a uniform chain and the length of chain between each, for the reach term at the
    weight of a loop spring. Under `[excluded_volume] arcs_cap = target` the excluded volume of a
    pair acts at the smaller of the radius of the stage and the distance its spring holds it at.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    exp64 = np.ascontiguousarray(exp_dist, dtype=np.float64)
    pw = np.ascontiguousarray(pos, dtype=np.float64)
    n = pw.shape[0]

    mask = exp64 > 1e-6
    avg = float(exp64[mask].mean()) if mask.any() else 1.0
    factor = float(s.arcs_repulsion_cutoff_factor)
    rep_inv = 1.0 / (factor * avg) if factor > 0.0 else 0.0

    excl_r0 = 0.0
    excl_w = 0.0
    excl_skip = int(s.exclusion_skip_neighbors)
    if bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_arcs):
        excl_r0 = float(s.exclusion_radius_arcs)
        if excl_r0 <= 0.0:
            excl_r0 = float(s.exclusion_auto_factor_arcs) * avg
        excl_w = float(s.exclusion_weight)

    cx = cy = cz = 0.0
    conf_r = 1.0
    conf_w = 0.0
    if bool(s.use_confinement) and bool(s.confinement_apply_to_arcs):
        cx, cy, cz = (float(pw[:, k].mean()) for k in range(3))
        conf_r = float(s.confinement_radius_arcs)
        if conf_r <= 0.0:
            conf_r = float(s.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))
        conf_w = float(s.confinement_weight)

    args = (
        exp64,
        float(s.spring_stretch_arcs),
        float(s.spring_squeeze_arcs),
        rep_inv,
        float(s.background_weight),
        cx,
        cy,
        cz,
        conf_r,
        conf_w,
        excl_r0,
        excl_w,
        excl_skip,
    )
    n_it = int(s.arcs_solver_iters if iters is None else iters)
    if backend not in ("numba", "jax"):
        raise ValueError(f"solver backend must be numba or jax, got {backend!r}")
    t0 = time.perf_counter()
    fac_w = float(s.factory_weight) if activity is not None else 0.0
    fac_r = max(float(s.factory_radius), 1e-6)
    act64 = np.ascontiguousarray(activity, dtype=np.float64) if fac_w > 0.0 else None
    if backend == "jax":
        from gnome3d.mc.jax.arcs_energy import (  # noqa: PLC0415
            DeviceArcsEnergy,
            DeviceFactoryEnergy,
        )

        base_dev: Any = DeviceArcsEnergy(exp64, args[1:], arc_w)
        fun: Any = base_dev
        fun_args: tuple[Any, ...] = ()
        if act64 is not None:
            fac_dev = DeviceFactoryEnergy(act64, fac_w, fac_r)

            def with_factories_dev(x: Any) -> tuple[float, F64Array]:
                e1, g1 = base_dev(x)
                e2, g2 = fac_dev(x)
                return e1 + e2, g1 + g2

            fun = with_factories_dev
    else:
        fun = arcs_energy_grad
        fun_args = (
            args if arc_w is None else (*args, True, np.ascontiguousarray(arc_w, dtype=np.float32))
        )
        if act64 is not None:
            base_cpu, base_args = fun, fun_args

            def with_factories_cpu(x: Any) -> tuple[float, F64Array]:
                e1, g1 = base_cpu(x, *base_args)
                e2, g2 = factory_energy_grad(x, act64, fac_w, fac_r)
                return e1 + e2, g1 + g2

            fun = with_factories_cpu
            fun_args = ()
    gyr_w = float(s.gyration_weight)
    if gyr_w > 0.0:
        if mids is None:
            raise ValueError("[springs] gyration_weight needs the anchors' genomic midpoints")
        smallest = (
            int(s.gyration_min_bp) if int(s.gyration_min_bp) > 0 else int(s.background_range_bp)
        )
        member, count, target = gyration_windows(mids, s.polymer_law(), smallest)
        base_g, base_g_args = fun, fun_args

        def with_gyration(x: Any) -> tuple[float, F64Array]:
            e1, g1 = base_g(x, *base_g_args)
            e2, g2 = gyration_energy_grad(
                np.asarray(x, dtype=np.float64), member, count, target, gyr_w
            )
            return float(e1) + e2, np.asarray(g1, dtype=np.float64) + g2

        fun = with_gyration
        fun_args = ()
    if reach is not None and reach[0].shape[0] > 0:
        reach_i = np.ascontiguousarray(reach[0], dtype=np.int32)
        reach_j = np.ascontiguousarray(reach[1], dtype=np.int32)
        reach_len = np.ascontiguousarray(reach[2], dtype=np.float64)
        reach_w = float(s.spring_stretch_arcs)
        base_r, base_r_args = fun, fun_args

        def with_reach(x: Any) -> tuple[float, F64Array]:
            e1, g1 = base_r(x, *base_r_args)
            e2, g2 = reach_energy_grad(
                np.asarray(x, dtype=np.float64), reach_i, reach_j, reach_len, reach_w
            )
            return float(e1) + e2, np.asarray(g1, dtype=np.float64) + g2

        fun = with_reach
        fun_args = ()
    if excl_w > 0.0 and s.exclusion_arcs_cap == "target":
        cap_i, cap_j, cap_r = capped_pairs(exp64, excl_r0, excl_skip)
        if cap_i.shape[0] > 0:
            base_c, base_c_args = fun, fun_args

            def with_cap(x: Any) -> tuple[float, F64Array]:
                e1, g1 = base_c(x, *base_c_args)
                e2, g2 = capped_exclusion_energy_grad(
                    np.asarray(x, dtype=np.float64), cap_i, cap_j, cap_r, excl_r0, excl_w
                )
                return float(e1) + e2, np.asarray(g1, dtype=np.float64) + g2

            fun = with_cap
            fun_args = ()
    if n_it <= 0:
        # No descent. The start is returned as it is.
        e0, _ = fun(pw.reshape(-1), *fun_args)
        return float(e0), np.ascontiguousarray(pw, dtype=np.float32)
    # With a tolerance the solve stops when an iteration improves the energy by less than that
    # fraction, the `ftol` of L-BFGS-B. The iteration count is a cap.
    options: dict[str, Any] = {"maxiter": n_it, "maxfun": 4 * n_it, "maxcor": 20}
    tol = float(s.arcs_solver_tol)
    if tol > 0.0:
        options["ftol"] = tol
    # GNOME3D_ARCS_TRACE is a file that receives the energy of every evaluation.
    trace_path = os.environ.get("GNOME3D_ARCS_TRACE", "")
    trace: list[float] = []
    objective: Any = fun
    if trace_path:

        def traced(x: Any, *a: Any) -> Any:
            e, grad = fun(x, *a)
            trace.append(float(e))
            return e, grad

        objective = traced

    with blas_threads():
        res: Any = minimize(
            objective,
            pw.reshape(-1),
            args=fun_args,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )
    if trace_path:
        with open(trace_path, "a") as fh:
            fh.write(f"# {n} anchors, {int(res.nit)} iterations, {res.message}\n")
            fh.write("\n".join(f"{e:.6f}" for e in trace) + "\n")
    # The rule that stopped a large solve is always logged.
    if n >= 2048:
        log.status(
            LOG,
            "arcs solver on %s: %d anchors, %d iterations, %d evaluations, %.1fs, %s",
            backend,
            n,
            int(res.nit),
            int(res.nfev),
            time.perf_counter() - t0,
            str(res.message),
        )
    return float(res.fun), np.ascontiguousarray(res.x.reshape(n, 3), dtype=np.float32)
