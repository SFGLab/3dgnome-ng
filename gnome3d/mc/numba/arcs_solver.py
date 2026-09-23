"""Solve the arcs stage instead of annealing it.

The arcs landscape is a funnel. Ten Monte Carlo starts from perturbed seeds land within one
percent of the same energy, with a coefficient of variation of a third of a percent, and
temperature makes no difference to where a run converges, so there are no basins for a
stochastic search to escape. Measured on real blocks, two hundred L-BFGS iterations reach the
Monte Carlo's energy about thirty six times faster, with the ensemble spread slightly wider
rather than narrower and the geometry, realised distance against the distance an arc asked for,
matching across the whole distribution.

The energy here is the one `mc_arcs_numba` scores, its arc term plus its confinement, with the
repulsion cutoff and the confinement centre and radius derived the way that driver derives them.
It implements every term the arcs stage carries.

Production solves; the annealer stays available as `solver = mc`. See
design/algorithm-improvements.md.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.mc.numba.terms import NO_W, njit
from gnome3d.types import F32Array, F64Array

if TYPE_CHECKING:
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
    """The arcs energy over a whole structure and its gradient, for a flattened `(3N,)` vector.

    With `use_w` each arc spring's constant is scaled by `arc_w[j, i]`, the loop's weight;
    the other terms never read it.

    Two counting conventions, both taken from the initialisers the MC scores with. The arc term
    sums unordered pairs, so visiting each from both ends means halving its energy while its
    gradient stays whole, the two halves being the derivative with respect to each end. The
    excluded volume sums ordered pairs and so is counted twice, which means a whole energy and a
    doubled gradient. Confinement is per anchor and single counted.

    Per anchor energies are summed after the loop rather than reduced inside it, because numba's
    parallel reduction does not cope with a scalar accumulator beside the array writes.
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
    """The factory term over a whole structure and its gradient, for a flattened `(3N,)` vector.

    An active anchor, one with `act` above zero, gains from the active anchors near it through
    `S_i = sum_j act_j exp(-d_ij / r)`, and its energy is `w act_i (log(1 + A) - log(1 + S_i))`
    with `A` the total activity, so joining a group lowers it, a second group lowers it little,
    and it is never below zero, which the Metropolis rule needs of every term. The gradient on
    anchor i collects its own `S_i` and its part in every other anchor's, so `S` is built in a
    first pass over the pairs and read in the second.
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


def solve_arcs(
    pos: F32Array,
    exp_dist: F64Array,
    s: Settings,
    iters: int | None = None,
    backend: str = "numba",
    arc_w: F32Array | None = None,
    activity: F32Array | None = None,
) -> tuple[float, F32Array]:
    """Minimise the arcs energy from `pos`. Returns `(energy, positions)`.

    Mirrors the derivations in `mc_arcs_numba` so the energy is the one the annealer reports.
    `backend` is where the energy is evaluated, `numba` on the CPU or `jax` on the device; the
    minimiser itself always runs on the host. The arcs stage passes the executor's choice.
    `arc_w` is each arc pair's spring weight, or None for every pair at one. `activity` is
    each anchor's activity for the factory term at `factory_weight`, or None for no term.
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
    # The stop rule. With a tolerance the solve ends when an iteration improves the energy by
    # less than that fraction of it, L-BFGS-B's own `ftol`, and the iteration count is a safety
    # cap; without one the cap is the only stop, which on a chromosome it always is.
    options: dict[str, Any] = {"maxiter": n_it, "maxfun": 4 * n_it, "maxcor": 20}
    tol = float(s.arcs_solver_tol)
    if tol > 0.0:
        options["ftol"] = tol
    # GNOME3D_ARCS_TRACE names a file that gets every evaluation's energy, for choosing the
    # tolerance from a real solve's trajectory.
    trace_path = os.environ.get("GNOME3D_ARCS_TRACE", "")
    trace: list[float] = []
    objective: Any = fun
    if trace_path:

        def traced(x: Any, *a: Any) -> Any:
            e, grad = fun(x, *a)
            trace.append(float(e))
            return e, grad

        objective = traced

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
    # Which rule stopped the solve is what decides whether more iterations buy anything, so
    # it is always visible for the large solves.
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
