"""Can a quasi Newton solver reach the arcs minimum, and in how much less work than the MC.

The arithmetic is attractive. Converging a 1,227 anchor block costs the Monte Carlo about 5.1e10
pair evaluations, where two thousand L-BFGS iterations cost 3e9. The landscape is a funnel, ten
starts landing within one percent, so there is nothing for a stochastic search to escape.

Two things argue the other way and are what this measures. The `1/d` repulsion is singular, so
the gradient is unbounded when two anchors approach. And the energy is piecewise smooth, with
kinks at the repulsion cutoff, at the stretch to squeeze crossover, and at the confinement
radius, where a quasi Newton method can stall.

Same block, same start, same energy as `mc_arcs_numba` minimises: the arcs pair term plus
confinement, with the centre and radius derived the way the driver derives them.

    python playground/lbfgs_vs_mc.py <arcs_real.pkl> [n_blocks]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
from numba import njit, prange  # type: ignore[reportMissingTypeStubs]  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def energy_grad(
    x: np.ndarray, exp: np.ndarray, sk: float, qk: float, rep_inv: float,
    cx: float, cy: float, cz: float, cr: float, cw: float,
) -> tuple[float, np.ndarray]:
    """The energy `mc_arcs_numba` minimises and its gradient, over the whole structure."""
    n = exp.shape[0]
    pos = x.reshape(n, 3)
    g = np.zeros((n, 3))
    # Per anchor energies summed after the loop rather than reduced inside it: numba's parallel
    # reduction does not cope with a scalar accumulator beside the array writes.
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
            if e < 0.0:
                v = 1.0 / dd - rep_inv
                if v > 0.0:
                    ei += 0.5 * v  # each unordered pair is visited twice
                    w = -1.0 / (dd * dd * dd)
                    gi0 += w * dx
                    gi1 += w * dy
                    gi2 += w * dz
            elif e >= 1e-6:
                rel = (d - e) / e
                k = sk if rel >= 0.0 else qk
                ei += 0.5 * rel * rel * k
                w = 2.0 * k * rel / (e * dd)
                gi0 += w * dx
                gi1 += w * dy
                gi2 += w * dz
        # confinement, one sided, single counted
        rx = pos[i, 0] - cx
        ry = pos[i, 1] - cy
        rz = pos[i, 2] - cz
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        if r > cr:
            u = (r - cr) / cr
            ei += cw * u * u
            w = 2.0 * cw * u / (cr * (r if r > 1e-10 else 1e-10))
            gi0 += w * rx
            gi1 += w * ry
            gi2 += w * rz
        g[i, 0] = gi0
        g[i, 1] = gi1
        g[i, 2] = gi2
        ener[i] = ei
    return ener.sum(), g.reshape(-1)


def settings_for(step: float) -> Settings:
    s = Settings()
    s.arcs_repulsion_cutoff_factor = 3.0
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.max_temp = 5.0
    s.dt_temp = 0.9999
    s.jump_scale = 50.0
    s.jump_coef = 20.0
    s.mc_stop_improvement = 0.999
    s.mc_stop_successes = 100
    s.mc_stop_steps = 50_000
    return s


def rg(p: np.ndarray) -> float:
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def main() -> None:
    blocks = pickle.load(open(sys.argv[1], "rb"))
    n_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for pos0, exp, step in blocks[:n_blocks]:
        n = pos0.shape[0]
        s = settings_for(step)
        # the driver's own derivations, so both arms minimise the same thing
        mask = exp > 1e-6
        rep_inv = 1.0 / (3.0 * float(exp[mask].mean()))
        avg = float(exp[mask].mean())
        cr = float(s.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))
        cx, cy, cz = (float(pos0[:, k].mean()) for k in range(3))
        cw = float(s.confinement_weight)
        sk, qk = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)

        print(f"\nREAL block N={n}, {int(mask.sum() // 2)} arc pairs")
        x0 = np.ascontiguousarray(pos0.astype(np.float64).reshape(-1))
        e0, _ = energy_grad(x0, exp, sk, qk, rep_inv, cx, cy, cz, cr, cw)
        print(f"  start energy {e0:,.1f}, Rg {rg(pos0):.3f}")

        p = pos0.copy()
        seed_numba(5)
        np.random.seed(5)
        t = time.perf_counter()
        e_mc = mc_arcs_numba(p, exp, step, s)
        w_mc = time.perf_counter() - t
        print(f"  MC       {w_mc:8.1f}s  energy {e_mc:11,.1f}  Rg {rg(p):8.3f}")

        for maxiter in (200, 2000, 20000):
            t = time.perf_counter()
            r = minimize(
                energy_grad, x0.copy(), args=(exp, sk, qk, rep_inv, cx, cy, cz, cr, cw),
                jac=True, method="L-BFGS-B",
                options={"maxiter": maxiter, "maxfun": 4 * maxiter, "maxcor": 20},
            )
            w = time.perf_counter() - t
            q = r.x.reshape(n, 3)
            print(
                f"  L-BFGS {maxiter:>6d} {w:8.1f}s  energy {r.fun:11,.1f}  Rg {rg(q):8.3f}  "
                f"{r.nit:>6d} iters {r.nfev:>7d} fev  vs MC {r.fun / e_mc:6.3f}x energy "
                f"{w_mc / w:6.2f}x speed",
                flush=True,
            )


if __name__ == "__main__":
    main()
