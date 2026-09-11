"""Are the structures a solver produces the same kind of structure the Monte Carlo produces.

Equal energy and equal spread are necessary but not sufficient. What the pipeline is judged on
is geometry, so this compares the two arms on what the arcs stage is actually for: how close arc
linked anchors end up relative to the distance their arc asked for, and how crowded the
structure is.

Arcs are realised at about 2.6 to 3.2 times their target under the Monte Carlo, which is the
equilibrium of a network of near equal links and is what sets the within block distance
behaviour. A solver that lands somewhere very different is not a drop in replacement whatever
its energy.

    python playground/lbfgs_quality.py <arcs_real.pkl> [n_starts]
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lbfgs_vs_mc import energy_grad, rg, settings_for  # noqa: E402


def report(label: str, structs: list[np.ndarray], exp: np.ndarray, cutoff: float) -> None:
    ii, jj = np.where(np.triu(exp > 1e-6, 1))
    tgt = exp[ii, jj]
    ratios, inside, nn = [], [], []
    for p in structs:
        d = np.linalg.norm(p[ii] - p[jj], axis=1)
        ratios.append(d / tgt)
        full = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(full, np.inf)
        inside.append((full < cutoff).mean())
        nn.append(np.min(full, axis=1).mean())
    r = np.concatenate(ratios)
    print(
        f"  {label:>16s} {np.median(r):>9.2f} {r.mean():>9.2f} "
        f"{np.percentile(r, 10):>8.2f} {np.percentile(r, 90):>8.2f} "
        f"{np.mean(inside):>9.1%} {np.mean(nn):>10.3f} {np.mean([rg(p) for p in structs]):>8.3f}"
    )


def main() -> None:
    blocks = pickle.load(open(sys.argv[1], "rb"))
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = np.random.default_rng(0)

    for pos0, exp, step in blocks[:2]:
        n = pos0.shape[0]
        s = settings_for(step)
        mask = exp > 1e-6
        avg = float(exp[mask].mean())
        cutoff = 3.0 * avg
        rep_inv = 1.0 / cutoff
        cr = float(s.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))
        cx, cy, cz = (float(pos0[:, k].mean()) for k in range(3))
        args = (exp, float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs), rep_inv,
                cx, cy, cz, cr, float(s.confinement_weight))
        starts = [
            np.ascontiguousarray(pos0 + rng.normal(0.0, step, size=pos0.shape).astype(np.float32))
            for _ in range(n_starts)
        ]
        print(f"\nREAL block N={n}, arc target mean {avg:.3f}, cutoff {cutoff:.3f}, "
              f"{n_starts} starts")
        print(f"  {'arm':>16s} {'d/target':>9s} {'mean':>9s} {'p10':>8s} {'p90':>8s} "
              f"{'in cutoff':>9s} {'nearest':>10s} {'Rg':>8s}")

        mc = []
        for k, st in enumerate(starts):
            p = st.copy()
            seed_numba(100 + k)
            np.random.seed(100 + k)
            mc_arcs_numba(p, exp, step, s)
            mc.append(p.astype(np.float64))
        report("MC", mc, exp, cutoff)

        for it in (200, 2000):
            out = []
            for st in starts:
                r = minimize(energy_grad, st.astype(np.float64).reshape(-1), args=args, jac=True,
                             method="L-BFGS-B",
                             options={"maxiter": it, "maxfun": 4 * it, "maxcor": 20})
                out.append(r.x.reshape(n, 3))
            report(f"L-BFGS {it}", out, exp, cutoff)


if __name__ == "__main__":
    main()
