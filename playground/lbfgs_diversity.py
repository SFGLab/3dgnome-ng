"""Does a solver still give an ensemble, or does it collapse every start onto one structure.

L-BFGS reaches the arcs minimum far faster than the Monte Carlo and at a slightly lower energy,
which only matters if the result is still an ensemble. Conformational heterogeneity is the
product here, and it currently comes from starting each run at a differently perturbed seed.

Runs both arms from the same set of perturbed starts and compares what comes out: the spread of
energies, and the structural spread, measured as the mean pairwise distance between structures
after superposition-free comparison of their distance matrices, which is invariant to rotation
and translation.

    python playground/lbfgs_diversity.py <arcs_real.pkl> [n_starts]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lbfgs_vs_mc import energy_grad, rg, settings_for  # noqa: E402


def dmat(p: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    return d[np.triu_indices(p.shape[0], 1)]


def diversity(structs: list[np.ndarray]) -> float:
    """Mean relative difference between the distance matrices of every pair of structures."""
    ds = [dmat(p) for p in structs]
    out = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            out.append(float(np.abs(ds[i] - ds[j]).mean() / max(ds[i].mean(), 1e-9)))
    return float(np.mean(out)) if out else 0.0


def main() -> None:
    blocks = pickle.load(open(sys.argv[1], "rb"))
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rng = np.random.default_rng(0)

    for pos0, exp, step in blocks[:2]:
        n = pos0.shape[0]
        s = settings_for(step)
        mask = exp > 1e-6
        rep_inv = 1.0 / (3.0 * float(exp[mask].mean()))
        avg = float(exp[mask].mean())
        cr = float(s.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))
        cx, cy, cz = (float(pos0[:, k].mean()) for k in range(3))
        cw = float(s.confinement_weight)
        sk, qk = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)
        args = (exp, sk, qk, rep_inv, cx, cy, cz, cr, cw)

        starts = [
            np.ascontiguousarray(
                pos0 + rng.normal(0.0, step, size=pos0.shape).astype(np.float32)
            )
            for _ in range(n_starts)
        ]
        print(f"\nREAL block N={n}, {n_starts} starts from the same perturbed seeds")
        print(f"  {'arm':>18s} {'sec':>8s} {'mean E':>11s} {'E cv':>7s} {'mean Rg':>8s} "
              f"{'Rg cv':>7s} {'diversity':>10s}")

        for label, run_one in (
            ("MC", lambda st, k: _mc(st, exp, step, s, k)),
            ("L-BFGS 200", lambda st, k: _lb(st, args, n, 200)),
            ("L-BFGS 2000", lambda st, k: _lb(st, args, n, 2000)),
        ):
            E, S = [], []
            t = time.perf_counter()
            for k, st in enumerate(starts):
                e, p = run_one(st, k)
                E.append(e)
                S.append(p)
            w = time.perf_counter() - t
            a = np.array(E)
            g = np.array([rg(p) for p in S])
            print(
                f"  {label:>18s} {w:>7.1f}s {a.mean():>11,.1f} {a.std() / a.mean():>6.2%} "
                f"{g.mean():>8.3f} {g.std() / g.mean():>6.2%} {diversity(S):>9.2%}",
                flush=True,
            )


def _mc(start, exp, step, s, k):
    p = start.copy()
    seed_numba(100 + k)
    np.random.seed(100 + k)
    return mc_arcs_numba(p, exp, step, s), p.astype(np.float64)


def _lb(start, args, n, maxiter):
    r = minimize(
        energy_grad, start.astype(np.float64).reshape(-1), args=args, jac=True,
        method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": 4 * maxiter, "maxcor": 20},
    )
    return float(r.fun), r.x.reshape(n, 3)


if __name__ == "__main__":
    main()
