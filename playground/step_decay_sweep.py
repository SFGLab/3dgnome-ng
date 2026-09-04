"""Does annealing the step size cut the arcs stage's rounds without costing quality.

The step size is held fixed for a whole run. cudaMMC shrinks it once per outer round beside the
temperature. A smaller step late in the anneal is accepted more often and moves less, so it can
refine a structure the fixed step keeps jostling, and the convergence test watches exactly that:
a score plateau together with too few accepts.

Runs the arcs MC to convergence on blocks with the density real converged ones have, at the
production schedule, and reports for each decay the rounds it needed, the wall time, the energy
it reached and the radius of gyration. A decay is only interesting if it cuts rounds while the
energy does not rise and the radius does not move.

    python playground/step_decay_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

DECAYS = (1.0, 0.9999, 0.999, 0.995, 0.99, 0.95)


def matrix(n: int, seed: int, arc_frac: float = 0.004) -> np.ndarray:
    """Mostly the repulsion sentinel with a few springs, the shape real blocks have."""
    rng = np.random.default_rng(seed)
    exp = np.full((n, n), -1.0)
    for _ in range(max(1, int(arc_frac * n * n / 2))):
        i, j = rng.integers(0, n, 2)
        if i != j:
            v = float(rng.uniform(0.2, 0.6))
            exp[i, j] = exp[j, i] = v
    np.fill_diagonal(exp, 0.0)
    return exp


def ball(n: int, r: float, seed: int) -> np.ndarray:
    """Uniform in a ball, so the neighbour density matches what real blocks converge to."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray((v * (r * rng.random(n) ** (1 / 3))[:, None]).astype(np.float32))


def settings(decay: float) -> Settings:
    """The production arcs schedule, from a real tuned config."""
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
    s.mc_step_decay_arcs = decay
    s.mc_step_decay_floor = 0.1
    return s


def rg(p: np.ndarray) -> float:
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def main() -> None:
    for n, inside in ((462, 0.216), (1227, 0.091)):
        exp = matrix(n, n + 1)
        cutoff = 3.0 * float(exp[exp > 1e-6].mean())
        start = ball(n, cutoff / inside ** (1 / 3), n)
        print(f"\nN={n}, {inside:.1%} of pairs inside the cutoff, production arcs schedule")
        print(f"  {'decay':>8s} {'wall':>9s} {'energy':>12s} {'vs fixed':>9s} {'Rg':>8s} {'vs fixed':>9s}")
        base_e = base_rg = base_w = 0.0
        for d in DECAYS:
            pos = start.copy()
            seed_numba(5)
            np.random.seed(5)
            s = settings(d)
            mc_arcs_numba(pos.copy(), exp, 0.05, s)  # warm the jit
            pos = start.copy()
            seed_numba(5)
            np.random.seed(5)
            t = time.perf_counter()
            score = mc_arcs_numba(pos, exp, 0.05, s)
            w = time.perf_counter() - t
            g = rg(pos)
            if d == 1.0:
                base_e, base_rg, base_w = score, g, w
            print(
                f"  {d:>8.4f} {w:>8.2f}s {score:>12.2f} {score / base_e:>8.3f}x "
                f"{g:>8.3f} {g / base_rg:>8.3f}x  {'' if d == 1.0 else f'{base_w / w:.2f}x wall'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
