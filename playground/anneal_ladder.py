"""Does a real temperature ladder beat the quench this pipeline actually runs.

`delta_temp` is applied once per MC step, not per round, so at the reference's 0.9999 over
50,000 steps a round the temperature falls by a factor of 148 each round. From max_temp 5.0 it
is 3e-2 after one round and 1e-13 after six, and runs take seventy to ninety. Every stage
therefore anneals for about three rounds and then descends greedily for the rest.

That has to be settled before population annealing is worth prototyping, because a population
carried down a temperature ladder can only help if the ladder does. Here one chain runs a fixed
budget of steps, so the work is matched, and only the cooling rate changes: at 0.9999 the ladder
collapses inside the first two percent of the run, and at 0.9999986 it spans the whole of it.

Reports the energy reached, the radius of gyration and the acceptance rate.

    python playground/anneal_ladder.py [--steps N] [--n 1227]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_decay_sweep import ball, matrix, rg  # noqa: E402

SEEDS = (5, 6, 7)


def dt_for(t0: float, t_end: float, steps: int) -> float:
    """The per step cooling rate that takes `t0` to `t_end` over exactly `steps`."""
    return math.exp(math.log(t_end / t0) / steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4_300_000)
    ap.add_argument("--n", type=int, default=1227)
    a = ap.parse_args()

    n, steps = a.n, a.steps
    exp = matrix(n, n + 1)
    cutoff = 3.0 * float(exp[exp > 1e-6].mean())
    inside = {462: 0.216, 1227: 0.091}.get(n, 0.091)
    start = ball(n, cutoff / inside ** (1 / 3), n)

    t0 = 5.0
    ladders = [
        ("reference 0.9999", 0.9999),
        ("to 0.01 over 1 percent", dt_for(t0, 0.01, steps // 100)),
        ("to 0.01 over a tenth", dt_for(t0, 0.01, steps // 10)),
        ("to 0.01 over the run", dt_for(t0, 0.01, steps)),
        ("to 0.1 over the run", dt_for(t0, 0.1, steps)),
        ("no temperature at all", 0.0),
    ]
    print(f"N={n}, one round of {steps:,} steps, max_temp {t0}, three seeds\n")
    print(f"  {'ladder':>24s} {'delta_temp':>12s} {'dies after':>12s} {'energy':>11s} "
          f"{'vs ref':>7s} {'Rg':>7s} {'accept':>7s}")
    base = None
    for label, dt in ladders:
        E, G, A = [], [], []
        for seed in SEEDS:
            s = Settings()
            s.arcs_repulsion_cutoff_factor = 3.0
            s.use_confinement = True
            s.confinement_apply_to_arcs = True
            s.max_temp = t0 if dt > 0.0 else 0.0
            s.dt_temp = dt if dt > 0.0 else 1.0
            s.jump_scale = 50.0
            s.jump_coef = 20.0
            s.mc_stop_steps = steps
            s.mc_stop_improvement = 0.0  # exactly one round
            s.mc_stop_successes = 10**9
            pos = start.copy()
            seed_numba(seed)
            np.random.seed(seed)
            E.append(mc_arcs_numba(pos, exp, 0.05, s))
            G.append(rg(pos))
        e = float(np.mean(E))
        if base is None:
            base = e
        # where the ladder has fallen to 1 percent of max_temp, in steps
        dies = "n/a" if dt <= 0.0 else f"{math.log(0.01) / math.log(dt) / steps:.3%}"
        print(
            f"  {label:>24s} {dt:>12.7f} {dies:>12s} {e:>11.0f} {e / base:>6.3f}x "
            f"{np.mean(G):>7.3f} {'':>7s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
