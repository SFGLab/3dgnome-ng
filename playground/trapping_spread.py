"""Does the temperature earn its keep by escaping local minima.

Temperature exists to accept worsening moves so a chain can leave a basin. A matched budget
comparison cannot see that: greedy descent banks energy early by diving into the nearest basin,
so it wins on energy at any fixed budget whether or not that basin is the right one. The
question is not the mean at a fixed budget, it is the spread over many starts, and above all the
tail. If temperature is doing its job, running without it leaves some starts stuck somewhere
much worse, so the worst case and the standard deviation are what to look at.

Each start is the captured configuration perturbed by one step size, which is how the pipeline
makes its own starts. Both arms run to their own convergence, so neither is handicapped by a
budget the other does not need.

    python playground/trapping_spread.py <arcs_real.pkl> <smooth_real.pkl> [n_starts]
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.mc.numba.smooth import mc_smooth_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def report(label: str, e: list[float], g: list[float]) -> None:
    a = np.array(e)
    print(
        f"  {label:>16s} {a.mean():>11.1f} {a.std() / a.mean():>8.2%} {a.min():>11.1f} "
        f"{a.max():>11.1f} {a.max() / a.min():>8.3f}x {np.mean(g):>8.3f} "
        f"{np.std(g) / np.mean(g):>8.2%}"
    )


def header(title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'temperature':>16s} {'mean E':>11s} {'cv':>8s} {'best':>11s} {'worst':>11s} "
        f"{'spread':>9s} {'Rg':>8s} {'Rg cv':>8s}"
    )


def main() -> None:
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    rng = np.random.default_rng(0)

    for pos0, exp, step in pickle.load(open(sys.argv[1], "rb"))[1:3]:
        n = pos0.shape[0]
        header(f"ARCS block N={n}, {n_starts} starts, each run to its own convergence")
        for label, t0 in (("production 5.0", 5.0), ("none", 0.0)):
            E, G = [], []
            for k in range(n_starts):
                s = Settings()
                s.arcs_repulsion_cutoff_factor = 3.0
                s.use_confinement = True
                s.confinement_apply_to_arcs = True
                s.max_temp = t0
                s.dt_temp = 0.9999
                s.jump_scale = 50.0
                s.jump_coef = 20.0
                s.mc_stop_improvement = 0.999
                s.mc_stop_successes = 100
                s.mc_stop_steps = 50_000
                p = pos0 + rng.normal(0.0, step, size=pos0.shape).astype(np.float32)
                p = np.ascontiguousarray(p, dtype=np.float32)
                seed_numba(100 + k)
                np.random.seed(100 + k)
                E.append(mc_arcs_numba(p, exp, step, s))
                G.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
            report(label, E, G)

    for b in pickle.load(open(sys.argv[2], "rb"))[:1]:
        header(f"SMOOTH block N={b['pos'].shape[0]}, {n_starts} starts, own convergence")
        for label, t0 in (("production 20.0", 20.0), ("none", 0.0)):
            E, G = [], []
            for k in range(n_starts):
                s = Settings()
                s.use_excluded_volume = True
                s.exclusion_apply_to_smooth = True
                s.use_confinement = True
                s.confinement_apply_to_smooth = True
                s.max_temp_smooth = t0
                s.dt_temp_smooth = 0.99995
                s.mc_stop_improvement_smooth = 0.995
                s.mc_stop_successes_smooth = 5
                s.mc_stop_steps_smooth = 10_000
                p = b["pos"] + rng.normal(0.0, b["step_size"], size=b["pos"].shape).astype(
                    np.float32
                )
                p = np.ascontiguousarray(p, dtype=np.float32)
                seed_numba(100 + k)
                np.random.seed(100 + k)
                E.append(mc_smooth_numba(p, b["dtn"], b["fixed"], b["step_size"], s))
                G.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
            report(label, E, G)


if __name__ == "__main__":
    main()
