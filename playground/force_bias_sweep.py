"""Does biasing an arcs proposal along the local descent direction cut the work.

A proposal is drawn isotropically, so most of a step's O(N) sweep goes into a direction the
energy does not want. The gradient comes free from that same sweep, so steering the draw costs
nothing and should make a proposal both likelier to be accepted and worth more when it is.

Runs real captured blocks to their own convergence at the production schedule, so neither arm is
handicapped by a budget the other does not need, and reports rounds, wall, energy and the radius
of gyration. A bias is only interesting if it cuts rounds without raising the energy.

    python playground/force_bias_sweep.py <arcs_real.pkl> [n_seeds]
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

BIASES = (0.0, 0.1, 0.25, 0.5, 0.75)


def main() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    lg = logging.getLogger("gnome3d.mc.numba")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)

    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for pos0, exp, step in pickle.load(open(sys.argv[1], "rb"))[:3]:
        print(f"\nREAL block N={pos0.shape[0]}, {int((exp > 1e-6).sum() // 2)} arc pairs, "
              f"{n_seeds} seeds")
        print(f"  {'bias':>6s} {'rounds':>7s} {'sec':>8s} {'energy':>11s} {'vs none':>8s} "
              f"{'Rg':>8s} {'accept':>7s} {'speed':>7s}")
        base = None
        for b in BIASES:
            R, E, G, W, A = [], [], [], [], []
            for seed in range(5, 5 + n_seeds):
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
                s.arcs_force_bias = b
                p = pos0.copy()
                seed_numba(seed)
                np.random.seed(seed)
                buf.truncate(0)
                buf.seek(0)
                t = time.perf_counter()
                E.append(mc_arcs_numba(p, exp, step, s))
                W.append(time.perf_counter() - t)
                lines = [ln for ln in buf.getvalue().splitlines() if "ok=" in ln]
                R.append(len(lines))
                A.append(np.mean([int(ln.split("ok=")[1].split("/")[0]) for ln in lines]) / 50_000)
                G.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
            e, r, w = float(np.mean(E)), float(np.mean(R)), float(np.mean(W))
            if base is None:
                base = (e, w)
            print(
                f"  {b:>6.2f} {r:>7.0f} {w:>7.1f}s {e:>11.1f} {e / base[0]:>7.3f}x "
                f"{np.mean(G):>8.3f} {np.mean(A):>6.1%} {base[1] / w:>6.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
