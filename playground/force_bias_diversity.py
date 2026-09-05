"""Does a strongly biased proposal still give an ensemble.

Force bias at 0.75 cuts the arcs wall by about 1.8 times, but a biased proposal is a step toward
deterministic descent: at a bias of one the move is purely downhill and every start would run the
same trajectory. Conformational heterogeneity is the product here, so the bias is only usable if
the spread survives it.

Same measure as `lbfgs_diversity.py`, the mean relative difference between the distance matrices
of every pair of structures, which needs no superposition.

    python playground/force_bias_diversity.py <arcs_real.pkl> [n_starts]
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lbfgs_diversity import diversity  # noqa: E402
from lbfgs_vs_mc import rg, settings_for  # noqa: E402

BIASES = (0.0, 0.5, 0.75, 0.9)


def main() -> None:
    blocks = pickle.load(open(sys.argv[1], "rb"))
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rng = np.random.default_rng(0)

    for pos0, exp, step in blocks[:2]:
        n = pos0.shape[0]
        starts = [
            np.ascontiguousarray(pos0 + rng.normal(0.0, step, size=pos0.shape).astype(np.float32))
            for _ in range(n_starts)
        ]
        print(f"\nREAL block N={n}, {n_starts} starts from the same perturbed seeds")
        print(f"  {'bias':>6s} {'sec':>8s} {'mean E':>11s} {'E cv':>7s} {'mean Rg':>8s} "
              f"{'Rg cv':>7s} {'diversity':>10s} {'speed':>7s}")
        base_w = None
        for b in BIASES:
            s = settings_for(step)
            s.arcs_force_bias = b
            E, S = [], []
            t = time.perf_counter()
            for k, st in enumerate(starts):
                p = st.copy()
                seed_numba(100 + k)
                np.random.seed(100 + k)
                E.append(mc_arcs_numba(p, exp, step, s))
                S.append(p.astype(np.float64))
            w = time.perf_counter() - t
            if base_w is None:
                base_w = w
            a = np.array(E)
            g = np.array([rg(p) for p in S])
            print(
                f"  {b:>6.2f} {w:>7.1f}s {a.mean():>11,.1f} {a.std() / a.mean():>6.2%} "
                f"{g.mean():>8.3f} {g.std() / g.mean():>6.2%} {diversity(S):>9.2%} "
                f"{base_w / w:>6.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
