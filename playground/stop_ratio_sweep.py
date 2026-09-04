"""What the arcs convergence ratio buys, on real blocks.

The arcs MC stops when a round improves the score by less than a fixed relative amount. On real
blocks that is the branch that ends every run: the plateau branch also needs the accept count
below its threshold and acceptance sits at 15 to 50 percent throughout, so it never fires. This
one number therefore sets the round count, and the round count is the arcs wall, which is the
slowest single block since eleven blocks run on sixteen workers.

Loosening it stops the run earlier by construction, so the question is only the exchange rate:
how much energy a saved round costs. Blocks come from `capture_arcs.py`.

    python playground/stop_ratio_sweep.py <arcs_real.pkl>
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

RATIOS = (0.99, 0.999, 0.9995, 0.9999, 0.99999)
SEEDS = (5, 6)


def rg(p: np.ndarray) -> float:
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def main() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    lg = logging.getLogger("gnome3d.mc.numba")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)

    blocks = pickle.load(open(sys.argv[1], "rb"))
    for pos0, exp, step in blocks[:3]:
        n = pos0.shape[0]
        print(f"\nREAL block N={n}, {int((exp > 1e-6).sum() // 2)} arc pairs")
        print(f"  {'ratio':>9s} {'rounds':>7s} {'sec':>8s} {'energy':>11s} "
              f"{'vs default':>11s} {'Rg':>8s} {'speedup':>8s}")
        base_r = base_e = base_w = None
        for ratio in RATIOS:
            R, E, G, W = [], [], [], []
            for seed in SEEDS:
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
                s.mc_stop_ratio_arcs = ratio
                p = pos0.copy()
                seed_numba(seed)
                np.random.seed(seed)
                buf.truncate(0)
                buf.seek(0)
                t = time.perf_counter()
                E.append(mc_arcs_numba(p, exp, step, s))
                W.append(time.perf_counter() - t)
                R.append(len(buf.getvalue().splitlines()))
                G.append(rg(p))
            e, r, w = float(np.mean(E)), float(np.mean(R)), float(np.mean(W))
            if ratio == 0.9999:
                base_r, base_e, base_w = r, e, w
            print(
                f"  {ratio:>9.5f} {r:>7.0f} {w:>7.1f}s {e:>11.1f} "
                f"{e / base_e if base_e else float('nan'):>10.3f}x {np.mean(G):>8.3f} "
                f"{base_w / w if base_w else float('nan'):>7.2f}x",
                flush=True,
            )
        print(f"    (default 0.9999 = {base_r:.0f} rounds, {base_w:.1f}s, energy {base_e:.1f})")


if __name__ == "__main__":
    main()
