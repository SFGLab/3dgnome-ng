"""Does a temperature ladder buy anything in the smooth stage, on real captured blocks.

The same question already answered for arcs, where a ladder spanning the run was two to four
percent worse than the reference quench and pure greedy descent tied it. Smooth has a different
energy, chain bonds and angles and excluded volume and confinement rather than arc springs and a
truncated repulsion, and it is about forty percent of the wall, so the arcs answer does not
transfer on its own.

One chain, a fixed step budget so the work is matched, only the cooling rate changing. Blocks
come from `capture_smooth.py`.

    python playground/anneal_ladder_smooth.py <smooth_real.pkl> [steps]
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.smooth import mc_smooth_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

T0 = 20.0
SEEDS = (5, 6, 7)


def rg(p: np.ndarray) -> float:
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def main() -> None:
    blocks = pickle.load(open(sys.argv[1], "rb"))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

    def dt_for(t_end: float, over: int) -> float:
        return math.exp(math.log(t_end / T0) / over)

    for b in blocks[:2]:
        n = b["pos"].shape[0]
        print(f"\nREAL smooth block N={n}, {int(b['fixed'].sum())} anchors, "
              f"one round of {steps:,} steps, max_temp {T0}")
        print(f"  {'ladder':>24s} {'delta_temp':>12s} {'energy':>12s} {'vs ref':>7s} {'Rg':>8s} "
              f"{'bond cv':>8s}")
        base = None
        for label, dt in (
            ("reference 0.99995", 0.99995),
            ("to 0.01 over a tenth", dt_for(0.01, steps // 10)),
            ("to 0.01 over the run", dt_for(0.01, steps)),
            ("to 1.0 over the run", dt_for(1.0, steps)),
            ("no temperature at all", 0.0),
        ):
            E, G, C = [], [], []
            for seed in SEEDS:
                s = Settings()
                s.use_excluded_volume = True
                s.exclusion_apply_to_smooth = True
                s.use_confinement = True
                s.confinement_apply_to_smooth = True
                s.max_temp_smooth = T0 if dt > 0.0 else 0.0
                s.dt_temp_smooth = dt if dt > 0.0 else 1.0
                s.mc_stop_steps_smooth = steps
                s.mc_stop_improvement_smooth = 0.0  # exactly one round
                s.mc_stop_successes_smooth = 10**9
                p = b["pos"].copy()
                seed_numba(seed)
                np.random.seed(seed)
                E.append(mc_smooth_numba(p, b["dtn"], b["fixed"], b["step_size"], s))
                G.append(rg(p))
                d = np.linalg.norm(np.diff(p.astype(np.float64), axis=0), axis=1)
                C.append(float(d.std() / d.mean()))
            e = float(np.mean(E))
            if base is None:
                base = e
            print(
                f"  {label:>24s} {dt:>12.7f} {e:>12.1f} {e / base:>6.3f}x "
                f"{np.mean(G):>8.3f} {np.mean(C):>8.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
