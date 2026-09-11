"""Is the JAX smooth launch cost per step, or a fixed cost per call.

`jax_batch_cost.py` timed one call of a fixed step budget and divided by the budget.
If a call carries a large fixed cost (transfer, trace lookup, device sync) that
division reports the fixed cost as a per step cost, and the apparent scaling with
batch width is the fixed cost growing with the arrays, not the arithmetic.

Times the same shape at several step budgets. A per step cost that falls as the
budget rises is fixed cost. One that holds is real.

    python jax_overhead.py <cif>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from gnome3d.mc.jax.smooth import mc_smooth_jax_batch  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

cif = sys.argv[1]
rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
P = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float32)
S = np.array([int(r[16]) for r in rows])
K = np.array([r[5] == "ALA" for r in rows])
o = np.argsort(S)
P, K = P[o], K[o]
bond = float(np.median(np.linalg.norm(np.diff(P.astype(np.float64), axis=0), axis=1)))
print(f"{len(P):,} beads, median bond {bond:.2f}\n")


def settings(steps: int) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 10.0
    s.exclusion_radius_smooth = 1.5 * bond
    s.exclusion_skip_neighbors = 1
    s.use_confinement = False
    s.spring_stretch = s.spring_squeeze = 10.0
    s.max_temp_smooth = 2.0
    s.mc_stop_steps_smooth = steps
    s.mc_stop_improvement_smooth = 0.0
    s.mc_stop_successes_smooth = 10**9
    s.mc_executor_jax_bucket_shapes = False
    return s


def problems(n: int, k: int) -> list[dict[str, object]]:
    pos = np.ascontiguousarray(P[:n])
    fixed = np.ascontiguousarray(K[:n])
    dtn = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1).astype(np.float32)
    return [
        {"pos": pos.copy(), "dtn": dtn, "fixed": fixed, "step_size": 0.5 * bond, "seed": i}
        for i in range(k)
    ]


for n, k in ((4096, 1), (4096, 32), (16384, 32)):
    if n > len(P):
        continue
    print(f"{n:,} beads x {k} regions")
    print(f"  {'steps':>10s} {'wall':>9s} {'us/step':>9s} {'us/step/region':>15s}")
    for steps in (20_000, 100_000, 500_000, 2_000_000):
        s = settings(steps)
        mc_smooth_jax_batch(problems(n, k), s)
        t = time.perf_counter()
        mc_smooth_jax_batch(problems(n, k), s)
        w = time.perf_counter() - t
        print(
            f"  {steps:>10,d} {w:>8.2f}s {w / steps * 1e6:>9.2f} {w / steps * 1e6 / k:>15.3f}",
            flush=True,
        )
    print()
