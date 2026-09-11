"""Does the JAX smooth step cost grow with the structure, or is it a latency floor.

The kernel evaluates every bead for the excluded volume term, so the arithmetic per step is
linear in the bead count. But the MC chain is sequential, one bead per step, so each step is a
separate small reduction on the device and may be bound by launch latency instead. Which of the
two it is decides whether a neighbour list is worth building for JAX at all: it cuts the
arithmetic and does nothing for the latency.

Times a fixed number of steps at several bead counts. Flat per step means latency, proportional
means arithmetic.

    python playground/jax_step_cost.py <cif> [steps]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.jax.smooth import mc_smooth_jax  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

cif = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000

rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
P = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float32)
S = np.array([int(r[16]) for r in rows])
K = np.array([r[5] == "ALA" for r in rows])
o = np.argsort(S)
P, K = P[o], K[o]
bond = float(np.median(np.linalg.norm(np.diff(P.astype(np.float64), axis=0), axis=1)))
print(f"{len(P):,} beads available, median bond {bond:.2f}, {STEPS:,} steps per timing\n")


def settings(use_excl: bool) -> Settings:
    s = Settings()
    s.use_excluded_volume = use_excl
    s.exclusion_apply_to_smooth = use_excl
    s.exclusion_weight = 10.0
    s.exclusion_radius_smooth = 1.5 * bond
    s.exclusion_skip_neighbors = 1
    s.use_confinement = False
    s.spring_stretch = s.spring_squeeze = 10.0
    s.max_temp_smooth = 2.0
    s.mc_stop_steps_smooth = STEPS
    s.mc_stop_improvement_smooth = 0.0
    s.mc_stop_successes_smooth = 10**9
    s.mc_executor_jax_bucket_shapes = False
    return s


def run(n: int, use_excl: bool) -> float:
    pos = np.ascontiguousarray(P[:n]).copy()
    fixed = np.ascontiguousarray(K[:n]).copy()
    dtn = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1).astype(np.float32)
    s = settings(use_excl)
    mc_smooth_jax(pos.copy(), dtn, fixed, 0.5 * bond, s)  # compile
    t = time.perf_counter()
    mc_smooth_jax(pos, dtn, fixed, 0.5 * bond, s)
    return time.perf_counter() - t


print(f"{'beads':>8s} {'chain only':>11s} {'chain + EV':>11s} {'EV share':>9s} {'us/step':>9s}")
for n in (1024, 2048, 4096, 8192, 16384):
    if n > len(P):
        continue
    t_off = run(n, False)
    t_on = run(n, True)
    print(
        f"{n:>8,d} {t_off:>10.2f}s {t_on:>10.2f}s "
        f"{100 * (1 - t_off / t_on):>8.0f}% {t_on / STEPS * 1e6:>9.2f}",
        flush=True,
    )

print("\nFlat us/step across bead counts means the kernel is latency bound and a neighbour list")
print("saves nothing. Growth with the bead count means the arithmetic dominates and one would.")
