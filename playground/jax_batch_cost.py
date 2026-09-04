"""How the JAX smooth kernel's cost scales with the number of regions in a launch.

A single chain on the device is latency bound: about ten microseconds a step whatever the bead
count, because the chain is sequential and each step is one small reduction. Batching many
regions into one launch shares that latency, so the cost per chain step falls as the batch
widens, until the arithmetic takes over and it stops falling.

Where that turn happens decides two things. Whether the batched kernel is worth its complexity
against a threaded CPU kernel, and whether a neighbour list would pay for itself: it cuts the
arithmetic, so it only helps on the side of the turn where the arithmetic dominates.

    python playground/jax_batch_cost.py <cif> [steps]
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
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000

rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
P = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float32)
S = np.array([int(r[16]) for r in rows])
K = np.array([r[5] == "ALA" for r in rows])
o = np.argsort(S)
P, K = P[o], K[o]
bond = float(np.median(np.linalg.norm(np.diff(P.astype(np.float64), axis=0), axis=1)))


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


def problems(n: int, k: int) -> list[dict[str, object]]:
    pos = np.ascontiguousarray(P[:n])
    fixed = np.ascontiguousarray(K[:n])
    dtn = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1).astype(np.float32)
    return [
        {"pos": pos.copy(), "dtn": dtn, "fixed": fixed, "step_size": 0.5 * bond, "seed": i}
        for i in range(k)
    ]


def run(n: int, k: int, use_excl: bool) -> float:
    s = settings(use_excl)
    mc_smooth_jax_batch(problems(n, k), s)  # compile
    t = time.perf_counter()
    mc_smooth_jax_batch(problems(n, k), s)
    return time.perf_counter() - t


for n in (4096, 16384):
    if n > len(P):
        continue
    print(f"\n{n:,} beads per region, {STEPS:,} steps\n")
    print(
        f"{'regions':>8s} {'chain only':>11s} {'chain + EV':>11s} {'EV share':>9s} "
        f"{'us/step':>9s} {'us/step/region':>15s} {'scaling':>8s}"
    )
    base = None
    for k in (1, 4, 8, 16, 32):
        t_off = run(n, k, False)
        t_on = run(n, k, True)
        per = t_on / STEPS * 1e6
        per_region = per / k
        if base is None:
            base = per_region
        print(
            f"{k:>8d} {t_off:>10.2f}s {t_on:>10.2f}s {100 * (1 - t_off / t_on):>8.0f}% "
            f"{per:>9.2f} {per_region:>15.3f} {base / per_region:>7.1f}x",
            flush=True,
        )

print("\nScaling near the region count means the latency is being shared and the kernel is still")
print("latency bound. Scaling that flattens means the arithmetic has taken over, and that is the")
print("regime where a neighbour list would cut the cost.")
