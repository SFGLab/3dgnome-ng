"""Benchmark numba vs torch backends for mc_heatmap.

Runs the same starting configuration through both backends with identical
settings and reports:
  - wall time per backend
  - final score per backend (stochastic - they won't be equal, but both
    should converge to similar magnitudes)

Usage:
    .venv/bin/python playground/bench_heatmap_mc.py [N] [stop_steps]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from gnome3d import log
from gnome3d.mc import mc_heatmap
from gnome3d.settings import Settings


def make_problem(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    # Random initial bead positions in a unit cube.
    pos = rng.standard_normal((n, 3)).astype(np.float32) * float(n) ** (1.0 / 3.0)

    # Synthetic expected-distance matrix: distance grows with |i-j| (1D-like
    # chain target) plus some long-range scatter so the heatmap MC has signal
    # both near the diagonal and far from it.
    idx = np.arange(n)
    gen = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
    exp_dist = np.sqrt(gen + 1.0)  # sub-linear in genomic distance

    # Zero out a fraction so the skip mask is non-trivial.
    keep_frac = 0.8
    mask = rng.random(exp_dist.shape) < keep_frac
    mask = mask | mask.T  # symmetric
    exp_dist = np.where(mask, exp_dist, 0.0).astype(np.float32)
    np.fill_diagonal(exp_dist, 0.0)

    diag_size = 1
    return pos, exp_dist, diag_size


def bench(label: str, backend: str, device: str | None = None) -> tuple[float, float, float]:
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    stop_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    pos, exp_dist, diag_size = make_problem(n_arg)

    s = Settings()
    s.mc_backend = backend
    if device is not None:
        s.torch_device = device
    s.mc_torch_candidates = 32
    s.mc_stop_steps_heatmap = stop_arg
    # Loosen schedule so it converges within a few batches.
    s.max_temp_heatmap = 5.0
    s.dt_temp_heatmap = 0.9999
    s.mc_stop_improvement_heatmap = 0.99
    s.mc_stop_successes_heatmap = 1

    pw = pos.copy()
    t0 = time.perf_counter()
    with log.scope(label):
        final = mc_heatmap(pw, exp_dist, diag_size, 0.5, s)
    t1 = time.perf_counter()

    print(
        f"{label:<22} backend={backend:<6} "
        f"N={n_arg:>4} steps/batch={stop_arg:>5}  "
        f"time={t1 - t0:6.2f}s  final_score={final:.2f}"
    )
    return t1 - t0, float(final), float(pw.mean())


if __name__ == "__main__":
    print("Warmup (JIT compile)...")
    bench("warmup numba", "numba")

    print("\n--- BENCHMARK ---")
    bench("numba", "numba")
    bench("torch (mps)", "torch", "mps")
    bench("torch (cpu)", "torch", "cpu")
