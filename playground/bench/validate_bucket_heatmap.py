"""Validate heatmap-MC shape bucketing: bucketed == non-bucketed.

Unlike the batched-IB path (statistical equivalence), heatmap bucketing should
be BIT-IDENTICAL: padding is fully inert (pad beads placed far -> EV=0, skip=True
heat rows -> heat=0) and `n_active` keeps the move sampler in [0, n), so with the
same RNG seed (same `label`) the real-bead trajectory is identical. The only
difference is extra +0.0 terms in the reductions, which IEEE preserves.

So the bar here is: max|pos_bucketed - pos_unbucketed| ~ 0 and score match.
We test N straddling bucket boundaries, with excluded volume ON (the term that
relies on far-placement inertness).

Run on the CUDA box:
    python playground/validate_bucket_heatmap.py
    JAX_LOG_COMPILES=1 python playground/validate_bucket_heatmap.py   # see bucket reuse
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from gnome3d import log  # noqa: E402
from gnome3d.mc_jax import _bucket_for, mc_heatmap_jax  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def make_heatmap(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, size=(n, 3)), axis=0).astype(np.float32)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    exp_dist = 1.0 + np.sqrt(sep)
    exp_dist[rng.random((n, n)) < 0.3] = 0.0  # ~30% no-contact cells
    exp_dist = ((exp_dist + exp_dist.T) / 2).astype(np.float32)  # symmetric
    np.fill_diagonal(exp_dist, 0.0)
    return pos, exp_dist


def run(pos, exp_dist, diag_size, bucket: bool):
    s = Settings()
    s.mc_backend = "jax"
    s.mc_backend_apply_to_heatmap = True
    s.use_excluded_volume = True
    s.exclusion_apply_to_heatmap = True
    s.mc_stop_steps_heatmap = 2000
    s.jax_bucket_shapes = bucket
    p = pos.copy()  # mc_heatmap_jax mutates in place
    with log.scope("val"):
        score = mc_heatmap_jax(p, exp_dist, diag_size, 0.5, s)
    return p, float(score)


def main():
    diag_size = 2
    print(
        f"{'N':>6} {'bucket':>7} {'max|Δpos|':>12} {'score off':>12} {'score on':>12} {'verdict':>8}"
    )
    print("-" * 62)
    all_ok = True
    for n in [200, 300, 500, 1000, 1500, 2011]:
        pos, exp_dist = make_heatmap(n)
        p_off, s_off = run(pos, exp_dist, diag_size, bucket=False)
        p_on, s_on = run(pos, exp_dist, diag_size, bucket=True)
        dmax = float(np.abs(p_off - p_on).max())
        sdiff = abs(s_off - s_on)
        ok = dmax < 1e-3 and sdiff < 1e-2
        all_ok = all_ok and ok
        print(
            f"{n:>6} {_bucket_for(n):>7} {dmax:>12.2e} {s_off:>12.4f} {s_on:>12.4f} "
            f"{'PASS' if ok else 'FAIL':>8}"
        )
    print(
        f"\n{'PASS' if all_ok else 'FAIL'}: bucketed heatmap MC "
        f"{'is bit-identical (within f32) to unbucketed' if all_ok else 'DIVERGES — padding not inert, inspect'}"
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
