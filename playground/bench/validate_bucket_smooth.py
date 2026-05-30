"""Validate smooth-MC shape bucketing: bucketed == unbucketed, bit-identically.

Smooth bucketing pads N to a bucket and masks pad beads out of every energy term
via a scalar `n_active` (chain bonds/angles + EV + confinement), zeroed heat rows,
and `movable` excludes them so they never move.  Because the move sampler draws
from the real `movable` set (unchanged by padding) and every term masks the pad
tail, the real-bead trajectory is IDENTICAL given the same RNG seed -> the result
must be bit-identical (extra +0.0 terms only), not merely statistically close.

We test N straddling bucket boundaries with the full production term set
(chain + EV + heat + CTCF orientation) so every masked path is exercised.

Run on the CUDA box:
    python playground/validate_bucket_smooth.py
    JAX_LOG_COMPILES=1 python playground/validate_bucket_smooth.py   # see bucket reuse
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from gnome3d.mc_jax import _bucket_for, mc_smooth_jax  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def make_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, size=(n, 3)), axis=0).astype(np.float32)
    dtn = np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32)
    dtn = np.append(dtn, dtn[-1]).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[::12] = True  # ~8% anchors
    # dense positive subanchor heat target with ~30% no-contact cells
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    heat = 1.0 + np.sqrt(sep)
    heat[rng.random((n, n)) < 0.3] = 0.0
    heat = ((heat + heat.T) / 2).astype(np.float32)
    np.fill_diagonal(heat, 0.0)
    return pos, dtn, fixed, heat


def make_orientation(fixed, half_window=4):
    n_anchors = int(fixed.sum())
    chars = ["L" if i % 2 == 0 else "R" for i in range(n_anchors)]
    neighbors, weights = {}, {}
    for k in range(n_anchors):
        nb = [j for j in range(max(0, k - half_window), min(n_anchors, k + half_window + 1)) if j != k]
        neighbors[k] = nb
        weights[k] = [1.0] * len(nb)
    return chars, neighbors, weights


def run(pos, dtn, fixed, heat, orn, bucket):
    s = Settings()
    s.mc_backend = "jax"
    s.mc_backend_apply_to_smooth = True
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.mc_stop_steps_smooth = 2000
    s.jax_bucket_shapes = bucket
    if not hasattr(s, "motif_weight"):
        s.motif_weight = 1.0
    kw = {}
    if orn:
        chars, nbrs, wts = make_orientation(fixed)
        kw = {"char_orientations": chars, "anchor_neighbors": nbrs, "anchor_neighbor_weights": wts}
    p = pos.copy()  # mutated in place
    score = mc_smooth_jax(p, dtn, fixed, 0.5, s, heat_dist=heat, label="val", **kw)
    return p, float(score)


def main():
    print(f"{'N':>6} {'bucket':>7} {'terms':>22} {'max|Δpos|':>12} {'Δscore':>10} {'verdict':>8}")
    print("-" * 70)
    all_ok = True
    for n in [200, 300, 1000, 1500, 2011]:
        pos, dtn, fixed, heat = make_problem(n)
        for orn, label in [(False, "chain+EV+heat"), (True, "chain+EV+heat+orient")]:
            p_off, s_off = run(pos, dtn, fixed, heat, orn, bucket=False)
            p_on, s_on = run(pos, dtn, fixed, heat, orn, bucket=True)
            dmax = float(np.abs(p_off - p_on).max())
            sdiff = abs(s_off - s_on)
            ok = dmax < 1e-3 and sdiff < 1e-2
            all_ok = all_ok and ok
            print(
                f"{n:>6} {_bucket_for(n):>7} {label:>22} {dmax:>12.2e} {sdiff:>10.2e} "
                f"{'PASS' if ok else 'FAIL':>8}"
            )
    print(
        f"\n{'PASS' if all_ok else 'FAIL'}: bucketed smooth MC "
        f"{'is bit-identical (within f32) to unbucketed' if all_ok else 'DIVERGES — a term is not masking padding, inspect'}"
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
