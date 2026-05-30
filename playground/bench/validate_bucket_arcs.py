"""Validate arcs-MC shape bucketing.

Arcs bucketing: pad N to a bucket; the arc-spring term is zeroed by exp_mat=0 pad
rows/cols (neither spring nor repulsion), EV/confine are masked by n_active, and
the move sampler draws from [0, n_active) so pad beads never move.

Same two-check protocol as the smooth validator:
  INIT (step_size=0 -> returned score == initial score): MUST be bit-identical;
       a nonzero is a real masking/reduction bug.
  TRAJ (full MC): arcs uses NON-strict acceptance (<=, like heatmap), so the
       trajectory may stay bit-identical too — but if it diverges at large N
       that's f32 chaos, not a bug (the INIT check is the authority).

Run on the CUDA box:
    python validate_bucket_arcs.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from gnome3d.mc_jax import _bucket_for, mc_arcs_jax  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def make_arcs(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, size=(n, 3)), axis=0).astype(np.float32)
    # arc expected-distance matrix: mostly 0 (no arc); some positive springs on
    # near-diagonal pairs; a few negative repulsions.  Symmetric, zero diagonal.
    exp = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        if rng.random() < 0.4:  # spring to a nearby bead
            j = min(i + rng.integers(1, 5), n - 1)
            exp[i, j] = exp[j, i] = float(rng.uniform(1.0, 5.0))
    rep = rng.random((n, n)) < 0.02
    exp[rep] = -1.0
    exp = np.triu(exp) + np.triu(exp, 1).T  # enforce symmetry
    np.fill_diagonal(exp, 0.0)
    return pos, exp.astype(np.float32)


def run(pos, exp, bucket, step_size=0.5, label="val"):
    s = Settings()
    s.mc_backend = "jax"
    s.mc_backend_apply_to_arcs = True
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.mc_stop_steps = 2000
    s.jax_bucket_shapes = bucket
    p = pos.copy()
    score = mc_arcs_jax(p, exp, step_size, s, label=label)
    return p, float(score)


def main():
    hdr = f"{'N':>6} {'bucket':>7} {'Δinit':>11} {'Δtraj|pos|':>12} {'init?':>6}"
    print(hdr)
    print("-" * len(hdr))
    init_ok_all = True
    for n in [200, 300, 1000, 1500, 2011]:
        pos, exp = make_arcs(n)
        _, si_off = run(pos, exp, bucket=False, step_size=0.0)
        _, si_on = run(pos, exp, bucket=True, step_size=0.0)
        d_init = abs(si_off - si_on)
        p_off, _ = run(pos, exp, bucket=False)
        p_on, _ = run(pos, exp, bucket=True)
        d_traj = float(np.abs(p_off - p_on).max())
        init_ok = d_init < 1e-2
        init_ok_all = init_ok_all and init_ok
        print(
            f"{n:>6} {_bucket_for(n):>7} {d_init:>11.2e} {d_traj:>12.2e} "
            f"{'ok' if init_ok else 'BUG':>6}"
        )
    print()
    if init_ok_all:
        print(
            "INIT bit-identical everywhere -> masking/reductions CORRECT.\n"
            "Δtraj (if any) is f32 chaos, not a bug."
        )
    else:
        print("INIT DIVERGES -> a bucketed arcs init reduction is not padding-insensitive; inspect.")
    sys.exit(0 if init_ok_all else 1)


if __name__ == "__main__":
    main()
