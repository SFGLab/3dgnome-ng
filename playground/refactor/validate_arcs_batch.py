"""Validate mc_arcs_jax_batch == sequential mc_arcs_jax (the new batched arcs
kernel).

Like the smooth region-batch validation, the bar is STATISTICAL: batched and
sequential seed RNG differently, so per-chain trajectories diverge under f32
chaos.  What this MUST catch: shape / in_axes / per-chain-convergence /
freeze-converged bugs in kernel_full_mp.  If batched runs clean, returns finite
per-IB scores at the right shapes, and lands in the sequential ballpark, the
multi-problem plumbing is correct.

Runs on CPU-JAX (slow) or GPU.
    python playground/refactor/validate_arcs_batch.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from gnome3d import log  # noqa: E402
from gnome3d.mc import mc_arcs_jax, mc_arcs_jax_batch  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def make_arc_ib(n: int, seed: int):
    """Synthetic IB: random anchor positions + an exp-distance matrix with a mix
    of arcs (positive targets) and default repulsion (-1)."""
    rng = np.random.default_rng(seed)
    pos = (rng.standard_normal((n, 3)) * 50.0).astype(np.float32)
    exp = np.full((n, n), -1.0, dtype=np.float64)
    np.fill_diagonal(exp, 0.0)
    for _ in range(n):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j:
            d = float(rng.uniform(20.0, 80.0))
            exp[i, j] = exp[j, i] = d
    return pos, exp


def main() -> int:
    log.setup(0)
    s = Settings()
    s.load_ini("data/GM12878/config_dryrun.ini")
    s.mc_backend = "jax"

    sizes = [40, 64, 100, 48]  # different N -> exercises bucketing + per-chain
    ibs = [make_arc_ib(n, seed=i) for i, n in enumerate(sizes)]
    step = 0.005

    print("  sequential mc_arcs_jax (per IB)...")
    seq_scores = [mc_arcs_jax(pos.copy(), exp, step, s) for pos, exp in ibs]

    print("  batched mc_arcs_jax_batch (one kernel)...")
    batch = mc_arcs_jax_batch(
        [{"pos": pos, "exp_dist": exp, "step_size": step} for pos, exp in ibs], s
    )

    ok = True
    # shapes + finiteness
    shapes_ok = all(
        bp.shape == (n, 3) and np.isfinite(bp).all() and np.isfinite(sc)
        for (sc, bp), n in zip(batch, sizes, strict=True)
    )
    ok &= shapes_ok
    print(f"  {'ok ' if shapes_ok else 'FAIL'} per-IB shapes (n,3) + finite scores")

    print(f"\n  {'N':>4}  {'seq score':>12}  {'batch score':>12}  {'rel diff':>9}")
    print("  " + "-" * 46)
    rels = []
    for (sc_b, _), sc_s, n in zip(batch, seq_scores, sizes, strict=True):
        rel = abs(sc_b - sc_s) / max(abs(sc_s), 1e-9)
        rels.append(rel)
        print(f"  {n:>4}  {sc_s:>12.3f}  {sc_b:>12.3f}  {rel:>8.1%}")
    ensemble_rel = abs(np.mean([b[0] for b in batch]) - np.mean(seq_scores)) / max(
        abs(np.mean(seq_scores)), 1e-9
    )
    ballpark = ensemble_rel < 0.5
    ok &= ballpark
    print(f"\n  ensemble rel diff = {ensemble_rel:.1%}")
    print(f"  {'ok ' if ballpark else 'FAIL'} batched lands in sequential ballpark")

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
