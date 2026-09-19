"""The cell grid in the JAX smooth kernel: exact against the full scan, and composes with
prefetching and the wall.

    JAX_PLATFORMS=cuda python harness/test_jax_grid.py

Runs on a GPU only. On the CPU backend XLA copies the whole grid on every relink scatter, so a
2,500 bead chain takes tens of minutes there, and the check is skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_smooth_levers import _jax_available, chain, check, settings, under  # noqa: E402


def compact_chain(n: int = 2500, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A dense random walk with anchors every 25 beads, so cells hold many beads."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n, 3)).astype(np.float64)
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    pos = np.cumsum(steps, axis=0) * 0.6
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[::25] = True
    fixed[-1] = True
    return pos, fixed, np.ones(n - 1, dtype=np.float64)


def _gpu() -> bool:
    if not _jax_available():
        print("  skip  JAX not available")
        return False
    import jax

    if jax.default_backend() == "cpu":
        print("  skip  the grid check needs a GPU; XLA on the CPU copies the grid per relink")
        return False
    return True


def test_grid_matches_full_scan() -> None:
    if not _gpu():
        return
    from gnome3d.mc import jax as mc_jax

    for seed in (0, 1):
        runs = {}
        for grid in (False, True):
            pos, fixed, dtn = compact_chain(seed=seed)
            s = settings(smooth_hard_wall=True, smooth_jax_grid=grid, smooth_jax_grid_min_beads=1)
            s.mc_smooth_chains = 1
            s.mc_stop_steps_smooth = 4000
            e = mc_jax.mc_smooth_jax(pos, dtn, fixed, 0.5, s)
            runs[grid] = (e, pos.copy())
        e0, p0 = runs[False]
        e1, p1 = runs[True]
        check(
            f"grid on and off follow the same trajectory (seed {seed})",
            np.allclose(p0, p1, atol=1e-3),
            f"max position difference {np.abs(p0 - p1).max():.2e}",
        )
        check(
            f"and carry the same energy (seed {seed})",
            abs(e0 - e1) <= 1e-3 * max(abs(e0), 1e-6),
            f"full {e0:.5f}, grid {e1:.5f}",
        )


def test_grid_with_prefetch_and_wall() -> None:
    if not _gpu():
        return
    from gnome3d.mc import jax as mc_jax

    means = {}
    for grid in (False, True):
        s = settings(
            smooth_hard_wall=True,
            smooth_prefetch=8,
            smooth_jax_grid=grid,
            smooth_jax_grid_min_beads=1,
        )
        probs = []
        starts = []
        for seed in range(1, 5):
            p_, f_, d_ = chain(seed=seed)
            starts.append(under(p_, 0.7))
            probs.append(
                {
                    "pos": p_,
                    "dtn": d_,
                    "fixed": f_,
                    "step_size": 0.5,
                    "settings": s,
                    "seed": seed,
                    "char_orientations": None,
                    "anchor_neighbors": None,
                    "anchor_neighbor_weights": None,
                    "heat_dist": None,
                    "compartment": None,
                }  # fmt: skip
            )
        out = mc_jax.mc_smooth_jax_batch(probs, s)
        ends = [under(np.asarray(o[1]), 0.7) for o in out]
        means[grid] = float(np.mean([float(o[0]) for o in out]))
        if grid:
            check(
                "grid with prefetch: the wall never lets the count rise in any chain",
                all(e <= b for e, b in zip(ends, starts, strict=True)),
                f"{starts} -> {ends}",
            )
    check(
        "grid with prefetch: the mean final energy over seeds matches the full scan's",
        abs(means[True] - means[False]) <= 0.5 * means[False],
        f"full {means[False]:.4f}, grid {means[True]:.4f}",
    )


def main() -> int:
    test_grid_matches_full_scan()
    test_grid_with_prefetch_and_wall()
    from test_smooth_levers import FAIL

    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
