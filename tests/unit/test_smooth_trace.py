"""Unit checks for the `on_round` callback of `mc_smooth_numba`."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.mc import numba as mc_numba
from gnome3d.settings import Settings


def chain(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(n, 3)), axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[0, n - 1]] = True
    return pos, fixed, np.ones(n - 1, dtype=np.float32)


def settings() -> Settings:
    s = Settings()
    s.use_excluded_volume = False
    s.use_confinement = False
    s.use_motif_orientation = False
    s.mc_smooth_chains = 1
    s.mc_stop_steps_smooth = 2000
    return s


def test_the_callback_is_called_at_least_once_per_run_rounds() -> None:
    frames: list[np.ndarray] = []
    pos, fixed, dtn = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(
        pos, dtn, fixed, 0.5, settings(), on_round=lambda p: frames.append(p.copy())
    )
    assert len(frames) >= 1, f"the callback is called at least once per run: {len(frames)} rounds"
    assert all(f.shape == pos.shape for f in frames), "each frame holds every bead"
    assert np.allclose(frames[-1], pos, atol=1e-6), "the last frame is the returned chain"
    assert len(frames) < 2 or not np.array_equal(frames[0], frames[-1]), (
        "frames differ, so they are snapshots not one array"
    )
    pos2, fixed2, dtn2 = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos2, dtn2, fixed2, 0.5, settings())
    assert np.array_equal(pos, pos2), "the callback changes nothing about the result"
