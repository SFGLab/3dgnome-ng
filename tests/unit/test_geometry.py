"""Unit checks for the chain smoothing of `gnome3d/geometry.py`."""

from __future__ import annotations

import numpy as np

from gnome3d.geometry import smooth_chain


def test_four_points_per_bond_and_the_last_bead() -> None:
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(20, 3)), axis=0)
    sm = smooth_chain(pos, per_bond=4)
    assert sm.shape == (19 * 4 + 1, 3), "four points per bond and the last bead: " + str(sm.shape)
    assert np.allclose(sm[::4], pos), "the curve passes through every bead"
    line = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], axis=1)
    assert np.allclose(smooth_chain(line, 5)[:, 1:], 0.0), "a straight chain stays straight"
    assert np.array_equal(smooth_chain(pos[:2], 4), pos[:2]), "two beads are returned as they are"
