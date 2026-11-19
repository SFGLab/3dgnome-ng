"""Geometry for drawing a structure."""

from __future__ import annotations

import numpy as np

from gnome3d.types import F64Array


def smooth_chain(pos: F64Array, per_bond: int = 4) -> F64Array:
    """Returns a Catmull-Rom spline through the beads with `per_bond` points per bond.

    Fewer than three beads or a `per_bond` of one return the beads unchanged.
    """
    p = np.asarray(pos, dtype=np.float64)
    if len(p) < 3 or per_bond <= 1:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0.0, 1.0, per_bond, endpoint=False)
    t2, t3 = t * t, t * t * t
    b0 = -0.5 * t3 + t2 - 0.5 * t
    b1 = 1.5 * t3 - 2.5 * t2 + 1.0
    b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    b3 = 0.5 * t3 - 0.5 * t2
    p0, p1, p2, p3 = ext[:-3], ext[1:-2], ext[2:-1], ext[3:]
    out = (
        b0[None, :, None] * p0[:, None, :]
        + b1[None, :, None] * p1[:, None, :]
        + b2[None, :, None] * p2[:, None, :]
        + b3[None, :, None] * p3[:, None, :]
    ).reshape(-1, 3)
    return np.vstack([out, p[-1]])
