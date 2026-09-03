"""The genomic excluded volume floor for the arcs stage.

An anchor pair with no arc has no target distance, only a scale free `1/d` repulsion, so
anchors inside a block settle into a ball whose size does not depend on genomic separation.
The floor gives such a pair an excluded volume radius that grows with separation,
`scale * (s / 1000)^nu`, with `nu` taken from the cell line's own contact probability curve.
It is one sided. Arc pairs keep their springs and can still pull distant loci together.

The scale is calibrated on the structure itself. `beta` over the median consecutive anchor
distance came out at 0.44 within six percent across five models from three data sources, so
the arcs stage anneals once, reads `bond_scale` off the result, and anneals again with the
floor on. See design/anchor-placement.md.
"""

from __future__ import annotations

import numpy as np

from gnome3d.types import F32Array, F64Array


def genomic_floor_matrix(
    anchor_genomic: list[tuple[int, int, int]], exp_dist: F64Array, scale: float, nu: float
) -> F64Array:
    """One excluded volume radius per anchor pair.

    Parameters
    ----------
    anchor_genomic
        (start, end, midpoint) in bp per anchor, in anchor index order.
    exp_dist
        The arcs expected distance matrix. Negative marks an arcless pair.
    scale, nu
        The floor is `scale * (separation / 1000)^nu` for arcless pairs and zero, which the
        term treats as skip, for arc pairs and the diagonal.
    """
    mid = np.array([m for _, _, m in anchor_genomic], dtype=np.float64)
    sep = np.abs(mid[:, None] - mid[None, :])
    r0 = scale * (sep / 1000.0) ** nu
    r0[np.asarray(exp_dist) >= 0.0] = 0.0
    np.fill_diagonal(r0, 0.0)
    return np.ascontiguousarray(r0, dtype=np.float64)


def arcs_without_repulsion(exp_dist: F64Array) -> F64Array:
    """A copy of the arcs matrix with arcless pairs set to zero, so the arcs term skips them
    and the floor is the only term they carry."""
    out = np.array(exp_dist, dtype=np.float64, copy=True)
    out[out < 0.0] = 0.0
    return out


def bond_scale(pos: F32Array | F64Array) -> float:
    """Median distance between consecutive anchors, the length the floor is calibrated on.
    Zero when there is no pair."""
    p = np.asarray(pos, dtype=np.float64)
    if p.shape[0] < 2:
        return 0.0
    return float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
