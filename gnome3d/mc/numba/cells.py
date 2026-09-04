"""A uniform cell grid over bead positions, so the excluded volume term visits the beads that
are near instead of every bead in the structure.

The term sums over pairs closer than `r0`, and only about fifty beads are ever that close
whatever the structure's size, because that is a local density. A full scan therefore does
hundreds of times more work than it needs on a chromosome: measured on a finished 60 Mb region,
42,480 beads scanned for the 57 that matter, and the term is 99 percent of an MC step.

The grid is a linked list per cell rather than a packed array, so a bead that moves is unlinked
from its old cell and linked into its new one in constant time. That keeps the cells exactly
`r0` wide with no margin for drift, which is what makes the neighbourhood small, and it keeps
the grid exact after every accepted move rather than approximately right until a rebuild.

The query and the relink themselves live in `terms.py`, because they run inside the step loop
and that module may not import this one. This module builds the grid and is what a driver
calls. The sum runs in ascending bead index, the order a full scan uses, so results are
identical bit for bit and a run's trajectory does not change.
"""

from __future__ import annotations

import numpy as np

from gnome3d.mc.numba.terms import cell_of_nb, local_excl_cells_nb, njit, relink_nb
from gnome3d.types import F64Array, I32Array, I64Array

__all__ = ["BUF", "MAX_CELLS", "build_grid", "cell_of", "grid_shape", "local_excl_cells", "relink"]

# Cells are grown until the grid fits in this many, so a sparse structure cannot ask for an
# enormous array. Larger cells only mean more candidates per query.
MAX_CELLS: int = 4_000_000
# A query collects the beads inside the radius into a fixed buffer. Overflow is reported rather
# than truncated and the caller falls back to the full scan, so correctness never depends on
# this being large enough.
BUF: int = 4096

cell_of = cell_of_nb
relink = relink_nb
local_excl_cells = local_excl_cells_nb


@njit(cache=True, nogil=True)
def grid_shape(pos: F64Array, cell: float) -> tuple[F64Array, I64Array, float]:
    """Origin, dimensions and the cell size actually used for `pos` at cell size `cell`."""
    lo = np.empty(3, dtype=np.float64)
    dim = np.empty(3, dtype=np.int64)
    lo[0] = pos[:, 0].min()
    lo[1] = pos[:, 1].min()
    lo[2] = pos[:, 2].min()
    ex = pos[:, 0].max() - lo[0]
    ey = pos[:, 1].max() - lo[1]
    ez = pos[:, 2].max() - lo[2]
    c = cell if cell > 1e-12 else 1.0
    while True:
        nx = int(ex / c) + 1
        ny = int(ey / c) + 1
        nz = int(ez / c) + 1
        if nx * ny * nz <= MAX_CELLS or c > 1e12:
            dim[0] = nx
            dim[1] = ny
            dim[2] = nz
            return lo, dim, c
        c *= 2.0


@njit(cache=True, nogil=True)
def build_grid(
    pos: F64Array, lo: F64Array, dim: I64Array, c: float
) -> tuple[I32Array, I32Array, I32Array]:
    """Link every bead into its cell. Returns the per cell heads, the next pointers and each
    bead's cell, the last so a move can unlink from the right chain."""
    n = pos.shape[0]
    head = np.full(int(dim[0]) * int(dim[1]) * int(dim[2]), -1, dtype=np.int32)
    nxt = np.full(n, -1, dtype=np.int32)
    where = np.empty(n, dtype=np.int32)
    for i in range(n):
        k = cell_of_nb(pos[i, 0], pos[i, 1], pos[i, 2], lo, dim, c)
        where[i] = k
        nxt[i] = head[k]
        head[k] = i
    return head, nxt, where
