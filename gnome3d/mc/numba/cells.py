"""A uniform cell grid over bead positions for the excluded volume term.

The term sums over pairs closer than `r0`. The grid is a linked list per cell. A bead that
moves is unlinked from its old cell and linked into its new one in constant time and the grid
is exact after every accepted move. The query and the relink are in `terms.py`, which may not
import this module. The sum runs in ascending bead index, the order of a full scan. The grid
and the scan return the same bits on ARM. On x86-64 the compiler adds the pairs of the scan in
another order under fastmath and the two agree to rounding.
"""

from __future__ import annotations

import numpy as np

from gnome3d.mc.numba.terms import cell_of_nb, local_excl_cells_nb, njit, relink_nb
from gnome3d.types import F64Array, I32Array, I64Array

__all__ = ["BUF", "MAX_CELLS", "build_grid", "cell_of", "grid_shape", "local_excl_cells", "relink"]

# The cell size grows until the grid fits in this many cells.
MAX_CELLS: int = 4_000_000
# The buffer of a query. On overflow the caller falls back to the full scan.
BUF: int = 4096

cell_of = cell_of_nb
relink = relink_nb
local_excl_cells = local_excl_cells_nb


@njit(cache=True, nogil=True)
def grid_shape(pos: F64Array, cell: float) -> tuple[F64Array, I64Array, float]:
    """Returns the origin, the dimensions and the cell size used for `pos` at cell size `cell`."""
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
    """Links every bead into its cell. Returns the head per cell, the next pointers and the cell
    of each bead.
    """
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
