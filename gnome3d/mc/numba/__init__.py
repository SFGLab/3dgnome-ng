"""The numba kernels.

`mc_heatmap_numba`, `mc_arcs_numba`, `mc_smooth_numba` and `mc_ib_numba` are the CPU entry
points. They share one inner kernel, `terms.batch_mc_nb` and one driver,
`common.run_outer_loop`.

  terms   the energy terms and the inner kernel
  common  the driver, the orientation arrays and array helpers
  arcs, smooth, heatmap, ib  one entry point each

The first import compiles the kernels, 10 to 30 s. Later runs load the numba disk cache.
"""

from __future__ import annotations

from gnome3d.mc.numba.arcs import mc_arcs_numba
from gnome3d.mc.numba.heatmap import mc_heatmap_numba
from gnome3d.mc.numba.ib import mc_ib_numba
from gnome3d.mc.numba.smooth import mc_smooth_numba
from gnome3d.mc.numba.terms import seed_numba

__all__ = [
    "mc_arcs_numba",
    "mc_heatmap_numba",
    "mc_ib_numba",
    "mc_smooth_numba",
    "seed_numba",
]
