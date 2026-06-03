"""Numba backend for the Monte-Carlo loops, split one file per kernel.

The four public entry points - `mc_heatmap_numba`, `mc_arcs_numba`,
`mc_smooth_numba`, `mc_ib_numba` - are the production CPU kernels.  They all
drive one unified inner kernel (`terms._batch_mc_nb`) via the shared driver
(`common._run_outer_loop`); they differ only in which structure-energy variant
and which optional terms (heat / orientation / EV / confinement) they wire up.

Layout:
  * `terms`   - all `_local_*_nb` / `_init_*_nb` term math + `_batch_mc_nb`
  * `common`  - the `_run_outer_loop` driver, orientation prep, array helpers
  * `arcs` / `smooth` / `heatmap` / `ib` - one public entry per kernel

This `__init__` re-exports the flat API so `from gnome3d.mc import numba as
mc_numba; mc_numba.mc_arcs_numba(...)` keeps working after the split.  On first
import the JIT functions compile (~10-30 s); subsequent runs load from numba's
disk cache.
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
