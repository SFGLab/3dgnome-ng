"""
Monte-Carlo backends, as a package.

  * `dispatch`  - the public `mc_arcs`/`mc_smooth`/`mc_heatmap`/`mc_ib` entries
    that route to numba or JAX per `settings.mc_backend*`, plus GNOME3D_MC_PROFILE
  * `numba`     - the numba (`@njit`) kernels
  * `jax`       - the JAX kernels: per-IB `mc_*_jax` and region-batched
    `mc_*_jax_batch`, the shape-bucketing, and the compiled-kernel cache

This `__init__` re-exports the flat API so `from gnome3d.mc import mc_smooth` (and
the batch/kernel entries) keep working after the split.  Backend submodules stay
import-cheap (numba compiles on first call; JAX imports lazily in `_ensure_jax`).
"""

from __future__ import annotations

from gnome3d.mc.jax import (
    mc_arcs_jax,
    mc_arcs_jax_batch,
    mc_heatmap_jax,
    mc_smooth_jax,
    mc_smooth_jax_batch,
)
from gnome3d.mc.numba import (
    mc_arcs_numba,
    mc_heatmap_numba,
    mc_ib_numba,
    mc_smooth_numba,
)

__all__ = [
    "mc_arcs_jax",
    "mc_arcs_jax_batch",
    "mc_arcs_numba",
    "mc_heatmap_jax",
    "mc_heatmap_numba",
    "mc_ib_numba",
    "mc_smooth_jax",
    "mc_smooth_jax_batch",
    "mc_smooth_numba",
]
