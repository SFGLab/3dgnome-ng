"""The Monte Carlo kernels.

  numba  the numba kernels
  jax    the JAX kernels, per IB and batched

Numba compiles on the first call. JAX is imported on first use.
"""

from __future__ import annotations

from gnome3d.mc.jax import (
    mc_arcs_jax,
    mc_arcs_jax_batch,
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
    "mc_heatmap_numba",
    "mc_ib_numba",
    "mc_smooth_jax",
    "mc_smooth_jax_batch",
    "mc_smooth_numba",
]
