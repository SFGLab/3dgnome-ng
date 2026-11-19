"""The JAX kernels.

Each kernel imports JAX on first use, builds and caches its compiled function and has an entry
for one problem and a batched entry that anneals several IBs in one vmapped kernel.

  smooth       chain bonds and angles, with excluded volume, heat and orientation
  arcs         anchor springs, with excluded volume and confinement
  arcs_energy  the energy of the arcs solver on the device
"""

from gnome3d.mc.jax.arcs import mc_arcs_jax, mc_arcs_jax_batch
from gnome3d.mc.jax.smooth import mc_smooth_jax, mc_smooth_jax_batch

__all__ = [
    "mc_arcs_jax",
    "mc_arcs_jax_batch",
    "mc_smooth_jax",
    "mc_smooth_jax_batch",
]
