"""JAX backend for the Monte-Carlo loops, split one file per kernel.

Each kernel family is self-contained - it lazily imports JAX (via
`jax_is_available`), builds and memoises its compiled kernel, and exposes a
single-problem entry plus (for arcs/smooth) a region-batched entry that anneals K
different IBs in one vmapped kernel.

Layout:
  * `smooth`  - chain bonds + angles (+ EV / heat / CTCF orientation); the hot path
  * `arcs`    - anchor springs to expected distances (+ EV / confinement)
  * `heatmap` - pairwise distance-to-expected (+ EV); simplest energy

This `__init__` re-exports the flat API so `from gnome3d.mc import jax as mc_jax;
mc_jax.mc_smooth_jax_batch(...)` keeps working after the split.  JAX is an optional
extras dep, imported lazily on first kernel use.
"""

from gnome3d.mc.jax.arcs import mc_arcs_jax, mc_arcs_jax_batch
from gnome3d.mc.jax.smooth import mc_smooth_jax, mc_smooth_jax_batch

__all__ = [
    "mc_arcs_jax",
    "mc_arcs_jax_batch",
    "mc_smooth_jax",
    "mc_smooth_jax_batch",
]
