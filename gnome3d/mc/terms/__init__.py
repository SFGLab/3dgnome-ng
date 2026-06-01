"""Composeable MC energy terms.

Each term module exposes a `Term` record (see `base.Term`) bundling its shared
param `namedtuple` and its numba + JAX local/init implementations.  Kernels are
*recipes* — ordered tuples of these terms — composed by `compose_*_nb` (numba)
or trace-time unrolling (JAX).  See `base` for the full contract.
"""

from __future__ import annotations

from gnome3d.mc.terms.arc_springs import ARC_SPRINGS
from gnome3d.mc.terms.base import Term, compose_init_nb, compose_local_nb
from gnome3d.mc.terms.chain import CHAIN
from gnome3d.mc.terms.confinement import CONFINEMENT
from gnome3d.mc.terms.excluded_volume import EXCLUDED_VOLUME
from gnome3d.mc.terms.heatmap import HEATMAP
from gnome3d.mc.terms.orientation import ORIENTATION
from gnome3d.mc.terms.subanchor_heat import SUBANCHOR_HEAT

__all__ = [
    "ARC_SPRINGS",
    "CHAIN",
    "CONFINEMENT",
    "EXCLUDED_VOLUME",
    "HEATMAP",
    "ORIENTATION",
    "SUBANCHOR_HEAT",
    "Term",
    "compose_init_nb",
    "compose_local_nb",
]
