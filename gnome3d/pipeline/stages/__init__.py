"""
Concrete stages — the per-IB pipeline as pure ``State -> State`` transforms.

Each stage wraps an existing validated kernel (numba `mc_arcs`/`mc_smooth`, the
densify arithmetic, the heat-dist estimate); the orchestration moves here, the
math does not.  Importing this package wires the serial (numba / CPU) runners
into the registry so a `SerialExecutor` can run a Dag built from these stages.

Stages live here (not in the core `pipeline` modules) because they're the only
part that touches `mc`/`mc_jax` — keeping `pipeline.stage`/`pipeline.registry`
as pure contract.
"""

from __future__ import annotations

from ..registry import register
from ..stage import StageKind
from .arcs import ArcsStage
from .arcs import _batch_run as _arcs_batch
from .arcs import _run as _arcs_run
from .densify import DensifyStage
from .densify import _run as _densify_run
from .heat import HeatDistStage
from .heat import _batch_run as _heat_batch
from .heat import _run as _heat_run
from .smooth import SmoothStage
from .smooth import _batch_run as _smooth_batch
from .smooth import _run as _smooth_run

register(StageKind.ARCS, serial=_arcs_run, batch=_arcs_batch)
register(StageKind.DENSIFY, serial=_densify_run)
register(StageKind.HEAT_DIST, serial=_heat_run, batch=_heat_batch)
register(StageKind.SMOOTH, serial=_smooth_run, batch=_smooth_batch)

__all__ = ["ArcsStage", "DensifyStage", "HeatDistStage", "SmoothStage"]
