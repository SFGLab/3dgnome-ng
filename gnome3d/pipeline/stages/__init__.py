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

from gnome3d.pipeline.registry import register
from gnome3d.pipeline.stage import StageKind
from gnome3d.pipeline.stages.arcs import ArcsStage
from gnome3d.pipeline.stages.arcs import _batch_run as _arcs_batch
from gnome3d.pipeline.stages.arcs import _run as _arcs_run
from gnome3d.pipeline.stages.densify import DensifyStage
from gnome3d.pipeline.stages.densify import _run as _densify_run
from gnome3d.pipeline.stages.heat import HeatDistStage
from gnome3d.pipeline.stages.heat import _batch_run as _heat_batch
from gnome3d.pipeline.stages.heat import _run as _heat_run
from gnome3d.pipeline.stages.smooth import SmoothStage
from gnome3d.pipeline.stages.smooth import _batch_run as _smooth_batch
from gnome3d.pipeline.stages.smooth import _run as _smooth_run

register(StageKind.ARCS, serial=_arcs_run, batch=_arcs_batch)
register(StageKind.DENSIFY, serial=_densify_run)
register(StageKind.HEAT_DIST, serial=_heat_run, batch=_heat_batch)
register(StageKind.SMOOTH, serial=_smooth_run, batch=_smooth_batch)

__all__ = ["ArcsStage", "DensifyStage", "HeatDistStage", "SmoothStage"]
