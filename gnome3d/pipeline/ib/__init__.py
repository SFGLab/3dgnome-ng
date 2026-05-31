"""
IB reconstruction domain: the per-interaction-block pipeline stages.

Once the coarse spine (`gnome3d.pipeline.coarse`) has positioned the cluster
graph, each interaction block reconstructs *independently* as a linear chain of
stages — arcs (anchor positioning) -> densify (insert subanchors) -> [heat]
(subanchor distance target) -> smooth (final MC).  Each stage wraps a validated
kernel (numba `mc_arcs`/`mc_smooth` in `gnome3d.mc`, the densify arithmetic, the
heat-dist estimate); the orchestration lives in the stage, the math does not.

Importing this package wires the serial (numba / CPU) and batched (JAX) runners
into the registry, so an executor can run a Dag built from these stages.  This is
the domain peer of `pipeline.coarse`: coarse owns the coupled positioning spine,
`ib` owns the isolated per-IB chains (and `chain.ib_chain_nodes`, the chain
assembly the coarse fan-out calls).
"""

from __future__ import annotations

from gnome3d.pipeline.ib.arcs import ArcsStage
from gnome3d.pipeline.ib.arcs import _batch_run as _arcs_batch
from gnome3d.pipeline.ib.arcs import _run as _arcs_run
from gnome3d.pipeline.ib.chain import ib_chain_nodes, ib_node_id
from gnome3d.pipeline.ib.densify import DensifyStage
from gnome3d.pipeline.ib.densify import _run as _densify_run
from gnome3d.pipeline.ib.heat import HeatDistStage
from gnome3d.pipeline.ib.heat import _batch_run as _heat_batch
from gnome3d.pipeline.ib.heat import _run as _heat_run
from gnome3d.pipeline.ib.smooth import SmoothStage
from gnome3d.pipeline.ib.smooth import _batch_run as _smooth_batch
from gnome3d.pipeline.ib.smooth import _run as _smooth_run
from gnome3d.pipeline.registry import register
from gnome3d.pipeline.stage import StageKind

register(StageKind.ARCS, serial=_arcs_run, batch=_arcs_batch)
register(StageKind.DENSIFY, serial=_densify_run)
register(StageKind.HEAT_DIST, serial=_heat_run, batch=_heat_batch)
register(StageKind.SMOOTH, serial=_smooth_run, batch=_smooth_batch)

__all__ = [
    "ArcsStage",
    "DensifyStage",
    "HeatDistStage",
    "SmoothStage",
    "ib_chain_nodes",
    "ib_node_id",
]
