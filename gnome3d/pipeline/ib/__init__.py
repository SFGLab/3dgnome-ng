"""The stages of one interaction block: arcs, densify, the optional distance estimate and smooth.

Importing this package registers the serial and the batch runners.
"""

from __future__ import annotations

from gnome3d.pipeline.ib.arcs import ArcsStage
from gnome3d.pipeline.ib.arcs import _batch_run as _arcs_batch
from gnome3d.pipeline.ib.arcs import _run as _arcs_run
from gnome3d.pipeline.ib.chain import ib_chain_nodes, ib_node_id
from gnome3d.pipeline.ib.densify import DensifyStage
from gnome3d.pipeline.ib.densify import _run as _densify_run
from gnome3d.pipeline.ib.estimate_dist import EstimateDistStage
from gnome3d.pipeline.ib.estimate_dist import _batch_run as _est_dist_batch
from gnome3d.pipeline.ib.estimate_dist import _run as _est_dist_run
from gnome3d.pipeline.ib.smooth import SmoothStage
from gnome3d.pipeline.ib.smooth import _batch_run as _smooth_batch
from gnome3d.pipeline.ib.smooth import _run as _smooth_run
from gnome3d.pipeline.registry import register
from gnome3d.pipeline.stage import StageKind

register(StageKind.ARCS, serial=_arcs_run, batch=_arcs_batch)
register(StageKind.DENSIFY, serial=_densify_run)
register(StageKind.ESTIMATE_DIST, serial=_est_dist_run, batch=_est_dist_batch)
register(StageKind.SMOOTH, serial=_smooth_run, batch=_smooth_batch)

__all__ = [
    "ArcsStage",
    "DensifyStage",
    "EstimateDistStage",
    "SmoothStage",
    "ib_chain_nodes",
    "ib_node_id",
]
