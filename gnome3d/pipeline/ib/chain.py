"""Builds the node chain of each IB, arcs -> densify -> [estimate_dist] -> smooth."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gnome3d.pipeline.dag import Node, NodeId
from gnome3d.pipeline.ib.arcs import ArcsStage
from gnome3d.pipeline.ib.densify import DensifyStage
from gnome3d.pipeline.ib.estimate_dist import EstimateDistStage
from gnome3d.pipeline.ib.smooth import SmoothStage
from gnome3d.pipeline.stage import StageKind
from gnome3d.pipeline.state import Seeded

if TYPE_CHECKING:
    from gnome3d.skeleton import IBSeed


def ib_node_id(ib_id: str, stage: StageKind) -> NodeId:
    """Returns the id of one node of an IB chain."""
    return f"{ib_id} :: {stage.value}"


def ib_chain_nodes(
    ibseeds: list[IBSeed], prefix: str = ""
) -> tuple[list[Node], dict[NodeId, Seeded]]:
    """Returns one linear chain per IB and the seeds of the roots.

    ESTIMATE_DIST is in the chain when `IBSeed.wants_heat` is set. `prefix` is put before the node
    ids. The chains of several ensemble members fit in one DAG.
    """
    nodes: list[Node] = []
    seeds: dict[NodeId, Seeded] = {}

    for ibs in ibseeds:
        chain: list[tuple[StageKind, object]] = [
            (StageKind.ARCS, ArcsStage()),
            (StageKind.DENSIFY, DensifyStage()),
        ]
        if ibs.wants_heat:
            chain.append((StageKind.ESTIMATE_DIST, EstimateDistStage()))
        chain.append((StageKind.SMOOTH, SmoothStage()))

        prev: NodeId | None = None
        for kind, stage in chain:
            nid = ib_node_id(prefix + ibs.ib_id, kind)
            nodes.append(Node(nid, stage, () if prev is None else (prev,)))  # type: ignore[arg-type]
            if prev is None:
                seeds[nid] = ibs.seed
            prev = nid

    return nodes, seeds
