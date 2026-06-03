"""
IB chain assembly: turn per-IB seeds into their `arcs -> densify -> [est_dist] -> smooth`
node chains.

This is the IB domain's answer to "what is one interaction block's pipeline?" -
the coarse domain decides *when* to fan out (its IB-positioning node's `expand`
hook), and calls `ib_chain_nodes` to get the actual chain.  Keeping the chain
shape here (not in `coarse`) means the coarse spine doesn't need to know the
per-IB stage sequence.
"""

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
    """The id of one per-IB chain node."""
    return f"{ib_id} :: {stage.value}"


def ib_chain_nodes(
    ibseeds: list[IBSeed], prefix: str = ""
) -> tuple[list[Node], dict[NodeId, Seeded]]:
    """One linear chain per IB: arcs -> densify -> [est_dist] -> smooth.  ESTIMATE_DIST is
    included only when the skeleton kept it (`IBSeed.wants_heat`, the sparse-signal
    early-out).  Each chain's root (arcs) is seeded with the IB's `Seeded`.

    `prefix` namespaces the node ids (e.g. per ensemble member) so chains from
    several runs can live in one DAG without colliding; read them back with the
    same prefix via `ib_node_id(prefix + ib_id, ...)`."""
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
