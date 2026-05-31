"""
The dataflow DAG: nodes with explicit dependencies, the unit an executor schedules.

The IB pipeline is a linear chain (`arcs -> densify -> [heat] -> smooth`); the
skeleton fans out to every IB; an ensemble fans each IB into E seed-varied
chains.  All of that is just graph shape over the same `Node`/`Dag`.  The only
scheduling primitive is `Dag.ready` — nodes whose deps are all done; executors
build their batching policy on top of it, so phases are emergent, not fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .stage import Stage
from .state import Seeded, State

NodeId: TypeAlias = str


@dataclass(frozen=True)
class Node:
    """A unit of work: run ``stage`` once ``deps`` have produced their states.
    A root node (no deps) takes its input from the skeleton seed (`Dag.seeds`)."""

    id: NodeId
    stage: Stage
    deps: tuple[NodeId, ...]


@dataclass
class Dag:
    """Nodes plus the externally-provided seed states for roots (the skeleton's
    per-IB ``Seeded`` outputs, keyed by the root node that consumes them)."""

    nodes: dict[NodeId, Node]
    seeds: dict[NodeId, Seeded]

    def ready(self, done: Sequence[NodeId]) -> list[Node]:
        """Nodes whose deps are all done and which haven't run yet."""
        done_set = set(done)
        return [
            n
            for n in self.nodes.values()
            if n.id not in done_set and all(d in done_set for d in n.deps)
        ]

    def inputs_for(self, node: Node, outputs: Mapping[NodeId, State]) -> tuple[State, ...]:
        """The input states a node's stage receives: its deps' outputs, or the
        skeleton seed for a root (no deps)."""
        if not node.deps:
            return (self.seeds[node.id],)
        return tuple(outputs[d] for d in node.deps)
