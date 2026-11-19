"""The DAG of nodes that an executor schedules.

A node may spawn new nodes when it completes. The coarse stages fan out into the IB chains
this way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from gnome3d.pipeline.stage import Stage
from gnome3d.pipeline.state import Seeded, State

NodeId: TypeAlias = str

# Takes the output state of a node and returns new nodes with the seed states of their roots.
ExpandFn: TypeAlias = Callable[[State], "tuple[list[Node], dict[NodeId, Seeded]]"]


@dataclass(frozen=True)
class Node:
    """One unit of work. `stage` runs when `deps` have produced their states.

    A root takes its input from `Dag.seeds[id]` when present. `expand` is called with the output
    state and adds nodes to the DAG.
    """

    id: NodeId
    stage: Stage
    deps: tuple[NodeId, ...]
    expand: ExpandFn | None = field(default=None, compare=False)


@dataclass
class Dag:
    """The nodes and the seed states of the roots. `add` merges nodes spawned at run time."""

    nodes: dict[NodeId, Node]
    seeds: dict[NodeId, Seeded]

    def ready(self, done: Sequence[NodeId]) -> list[Node]:
        """Returns the nodes whose deps are done and that have not run."""
        done_set = set(done)
        return [
            n
            for n in self.nodes.values()
            if n.id not in done_set and all(d in done_set for d in n.deps)
        ]

    def inputs_for(self, node: Node, outputs: Mapping[NodeId, State]) -> tuple[State, ...]:
        """Returns the input states of a node: the outputs of its deps, the seed of a seeded
        root or `()`.
        """
        if not node.deps:
            seed = self.seeds.get(node.id)
            return (seed,) if seed is not None else ()

        return tuple(outputs[d] for d in node.deps)

    def add(self, nodes: list[Node], seeds: dict[NodeId, Seeded]) -> None:
        """Merges spawned nodes and their root seeds into the DAG."""
        for n in nodes:
            self.nodes[n.id] = n

        self.seeds.update(seeds)
