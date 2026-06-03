"""
The dataflow DAG: nodes with explicit dependencies, the unit an executor schedules.

The IB pipeline is a linear chain (`arcs -> densify -> [heat] -> smooth`); the
coarse levels are an upstream chain (`hierarchy -> chr -> segment -> ib`); the
skeleton fans out to every IB; an ensemble fans each IB into E seed-varied
chains.  All of that is just graph shape over the same `Node`/`Dag`.

Two scheduling primitives:
  * `ready` - nodes whose deps are all done (executors build batching on top).
  * dynamic expansion - a node may carry an `expand` hook that, on completion,
    spawns *new* nodes into the running DAG.  This is how the coarse positioning
    fans out: the IB chains aren't known until the hierarchy is built and the
    IB centroids positioned, so a fan-out node emits them at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from gnome3d.pipeline.stage import Stage
from gnome3d.pipeline.state import Seeded, State

NodeId: TypeAlias = str

# A node's expansion hook: given the node's output state, return new nodes to add
# to the running DAG plus the seed states for any new roots among them.
ExpandFn: TypeAlias = Callable[[State], "tuple[list[Node], dict[NodeId, Seeded]]"]


@dataclass(frozen=True)
class Node:
    """A unit of work: run `stage`` once ``deps`` have produced their states.

    A root node (no deps) takes its input from `Dag.seeds[id]` if present, else
    no input (a true source - e.g. the coarse hierarchy builder).  ``expand``, if
    set, is called with this node's output state after it runs and spawns more
    nodes into the DAG (the coarse fan-out into per-IB chains)."""

    id: NodeId
    stage: Stage
    deps: tuple[NodeId, ...]
    expand: ExpandFn | None = field(default=None, compare=False)


@dataclass
class Dag:
    """Nodes plus the seed states for roots that consume one (per-IB ``Seeded``;
    true sources have no seed).  Mutable: `add` merges nodes spawned at runtime
    by an `expand` hook."""

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
        """The input states a node's stage receives: its deps' outputs, the
        skeleton seed for a seeded root, or `()` for a true source."""
        if not node.deps:
            seed = self.seeds.get(node.id)
            return (seed,) if seed is not None else ()

        return tuple(outputs[d] for d in node.deps)

    def add(self, nodes: list[Node], seeds: dict[NodeId, Seeded]) -> None:
        """Merge runtime-spawned nodes (and their root seeds) into the DAG."""
        for n in nodes:
            self.nodes[n.id] = n

        self.seeds.update(seeds)
