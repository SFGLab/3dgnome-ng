"""
Executors: turn a Dag into per-node output states.

Both walk the DAG by repeatedly taking the *ready* set (`Dag.ready`) and running
it; they differ only in how a ready set is dispatched:

  * `SerialExecutor` — run each ready node on its own, via the kind's serial
    runner (numba / JAX K=1).  Independent nodes own disjoint data, so this is
    also where CPU-thread parallelism would slot in.
  * `BatchExecutor` — group the ready set by ``(kind, bucket)`` and run each
    group through the kind's batch runner in one shot (the `mc_*_jax_batch`
    entries).  Reproduces today's JaxSolver phase pipeline, but emergent from
    which nodes are ready together rather than a hardcoded phase list.

Result-equivalence note: a node's RNG seed lives in its `Seeded.seed`, not in
scheduling order, so bead output is independent of drain policy — serial and
batched (and any future pipelined policy) produce the same structures, which is
what lets the fixed-seed-exact gate hold across executors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.registry import runners_for
from gnome3d.pipeline.stage import Result, StageKind
from gnome3d.pipeline.state import State


@runtime_checkable
class Executor(Protocol):
    """Run every node in dependency order, returning each node's output state.
    Beads are read off the terminal (`Smoothed`) nodes."""

    def run(self, dag: Dag) -> dict[NodeId, State]: ...


def _ready_or_raise(dag: Dag, done: list[NodeId]) -> list[Node]:
    ready = dag.ready(done)
    if not ready and len(done) < len(dag.nodes):
        missing = set(dag.nodes) - set(done)
        raise RuntimeError(f"DAG stalled (cycle or missing dep) — unrun nodes: {sorted(missing)}")
    return ready


def _finish(dag: Dag, node: Node, output: State, outputs: dict, done: list[NodeId]) -> None:
    """Record a node's output, mark it done, and run its `expand` hook (if any) —
    spawning new nodes into the DAG.  Newly added nodes are picked up by the next
    `ready` pass, so the loops naturally drain the grown graph."""
    outputs[node.id] = output
    done.append(node.id)
    if node.expand is not None:
        new_nodes, new_seeds = node.expand(output)
        dag.add(new_nodes, new_seeds)


class SerialExecutor:
    """Run nodes one at a time through their kind's serial runner."""

    def run(self, dag: Dag) -> dict[NodeId, State]:
        outputs: dict[NodeId, State] = {}
        done: list[NodeId] = []
        while len(done) < len(dag.nodes):
            for node in _ready_or_raise(dag, done):
                inputs = dag.inputs_for(node, outputs)
                runner = runners_for(node.stage.kind).serial
                if runner is None:
                    raise RuntimeError(f"no serial runner registered for {node.stage.kind}")
                result: Result = runner(node.stage.to_problem(inputs))
                _finish(dag, node, node.stage.apply(inputs, result), outputs, done)
        return outputs


class BatchExecutor:
    """Group each ready set by ``(kind, bucket)`` and run each group through its
    batch runner (falling back to the serial runner mapped over the group when a
    kind has no batch runner, e.g. DENSIFY).

    ``batch_kinds`` selects which kinds actually use their batch (GPU) runner;
    kinds outside it run serially even if a batch runner is registered.  This is
    how the per-term backend flags (``mc_backend_apply_to_*``) map onto the
    pipeline — e.g. smooth/heat batched on JAX while arcs stays numba, matching
    the old JaxSolver.  ``None`` means "batch every kind that can" (validation)."""

    def __init__(self, batch_kinds: set[StageKind] | None = None) -> None:
        self._batch_kinds = batch_kinds

    def _use_batch(self, kind: StageKind, runners) -> bool:  # type: ignore[no-untyped-def]
        return runners.batch is not None and (
            self._batch_kinds is None or kind in self._batch_kinds
        )

    def run(self, dag: Dag) -> dict[NodeId, State]:
        outputs: dict[NodeId, State] = {}
        done: list[NodeId] = []
        while len(done) < len(dag.nodes):
            ready = _ready_or_raise(dag, done)

            # Group the ready set so every batch is shape- and kind-uniform.
            groups: dict[tuple[StageKind, int], list[tuple[Node, tuple[State, ...]]]] = defaultdict(
                list
            )
            for node in ready:
                inputs = dag.inputs_for(node, outputs)
                groups[(node.stage.kind, node.stage.bucket(inputs))].append((node, inputs))

            for (kind, _bucket), members in groups.items():
                runners = runners_for(kind)
                problems = [node.stage.to_problem(inp) for node, inp in members]
                if self._use_batch(kind, runners):
                    results = runners.batch(problems)
                elif runners.serial is not None:
                    results = [runners.serial(p) for p in problems]
                else:
                    raise RuntimeError(f"no runner registered for {kind}")
                for (node, inputs), result in zip(members, results, strict=True):
                    _finish(dag, node, node.stage.apply(inputs, result), outputs, done)
        return outputs
