"""
Executors: turn a Dag into per-node output states.

All walk the DAG by repeatedly taking the *ready* set (`Dag.ready`) and running
it; they differ only in how a ready set is dispatched.  `MixedExecutor` is the
workhorse: it assigns each `StageKind` a strategy —

  * serial   — run each node on its own via the kind's serial (numba) runner;
  * threaded — run the kind's independent ready nodes across a numba thread pool
    (the kernels are nogil + thread-local RNG, so CPU-parallel and deterministic);
  * batch    — group the kind's nodes by ``(kind, bucket)`` and run each group
    through its batch (JAX) runner in one shot (the `mc_*_jax_batch` entries).

`SerialExecutor` / `ThreadedExecutor` / `BatchExecutor` are just `MixedExecutor`
with a uniform strategy; `pick_executor` builds a per-stage mix from settings.
The COARSE spine is always inline-serial (one RNG stream across its stages).

Result-equivalence note: a node's RNG seed lives in its `Seeded.seed`, not in
scheduling order, so bead output is independent of strategy — serial, threaded,
and batched produce the same structures, which is what lets the fixed-seed-exact
gate hold across executors.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from gnome3d import log
from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.registry import runners_for
from gnome3d.pipeline.stage import Result, StageKind
from gnome3d.pipeline.state import State

LOG = log.get("executor")

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


def _finish(
    dag: Dag, node: Node, output: State, outputs: dict[NodeId, State], done: list[NodeId]
) -> None:
    """Record a node's output, mark it done, and run its `expand` hook (if any) —
    spawning new nodes into the DAG.  Newly added nodes are picked up by the next
    `ready` pass, so the loops naturally drain the grown graph."""
    outputs[node.id] = output
    done.append(node.id)
    if node.expand is not None:
        new_nodes, new_seeds = node.expand(output)
        dag.add(new_nodes, new_seeds)


def _run_node(node: Node, inputs: tuple[State, ...]) -> State:
    """Compute one node's output (serial runner + apply) — the unit of work a
    serial step or a worker thread runs.

    Pure w.r.t. shared state: `to_problem`/`apply` only read the (already-extracted)
    input states and construct a new state; the kernel re-seeds the thread-local
    Python RNG and numba's thread-local RNG from the node's seed, so concurrent
    nodes don't share RNG.  Bookkeeping (recording the output, `expand`) stays on
    the main thread."""
    runner = runners_for(node.stage.kind).serial
    if runner is None:
        raise RuntimeError(f"no serial runner registered for {node.stage.kind}")
    result: Result = runner(node.stage.to_problem(inputs))
    return node.stage.apply(inputs, result)


# Per-kind scheduling strategies.
SERIAL = "serial"  # numba, one node at a time
THREADED = "threaded"  # numba, independent nodes across a thread pool
BATCH = "batch"  # JAX, same-(kind,bucket) nodes in one vmapped launch

# The per-IB kinds (everything except the COARSE spine, which is always serial).
_IB_KINDS = (StageKind.ARCS, StageKind.DENSIFY, StageKind.HEAT_DIST, StageKind.SMOOTH)


class MixedExecutor:
    """Run the DAG with a *per-kind* scheduling strategy.

    ``strategy`` maps each `StageKind` to ``SERIAL`` | ``THREADED`` | ``BATCH``
    (kinds absent default to serial).  This single executor subsumes the serial /
    threaded / batch executors — they're just uniform-strategy cases — and lets
    stages mix backends, e.g. smooth ``BATCH`` (JAX/GPU) while arcs ``THREADED``
    (numba/CPU).  ``pick_executor`` builds it from the per-stage ``mc_executor_*``
    settings.

    All three strategies are RNG-isolated per node (numba's RNG is per-OS-thread;
    the Python noise re-seeds a thread-local RNG), so the structures are
    byte-identical no matter the strategy — the executor-equivalence the fixed-seed
    gate relies on.  ``COARSE`` is always forced inline-serial: the coarse spine
    flows one RNG stream across its stages without re-seeding (so it can't be
    threaded/batched) and carries the fan-out `expand`; it's a linear chain anyway."""

    def __init__(self, strategy: dict[StageKind, str], max_workers: int | None = None) -> None:
        self._strategy = dict(strategy)
        self._max_workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 1)

    def _strat(self, kind: StageKind) -> str:
        if kind is StageKind.COARSE:
            return SERIAL
        return self._strategy.get(kind, SERIAL)

    def run(self, dag: Dag) -> dict[NodeId, State]:
        outputs: dict[NodeId, State] = {}
        done: list[NodeId] = []
        pool = (
            ThreadPoolExecutor(max_workers=self._max_workers)
            if any(v == THREADED for v in self._strategy.values())
            else None
        )
        try:
            while len(done) < len(dag.nodes):
                by_kind: dict[StageKind, list[Node]] = defaultdict(list)
                for node in _ready_or_raise(dag, done):
                    by_kind[node.stage.kind].append(node)
                for kind, nodes in by_kind.items():
                    self._dispatch(self._strat(kind), kind, nodes, dag, outputs, done, pool)
        finally:
            if pool is not None:
                pool.shutdown()
        return outputs

    def _dispatch(self, strat, kind, nodes, dag, outputs, done, pool):  # type: ignore[no-untyped-def]
        if strat == BATCH:
            LOG.info(f"running {len(nodes)} {kind} nodes in batch...")
            self._run_batch(kind, nodes, dag, outputs, done)
        elif strat == THREADED and pool is not None:
            LOG.info(f"running {len(nodes)} {kind} nodes across {self._max_workers} threads...")
            # Compute on the pool; finish (record + expand) on the main thread.
            jobs = [(n, dag.inputs_for(n, outputs)) for n in nodes]
            futures = [pool.submit(_run_node, n, inp) for n, inp in jobs]
            for (node, _inp), fut in zip(jobs, futures, strict=True):
                _finish(dag, node, fut.result(), outputs, done)
        else:  # SERIAL (and the COARSE spine)
            LOG.info(f"running {len(nodes)} {kind} nodes serially...")
            for node in nodes:
                inputs = dag.inputs_for(node, outputs)
                _finish(dag, node, _run_node(node, inputs), outputs, done)

    @staticmethod
    def _run_batch(kind, nodes, dag, outputs, done):  # type: ignore[no-untyped-def]
        """Group this kind's ready nodes by bucket and run each bucket through the
        batch runner in one launch (serial fallback for kinds with no batch runner,
        e.g. DENSIFY)."""
        runners = runners_for(kind)
        buckets: dict[int, list[tuple[Node, tuple[State, ...]]]] = defaultdict(list)
        for node in nodes:
            inputs = dag.inputs_for(node, outputs)
            buckets[node.stage.bucket(inputs)].append((node, inputs))
        for members in buckets.values():
            problems = [node.stage.to_problem(inp) for node, inp in members]
            if runners.batch is not None:
                results = runners.batch(problems)
            elif runners.serial is not None:
                results = [runners.serial(p) for p in problems]
            else:
                raise RuntimeError(f"no runner registered for {kind}")
            for (node, inputs), result in zip(members, results, strict=True):
                _finish(dag, node, node.stage.apply(inputs, result), outputs, done)


class SerialExecutor(MixedExecutor):
    """Every kind serial with one node at a time (the deterministic baseline)."""

    def __init__(self) -> None:
        super().__init__({})


class ThreadedExecutor(MixedExecutor):
    """Every per-IB kind threaded across `max_workers` numba threads (CPU)."""

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__(dict.fromkeys(_IB_KINDS, THREADED), max_workers)


class BatchExecutor(MixedExecutor):
    """The kinds in `batch_kinds` batched on JAX, the rest serial.  `None` =
    batch every kind that has a batch runner (validation)."""

    def __init__(self, batch_kinds: set[StageKind] | None = None) -> None:
        kinds = _IB_KINDS if batch_kinds is None else batch_kinds
        super().__init__(dict.fromkeys(kinds, BATCH))
