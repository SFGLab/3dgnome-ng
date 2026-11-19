"""Executors run a Dag and return the output state of each node.

Each walks the DAG by its ready set. A strategy decides how a ready set of one kind runs.
Serial runs each node with the serial runner, threaded runs independent nodes on a thread pool
and batch runs each `(kind, bucket)` group through the batch runner. COARSE is always serial.
Its stages share one RNG stream. The seed of a node is in its `Seeded.seed`. The
output does not depend on the strategy.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Protocol, runtime_checkable

from gnome3d import log
from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.multigpu import run_group_parallel, run_sharded, visible_devices
from gnome3d.pipeline.registry import runners_for
from gnome3d.pipeline.stage import Result, StageKind
from gnome3d.pipeline.state import State

LOG = log.get("executor")


# A dispatch of one node logs only when it is slow.
_QUIET_MAX_NODES = 1
_QUIET_MAX_SECS = 0.2


def _log_dispatch_start(kind: StageKind, strategy: str, n: int, detail: str = "") -> None:
    """Logs the start of a dispatch, `arcs[batch]: 50 nodes (group 1/1, ...), running...`."""
    if n <= _QUIET_MAX_NODES:
        return
    log.status(LOG, "  %s[%s]: %d nodes%s, running...", kind.value, strategy, n, detail)


def _log_dispatch_done(kind: StageKind, strategy: str, n: int, secs: float) -> None:
    """Logs the end of a dispatch, `arcs[batch]: 50 nodes in 11.2s`."""
    if n <= _QUIET_MAX_NODES and secs < _QUIET_MAX_SECS:
        return
    log.status(LOG, "  %s[%s]: %d nodes in %.1fs", kind.value, strategy, n, secs)


@runtime_checkable
class Executor(Protocol):
    """Runs every node in dependency order and returns the output state of each."""

    def run(self, dag: Dag) -> dict[NodeId, State]: ...


class ExecutorStrategy(StrEnum):
    """The scheduling strategy of a stage kind."""

    SERIAL = "serial"  # numba, one node at a time
    THREADED = "threaded"  # numba, independent nodes on a thread pool
    BATCH = "batch"  # JAX, the nodes of one (kind, bucket) in one launch


def _ready_or_raise(dag: Dag, done: list[NodeId]) -> list[Node]:
    ready = dag.ready(done)
    if not ready and len(done) < len(dag.nodes):
        missing = set(dag.nodes) - set(done)
        raise ValueError(f"DAG stalled (cycle or missing dep) with unrun nodes: {sorted(missing)}")

    return ready


def _finish(
    dag: Dag, node: Node, output: State, outputs: dict[NodeId, State], done: list[NodeId]
) -> None:
    """Records the output of a node, marks it done and runs its `expand` hook."""
    outputs[node.id] = output
    done.append(node.id)

    if node.expand is not None:
        new_nodes, new_seeds = node.expand(output)
        dag.add(new_nodes, new_seeds)


def _run_serial(node: Node, inputs: tuple[State, ...]) -> State:
    """Computes the output of one node with the serial runner.

    The kernel seeds the thread local RNGs from the node's seed. Threads may run this.
    Recording the output and `expand` stay on the main thread.
    """
    runner = runners_for(node.stage.kind).serial
    if runner is None:
        raise RuntimeError(f"no serial runner registered for {node.stage.kind}")

    result: Result = runner(node.stage.to_problem(inputs))
    return node.stage.apply(inputs, result)


@runtime_checkable
class Strategy(Protocol):
    """Dispatches one ready set of nodes of one kind. Outputs are recorded on the main thread."""

    def dispatch(
        self,
        kind: StageKind,
        nodes: list[Node],
        dag: Dag,
        outputs: dict[NodeId, State],
        done: list[NodeId],
    ) -> None: ...

    def close(self) -> None: ...


class SerialStrategy:
    """Runs nodes one at a time on the calling thread."""

    def dispatch(
        self,
        kind: StageKind,
        nodes: list[Node],
        dag: Dag,
        outputs: dict[NodeId, State],
        done: list[NodeId],
    ) -> None:
        _log_dispatch_start(kind, "serial", len(nodes))
        t0 = time.perf_counter()
        for node in nodes:
            inputs = dag.inputs_for(node, outputs)
            _finish(dag, node, _run_serial(node, inputs), outputs, done)
        _log_dispatch_done(kind, "serial", len(nodes), time.perf_counter() - t0)

    def close(self) -> None:
        pass


class ThreadedStrategy:
    """Runs independent ready nodes on a shared thread pool, created on the first dispatch."""

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None

    def _ensure_pool(self) -> ThreadPoolExecutor:
        pool = self._pool
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=self._max_workers)
            self._pool = pool
        return pool

    def dispatch(
        self,
        kind: StageKind,
        nodes: list[Node],
        dag: Dag,
        outputs: dict[NodeId, State],
        done: list[NodeId],
    ) -> None:
        pool = self._ensure_pool()
        _log_dispatch_start(kind, "threaded", len(nodes), f" ({self._max_workers} workers)")
        t0 = time.perf_counter()
        jobs = [(n, dag.inputs_for(n, outputs)) for n in nodes]
        futures = [pool.submit(_run_serial, n, inp) for n, inp in jobs]
        for (node, _inp), fut in zip(jobs, futures, strict=True):
            _finish(dag, node, fut.result(), outputs, done)
        _log_dispatch_done(kind, "threaded", len(nodes), time.perf_counter() - t0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


class BatchStrategy:
    """Groups the ready nodes of a kind by their batch key and runs each group through the batch
    runner in one launch. A kind with no batch runner runs serially.

    The key is the stage's `batch_key`, the energy term signature and the shape bucket. The
    kernels read their flags from `problems[0]`. A group must be uniform in them.

    Multi GPU follows `mc_multigpu_mode`. Under "groups" each group stays whole on one device and
    groups run side by side, which is byte identical for any device count. Under "within" one
    group is split across devices, which changes the random draw. "off" uses one device. Results
    are applied in group order.
    """

    def __init__(self, multigpu_mode: str = "groups") -> None:
        self._devices: list[object] | None = None  # the visible JAX devices, read on first use
        self._mode = str(multigpu_mode).strip().lower()

    def _shard_limit(self) -> int:
        return 1 if self._mode == "off" else 0

    def _shard_devices(self) -> list[object]:
        if self._devices is None:
            self._devices = visible_devices(limit=self._shard_limit())
        return self._devices

    def dispatch(
        self,
        kind: StageKind,
        nodes: list[Node],
        dag: Dag,
        outputs: dict[NodeId, State],
        done: list[NodeId],
    ) -> None:
        runners = runners_for(kind)
        groups: dict[object, list[tuple[Node, tuple[State, ...]]]] = defaultdict(list)
        for node in nodes:
            inputs = dag.inputs_for(node, outputs)
            key_fn = getattr(node.stage, "batch_key", None)
            key = key_fn(inputs) if key_fn is not None else node.stage.bucket(inputs)
            groups[key].append((node, inputs))

        ordered = list(groups.items())
        devices = self._shard_devices() if runners.batch is not None else []
        by_group = self._mode == "groups" and len(devices) > 1
        results: list[list[Result] | None] = [None] * len(ordered)

        def _run_group(gi: int) -> None:
            key, members = ordered[gi]
            problems = [node.stage.to_problem(inp) for node, inp in members]
            stage0 = members[0][0].stage
            key_desc = (
                stage0.describe_batch_key(key)
                if hasattr(stage0, "describe_batch_key")
                else f"key={key}"
            )
            gpu_tag = f", {len(devices)} GPUs" if len(devices) > 1 else ""
            _log_dispatch_start(
                kind,
                "batch",
                len(members),
                f" (group {gi + 1}/{len(ordered)}, {key_desc}{gpu_tag})",
            )

            t0 = time.perf_counter()
            if runners.batch is not None:
                results[gi] = (
                    runners.batch(problems)
                    if by_group
                    else run_sharded(runners.batch, problems, devices)
                )
            elif runners.serial is not None:
                results[gi] = [runners.serial(p) for p in problems]
            else:
                raise RuntimeError(f"no runner registered for {kind}")

            _log_dispatch_done(kind, "batch", len(members), time.perf_counter() - t0)

        if by_group:
            run_group_parallel(_run_group, len(ordered), devices)
        else:
            for gi in range(len(ordered)):
                _run_group(gi)

        # Applied on the calling thread in group order. The growth of the DAG is deterministic.
        for (_key, members), result in zip(ordered, results, strict=True):
            for (node, inputs), one in zip(members, result or [], strict=True):
                _finish(dag, node, node.stage.apply(inputs, one), outputs, done)

    def close(self) -> None:
        pass


class MixedExecutor:
    """Runs the DAG with one strategy per stage kind.

    `strategy` maps each `StageKind` to SERIAL, THREADED or BATCH. A kind that is absent runs
    serially. The structures are byte identical under any strategy. Each node seeds its own
    RNGs. COARSE is always serial.
    """

    def __init__(
        self,
        strategy: Mapping[StageKind, ExecutorStrategy],
        max_workers: int | None = None,
        multigpu_mode: str = "groups",
    ) -> None:
        self._strategy_map: dict[StageKind, ExecutorStrategy] = dict(strategy)
        self._multigpu_mode: str = multigpu_mode
        self._max_workers: int = (
            max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 1)
        )
        self._strategies: dict[ExecutorStrategy, Strategy] = {}

    def _strat_for(self, kind: StageKind) -> ExecutorStrategy:
        if kind is StageKind.COARSE:
            return ExecutorStrategy.SERIAL

        return self._strategy_map.get(kind, ExecutorStrategy.SERIAL)

    def _get_strategy(self, name: ExecutorStrategy) -> Strategy:
        cached = self._strategies.get(name)
        if cached is not None:
            return cached

        strat: Strategy
        if name is ExecutorStrategy.SERIAL:
            strat = SerialStrategy()
        elif name is ExecutorStrategy.THREADED:
            strat = ThreadedStrategy(self._max_workers)
        elif name is ExecutorStrategy.BATCH:
            strat = BatchStrategy(self._multigpu_mode)
        else:
            raise ValueError(f"unknown executor strategy: {name!r}")

        self._strategies[name] = strat
        return strat

    def run(self, dag: Dag) -> dict[NodeId, State]:
        outputs: dict[NodeId, State] = {}
        done: list[NodeId] = []
        try:
            while len(done) < len(dag.nodes):
                by_kind: dict[StageKind, list[Node]] = defaultdict(list)

                for node in _ready_or_raise(dag, done):
                    by_kind[node.stage.kind].append(node)

                for kind, nodes in by_kind.items():
                    self._get_strategy(self._strat_for(kind)).dispatch(
                        kind, nodes, dag, outputs, done
                    )
        finally:
            for strat in self._strategies.values():
                strat.close()

            self._strategies.clear()

        return outputs


class SerialExecutor(MixedExecutor):
    """Runs every kind serially."""

    def __init__(self) -> None:
        super().__init__(dict.fromkeys(StageKind, ExecutorStrategy.SERIAL))


class ThreadedExecutor(MixedExecutor):
    """Runs every IB kind on `max_workers` numba threads."""

    _PER_IB: frozenset[StageKind] = frozenset(
        {
            StageKind.ARCS,
            StageKind.ESTIMATE_DIST,
            StageKind.SMOOTH,
        }
    )

    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__(
            strategy={
                kind: (
                    ExecutorStrategy.THREADED if kind in self._PER_IB else ExecutorStrategy.SERIAL
                )
                for kind in StageKind
            },
            max_workers=max_workers,
        )


class BatchExecutor(MixedExecutor):
    """Batches the kinds in `batch_kinds` on JAX and runs the rest serially. None batches every
    kind that has a batch runner.
    """

    _BATCHABLE: frozenset[StageKind] = frozenset(
        {
            StageKind.ARCS,
            StageKind.ESTIMATE_DIST,
            StageKind.SMOOTH,
        }
    )

    def __init__(self, batch_kinds: set[StageKind] | None = None) -> None:
        def pick(kind: StageKind) -> ExecutorStrategy:
            if kind not in self._BATCHABLE:
                return ExecutorStrategy.SERIAL
            if batch_kinds is None or kind in batch_kinds:
                return ExecutorStrategy.BATCH
            return ExecutorStrategy.SERIAL

        super().__init__({kind: pick(kind) for kind in StageKind})
