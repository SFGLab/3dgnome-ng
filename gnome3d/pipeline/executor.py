"""
Executors: turn a Dag into per-node output states.

All walk the DAG by repeatedly taking the *ready* set (`Dag.ready`) and running
it; they differ only in how a ready set is dispatched.  Each dispatch strategy
is its own small class (`SerialStrategy`, `ThreadedStrategy`, `BatchStrategy`) -

  * serial   - run each node on its own via the kind's serial (numba) runner;
  * threaded - run the kind's independent ready nodes across a numba thread pool
    (the kernels are nogil + thread-local RNG, so CPU-parallel and deterministic);
  * batch    - group the kind's nodes by ``(kind, bucket)`` and run each group
    through its batch (JAX) runner in one shot (the `mc_*_jax_batch` entries).

`MixedExecutor` is the workhorse: it assigns each `StageKind` a strategy and
delegates ready-set dispatch to that strategy.  `SerialExecutor` /
`ThreadedExecutor` / `BatchExecutor` are uniform-strategy convenience
constructors; `pick_executor` builds a per-stage mix from settings.  The COARSE
spine is always inline-serial (one RNG stream across its stages).

Result-equivalence note: a node's RNG seed lives in its `Seeded.seed`, not in
scheduling order, so bead output is independent of strategy - serial, threaded,
and batched produce the same structures, which is what lets the fixed-seed-exact
gate hold across executors.
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


# Single-node dispatches (e.g. the coarse chr/segment/ib seeding stages) finish near-instantly and
# spam two lines each; stay quiet for them unless they turn out slow. Multi-node dispatches and any
# slow node still log. Heavy single-node stages (arcs/smooth) print their own kernel progress.
_QUIET_MAX_NODES = 1
_QUIET_MAX_SECS = 0.2


def _log_dispatch_start(kind: StageKind, strategy: str, n: int, detail: str = "") -> None:
    """Standard executor dispatch START line, matching the JAX-kernel format:
    ``arcs[batch]: 50 nodes (group 1/1, ...), running...``. Suppressed for single trivial nodes."""
    if n <= _QUIET_MAX_NODES:
        return
    log.status(LOG, "  %s[%s]: %d nodes%s, running...", kind.value, strategy, n, detail)


def _log_dispatch_done(kind: StageKind, strategy: str, n: int, secs: float) -> None:
    """Standard executor dispatch DONE line: ``arcs[batch]: 50 nodes in 11.2s``. Suppressed for a
    single node that finished fast (keeps the line if it was actually slow)."""
    if n <= _QUIET_MAX_NODES and secs < _QUIET_MAX_SECS:
        return
    log.status(LOG, "  %s[%s]: %d nodes in %.1fs", kind.value, strategy, n, secs)


@runtime_checkable
class Executor(Protocol):
    """Run every node in dependency order, returning each node's output state.
    Beads are read off the terminal (`Smoothed`) nodes."""

    def run(self, dag: Dag) -> dict[NodeId, State]: ...


class ExecutorStrategy(StrEnum):
    """Strategies for running a DAG with a *per-kind* scheduling strategy."""

    SERIAL = "serial"  # numba, one node at a time
    THREADED = "threaded"  # numba, independent nodes across a thread pool
    BATCH = "batch"  # JAX, same-(kind,bucket) nodes in one vmapped launch


# --- helpers -----------------------------------------------------------------


def _ready_or_raise(dag: Dag, done: list[NodeId]) -> list[Node]:
    ready = dag.ready(done)
    if not ready and len(done) < len(dag.nodes):
        missing = set(dag.nodes) - set(done)
        raise ValueError(f"DAG stalled (cycle or missing dep) with unrun nodes: {sorted(missing)}")

    return ready


def _finish(
    dag: Dag, node: Node, output: State, outputs: dict[NodeId, State], done: list[NodeId]
) -> None:
    """Record a node's output, mark it done, and run its `expand` hook (if any),
    spawning new nodes into the DAG.  Newly added nodes are picked up by the next
    `ready` pass, so the loops naturally drain the grown graph."""
    outputs[node.id] = output
    done.append(node.id)

    if node.expand is not None:
        new_nodes, new_seeds = node.expand(output)
        dag.add(new_nodes, new_seeds)


def _run_serial(node: Node, inputs: tuple[State, ...]) -> State:
    """Compute one node's output (serial runner + apply) - the unit of work a
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


# --- strategies --------------------------------------------------------------


@runtime_checkable
class Strategy(Protocol):
    """Dispatch one ready set of same-`kind` nodes.  Records outputs and runs
    `expand` hooks on the main thread via `_finish`."""

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
    """Run nodes one at a time on the calling thread (the deterministic baseline,
    and the only legal strategy for the COARSE spine)."""

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
    """Run independent ready nodes across a shared `ThreadPoolExecutor`.  The
    pool is lazily created on first dispatch and torn down via `close()`."""

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
        # Compute on the pool; finish (record + expand) on the main thread.
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
    """Group this kind's ready nodes by their *batch key* and run each group
    through the batch runner in one launch (serial fallback for kinds with no
    batch runner, e.g. DENSIFY).

    The key is the stage's ``batch_key`` - ``(energy-term signature,
    shape-ladder bucket)`` - not the raw size.  Grouping by the ladder bucket
    keeps one compiled kernel + one wide launch per bucket (vs one per distinct
    size), and keeps each batch uniform in the flags ``mc_*_jax_batch`` reads
    from ``problems[0]``.  Each group is timed and logged (always-on) so a slow
    launch is never a silent stall.

    Multi-GPU follows ``mc_multigpu_mode``.  Under the default "groups" the groups themselves are
    the unit of parallelism: each one stays whole on a single device and different groups run side
    by side (`multigpu.run_group_parallel`).  That is what scales here, because the batch key
    splits a dispatch into many narrow groups - a chr1 smooth dispatch is 252 groups, most holding
    one or two IBs, so splitting inside a group has almost nothing to split.  Keeping a group
    intact also leaves every IB at the launch position it would have had on one device, and the
    kernels key their per-step RNG on that position, so output is byte-identical whatever the
    device count.

    "within" is the older behaviour, sharding one group's IBs across devices (`run_sharded`); it
    helps only while groups are wider than the device count and it does change the random draw.
    "off" pins everything to one device.  Results are applied in group order regardless."""

    def __init__(self, multigpu_mode: str = "groups") -> None:
        self._devices: list[object] | None = None  # visible JAX devices (lazy, jax-only)
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
        # "groups" keeps each group whole on one device and runs different groups side by side.
        # "within" splits a single group across devices, which only helps while groups are wide.
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

        # Applied on the calling thread in group order, so DAG growth stays deterministic
        # however the groups were interleaved across devices.
        for (_key, members), result in zip(ordered, results, strict=True):
            for (node, inputs), one in zip(members, result or [], strict=True):
                _finish(dag, node, node.stage.apply(inputs, one), outputs, done)

    def close(self) -> None:
        pass


# --- executors ---------------------------------------------------------------


class MixedExecutor:
    """Run the DAG with a *per-kind* scheduling strategy.

    ``strategy`` maps each `StageKind` to ``SERIAL`` | ``THREADED`` | ``BATCH``
    (kinds absent default to serial).  Dispatch is delegated to a `Strategy`
    instance per chosen value - `SerialStrategy`, `ThreadedStrategy`,
    `BatchStrategy` - so this executor stays a thin scheduler over them; the
    convenience subclasses (`SerialExecutor`, `ThreadedExecutor`, `BatchExecutor`)
    are just uniform-strategy mixes.  ``pick_executor`` builds it from the
    per-stage ``mc_executor_*`` settings.

    All three strategies are RNG-isolated per node (numba's RNG is per-OS-thread;
    the Python noise re-seeds a thread-local RNG), so the structures are
    byte-identical no matter the strategy - the executor-equivalence the fixed-seed
    gate relies on.  ``COARSE`` is always forced inline-serial: the coarse spine
    flows one RNG stream across its stages without re-seeding (so it can't be
    threaded/batched) and carries the fan-out `expand`; it's a linear chain anyway."""

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
    """Every kind serial with one node at a time (the deterministic baseline)."""

    def __init__(self) -> None:
        super().__init__(dict.fromkeys(StageKind, ExecutorStrategy.SERIAL))


class ThreadedExecutor(MixedExecutor):
    """Every per-IB kind threaded across `max_workers` numba threads (CPU)."""

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
    """The kinds in `batch_kinds` batched on JAX, the rest serial.  `None` =
    batch every kind that has a batch runner (validation)."""

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
