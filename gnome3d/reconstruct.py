"""
Pipeline driver: a single self-expanding DAG -> executor -> beads.

The top-level orchestration.  `simulate` / `cli` call `reconstruct`; it builds the
coarse `CoarseState`, assembles the unified coarse DAG (`build_coarse_dag`) whose
IB-positioning node `expand`s into the per-IB `arcs -> densify -> [heat] -> smooth`
chains, runs it under an executor, and returns ``dict[chr -> list[BeadOut]]``.

It is a leaf consumer - nothing imports it back, so there is no cycle.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from gnome3d import log, skeleton
from gnome3d.pipeline import Dag, Node
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse.stages import build_coarse_dag
from gnome3d.pipeline.executor import (
    Executor,
    ExecutorStrategy,
    MixedExecutor,
    SerialExecutor,
)
from gnome3d.pipeline.ib import ib_chain_nodes, ib_node_id
from gnome3d.pipeline.stage import StageKind
from gnome3d.pipeline.state import Seeded, Smoothed, State
from gnome3d.util import jax_is_available

if TYPE_CHECKING:
    from gnome3d.data import ContactData
    from gnome3d.settings import Settings
    from gnome3d.skeleton import IBSeed
    from gnome3d.types import BeadOut, BedRegion

LOG = log.get("reconstruct")

# Well-separated RNG offset per ensemble member, so structure i's per-IB seeds
# (and coarse seed) are distinct from structure j's.  See coarse.seed_global_rng.
MEMBER_SEED_STRIDE = 2_000_003


def _auto_strategy(kind: StageKind, settings: Settings) -> ExecutorStrategy:
    """Resolve ``mc_executor_<stage> = auto``."""
    prefer_jax = jax_is_available()
    if prefer_jax and kind in (StageKind.SMOOTH, StageKind.ESTIMATE_DIST):
        return ExecutorStrategy.BATCH

    return (
        ExecutorStrategy.THREADED
        if int(settings.mc_executor_threaded_workers) > 1
        else ExecutorStrategy.SERIAL
    )


def _resolve_strategy(value: str, kind: StageKind, settings: Settings) -> ExecutorStrategy:
    """One ``mc_executor_<stage>`` value (serial|threaded|batch|auto) -> strategy.
    A batch choice that can't run (small-IB-boost's per-IB springs aren't in the
    batched kernels; DENSIFY has no batch kernel) downgrades to threaded/serial."""
    v = str(value).strip().lower()
    chosen = (
        ExecutorStrategy(v)
        if v in (ExecutorStrategy.SERIAL, ExecutorStrategy.THREADED, ExecutorStrategy.BATCH)
        else _auto_strategy(kind, settings)
    )

    if chosen == ExecutorStrategy.BATCH and (kind is StageKind.DENSIFY):
        chosen = (
            ExecutorStrategy.THREADED
            if int(settings.mc_executor_threaded_workers) > 1
            else ExecutorStrategy.SERIAL
        )

    return chosen


def pick_executor(settings: Settings) -> Executor:
    """Build the executor from the per-stage ``mc_executor_*`` settings.

    Each IB stage names its executor (serial | threaded | batch | auto); this maps
    them to a per-kind strategy and returns one `MixedExecutor`."""
    strategy = {
        StageKind.ARCS: _resolve_strategy(
            settings.mc_executor_arcs, StageKind.ARCS, settings
        ),
        StageKind.DENSIFY: _resolve_strategy(
            settings.mc_executor_densify, StageKind.DENSIFY, settings
        ),
        StageKind.ESTIMATE_DIST: _resolve_strategy(
            settings.mc_executor_heat, StageKind.ESTIMATE_DIST, settings
        ),
        StageKind.SMOOTH: _resolve_strategy(
            settings.mc_executor_smooth, StageKind.SMOOTH, settings
        ),
    }

    return MixedExecutor(strategy, max_workers=int(settings.mc_executor_threaded_workers))


def _beads(output: State) -> list[BeadOut]:
    """The beads of a terminal smooth node, narrowed from the opaque output."""
    assert isinstance(output, Smoothed)
    return output.beads


def reconstruct(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    executor: Executor | None = None,
    seed_offset: int = 0,
) -> dict[str, list[BeadOut]]:
    """Reconstruct via the single self-expanding DAG: the coarse spine
    (hierarchy -> chr -> segment -> ib) and the per-IB chains are one graph, with
    the IB-positioning node's `expand` fanning out the chains at run time.

    Returns one bead list per chr, sorted by genomic start.  `seed_offset`
    selects the ensemble member (distinct structures from the same inputs); 0 is
    the canonical single structure.  The coarse spine's root stage seeds the
    global RNG from the seed it carries (`build_state` / `build_coarse_dag` use no
    RNG), so no external seeding is needed."""
    state = cb.build_state(settings, data, chrs, region)
    dag, ib_sink = build_coarse_dag(state, seed_offset)

    LOG.info(f"running DAG with {len(dag.nodes)} nodes and {len(dag.seeds)} seeds...")
    outputs = (executor or SerialExecutor()).run(dag)

    per_chr: dict[str, list[BeadOut]] = defaultdict(list)
    for ibs in ib_sink:
        beads = _beads(outputs[ib_node_id(ibs.ib_id, StageKind.SMOOTH)])
        per_chr[ibs.chr_].extend(beads)

    return {
        chr_: sorted(beads, key=lambda b: b.start)
        for chr_, beads in per_chr.items()
    }


def reconstruct_ensemble(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    n: int = 1,
    executor: Executor | None = None,
    base_seed_offset: int = 0,
) -> list[dict[str, list[BeadOut]]]:
    """Reconstruct an ensemble of `n` structures, batching the per-IB chains
    *across members* so same-shaped IBs from different members fill one kernel
    launch.

    Two phases:
      1. Run each member's coarse spine *sequentially* and gather its IB seeds.
         The spine seeds + flows the global RNG, so members cannot interleave;
         the gather is RNG-free.
      2. Merge all members' IB chains into one DAG (node ids namespaced per member)
         and run it under `executor` - a `BatchExecutor` then groups same-(kind,
         bucket) chains across members into wide launches.

    Member m uses ``base_seed_offset + m * MEMBER_SEED_STRIDE``, so member m is
    byte-identical to ``reconstruct(seed_offset=that)`` (the coarse spine + IB
    seeds match; the IB chains re-seed per node, independent of batching)."""
    executor = executor or SerialExecutor()

    # Phase 1: per-member coarse spine (sequential, RNG-safe) + seed gather.
    member_seeds: list[list[IBSeed]] = []
    for m in range(n):
        off = base_seed_offset + m * MEMBER_SEED_STRIDE
        state = cb.build_state(settings, data, chrs, region)
        spine, _ = build_coarse_dag(state, off, fan_out=False)
        executor.run(spine)
        member_seeds.append(skeleton.gather_all_ib_seeds(state, off))

    n_ib = sum(len(s) for s in member_seeds)
    LOG.info(f"ensemble: {n} members, {n_ib} IB chains total")

    # Phase 2: one DAG over all members' IB chains, namespaced by member.
    nodes: dict[str, Node] = {}
    seeds: dict[str, Seeded] = {}
    for m, ibseeds in enumerate(member_seeds):
        chain_nodes, chain_seeds = ib_chain_nodes(ibseeds, prefix=f"m{m} :: ")
        nodes.update({nd.id: nd for nd in chain_nodes})
        seeds.update(chain_seeds)

    LOG.info(f"running ensemble DAG with {len(nodes)} nodes and {len(seeds)} seeds...")
    outputs = executor.run(Dag(nodes=nodes, seeds=seeds))

    # Phase 3: collect each member's beads, sorted per chr.
    results: list[dict[str, list[BeadOut]]] = []
    for m, ibseeds in enumerate(member_seeds):
        per_chr: dict[str, list[BeadOut]] = defaultdict(list)
        for ibs in ibseeds:
            beads = _beads(outputs[ib_node_id(f"m{m} :: {ibs.ib_id}", StageKind.SMOOTH)])
            per_chr[ibs.chr_].extend(beads)

        results.append({c: sorted(b, key=lambda x: x.start) for c, b in per_chr.items()})

    return results
