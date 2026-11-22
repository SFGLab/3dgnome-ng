"""The pipeline driver. Builds the coarse DAG, runs it under an executor and returns the beads
per chromosome.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from gnome3d import log, skeleton
from gnome3d.mc.jax.util import jax_is_available
from gnome3d.pipeline import Dag, Node
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse.stages import build_coarse_dag
from gnome3d.pipeline.executor import (
    Executor,
    ExecutorStrategy,
    MixedExecutor,
    SerialExecutor,
)
from gnome3d.pipeline.holes import fill_holes
from gnome3d.pipeline.ib import ib_chain_nodes, ib_node_id
from gnome3d.pipeline.relax import relax_blocks
from gnome3d.pipeline.stage import StageKind
from gnome3d.pipeline.state import Seeded, Smoothed, State
from gnome3d.pipeline.stitch import stitch_blocks

if TYPE_CHECKING:
    from gnome3d.data import ContactData
    from gnome3d.settings import Settings
    from gnome3d.skeleton import IBSeed
    from gnome3d.types import BeadOut, BedRegion, BoolArray

LOG = log.get("reconstruct")

# The seed offset between two ensemble members.
MEMBER_SEED_STRIDE = 2_000_003


# The worker count from which the threaded smooth stage is faster than the device, design/kernel-
# performance.md section 14.
_SMOOTH_THREADED_MIN_WORKERS = 4


def _auto_strategy(kind: StageKind, settings: Settings) -> ExecutorStrategy:
    """Resolves `mc_executor_<stage> = auto`. The smooth stage runs threaded on numba from
    `_SMOOTH_THREADED_MIN_WORKERS` workers and on the device below that when JAX is installed.
    """
    workers = int(settings.mc_executor_threaded_workers)
    if kind is StageKind.SMOOTH and workers >= _SMOOTH_THREADED_MIN_WORKERS:
        return ExecutorStrategy.THREADED
    prefer_jax = jax_is_available()
    if prefer_jax and kind in (StageKind.SMOOTH, StageKind.ESTIMATE_DIST):
        return ExecutorStrategy.BATCH

    return (
        ExecutorStrategy.THREADED
        if int(settings.mc_executor_threaded_workers) > 1
        else ExecutorStrategy.SERIAL
    )


def _resolve_strategy(value: str, kind: StageKind, settings: Settings) -> ExecutorStrategy:
    """Maps one `mc_executor_<stage>` value to a strategy.

    A batch choice that cannot run becomes threaded or serial. The small IB boost has per IB
    springs and DENSIFY has no batch kernel.
    """
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


def stage_strategy(settings: Settings, kind: StageKind) -> ExecutorStrategy:
    """Returns the strategy of one stage from its `mc_executor_<stage>` value."""
    value = {
        StageKind.ARCS: settings.mc_executor_arcs,
        StageKind.DENSIFY: settings.mc_executor_densify,
        StageKind.ESTIMATE_DIST: settings.mc_executor_estimate_dist,
        StageKind.SMOOTH: settings.mc_executor_smooth,
    }[kind]
    return _resolve_strategy(value, kind, settings)


def pick_executor(settings: Settings) -> Executor:
    """Builds the executor from the `mc_executor_*` settings."""
    strategy = {
        StageKind.ARCS: _resolve_strategy(settings.mc_executor_arcs, StageKind.ARCS, settings),
        StageKind.DENSIFY: _resolve_strategy(
            settings.mc_executor_densify, StageKind.DENSIFY, settings
        ),
        StageKind.ESTIMATE_DIST: _resolve_strategy(
            settings.mc_executor_estimate_dist, StageKind.ESTIMATE_DIST, settings
        ),
        StageKind.SMOOTH: _resolve_strategy(
            settings.mc_executor_smooth, StageKind.SMOOTH, settings
        ),
    }

    return MixedExecutor(
        strategy,
        max_workers=int(settings.mc_executor_threaded_workers),
        multigpu_mode=settings.mc_multigpu_mode,
    )


def _beads(output: State) -> list[BeadOut]:
    """Returns the beads of a terminal smooth node."""
    assert isinstance(output, Smoothed)
    return output.beads


def _fixed(output: State) -> BoolArray:
    """Returns the beads of a terminal smooth node that the smooth stage did not move."""
    assert isinstance(output, Smoothed)
    return output.fixed


def _assemble(
    per_chr: dict[str, list[list[BeadOut]]],
    settings: Settings,
    seed: int = 0,
    fixed: dict[str, list[BoolArray]] | None = None,
) -> dict[str, list[BeadOut]]:
    """Joins the beads of each chromosome's blocks into one list in genomic order.

    With `use_hole_beads` the holes between blocks get beads first, drawn from generators seeded
    by `seed`. With `use_boundary_stitch` the blocks are stitched at their edges. `fixed` gives
    the nodes of each block of a uniform chain, which the relaxation holds.
    """
    if fixed is not None and (settings.use_hole_beads or settings.use_boundary_stitch):
        raise ValueError(
            "[main] uniform_chain is on with hole beads or the boundary stitch. Both read the"
            " anchors from the kind of a bead and are not stated on the nodes of the uniform"
            " chain. Set [holes] use_hole_beads and [boundary_stitch] use_boundary_stitch to no."
        )
    out: dict[str, list[BeadOut]] = {}
    for chr_, blocks in per_chr.items():
        blocks = fill_holes(blocks, settings, chr_, seed)
        if settings.use_boundary_stitch:
            blocks = stitch_blocks(blocks, settings)
        if settings.use_cross_block_relax:
            blocks = relax_blocks(blocks, settings, None if fixed is None else fixed[chr_])
        out[chr_] = sorted((b for block in blocks for b in block), key=lambda b: b.start)
    return out


def reconstruct(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    executor: Executor | None = None,
    seed_offset: int = 0,
) -> dict[str, list[BeadOut]]:
    """Reconstructs one structure and returns a bead list per chromosome, sorted by genomic start.

    `seed_offset` selects the ensemble member. The root stage of the coarse DAG seeds the global
    RNG from it.
    """
    state = cb.build_state(settings, data, chrs, region)
    dag, ib_sink = build_coarse_dag(state, seed_offset)

    LOG.info(f"running DAG with {len(dag.nodes)} nodes and {len(dag.seeds)} seeds...")
    outputs = (executor or SerialExecutor()).run(dag)

    per_chr: dict[str, list[list[BeadOut]]] = defaultdict(list)
    fixed: dict[str, list[BoolArray]] = defaultdict(list)
    for ibs in ib_sink:
        out = outputs[ib_node_id(ibs.ib_id, StageKind.SMOOTH)]
        per_chr[ibs.chr_].append(_beads(out))
        fixed[ibs.chr_].append(_fixed(out))

    return _assemble(per_chr, settings, seed_offset, fixed if settings.uniform_chain else None)


def reconstruct_ensemble(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    n: int = 1,
    executor: Executor | None = None,
    base_seed_offset: int = 0,
) -> list[dict[str, list[BeadOut]]]:
    """Reconstructs `n` structures and batches the IB chains across members.

    The coarse stages of each member run one after another. They use the global RNG. The
    IB chains of all members then run as one DAG. Member m uses
    `base_seed_offset + m * MEMBER_SEED_STRIDE` and is byte identical to `reconstruct` at that
    offset.
    """
    executor = executor or SerialExecutor()

    member_seeds: list[list[IBSeed]] = []
    for m in range(n):
        off = base_seed_offset + m * MEMBER_SEED_STRIDE
        state = cb.build_state(settings, data, chrs, region)
        spine, _ = build_coarse_dag(state, off, fan_out=False)
        executor.run(spine)
        member_seeds.append(skeleton.gather_all_ib_seeds(state, off))

    n_ib = sum(len(s) for s in member_seeds)
    LOG.info(f"ensemble: {n} members, {n_ib} IB chains total")

    nodes: dict[str, Node] = {}
    seeds: dict[str, Seeded] = {}
    for m, ibseeds in enumerate(member_seeds):
        chain_nodes, chain_seeds = ib_chain_nodes(ibseeds, prefix=f"m{m} :: ")
        nodes.update({nd.id: nd for nd in chain_nodes})
        seeds.update(chain_seeds)

    LOG.info(f"running ensemble DAG with {len(nodes)} nodes and {len(seeds)} seeds...")
    outputs = executor.run(Dag(nodes=nodes, seeds=seeds))

    results: list[dict[str, list[BeadOut]]] = []
    for m, ibseeds in enumerate(member_seeds):
        per_chr: dict[str, list[list[BeadOut]]] = defaultdict(list)
        fixed: dict[str, list[BoolArray]] = defaultdict(list)
        for ibs in ibseeds:
            nid = ib_node_id(f"m{m} :: {ibs.ib_id}", StageKind.SMOOTH)
            per_chr[ibs.chr_].append(_beads(outputs[nid]))
            fixed[ibs.chr_].append(_fixed(outputs[nid]))

        results.append(
            _assemble(
                per_chr,
                settings,
                base_seed_offset + m * MEMBER_SEED_STRIDE,
                fixed if settings.uniform_chain else None,
            )
        )

    return results
