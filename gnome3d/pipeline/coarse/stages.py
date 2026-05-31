"""
Coarse positioning stages: the cluster-tree spine (hierarchy / chr / segment /
IB centroid) expressed as pipeline nodes, plus the fan-out into the per-IB chains.

These wrap the RNG-ordered ops in `coarse` (the same functions the
legacy `skeleton` path drives) as `COARSE`-kind stages over a `CoarsePhase`
carrier.  Unlike the per-IB kernels, a coarse stage's *work is its `apply`*: it
mutates the one shared cluster graph in place (positions a level), and the
registered COARSE runner is a no-op passthrough.  That's deliberate — the coarse
spine is a coupled serial sequence over one graph (never batched), so there's no
kernel to fan a `to_problem` into; the kernel split / batching machinery is for
the isolated IB fan-out downstream.

`build_coarse_dag` assembles the spine — picking the branch (random-walk /
single-segment / chr+segment) exactly as `coarse.reconstruct_heatmap`
does — and gives the terminal IB-positioning node an `expand` hook that reads the
positioned graph and spawns each IB's `arcs -> densify -> [heat] -> smooth` chain.

RNG note — exactness: the spine is a *linear chain* (one ready node at a time),
so the executor runs the ops in the same order the old engine did; the ops don't
re-seed (they consume the one global stream seeded once upstream), so the coarse
layout — and thus every IB seed and the final fold — is byte-preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gnome3d.hierarchy import Level, set_level
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.registry import register
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.stages.arcs import ArcsStage
from gnome3d.pipeline.stages.densify import DensifyStage
from gnome3d.pipeline.stages.heat import HeatDistStage
from gnome3d.pipeline.stages.smooth import SmoothStage
from gnome3d.pipeline.state import CoarsePhase, Seeded, State

if TYPE_CHECKING:
    from gnome3d.pipeline.coarse import CoarseState
    from gnome3d.skeleton import IBSeed


# COARSE runner: identity.  A coarse stage does its work in `apply` (mutating the
# shared cluster graph); there is no kernel to dispatch, so the runner passes the
# (empty) problem through unchanged.
def _coarse_identity(problem: Problem) -> Result:
    return problem


register(StageKind.COARSE, serial=_coarse_identity)


# --- node id helpers --------------------------------------------------------

_COARSE = "coarse"


def _coarse_id(level: str) -> NodeId:
    return f"{_COARSE} :: {level}"


def ib_node_id(ib_id: str, stage_name: str) -> NodeId:
    """The id of one per-IB chain node — must match `reconstruct.build_dag` so the
    two paths address the same nodes."""
    return f"{ib_id} :: {stage_name}"


# --- coarse stages (work-in-apply graph transforms) -------------------------


class _CoarseStage:
    """Base for the COARSE-kind graph stages: a no-op problem/runner, all work in
    `apply` (which mutates the shared cluster graph and returns the carrier)."""

    kind = StageKind.COARSE

    def bucket(self, inputs: tuple[State, ...]) -> int:
        return 0

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        return {}

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _phase(inputs: tuple[State, ...]) -> CoarsePhase:
        phase = inputs[0]
        assert isinstance(phase, CoarsePhase)
        return phase


class RootStage(_CoarseStage):
    """Source: seed the coarse spine's RNG stream and wrap the pre-built
    `CoarseState` into the spine's first carrier.

    Seeding here (rather than in the caller before the run) makes the seed state
    the pipeline owns.  It is safe to seed at this point — `build_state` and
    `build_coarse_dag` consume no RNG — so the first real consumer (chr/segment)
    sees the exact same stream state as the legacy path."""

    def __init__(self, state: CoarseState, seed_offset: int = 0) -> None:
        self._state = state
        self._seed_offset = seed_offset

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        seed = cb.seed_global_rng(self._seed_offset)
        return CoarsePhase(self._state, seed)


class ChrLevelStage(_CoarseStage):
    """Position chromosome roots via the inter-chr heatmap MC (multi-chr only)."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        cb.reconstruct_chromosome_level(phase.state)
        return phase


class SegmentStage(_CoarseStage):
    """Position segment-level beads via the segment heatmap MC."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.reconstruct_segment_level(phase.state, lvl)
        return phase


class WalkStage(_CoarseStage):
    """Position segment-level beads via a chained random walk (random_walk mode)."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.random_walk_segment_level(phase.state, lvl)
        return phase


class OriginStage(_CoarseStage):
    """Single-segment single-chr case: place at origin + interpolate children."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.place_single_segment(phase.state, lvl)
        return phase


class IBPositionStage(_CoarseStage):
    """Position every chromosome's IB centroids (interpolate / walk + IB-MC).  Its
    node carries the `expand` hook that fans out the per-IB reconstruction chains."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        state = phase.state
        lvl = _seg_level(state)
        for chr_ in state.chrs:
            segs = lvl.get(chr_, [])
            if segs:
                cb.position_interaction_blocks(state, segs)
        return phase


def _seg_level(state: CoarseState) -> dict[str, list[int]]:
    return set_level(Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs)


# --- the per-IB chain (mirrors reconstruct.build_dag) -----------------------


def _ib_chain_nodes(ibseeds: list[IBSeed]) -> tuple[list[Node], dict[NodeId, Seeded]]:
    """One linear chain per IB: arcs -> densify -> [heat] -> smooth.  HEAT_DIST is
    included only when the skeleton kept it (`IBSeed.wants_heat`)."""
    nodes: list[Node] = []
    seeds: dict[NodeId, Seeded] = {}
    for ibs in ibseeds:
        chain: list[tuple[StageKind, object]] = [
            (StageKind.ARCS, ArcsStage()),
            (StageKind.DENSIFY, DensifyStage()),
        ]
        if ibs.wants_heat:
            chain.append((StageKind.HEAT_DIST, HeatDistStage()))
        chain.append((StageKind.SMOOTH, SmoothStage()))

        prev: NodeId | None = None
        for kind, stage in chain:
            nid = ib_node_id(ibs.ib_id, kind.value)
            nodes.append(Node(nid, stage, () if prev is None else (prev,)))  # type: ignore[arg-type]
            if prev is None:
                seeds[nid] = ibs.seed
            prev = nid
    return nodes, seeds


def _expand_into_ib_chains(seed_offset: int, sink: list[IBSeed]) -> object:
    """Build the IB-positioning node's `expand` hook.  On completion it reads the
    positioned graph, gathers every IB's `Seeded` (via the shared
    `skeleton.gather_ib_seeds`), records them in `sink` (so the caller can map
    smooth outputs back to chromosomes), and returns the per-IB chain nodes."""

    def expand(output: State) -> tuple[list[Node], dict[NodeId, Seeded]]:
        # Imported lazily to avoid an import cycle (skeleton -> coarse,
        # this module -> skeleton at call time only).
        from gnome3d import skeleton

        assert isinstance(output, CoarsePhase)
        state = output.state
        lvl = _seg_level(state)
        ibseeds: list[IBSeed] = []
        for chr_ in state.chrs:
            ibseeds.extend(skeleton.gather_ib_seeds(state, chr_, lvl, seed_offset))
        sink.extend(ibseeds)
        return _ib_chain_nodes(ibseeds)

    return expand


# --- the unified coarse DAG -------------------------------------------------


def build_coarse_dag(state: CoarseState, seed_offset: int = 0) -> tuple[Dag, list[IBSeed]]:
    """Assemble the coarse spine for `state` and return ``(dag, ib_sink)``.

    The spine branches exactly as `coarse.reconstruct_heatmap`:
      * ``random_walk``  -> root -> walk    -> ib
      * single segment   -> root -> origin  -> ib
      * multi-chromosome -> root -> chr     -> segment -> ib
      * single-chr/multi -> root -> segment -> ib

    ``ib_sink`` is filled at run time by the IB node's `expand` with the gathered
    `IBSeed`s (empty until the DAG runs) so the caller can group the terminal
    smooth beads by chromosome.
    """
    lvl = _seg_level(state)
    total_segs = sum(len(v) for v in lvl.values())
    single_seg = len(state.chrs) == 1 and total_segs <= 1

    nodes: dict[NodeId, Node] = {}

    def chain(stage: _CoarseStage, level: str, dep: NodeId | None, expand: object = None) -> NodeId:
        nid = _coarse_id(level)
        deps: tuple[NodeId, ...] = () if dep is None else (dep,)
        nodes[nid] = Node(nid, stage, deps, expand)  # type: ignore[arg-type]
        return nid

    prev = chain(RootStage(state, seed_offset), "root", None)

    if state.s.random_walk:
        prev = chain(WalkStage(), "walk", prev)
    elif single_seg:
        prev = chain(OriginStage(), "origin", prev)
    else:
        if len(state.chrs) > 1:
            prev = chain(ChrLevelStage(), "chr", prev)
        prev = chain(SegmentStage(), "segment", prev)

    ib_sink: list[IBSeed] = []
    chain(IBPositionStage(), "ib", prev, _expand_into_ib_chains(seed_offset, ib_sink))

    return Dag(nodes=nodes, seeds={}), ib_sink
