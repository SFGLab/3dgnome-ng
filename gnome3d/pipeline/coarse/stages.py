"""The coarse stages as pipeline nodes and the fan out into the IB chains.

A coarse stage does its work in `apply`, which positions one level of the shared cluster graph
in place. Its runner returns the problem unchanged. `build_coarse_dag` builds the chain of
stages and gives the last node an `expand` hook that spawns the chain of each IB.

The stages are one linear chain and draw from one global RNG stream, seeded once at the root.
Their order must not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gnome3d.hierarchy import Level, set_level
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.ib import ib_chain_nodes
from gnome3d.pipeline.registry import register
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import CoarsePhase, State

if TYPE_CHECKING:
    from gnome3d.pipeline.coarse import CoarseState
    from gnome3d.pipeline.state import Seeded
    from gnome3d.skeleton import IBSeed


def _coarse_identity(problem: Problem) -> Result:
    return problem


register(StageKind.COARSE, serial=_coarse_identity)


_COARSE = "coarse"


def _coarse_id(level: str) -> NodeId:
    return f"{_COARSE} :: {level}"


class _CoarseStage:
    """Base of the coarse stages. The work is in `apply`, which mutates the shared cluster graph
    and returns the payload.
    """

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
    """Seeds the global RNG and wraps the built `CoarseState` into the first payload.

    `build_state` and `build_coarse_dag` draw no random number. Seeding here is early enough.
    """

    def __init__(self, state: CoarseState, seed_offset: int = 0) -> None:
        self._state = state
        self._seed_offset = seed_offset

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        seed = cb.seed_global_rng(self._seed_offset)
        return CoarsePhase(self._state, seed)


class ChrLevelStage(_CoarseStage):
    """Positions the chromosome roots by the inter chromosomal heatmap MC."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        cb.reconstruct_chromosome_level(phase.state)
        return phase


class SegmentStage(_CoarseStage):
    """Positions the segment beads by the segment heatmap MC."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.reconstruct_segment_level(phase.state, lvl)
        return phase


class WalkStage(_CoarseStage):
    """Positions the segment beads on a chained random walk, under `random_walk`."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.random_walk_segment_level(phase.state, lvl)
        return phase


class OriginStage(_CoarseStage):
    """Places one segment of one chromosome at the origin and interpolates its children."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        lvl = _seg_level(phase.state)
        cb.place_single_segment(phase.state, lvl)
        return phase


class IBPositionStage(_CoarseStage):
    """Positions the IB centroids of every chromosome. Its node carries the `expand` hook."""

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        phase = self._phase(inputs)
        state = phase.state
        lvl = _seg_level(state)
        for chr_ in state.chrs:
            segs = lvl.get(chr_, [])
            if segs:
                cb.position_interaction_blocks(state, segs, chr_)
        return phase


def _seg_level(state: CoarseState) -> dict[str, list[int]]:
    return set_level(Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs)


def _expand_into_ib_chains(seed_offset: int, sink: list[IBSeed]) -> object:
    """Builds the `expand` hook of the IB positioning node.

    The hook gathers the `Seeded` of every IB from the positioned graph, appends them to `sink`
    and returns the chain nodes of each IB.
    """

    def expand(output: State) -> tuple[list[Node], dict[NodeId, Seeded]]:
        # Imported here. Skeleton imports coarse.
        from gnome3d import skeleton

        assert isinstance(output, CoarsePhase)
        ibseeds = skeleton.gather_all_ib_seeds(output.state, seed_offset)
        sink.extend(ibseeds)
        return ib_chain_nodes(ibseeds)

    return expand


def build_coarse_dag(
    state: CoarseState, seed_offset: int = 0, *, fan_out: bool = True
) -> tuple[Dag, list[IBSeed]]:
    """Builds the coarse DAG of `state` and returns `(dag, ib_sink)`.

      random walk             root -> walk    -> ib
      one segment             root -> origin  -> ib
      several chromosomes     root -> chr     -> segment -> ib
      one chromosome          root -> segment -> ib

    With `fan_out` the last node spawns the IB chains at run time and fills `ib_sink`. Without it
    the DAG places the cluster tree alone, which the ensemble driver uses.
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
    expand = _expand_into_ib_chains(seed_offset, ib_sink) if fan_out else None
    chain(IBPositionStage(), "ib", prev, expand)

    return Dag(nodes=nodes, seeds={}), ib_sink
