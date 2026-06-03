"""
Sealed state progression for an IB reconstruction.

A task's data is a *closed chain of frozen states*, each adding exactly what its
producing stage yields:

    Seeded ──arcs──▶ Arced ──densify──▶ Densified ──est_dist──▶ DistEstimated ──smooth──▶ Smoothed

A stage is typed by the state it consumes and the state it produces, so you
cannot reach for `pos` before densify - the type doesn't carry it.  States are
immutable; a stage returns the next state rather than mutating a shared bag.
No field references the solver or the global cluster graph: everything an IB
needs is copied in, which is what makes the task isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from gnome3d.pipeline.coarse import CoarseState
    from gnome3d.settings import Settings
    from gnome3d.types import BeadOut, BoolArray, F32Array, F64Array, I8Array

# (bead_index_in_densified, anchor_index_in_ib) - abstract, no global cluster ref.
AnchorMapEntry = tuple[int, int]


class Orientation(IntEnum):
    """CTCF motif orientation per anchor.  Stored as an `int8` array (one byte
    each, vectorizable) instead of a `list[str]` of ``'N'/'L'/'R'`` - same info,
    ~50x less memory and the form the kernels actually want (``orn == LEFT``).
    """

    NONE = 0  # no motif call ('N')
    LEFT = 1  # 'L' - orientation vector is negated downstream
    RIGHT = 2  # 'R'


@dataclass(frozen=True)
class Seeded:
    """Skeleton output / chain input: everything an isolated IB needs, copied in
    (no solver or cluster references)."""

    settings: Settings
    seed: int  # RNG seed = f(ib_id, ensemble_member)
    anchor_seed_pos: F32Array  # (A, 3) seed positions from IB positioning
    exp_dist: F64Array  # (A, A) arc expected-distance matrix
    orientations: I8Array | None  # (A,) Orientation codes, or None when motif off
    anchor_neighbors: dict[int, list[int]] | None  # CTCF motif graph (anchor-local)
    anchor_neighbor_weights: dict[int, list[float]] | None
    subanchor_heat_raw: F64Array | None  # raw subanchor contact heatmap (or None)
    anchor_genomic: list[tuple[int, int, int]]  # (start_bp, end_bp, midpoint_bp) per anchor
    step_size_arcs: float


@dataclass(frozen=True)
class Arced(Seeded):
    """After arc-spring MC: anchors are positioned."""

    anchor_pos: F32Array  # (A, 3)


@dataclass(frozen=True)
class Densified(Arced):
    """After densify: subanchor beads inserted; arrays are now bead-shaped (N)."""

    pos: F32Array  # (N, 3) anchors + subanchors
    fixed: BoolArray  # (N,) which beads are pinned during smooth
    dtn: F32Array  # (N-1,) consecutive-bead bond targets
    anchor_map: list[AnchorMapEntry]
    bead_starts: list[int]  # (N,) genomic start per bead
    bead_ends: list[int]  # (N,) genomic end per bead
    step_size_smooth: float


@dataclass(frozen=True)
class DistEstimated(Densified):
    """After heat-dist estimate: the subanchor distance target (None when the
    sparse-signal early-out skipped it -> smooth runs without heat)."""

    heat_dist: F64Array | None  # (N, N)


@dataclass(frozen=True)
class Smoothed(DistEstimated):
    """Terminal: final positions and the write-back beads."""

    final_pos: F32Array  # (N, 3)
    beads: list[BeadOut]


@dataclass(frozen=True)
class CoarsePhase:
    """Carrier for the coarse positioning spine (hierarchy -> chr -> segment ->
    ib), threaded as a *separate* payload from the Seeded IB progression.

    It wraps the shared `CoarseState`: an immutable field set whose cluster
    objects mutate in place as each coarse stage positions a level (a graph
    algorithm - there is one graph, the stages write `.pos` into it).  A coarse
    stage takes a `CoarsePhase`, mutates the graph, and returns a `CoarsePhase`
    over the same state; the terminal IB-positioning node's `expand` reads the
    positioned graph to fan out the per-IB `Seeded` chains.

    `seed` is the coarse spine's RNG seed - the root stage seeds the global stream
    from it and the spine *flows* that one stream in dependency order (it does not
    re-seed per stage, unlike the per-IB `Seeded.seed`; the coarse levels are a
    coupled linear sequence, never reordered).  Carried here so the seed is state
    the pipeline owns, not a side-effect the caller applies before the run.
    """

    state: CoarseState
    seed: int


# The set of payloads that flow through the DAG.  The `Seeded` subclasses are the
# per-IB progression; `CoarsePhase` is the upstream coarse spine's carrier.
State: TypeAlias = Seeded | Arced | Densified | DistEstimated | Smoothed | CoarsePhase
