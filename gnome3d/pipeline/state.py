"""The states of an IB reconstruction.

    Seeded -> Arced -> Densified -> DistEstimated -> Smoothed

Each state is frozen and adds the output of its stage. No field refers to the cluster graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from gnome3d.pipeline.coarse import CoarseState
    from gnome3d.settings import Settings
    from gnome3d.types import (
        BeadOut,
        BoolArray,
        CompartmentInterval,
        F32Array,
        F64Array,
        I8Array,
        I32Array,
    )

# (bead index after densify, anchor index in the IB)
AnchorMapEntry = tuple[int, int]


class Orientation(IntEnum):
    """The CTCF motif orientation of an anchor, stored as int8."""

    NONE = 0  # N
    LEFT = 1  # L, the orientation vector is negated downstream
    RIGHT = 2  # R


@dataclass(frozen=True)
class Seeded:
    """The input of an IB chain. Everything is copied in."""

    settings: Settings
    seed: int  # RNG seed
    anchor_seed_pos: F32Array  # (A, 3) seed positions
    exp_dist: F64Array  # (A, A) expected distances of the arcs stage
    orientations: I8Array | None  # (A,) Orientation codes, None with the motif term off
    anchor_neighbors: dict[int, list[int]] | None  # motif neighbours by anchor index
    anchor_neighbor_weights: dict[int, list[float]] | None
    subanchor_heat_raw: F32Array | None  # (N, N) raw subanchor contact heatmap
    anchor_genomic: list[tuple[int, int, int]]  # (start_bp, end_bp, midpoint_bp) per anchor
    step_size_arcs: float
    # The slice of the chromosome's compartment track that this IB spans. None without a track or
    # with the term off.
    track_compartments: list[CompartmentInterval] | None
    # The spring weight of each arc pair, None at exponent zero. Keyword only. The positional
    # fields of the subclasses keep their places.
    arc_w: F32Array | None = field(default=None, kw_only=True)
    # The activity of each anchor for the factory term.
    activity: F32Array | None = field(default=None, kw_only=True)
    chrom: str = field(default="", kw_only=True)
    # Under `uniform_chain` the nodes of the block, the beads that carry a loop end in this
    # conformation, and their positions from the joint arcs solve. `orientations` and the
    # neighbour lists are then per node.
    node_bead: I32Array | None = field(default=None, kw_only=True)
    node_pos: F32Array | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Arced(Seeded):
    """After the arcs stage. The anchors are placed."""

    anchor_pos: F32Array  # (A, 3)


@dataclass(frozen=True)
class Densified(Arced):
    """After densify. Arrays have one entry per bead."""

    pos: F32Array  # (N, 3) anchors and subanchors
    fixed: BoolArray  # (N,) beads that smooth does not move
    dtn: F32Array  # (N-1,) bond targets
    anchor_map: list[AnchorMapEntry]
    bead_starts: list[int]  # (N,) genomic start
    bead_ends: list[int]  # (N,) genomic end
    step_size_smooth: float
    # (N,) compartment per bead
    bead_compartment: I8Array | None
    # (N,) beads under an anchor, for the kind of the output, under `uniform_chain`.
    covered: BoolArray | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class DistEstimated(Densified):
    """After the distance estimate. `heat_dist` is None when the heat signal is too sparse."""

    heat_dist: F32Array | None  # (N, N) float32


@dataclass(frozen=True)
class Smoothed(DistEstimated):
    """The final state."""

    final_pos: F32Array  # (N, 3)
    beads: list[BeadOut]


@dataclass(frozen=True)
class CoarsePhase:
    """The payload of the coarse stages.

    It wraps the shared `CoarseState`, whose clusters each coarse stage positions in place.
    `seed` seeds the global RNG at the root stage. The coarse stages draw from that one stream in
    order and do not seed again.
    """

    state: CoarseState
    seed: int


State: TypeAlias = Seeded | Arced | Densified | DistEstimated | Smoothed | CoarsePhase
