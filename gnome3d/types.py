from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

F32Array: TypeAlias = NDArray[np.float32]
F64Array: TypeAlias = NDArray[np.float64]
I8Array: TypeAlias = NDArray[np.int8]
I32Array: TypeAlias = NDArray[np.int32]
I64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StrArray: TypeAlias = NDArray[np.str_]

# (chr1, pos1, chr2, pos2, score)
SingletonContact: TypeAlias = tuple[str, int, str, int, int]

BeadKind: TypeAlias = Literal["anchor", "subanchor"]


class BeadOut(NamedTuple):
    """One bead of a reconstructed structure.

    `start` and `end` are the genomic range in bp. `kind` is "anchor" for a loop anchor and
    "subanchor" for a bead that densification inserts.
    """

    start: int
    end: int
    x: float
    y: float
    z: float
    kind: BeadKind

    @property
    def midpoint(self) -> int:
        return (self.start + self.end) // 2

    @property
    def span(self) -> int:
        return self.end - self.start + 1


class AnchorSpans(NamedTuple):
    """The anchors of a block as spans of beads of the uniform chain.

    Anchor k holds the beads `lo[k]` to `hi[k] - 1`. `knot[k]` is the bead at its midpoint.
    `home[k]` is its position from the arcs stage, where the smooth stage holds the centroid of
    the span.
    """

    lo: I32Array
    hi: I32Array
    knot: I32Array
    home: F64Array


@dataclass
class Anchor:
    chr: str
    start: int
    end: int
    orientation: str = "N"  # L, R or N

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def length(self) -> int:
        return self.end - self.start + 1

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end


@dataclass
class RawArc:
    """An arc in genomic coordinates."""

    start: int  # midpoint of the left anchor, bp
    end: int  # midpoint of the right anchor, bp
    score: float
    factor: int = 0


@dataclass
class InteractionArc:
    """An arc between two anchors of one chromosome."""

    start: int  # index into the chromosome's anchor list
    end: int  # index into the chromosome's anchor list
    score: int
    eff_score: int = 0
    factor: int = 0
    genomic_start: int = 0
    genomic_end: int = 0


AnchorMap: TypeAlias = dict[str, list["Anchor"]]

ArcMap: TypeAlias = dict[str, list[InteractionArc]]

RawArcMap: TypeAlias = dict[str, list["RawArc"]]

BreakpointMap: TypeAlias = dict[str, list[int]]


# Index into the global cluster list.
ClusterIndex: TypeAlias = int

# Index into `ArcMap[chr]`.
LocalArcIndex: TypeAlias = int

# bp
GenomicPos: TypeAlias = int

# The chromosome level cluster of each chromosome.
ChrRootMap: TypeAlias = dict[str, ClusterIndex]

# The first anchor cluster of each chromosome.
ChrFirstClusterMap: TypeAlias = dict[str, ClusterIndex]

# The cluster indices of each chromosome at one level of the tree.
ChrLevel: TypeAlias = dict[str, list[ClusterIndex]]


@dataclass
class BedRegion:
    chr: str
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end


class Compartment(IntEnum):
    """The A or B call of one bead, stored as int8.

    Positive codes are A, negative codes are B and the magnitude is the subcompartment rank, as in
    MultiMM. A bare A reads as A2 and a bare B as B1.
    """

    B2 = -2
    B1 = -1
    NONE = 0
    A2 = 1
    A1 = 2

    @property
    def is_a(self) -> bool:
        return self.value > 0

    @property
    def is_b(self) -> bool:
        return self.value < 0


@dataclass
class CompartmentInterval:
    """One compartment call over a genomic range.

    `score` is the signed strength in [-1, 1], positive for A and 0.0 for a source with labels
    only. `cls` is `NONE` for a value that is not phased yet.
    """

    chr: str
    start: int
    end: int
    cls: int = int(Compartment.NONE)
    score: float = 0.0


@dataclass
class SignalInterval:
    """One binned value of a continuous track such as ATAC-seq."""

    chr: str
    start: int
    end: int
    value: float


CompartmentMap: TypeAlias = dict[str, list["CompartmentInterval"]]

SignalMap: TypeAlias = dict[str, list["SignalInterval"]]


def empty_anchor_map() -> AnchorMap:
    return {}


def empty_arc_map() -> ArcMap:
    return {}


def empty_raw_arc_map() -> RawArcMap:
    return {}


def empty_breakpoint_map() -> BreakpointMap:
    return {}


def empty_singleton_list() -> list[SingletonContact]:
    return []


def empty_compartment_map() -> CompartmentMap:
    return {}


def empty_signal_map() -> SignalMap:
    return {}


def zero_pos() -> F32Array:
    return np.zeros(3, dtype=np.float32)


def empty_cluster_index_list() -> list[ClusterIndex]:
    return []


def empty_local_arc_index_list() -> list[LocalArcIndex]:
    return []
