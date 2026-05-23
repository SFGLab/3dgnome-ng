from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Numpy array aliases used throughout the package.
F32Array: TypeAlias = NDArray[np.float32]
F64Array: TypeAlias = NDArray[np.float64]
I32Array: TypeAlias = NDArray[np.int32]
I64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StrArray: TypeAlias = NDArray[np.str_]

# (chr1, pos1, chr2, pos2, score) singleton contact tuple.
SingletonContact: TypeAlias = tuple[str, int, str, int, int]

# (genomic_midpoint_bp, x, y, z) bead position output tuple.
BeadOut: TypeAlias = tuple[int, float, float, float]


@dataclass
class Anchor:
    chr: str
    start: int
    end: int
    orientation: str = "N"  # 'L', 'R', or 'N'

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def length(self) -> int:
        return self.end - self.start + 1

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end


@dataclass
class RawArc:
    """Arc in genomic coordinates (before anchor-index mapping)."""
    start: int  # genomic midpoint of left anchor
    end: int  # genomic midpoint of right anchor
    score: float
    factor: int = 0


@dataclass
class InteractionArc:
    """Arc after anchor-index mapping."""
    start: int  # index into anchors list (local, chr-relative)
    end: int  # index into anchors list (local, chr-relative)
    score: int
    eff_score: int = 0
    factor: int = 0
    genomic_start: int = 0
    genomic_end: int = 0


# Arc-collection mappings used everywhere downstream.

AnchorMap: TypeAlias = dict[str, list["Anchor"]]

ArcMap: TypeAlias = dict[str, list[InteractionArc]]

RawArcMap: TypeAlias = dict[str, list["RawArc"]]

BreakpointMap: TypeAlias = dict[str, list[int]]


# Cluster-hierarchy types.

# A cluster index is a position in Solver.clusters (or the global list returned by build_cluster_tree).
ClusterIndex: TypeAlias = int

# A local arc index (chr-relative) - position in ArcMap[chr_].
LocalArcIndex: TypeAlias = int

# A genomic position in basepairs.
GenomicPos: TypeAlias = int

# Per-chromosome root cluster index (level = LVL_CHROMOSOME).
ChrRootMap: TypeAlias = dict[str, ClusterIndex]

# Per-chromosome index of the first anchor cluster (level = LVL_ANCHOR).
ChrFirstClusterMap: TypeAlias = dict[str, ClusterIndex]

# A traversal "current level" snapshot: chr -> list of cluster indices at that depth.
ChrLevel: TypeAlias = dict[str, list[ClusterIndex]]


@dataclass
class BedRegion:
    chr: str
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end

# Empty initializers

def empty_anchor_map() -> AnchorMap:
    return {}


def empty_arc_map() -> ArcMap:
    return {}


def empty_breakpoint_map() -> BreakpointMap:
    return {}


def empty_singleton_list() -> list[SingletonContact]:
    return []


def zero_pos() -> F32Array:
    return np.zeros(3, dtype=np.float32)


def empty_cluster_index_list() -> list[ClusterIndex]:
    return []


def empty_local_arc_index_list() -> list[LocalArcIndex]:
    return []
