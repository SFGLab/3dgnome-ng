"""Core dataclasses mirroring Cluster / InteractionArc / Anchor from cudaMMC."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Anchor:
    """A CTCF anchor site (one BED entry)."""
    chrom: str
    start: int
    end: int
    orientation: str = "N"  # 'L', 'R', or 'N'
    name: str = ""

    @property
    def mid(self) -> int:
        return (self.start + self.end) // 2

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class InteractionArc:
    """A ChIA-PET arc between two anchor indices (after mark-arcs mapping)."""
    start: int        # anchor index (left)
    end: int          # anchor index (right)
    genomic_start: int = 0
    genomic_end: int = 0
    score: int = 0    # raw PET count
    eff_score: int = 0  # effective score (0 for non-summary multi-factor arcs)
    factor: int = -1  # -1 = combined/summary arc


@dataclass
class RawArc:
    """A ChIA-PET arc as read from file (genomic positions, not anchor indices)."""
    chrom1: str
    start1: int
    end1: int
    chrom2: str
    start2: int
    end2: int
    score: int = 1
    factor: int = 0


@dataclass
class Cluster:
    """
    One bead in the hierarchical model.

    Levels (cluster.level field):
        4 = anchor (leaf)
        3 = interaction block (IB)
        2 = segment
        1 = chromosome root
    """
    chrom: str
    start: int           # genomic start
    end: int             # genomic end
    genomic_pos: int     # representative genomic coordinate (midpoint or anchor mid)
    level: int = 4       # 1..4 as above
    orientation: str = "N"

    # 3-D position  (x, y, z) — initialised to 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    parent: int = -1
    children: List[int] = field(default_factory=list)

    # arc indices (into a per-chromosome arc list)
    arcs: List[int] = field(default_factory=list)

    is_fixed: bool = False
    dist_to_next: float = 0.0  # expected distance to next sibling in chain

    # genomic extents at the *base* (leaf) level covered by this cluster
    base_start: int = 0
    base_end: int = 0

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    def set_pos(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z
