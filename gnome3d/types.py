from dataclasses import dataclass


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


@dataclass
class BedRegion:
    chr: str
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end
