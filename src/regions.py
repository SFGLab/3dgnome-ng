"""
Parse and apply chromosome/region filter specifications.

Supported syntax (comma-separated, combinable):
    chr1              single chromosome
    chr1-chr22        inclusive range of numbered/named chromosomes
    chrX              named chromosome
    chr14:1:2500000   specific genomic region (1-based, inclusive)

Default: 'chr1-chr22,chrX'

Chromosome order for range expansion follows the standard human karyotype:
    chr1..chr22, chrX, chrY, chrM
"""

import re
from typing import Dict, List, Optional, Tuple

# canonical order for range expansion
_CANONICAL_ORDER: List[str] = (
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
)
_CANONICAL_INDEX: Dict[str, int] = {c: i for i, c in enumerate(_CANONICAL_ORDER)}


# ── Region dataclass ─────────────────────────────────────────────────────────

class Region:
    """One selected region: a chromosome + optional coordinate window."""

    def __init__(self, chrom: str,
                 start: Optional[int] = None,
                 end: Optional[int] = None):
        self.chrom = chrom
        self.start = start  # None = whole chromosome
        self.end = end      # None = whole chromosome

    @property
    def is_whole_chrom(self) -> bool:
        return self.start is None

    def contains_anchor(self, anchor_start: int, anchor_end: int) -> bool:
        """True when an anchor interval overlaps this region."""
        if self.is_whole_chrom:
            return True
        s = self.start or 0
        e = self.end or 2**62
        return anchor_start <= e and anchor_end >= s

    def __repr__(self) -> str:
        if self.is_whole_chrom:
            return self.chrom
        return f"{self.chrom}:{self.start}:{self.end}"


# ── Parser ────────────────────────────────────────────────────────────────────

def _chrom_range(lo: str, hi: str) -> List[str]:
    """Expand 'chrA-chrB' to the list of chromosomes between them (inclusive)."""
    if lo not in _CANONICAL_INDEX or hi not in _CANONICAL_INDEX:
        # fall back: just return both endpoints
        return [lo, hi] if lo != hi else [lo]
    i_lo = _CANONICAL_INDEX[lo]
    i_hi = _CANONICAL_INDEX[hi]
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo
    return _CANONICAL_ORDER[i_lo: i_hi + 1]


def parse_region_spec(spec: str) -> List[Region]:
    """
    Parse a region specification string into a list of Region objects.

    Examples:
        'chr1-chr22,chrX'         → 23 whole-chromosome regions
        'chr14:1:2500000'         → chr14 region [1, 2500000]
        'chr1,chr3-chr5,chrX'     → chr1, chr3, chr4, chr5, chrX
    """
    regions: List[Region] = []

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        # region with coordinates: chrN:start:end
        if re.match(r'^chr\S+:\d+:\d+$', token):
            parts = token.split(":")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            regions.append(Region(chrom, start, end))
            continue

        # chromosome range: chrA-chrB  (both sides must start with 'chr')
        range_match = re.match(r'^(chr\S+?)-(chr\S+)$', token)
        if range_match:
            lo, hi = range_match.group(1), range_match.group(2)
            for c in _chrom_range(lo, hi):
                regions.append(Region(c))
            continue

        # single chromosome
        if token.startswith("chr") or token.isdigit():
            # tolerate bare numbers like '1' → 'chr1'
            chrom = token if token.startswith("chr") else f"chr{token}"
            regions.append(Region(chrom))
            continue

        raise ValueError(f"Cannot parse region token: {token!r}")

    return regions


def default_regions() -> List[Region]:
    return parse_region_spec("chr1-chr22,chrX")


# ── Filtering helpers ─────────────────────────────────────────────────────────

def build_filter(regions: List[Region]) -> Dict[str, List[Region]]:
    """Group regions by chromosome for O(1) chrom lookup."""
    d: Dict[str, List[Region]] = {}
    for r in regions:
        d.setdefault(r.chrom, []).append(r)
    return d


def chrom_included(chrom: str, flt: Dict[str, List[Region]]) -> bool:
    return chrom in flt


def filter_anchors(anchors, flt: Dict[str, List[Region]]):
    """
    Return a sub-list of anchors that fall inside any matching region.
    `anchors` is a list of Anchor objects.
    """
    result = []
    for a in anchors:
        regions = flt.get(a.chrom, [])
        for r in regions:
            if r.contains_anchor(a.start, a.end):
                result.append(a)
                break
    return result
