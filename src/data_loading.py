"""Parse ChIA-PET input files (anchors BED, PET-clusters BEDPE, singletons BEDPE)."""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data_structures import Anchor, InteractionArc, RawArc


# ── BED / BEDPE readers ──────────────────────────────────────────────────────

def load_anchors(bed_path: str) -> Dict[str, List[Anchor]]:
    """Return {chrom: [Anchor, ...]} sorted by start position."""
    result: Dict[str, List[Anchor]] = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 else ""
            orientation = "N"
            if len(parts) > 5:
                strand = parts[5]
                if strand == "+":
                    orientation = "R"
                elif strand == "-":
                    orientation = "L"
            result[chrom].append(Anchor(chrom, start, end, orientation, name))

    for chrom in result:
        result[chrom].sort(key=lambda a: a.start)

    return dict(result)


def _parse_bedpe_line(parts: List[str], factor: int) -> Optional[RawArc]:
    if len(parts) < 6:
        return None
    chrom1, s1, e1 = parts[0], int(parts[1]), int(parts[2])
    chrom2, s2, e2 = parts[3], int(parts[4]), int(parts[5])
    score = 1
    if len(parts) > 7:
        try:
            score = int(parts[7])
        except ValueError:
            pass
    elif len(parts) > 6:
        try:
            score = int(parts[6])
        except ValueError:
            pass
    return RawArc(chrom1, s1, e1, chrom2, s2, e2, score, factor)


def load_pet_clusters(bedpe_path: str, factor: int = 0) -> List[RawArc]:
    """Load PET-cluster BEDPE file."""
    arcs: List[RawArc] = []
    with open(bedpe_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            arc = _parse_bedpe_line(line.split("\t"), factor)
            if arc is not None:
                arcs.append(arc)
    return arcs


def load_singletons(bedpe_path: str, factor: int = 0) -> List[RawArc]:
    """Singletons have score=1 by convention."""
    return load_pet_clusters(bedpe_path, factor)


# ── Arc → anchor mapping (CPU equivalent of ParallelMarkArcs.cu) ─────────────

def _anchor_contains(anchor: Anchor, pos: int) -> bool:
    return anchor.start <= pos <= anchor.end


def mark_arcs(raw_arcs: List[RawArc],
              anchors_by_chr: Dict[str, List[Anchor]],
              ignore_missing: bool = False
              ) -> Tuple[Dict[str, List[InteractionArc]], Dict[str, int]]:
    """
    Map raw arcs (genomic positions) to anchor indices.

    Returns:
        arcs_by_chr  : {chrom: [InteractionArc, ...]}
        arcs_cnt     : {chrom: count}
    """
    # group raw arcs by chromosome pair
    raw_by_chr: Dict[str, List[RawArc]] = defaultdict(list)
    for arc in raw_arcs:
        if arc.chrom1 == arc.chrom2:
            raw_by_chr[arc.chrom1].append(arc)
        # inter-chromosomal arcs are stored under a canonical key
        else:
            key = _inter_key(arc.chrom1, arc.chrom2)
            raw_by_chr[key].append(arc)

    arcs_by_chr: Dict[str, List[InteractionArc]] = {}
    arcs_cnt: Dict[str, int] = {}

    for chrom, chr_anchors in anchors_by_chr.items():
        raw = raw_by_chr.get(chrom, [])
        mapped: List[InteractionArc] = []
        # temporary buffer: {end_anchor_idx: [InteractionArc]}
        tmp: Dict[int, List[InteractionArc]] = defaultdict(list)
        last_start = -1

        raw_sorted = sorted(raw, key=lambda r: r.start1)

        for raw_arc in raw_sorted:
            ai = _find_anchor(chr_anchors, raw_arc.start1)
            bi = _find_anchor(chr_anchors, raw_arc.start2)
            if ai == -1 or bi == -1:
                if not ignore_missing:
                    pass  # silently skip (matching cudaMMC's "! error" but continuing)
                continue
            if ai == bi:
                continue  # skip looping arcs
            # ensure left < right
            if ai > bi:
                ai, bi = bi, ai
            ia = InteractionArc(ai, bi,
                                raw_arc.start1, raw_arc.start2,
                                raw_arc.score, raw_arc.score, raw_arc.factor)
            tmp[bi].append(ia)

            # flush when start anchor changes
            if ai != last_start and last_start != -1:
                _flush_tmp(tmp, mapped, last_start)
                tmp.clear()
            last_start = ai

        # final flush
        _flush_tmp(tmp, mapped, last_start)

        arcs_by_chr[chrom] = mapped
        arcs_cnt[chrom] = len(mapped)

    return arcs_by_chr, arcs_cnt


def _find_anchor(anchors: List[Anchor], pos: int) -> int:
    """Binary-search for the anchor that contains pos. Returns -1 if not found."""
    lo, hi = 0, len(anchors) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a = anchors[mid]
        if pos < a.start:
            hi = mid - 1
        elif pos > a.end:
            lo = mid + 1
        else:
            return mid
    return -1


def _inter_key(c1: str, c2: str) -> str:
    a, b = sorted([c1, c2])
    return f"{a}:{b}"


def _flush_tmp(tmp: Dict[int, List[InteractionArc]],
               out: List[InteractionArc], start_anchor: int) -> None:
    """Merge arcs with same (start, end) but different factors; append to out."""
    for end_anchor, arcs in tmp.items():
        if len(arcs) == 1:
            out.append(arcs[0])
            continue

        arcs.sort(key=lambda a: a.factor)
        factors = [a.factor for a in arcs]
        multiple_factors = len(set(factors)) > 1

        total_score = 0
        factor_score = 0
        first_of_factor = 0
        for j in range(len(arcs) + 1):
            if j == len(arcs) or (j > 0 and arcs[j].factor != arcs[j - 1].factor):
                arcs[first_of_factor].score = factor_score
                arcs[first_of_factor].eff_score = 0 if multiple_factors else factor_score
                out.append(arcs[first_of_factor])
                first_of_factor = j
                total_score += factor_score
                factor_score = 0
            if j < len(arcs):
                factor_score += arcs[j].score

        if multiple_factors:
            summary = InteractionArc(start_anchor, end_anchor,
                                     arcs[0].genomic_start, arcs[0].genomic_end,
                                     0, total_score, -1)
            out.append(summary)
