"""Parse ChIA-PET input files (anchors BED, PET-clusters BEDPE, singletons
BEDPE, optional segment-split breakpoints BED).

Mirrors cudaMMC `Anchor.cpp`, `InteractionArcs.cpp` and the BED reader paths in
`LooperSolver.cpp:41-44` (segments_predefined).  See AGENTS.md "Data flow per
chromosome".
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data_structures import Anchor, InteractionArc, RawArc


# ── BED / BEDPE readers ──────────────────────────────────────────────────────

def load_anchors(bed_path: str) -> Dict[str, List[Anchor]]:
    """Return {chrom: [Anchor, ...]} sorted by start position.

    cudaMMC `Anchor.cpp` parses (chrom, start, end, name, score, strand).
    Strand '+' → 'R', '-' → 'L', anything else → 'N' (initial-hint label;
    actual orientation vector is recomputed geometrically in MC — see
    `ChromosomeTree.calc_orientation`).
    """
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


def _parse_bedpe_line(parts: List[str], factor: int,
                      score_col: int = 7) -> Optional[RawArc]:
    """Parse one BEDPE row.

    cudaMMC `Cluster.cpp` expects the PET count in a fixed column.  Default
    here is column index 7 (0-based) which matches the cudaMMC layout (chrom1,
    s1, e1, chrom2, s2, e2, name, score, ...).  Override via
    `Settings.bedpe_score_column` if your BEDPE puts the count elsewhere.
    """
    if len(parts) < 6:
        return None
    chrom1, s1, e1 = parts[0], int(parts[1]), int(parts[2])
    chrom2, s2, e2 = parts[3], int(parts[4]), int(parts[5])
    score = 1
    if len(parts) > score_col:
        try:
            score = int(parts[score_col])
        except ValueError:
            try:
                score = int(float(parts[score_col]))
            except ValueError:
                pass
    return RawArc(chrom1, s1, e1, chrom2, s2, e2, score, factor)


def load_pet_clusters(bedpe_path: str, factor: int = 0,
                       score_col: int = 7) -> List[RawArc]:
    """Load PET-cluster BEDPE file. cudaMMC `InteractionArcs.cpp` equivalent."""
    arcs: List[RawArc] = []
    with open(bedpe_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            arc = _parse_bedpe_line(line.split("\t"), factor, score_col)
            if arc is not None:
                arcs.append(arc)
    return arcs


def load_singletons(bedpe_path: str, factor: int = 0,
                     score_col: int = 7) -> List[RawArc]:
    """Singletons have score=1 by convention."""
    return load_pet_clusters(bedpe_path, factor, score_col)


# ── Segment-split breakpoints BED reader ──────────────────────────────────
# cudaMMC `LooperSolver.cpp:41-44`:
#   if (Settings::dataSegmentsSplit.size() > 0)
#     segments_predefined.fromFile(Settings::dataSegmentsSplit);

def load_segments_split(bed_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Return {chrom: [(start, end), ...]} for a predefined-segments BED.

    Used by `find_segments` (Branch A — see cudaMMC `LooperSolver.cpp:911-962`)
    to promote arc-sweep gaps whose anchor coordinate is contained in any
    predefined region into segment boundaries.
    """
    if not bed_path or not os.path.exists(bed_path):
        return {}
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            out[parts[0]].append((int(parts[1]), int(parts[2])))
    for chrom in out:
        out[chrom].sort()
    return dict(out)


# ── Arc → anchor mapping (CPU equivalent of ParallelMarkArcs.cu) ─────────────

def _anchor_contains(anchor: Anchor, pos: int) -> bool:
    return anchor.start <= pos <= anchor.end


def mark_arcs(raw_arcs: List[RawArc],
              anchors_by_chr: Dict[str, List[Anchor]],
              ignore_missing: bool = False
              ) -> Tuple[Dict[str, List[InteractionArc]], Dict[str, int]]:
    """Map raw arcs (genomic positions) to anchor indices.

    cudaMMC `InteractionArcs.cpp:60-141` — for each chromosome iterate raw arcs
    in start order, find the anchor containing each endpoint, deduplicate by
    factor, emit a summary arc when multiple factors collide.

    Arcs whose endpoint is a 1-bp anchor (`length() <= 1`, cudaMMC
    `InteractionArcs.cpp:65`) are silently skipped to match the upstream filter
    — see AUDIT §I1.

    Returns:
        arcs_by_chr  : {chrom: [InteractionArc, ...]}
        arcs_cnt     : {chrom: count}
    """
    raw_by_chr: Dict[str, List[RawArc]] = defaultdict(list)
    for arc in raw_arcs:
        if arc.chrom1 == arc.chrom2:
            raw_by_chr[arc.chrom1].append(arc)
        else:
            # cudaMMC keeps inter-chrom arcs in a separate stream
            # (dataSingletonsInter etc.); see AUDIT §I5.  We aggregate under a
            # synthetic key but downstream `solver.py` only reads per-chrom
            # keys, so these arcs are effectively dropped.  Made explicit.
            key = _inter_key(arc.chrom1, arc.chrom2)
            raw_by_chr[key].append(arc)

    arcs_by_chr: Dict[str, List[InteractionArc]] = {}
    arcs_cnt: Dict[str, int] = {}

    for chrom, chr_anchors in anchors_by_chr.items():
        raw = raw_by_chr.get(chrom, [])
        mapped: List[InteractionArc] = []
        tmp: Dict[int, List[InteractionArc]] = defaultdict(list)
        last_start = -1

        raw_sorted = sorted(raw, key=lambda r: r.start1)

        for raw_arc in raw_sorted:
            ai = _find_anchor(chr_anchors, raw_arc.start1)
            bi = _find_anchor(chr_anchors, raw_arc.start2)
            if ai == -1 or bi == -1:
                if not ignore_missing:
                    # cudaMMC prints `! error: non-matching arc` and continues
                    # (`InteractionArcs.cpp:73-77`); we silently skip.
                    pass
                continue
            if ai == bi:
                continue
            if ai > bi:
                ai, bi = bi, ai
            ia = InteractionArc(ai, bi,
                                raw_arc.start1, raw_arc.start2,
                                raw_arc.score, raw_arc.score, raw_arc.factor)
            tmp[bi].append(ia)

            if ai != last_start and last_start != -1:
                _flush_tmp(tmp, mapped, last_start)
                tmp.clear()
            last_start = ai

        _flush_tmp(tmp, mapped, last_start)

        arcs_by_chr[chrom] = mapped
        arcs_cnt[chrom] = len(mapped)

    return arcs_by_chr, arcs_cnt


def _find_anchor(anchors: List[Anchor], pos: int) -> int:
    """Binary-search for the anchor that contains `pos`.

    cudaMMC `InteractionArcs.cpp:64-71` does an O(A·N) linear scan with the
    additional guard `if (anchors[chr][j].length() > 1)` at line 65 — single-
    base anchors are never matched.  We binary-search but apply the same
    `length() > 1` guard (AUDIT §I1).

    Returns the index of the matching anchor, or -1.
    """
    lo, hi = 0, len(anchors) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a = anchors[mid]
        if pos < a.start:
            hi = mid - 1
        elif pos > a.end:
            lo = mid + 1
        else:
            # cudaMMC InteractionArcs.cpp:65: only match anchors with length > 1
            if a.length > 1:
                return mid
            return -1
    return -1


def _inter_key(c1: str, c2: str) -> str:
    a, b = sorted([c1, c2])
    return f"{a}:{b}"


def _flush_tmp(tmp: Dict[int, List[InteractionArc]],
               out: List[InteractionArc], start_anchor: int) -> None:
    """Merge arcs with same (start, end) but different factors; append to out.

    cudaMMC `InteractionArcs.cpp:88-141` — sort by factor; for each
    factor-group keep the first arc with `score = factor_score`,
    `eff_score = 0 if multiple_factors else factor_score`; if multiple factors
    exist append a trailing summary arc `score=0, eff_score=total`.
    """
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
