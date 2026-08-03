"""
Mirrors Reference LooperSolver::createTreeChromosome(), findGaps(), findSplit().

The Reference code starts anchors at level=4, IBs at 3, segments at 2, chr root at 1.
We replicate this numbering so that setLevel()/levelDown() work identically.
"""

from __future__ import annotations

from dataclasses import field
from enum import IntEnum

import numpy as np

from gnome3d import log
from gnome3d.types import *

LOG = log.get("hierarchy")


class Level(IntEnum):
    """Cluster-tree depth, mirroring the Reference numbering (chr root at 1,
    anchors the deepest leaves at 4) so that setLevel()/levelDown() depth
    arithmetic - e.g. ``Level.SEGMENT - Level.CHROMOSOME`` steps - matches.
    IntEnum so members behave as ints in that arithmetic and in comparisons."""

    CHROMOSOME = 1
    SEGMENT = 2
    INTERACTION_BLOCK = 3
    ANCHOR = 4  # leaf level - original anchor clusters


@dataclass
class Cluster:
    start: int
    end: int
    level: int = Level.ANCHOR
    parent: ClusterIndex = -1

    # Indices into the global Solver.clusters list.
    children: list[ClusterIndex] = field(default_factory=empty_cluster_index_list)

    # Local (chr-relative) arc indices - positions in ArcMap[chr_].
    arcs: list[LocalArcIndex] = field(default_factory=empty_local_arc_index_list)

    orientation: str = "N"
    pos: F32Array = field(default_factory=zero_pos)
    is_fixed: bool = False
    dist_to_next: float = 0.0

    @property
    def genomic_pos(self) -> int:
        return (self.start + self.end) // 2

    def contains(self, pos: int) -> bool:
        return self.start <= pos <= self.end


def _other_end(
    arcs_chr: list[InteractionArc],
    arc_idx: LocalArcIndex,
    cluster_idx: ClusterIndex,
) -> ClusterIndex:
    """Return the other end of arc arc_idx from cluster_idx's perspective."""
    a = arcs_chr[arc_idx]
    if a.start == cluster_idx:
        return a.end
    if a.end == cluster_idx:
        return a.start
    return -1


def find_gaps(
    clusters: list[Cluster],
    chr_first: ClusterIndex,
    chr_arcs: list[InteractionArc],
) -> list[ClusterIndex]:
    """
    Find gap positions: anchor indices where no arc "covers" position i.
    Mirrors Reference LooperSolver::findGaps().

    Sweeps through anchors from chr_first to the end, tracking arc_count.
    arc_count += 1 when an arc starts at i (other_end > i)
    arc_count -= 1 when an arc ends at i (other_end < i)
    A gap is any position where arc_count == 0 after processing.

    Returns list of global cluster indices (gap positions).
    """
    gaps: list[ClusterIndex] = [chr_first]
    arc_count = 0

    n_clusters = len(clusters)
    for i in range(chr_first, n_clusters):
        if clusters[i].level != Level.ANCHOR:
            break
        for arc_idx in clusters[i].arcs:
            other = _other_end(chr_arcs, arc_idx, i)
            if other == i:
                continue
            if other > i:
                arc_count += 1
            else:
                arc_count -= 1

        if arc_count == 0:
            gaps.append(i)

    # Ensure last anchor is in gaps
    last = n_clusters - 1
    while last > chr_first and clusters[last].level != Level.ANCHOR:
        last -= 1
    if gaps[-1] != last:
        gaps.append(last)

    return gaps


def find_boundary_gaps(
    clusters: list[Cluster],
    chr_first: ClusterIndex,
    boundaries: list[int],
) -> list[ClusterIndex]:
    """Anchor indices whose following inter-anchor gap contains a boundary.

    The companion to `find_gaps`, which splits where ChIA-PET arc coverage falls
    to zero. That is a property of the assay's depth: where the library is shallow
    no arc spans, a block boundary appears, and the reconstruction then folds and
    places the two sides independently. These boundaries come from the contact map
    instead, so a block edge means the data shows insulation there.

    `boundaries` is sorted genomic positions for this chromosome. Returns global
    cluster indices, in the same form `find_gaps` uses, so the two can be merged.
    """
    out: list[ClusterIndex] = []
    if not boundaries:
        return out

    n_clusters = len(clusters)
    bp = np.asarray(boundaries, dtype=np.int64)
    for i in range(chr_first, n_clusters - 1):
        if clusters[i].level != Level.ANCHOR:
            break
        if clusters[i + 1].level != Level.ANCHOR:
            break
        lo, hi = clusters[i].end, clusters[i + 1].start
        if hi < lo:
            lo, hi = hi, lo
        j = int(np.searchsorted(bp, lo, side="left"))
        if j < bp.size and bp[j] <= hi:
            out.append(i)
    return out


def find_split_predefined(
    gaps: list[ClusterIndex],
    clusters: list[Cluster],
    breakpoints: list[int],
) -> list[ClusterIndex]:
    """
    Use predefined breakpoints to select which gaps are segment boundaries.
    Mirrors Reference LooperSolver::findSplit() (predefined branch).

    gaps:        list of gap indices (global cluster indices)
    clusters:    global cluster list
    breakpoints: list of breakpoint positions (sorted genomic coordinates) for this chromosome

    Returns subset of gap indices that are also segment boundaries.
    """
    splits: list[ClusterIndex] = [gaps[0]]

    bp_idx = 0
    n_bp = len(breakpoints)

    for i in range(1, len(gaps) - 1):
        if bp_idx >= n_bp:
            break

        gap_start = clusters[gaps[i]].end
        gap_end = clusters[gaps[i] + 1].start if gaps[i] + 1 < len(clusters) else gap_start

        # Advance breakpoint index past positions before the gap start
        while bp_idx < n_bp and breakpoints[bp_idx] < gap_start:
            bp_idx += 1

        if bp_idx < n_bp and gap_start <= breakpoints[bp_idx] <= gap_end:
            splits.append(gaps[i])
            bp_idx += 1

    if not splits or splits[-1] != gaps[-1]:
        splits.append(gaps[-1])

    return splits


def build_cluster_tree(
    anchors: AnchorMap,
    arcs: ArcMap,
    breakpoints: BreakpointMap,
    chrs: list[str],
    ib_splits: BreakpointMap | None = None,
    ib_split_source: str = "arcs",
) -> tuple[list[Cluster], ChrRootMap, ChrFirstClusterMap]:
    """
    Build the full cluster hierarchy for all chromosomes.
    Mirrors Reference LooperSolver::createTreeGenome() + createTreeChromosome().

    anchors:     dict[chr -> list[Anchor]]
    arcs:        dict[chr -> list[InteractionArc]] (anchor-index based, local per chr)
    breakpoints: dict[chr -> list[int]] of segment split positions
    chrs:        ordered list of chromosome names

    Returns:
        clusters:          list of Cluster objects (global, all chromosomes)
        chr_root:          dict[chr -> int] index of chromosome root cluster
        chr_first_cluster: dict[chr -> int] index of first anchor cluster per chr
    """

    clusters: list[Cluster] = []
    chr_root: ChrRootMap = {}
    chr_first_cluster: ChrFirstClusterMap = {}

    for chr_ in chrs:
        chr_first: ClusterIndex = len(clusters)
        chr_first_cluster[chr_] = chr_first

        chr_anchors = anchors.get(chr_, [])
        chr_arcs = arcs.get(chr_, [])

        if not chr_anchors:
            continue

        # --- level 4: create one cluster per anchor ---
        for a in chr_anchors:
            c = Cluster(start=a.start, end=a.end, level=Level.ANCHOR, orientation=a.orientation)
            clusters.append(c)

        # Shift arc indices from local (0..n_anchors) to global (chr_first..)
        # and register arcs on their anchor clusters
        for arc_i, arc in enumerate(chr_arcs):
            gs: ClusterIndex = arc.start + chr_first
            ge: ClusterIndex = arc.end + chr_first
            arc.start = gs
            arc.end = ge
            clusters[gs].arcs.append(arc_i)
            clusters[ge].arcs.append(arc_i)

        # --- find gaps and splits ---
        gaps = find_gaps(clusters, chr_first, chr_arcs)

        # Interaction blocks normally end where arc coverage does. `ib_split_source`
        # can add or substitute boundaries taken from the contact map, so a block
        # edge reflects measured insulation rather than the depth of the ChIA-PET
        # library. "arcs" is the reference behaviour.
        if str(ib_split_source).strip().lower() == "tads" and ib_splits:
            extra = find_boundary_gaps(clusters, chr_first, ib_splits.get(chr_, []))
            # Keep the first and last, which bound the chromosome rather than
            # marking a gap, or the block list loses its endpoints.
            keep = {gaps[0], gaps[-1]} if gaps else set()
            merged = sorted(keep | set(extra))
            gaps = merged if len(merged) > 1 else gaps

        chr_bp = breakpoints.get(chr_, [])
        if chr_bp:
            splits = find_split_predefined(gaps, clusters, chr_bp)
        else:
            splits = list(gaps)

        # --- level 3: create interaction block (IB) clusters ---
        next_split_idx = 1
        # Segment cluster indices accumulated for the chromosome root.
        root_children: list[ClusterIndex] = []
        # Index of the first IB cluster of the current segment.
        current_seg_ib_start: ClusterIndex = len(clusters)

        for i in range(1, len(gaps)):
            prev_gap: ClusterIndex = gaps[i - 1] if i == 1 else gaps[i - 1] + 1
            curr_gap: ClusterIndex = gaps[i]

            start_pos = clusters[prev_gap].start
            end_pos = clusters[curr_gap].end

            ib = Cluster(start=start_pos, end=end_pos, level=Level.INTERACTION_BLOCK)
            ib_idx: ClusterIndex = len(clusters)

            # Set anchors as children of IB
            for k in range(prev_gap, curr_gap + 1):
                ib.children.append(k)
                clusters[k].parent = ib_idx

            clusters.append(ib)

            # Check if this gap is a segment split
            if gaps[i] == splits[next_split_idx]:
                seg_end_ib_idx: ClusterIndex = len(clusters) - 1

                seg_start_pos = clusters[current_seg_ib_start].start
                seg_end_pos = clusters[seg_end_ib_idx].end

                seg = Cluster(start=seg_start_pos, end=seg_end_pos, level=Level.SEGMENT)
                seg_idx: ClusterIndex = len(clusters)

                for k in range(current_seg_ib_start, seg_end_ib_idx + 1):
                    seg.children.append(k)
                    clusters[k].parent = seg_idx

                root_children.append(seg_idx)
                clusters.append(seg)

                current_seg_ib_start = len(clusters)
                next_split_idx = min(next_split_idx + 1, len(splits) - 1)

        # --- level 1: chromosome root ---
        if root_children:
            root_start = clusters[root_children[0]].start
            root_end = clusters[root_children[-1]].end
            root = Cluster(start=root_start, end=root_end, level=Level.CHROMOSOME)
            root_idx: ClusterIndex = len(clusters)
            for k in root_children:
                root.children.append(k)
                clusters[k].parent = root_idx
            clusters.append(root)
            chr_root[chr_] = root_idx
        else:
            LOG.warning("no root children for %s", chr_)

    return clusters, chr_root, chr_first_cluster


# Level traversal helpers


def set_top_level(chr_root: ChrRootMap, chrs: list[str]) -> ChrLevel:
    """Returns current_level = {chr: [chr_root[chr]]} for each chr."""
    return {chr_: [chr_root[chr_]] for chr_ in chrs if chr_ in chr_root}


def level_down(
    current_level: ChrLevel,
    clusters: list[Cluster],
    chrs: list[str],
) -> ChrLevel:
    """
    Move one level deeper in the hierarchy.
    Mirrors Reference LooperSolver::levelDown().
    """
    new_level: ChrLevel = {}
    for chr_ in chrs:
        tmp: list[ClusterIndex] = []
        for idx in current_level.get(chr_, []):
            if not clusters[idx].children:
                tmp.append(idx)
            else:
                tmp.extend(clusters[idx].children)
        new_level[chr_] = tmp
    return new_level


def set_level(
    level: int,
    chr_root: ChrRootMap,
    clusters: list[Cluster],
    chrs: list[str],
) -> ChrLevel:
    """
    Set current_level to correspond to the given level number.
    Mirrors Reference LooperSolver::setLevel(level) which calls setTopLevel() then
    calls levelDown() `level` times.
    """
    current = set_top_level(chr_root, chrs)
    for _ in range(level):
        current = level_down(current, clusters, chrs)
    return current
