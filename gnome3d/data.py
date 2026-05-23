"""
All data the solver needs lives here. Two factory methods cover the two
supported input paths:

ContactData.from_files(settings, chrs, region)
    Reads anchors, arcs, breakpoints, and singletons from the files
    referenced by `settings`.

ContactData.from_dataframes(anchors_df, arcs_df, ...)
    Converts pandas DataFrames into the same internal representation.
    Expected columns:
        anchors_df:     chr, start, end[, orientation]
        arcs_df:        chr_a, start_a, end_a, chr_b, start_b, end_b, score
        breakpoints_df: chr, pos
        singletons_df:  chr1, pos1, chr2, pos2, score

Once constructed, a ContactData instance is file-independent and can be
passed directly to Solver.load().
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

from .io import load_anchors, load_arcs, load_breakpoints, load_singletons
from .types import *

if TYPE_CHECKING:
    from .settings import Settings

@dataclass
class ContactData:
    """
    anchors:     dict[chr -> list[Anchor]] - anchor beads (after empty-anchor removal)
    arcs:        dict[chr -> list[InteractionArc]] - mapped arcs (after mark_arcs)
    breakpoints: dict[chr -> list[int]] - segment boundary positions
    singletons:  list of (chr1, pos1, chr2, pos2, score) contacts
                 used to build the segment-level heatmap
    """
    anchors: AnchorMap = field(default_factory=empty_anchor_map)
    arcs: ArcMap = field(default_factory=empty_arc_map)
    breakpoints: BreakpointMap = field(default_factory=empty_breakpoint_map)
    singletons: list[SingletonContact] = field(default_factory=empty_singleton_list)
    # Long-range arcs (gap > max_pet_length): not anchor-mapped, folded into the
    # segment heatmap by Solver. Mirrors Reference InteractionArcs::long_arcs.
    long_arcs: RawArcMap = field(default_factory=empty_raw_arc_map)

    @classmethod
    def from_files(
        cls,
        settings: Settings,
        chrs: list[str],
        region: BedRegion | None = None,
    ) -> ContactData:
        """
        Load all engine inputs from the files named in `settings`.

        Parameters
        ----------
        settings : Settings
            Parsed config.  Must have data_dir and file name attributes set.
        chrs : list[str]
            Chromosome names to load (e.g. ['chr1']).
        region : BedRegion or None
            Genomic window to restrict to.  None = whole chromosome(s).
        """
        chr_set = set(chrs)
        s = settings

        print("[data] load anchors")
        anchors = load_anchors(s.data_path(s.data_anchors), chr_set, region)

        print("[data] load arcs")
        raw_arcs, long_arcs = load_arcs(
            s.data_path(s.data_pet_clusters), chr_set, region, s.max_pet_length
        )

        print("[data] mark arcs")
        arcs = mark_arcs(anchors, raw_arcs)

        print("[data] remove empty anchors")
        anchors = remove_empty_anchors(anchors, arcs)

        print("[data] load breakpoints")
        breakpoints = load_breakpoints(s.data_path(s.data_segment_split), chrs)

        print("[data] load singletons")
        singletons = load_singletons(s.data_path(s.data_singletons), chr_set, region)

        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            long_arcs=long_arcs,
        )

    @classmethod
    def from_dataframes(
        cls,
        anchors_df: Any,
        arcs_df: Any,
        breakpoints_df: Any | None = None,
        singletons_df: Any | None = None,
        chrs: list[str] | None = None,
        region: BedRegion | None = None,
        max_pet_length: int = 1_000_000,
    ) -> ContactData:
        """
        Build ContactData from pandas DataFrames.

        Parameters
        ----------
        anchors_df : DataFrame
            Columns: chr, start, end[, orientation]
        arcs_df : DataFrame
            Columns: chr_a, start_a, end_a, chr_b, start_b, end_b, score
        breakpoints_df : DataFrame or None
            Columns: chr, pos
        singletons_df : DataFrame or None
            Columns: chr1, pos1, chr2, pos2, score
        chrs : list[str] or None
            Restrict to these chromosomes.  None = all chromosomes in anchors_df.
        region : BedRegion or None
            Genomic window filter.
        max_pet_length : int
            Arcs longer than this are discarded.
        """
        chr_set: set[str] = (
            set(chrs) if chrs is not None else {str(c) for c in anchors_df["chr"].unique()}
        )

        # anchors
        anchors: AnchorMap = {}
        for _, row in anchors_df.iterrows():
            c = str(row["chr"])
            if c not in chr_set:
                continue
            st, en = int(row["start"]), int(row["end"])
            if region is not None and not (region.contains(st) or region.contains(en)):
                continue
            ori = str(row["orientation"]) if "orientation" in row.index else "N"
            anchors.setdefault(c, []).append(Anchor(c, st, en, ori))

        # raw arcs -> mark -> remove empty
        raw_arcs: RawArcMap = {}
        long_arcs: RawArcMap = {}
        for _, row in arcs_df.iterrows():
            ca, cb = str(row["chr_a"]), str(row["chr_b"])
            if ca != cb or ca not in chr_set:
                continue
            posa = (int(row["start_a"]) + int(row["end_a"])) // 2
            posb = (int(row["start_b"]) + int(row["end_b"])) // 2
            if posa > posb:
                posa, posb = posb, posa
            if region is not None and not (region.contains(posa) and region.contains(posb)):
                continue
            arc = RawArc(posa, posb, float(row["score"]))
            if posb - posa > max_pet_length:
                long_arcs.setdefault(ca, []).append(arc)
                continue
            lst = raw_arcs.setdefault(ca, [])
            p = len(lst)
            while p > 0 and lst[p - 1].start > arc.start:
                p -= 1
            lst.insert(p, arc)

        arcs = mark_arcs(anchors, raw_arcs)
        anchors = remove_empty_anchors(anchors, arcs)

        # breakpoints
        breakpoints: BreakpointMap = {}
        if breakpoints_df is not None:
            for _, row in breakpoints_df.iterrows():
                c = str(row["chr"])
                if c not in chr_set:
                    continue
                breakpoints.setdefault(c, []).append(int(row["pos"]))

        # singletons
        singletons: list[SingletonContact] = []
        if singletons_df is not None:
            for _, row in singletons_df.iterrows():
                c1, c2 = str(row["chr1"]), str(row["chr2"])
                if c1 not in chr_set or c2 not in chr_set:
                    continue
                p1, p2 = int(row["pos1"]), int(row["pos2"])
                sc = int(row["score"])
                if region is not None and not (region.contains(p1) and region.contains(p2)):
                    continue
                singletons.append((c1, p1, c2, p2, sc))

        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            long_arcs=long_arcs,
        )


# map RawArcs -> anchor-indexed InteractionArcs

def mark_arcs(
    anchors: AnchorMap,
    raw_arcs: RawArcMap,
) -> ArcMap:
    """
    Map genomic-position arcs to anchor-index arcs.
    Mirrors Reference InteractionArcs::markArcs().

    anchors:  dict[chr -> list[Anchor]]
    raw_arcs: dict[chr -> list[RawArc]] (sorted by start)

    Returns dict[chr -> list[InteractionArc]].
    """
    import bisect

    arcs: ArcMap = {}

    for chr_ in anchors:
        chr_anchors = anchors[chr_]
        chr_raw = raw_arcs.get(chr_, [])

        # Binary-search index: anchor start positions (anchors assumed sorted by start).
        # For a query pos, bisect gives the last anchor with start <= pos; we then
        # verify pos <= anchor.end.  Anchors in ChIA-PET data don't overlap, so at
        # most one candidate needs checking.
        anc_starts = [a.start for a in chr_anchors]

        def find_anchor(pos: int) -> int:
            i = bisect.bisect_right(anc_starts, pos) - 1
            while i >= 0:
                a = chr_anchors[i]
                if a.length() > 1 and a.start <= pos <= a.end:
                    return i
                i -= 1
            return -1

        result: list[InteractionArc] = []
        tmp_arcs: dict[int, list[InteractionArc]] = {}  # end_idx -> staged arcs for current start group
        last_start = -1

        def flush(target_list: list[InteractionArc]) -> None:
            for end_idx, arcs_group in sorted(tmp_arcs.items()):
                if len(arcs_group) == 1:
                    target_list.append(arcs_group[0])
                else:
                    arcs_group.sort(key=lambda a: a.factor)
                    multiple_factors = any(
                        arcs_group[j].factor != arcs_group[j - 1].factor
                        for j in range(1, len(arcs_group))
                    )
                    total_score = 0
                    factor_score = 0
                    first_of_factor = 0
                    for j in range(len(arcs_group) + 1):
                        if j == len(arcs_group) or (j > 0 and arcs_group[j].factor != arcs_group[j - 1].factor):
                            arcs_group[first_of_factor].score = factor_score
                            arcs_group[first_of_factor].eff_score = 0 if multiple_factors else factor_score
                            target_list.append(arcs_group[first_of_factor])
                            first_of_factor = j
                            total_score += factor_score
                            factor_score = 0
                        if j < len(arcs_group):
                            factor_score += arcs_group[j].score
                    if multiple_factors:
                        summary = InteractionArc(
                            start=arcs_group[0].start,
                            end=end_idx,
                            score=0,
                            eff_score=total_score,
                            factor=-1,
                        )
                        target_list.append(summary)
            tmp_arcs.clear()

        for raw in chr_raw:
            st = find_anchor(raw.start)
            end = find_anchor(raw.end)
            if st == -1 or end == -1 or st == end:
                continue
            if st != last_start:
                flush(result)
                last_start = st
            arc = InteractionArc(
                start=st,
                end=end,
                score=int(raw.score),
                eff_score=0,
                factor=0,
                genomic_start=raw.start,
                genomic_end=raw.end,
            )
            tmp_arcs.setdefault(end, []).append(arc)
        flush(result)

        arcs[chr_] = result
        print(f"  marked arcs {chr_}: {len(result)}")

    return arcs


# keep only anchors that are endpoints of at least one arc

def remove_empty_anchors(
    anchors: AnchorMap,
    arcs: ArcMap,
) -> AnchorMap:
    """
    Remove anchors that are not endpoints of any arc.
    Mirrors Reference InteractionArcs::removeEmptyAnchors().

    Returns new anchors dict (original is not modified).
    Also updates arc start/end indices to reflect removed anchors.
    """
    new_anchors: AnchorMap = {}
    index_maps: dict[str, dict[int, int]] = {}

    for chr_ in anchors:
        chr_anchors = anchors[chr_]
        chr_arcs = arcs.get(chr_, [])
        n = len(chr_anchors)

        # Mark which anchors are used
        used = [False] * n
        for arc in chr_arcs:
            if 0 <= arc.start < n:
                used[arc.start] = True
            if 0 <= arc.end < n:
                used[arc.end] = True

        # Build new list and index map
        new_list: list[Anchor] = []
        idx_map: dict[int, int] = {}
        for i, anchor in enumerate(chr_anchors):
            if used[i]:
                idx_map[i] = len(new_list)
                new_list.append(anchor)

        removed = n - len(new_list)
        print(f"  removed empty anchors {chr_}: {removed}")

        new_anchors[chr_] = new_list
        index_maps[chr_] = idx_map

    # Remap arc indices
    for chr_ in arcs:
        idx_map = index_maps.get(chr_, {})
        valid_arcs: list[InteractionArc] = []
        for arc in arcs[chr_]:
            ns = idx_map.get(arc.start, -1)
            ne = idx_map.get(arc.end, -1)
            if ns >= 0 and ne >= 0:
                arc.start = ns
                arc.end = ne
                valid_arcs.append(arc)
        arcs[chr_] = valid_arcs

    return new_anchors
