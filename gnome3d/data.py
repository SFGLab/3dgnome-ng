"""
All data the solver needs lives here. Two factory methods cover the two
supported input paths:

ContactData.from_files(settings, chrs, region)
    Reads anchors, arcs, breakpoints, and singletons from the files
    referenced by `settings`.

ContactData.from_dataframes(anchors_df, arcs_df, ...)
    Converts pandas DataFrames into the same internal representation.
    Expected columns:
        anchors_df:       chr, start, end[, orientation]
        arcs_df:          chr_a, start_a, end_a, chr_b, start_b, end_b, score
        breakpoints_df:   chr, pos
        singletons_df:    chr1, pos1, chr2, pos2, score
        compartments_df:  chr, start, end[, label][, value]
        accessibility_df: chr, start, end, value
        phasing_df:       chr, start, end, value

Once constructed, a ContactData instance is file-independent and can be
passed directly to Solver.load().
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

from gnome3d import log
from gnome3d.io import (
    compartment_from_label,
    filter_singletons,
    load_anchors,
    load_arcs,
    load_breakpoints,
    load_compartments,
    load_signal,
    load_singletons,
)
from gnome3d.polymer import (
    ArcStrengthFit,
    ContactFit,
    fit_arc_strength,
    fit_contact_exponent,
)
from gnome3d.tracks import normalize_signal_map, phase_compartments
from gnome3d.types import *

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("data")


@dataclass
class ContactData:
    """
    anchors:     dict[chr -> list[Anchor]] - anchor beads (after empty-anchor removal)
    arcs:        dict[chr -> list[InteractionArc]] - mapped arcs (after mark_arcs)
    breakpoints: dict[chr -> list[int]] - segment boundary positions
    singletons:  list of (chr1, pos1, chr2, pos2, score) contacts
                 used to build the segment-level heatmap
    compartments: dict[chr -> list[CompartmentInterval]] - A/B calls, already phased
    accessibility: dict[chr -> list[SignalInterval]] - ATAC-seq, rescaled to [0, 1]
    """

    anchors: AnchorMap = field(default_factory=empty_anchor_map)
    arcs: ArcMap = field(default_factory=empty_arc_map)
    breakpoints: BreakpointMap = field(default_factory=empty_breakpoint_map)
    singletons: list[SingletonContact] = field(default_factory=empty_singleton_list)
    # The polymer law's own fits, made on the whole loaded input before any region filter so a
    # small region can still supply an exponent. None until the law is on.
    contact_fit: ContactFit | None = None
    arc_fit: ArcStrengthFit | None = None
    # Long-range arcs (gap > max_pet_length): not anchor-mapped, folded into the
    # segment heatmap by Solver. Mirrors Reference InteractionArcs::long_arcs.
    long_arcs: RawArcMap = field(default_factory=empty_raw_arc_map)
    # Epigenomic tracks driving the opt-in compartment and accessibility energy
    # terms.  Empty when no track is configured, which leaves those terms inert.
    compartments: CompartmentMap = field(default_factory=empty_compartment_map)
    accessibility: SignalMap = field(default_factory=empty_signal_map)

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

        LOG.info("load anchors")
        anchors = load_anchors(s.data_path(s.data_anchors), chr_set, region)

        LOG.info("load arcs")
        raw_arcs, long_arcs = load_arcs(
            s.data_path(s.data_pet_clusters), chr_set, region, s.max_pet_length
        )

        with log.step(LOG, "mark arcs"):
            arcs = mark_arcs(anchors, raw_arcs)

        with log.step(LOG, "remove empty anchors"):
            anchors = remove_empty_anchors(anchors, arcs)

        LOG.info("load breakpoints")
        breakpoints = load_breakpoints(s.data_path(s.data_segment_split), chrs)

        LOG.info("load singletons")
        contact_fit: ContactFit | None = None
        arc_fit: ArcStrengthFit | None = None
        if s.use_polymer_law:
            # Fit on the whole chromosome set, then cut to the region. A region alone is too
            # narrow a band to read a decay off.
            singletons = load_singletons(s.data_path(s.data_singletons), chr_set, None)
            contact_fit = fit_contact_exponent(singletons)
            arc_fit = fit_arc_strength([a for al in arcs.values() for a in al])
            singletons = filter_singletons(singletons, region)
        else:
            singletons = load_singletons(s.data_path(s.data_singletons), chr_set, region)

        # Optional second file for inter-chromosomal singletons (matches the
        # Reference `data_singletons_inter` config key).  Only meaningful for
        # multi-chromosome runs - inter-chr contacts feed the chr-level MC.
        if s.data_singletons_inter and len(chrs) > 1:
            inter_path = s.data_path(s.data_singletons_inter)
            LOG.info("load inter-chr singletons")
            inter = load_singletons(inter_path, chr_set, region)
            singletons.extend(inter)

        compartments: CompartmentMap = {}
        if s.data_compartments:
            LOG.info("load compartments")
            compartments = load_compartments(s.data_path(s.data_compartments), chr_set, region)

        accessibility: SignalMap = {}
        if s.data_accessibility:
            LOG.info("load accessibility")
            accessibility = load_signal(s.data_path(s.data_accessibility), chr_set, region)

        phasing: SignalMap = {}
        if s.data_phasing_track:
            LOG.info("load phasing track")
            phasing = load_signal(s.data_path(s.data_phasing_track), chr_set, region)

        compartments, accessibility = _finalize_tracks(
            compartments,
            accessibility,
            phasing,
            anchors,
            mode=s.accessibility_mode,
            percentile=s.accessibility_percentile,
        )

        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            contact_fit=contact_fit,
            arc_fit=arc_fit,
            long_arcs=long_arcs,
            compartments=compartments,
            accessibility=accessibility,
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
        compartments_df: Any | None = None,
        accessibility_df: Any | None = None,
        phasing_df: Any | None = None,
        accessibility_mode: str = "log",
        accessibility_percentile: float = 80.0,
        polymer_law: bool = False,
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
        compartments_df : DataFrame or None
            Columns: chr, start, end[, label][, value].  `label` wins when both
            are present.  A value-only frame is phased the same way a file is.
        accessibility_df : DataFrame or None
            Columns: chr, start, end, value
        phasing_df : DataFrame or None
            Columns: chr, start, end, value.  Used to orient a value-only
        accessibility_mode : str
            `log` or `binary`, how the accessibility signal is normalised.
        accessibility_percentile : float
            Under `binary`, the percentile at or above which a bead is open.
        polymer_law : bool
            Fit the contact exponent and the loop strength on the frames, as `from_files`
            does when `use_polymer_law` is on.
            compartment frame when no accessibility frame is given.
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
        contact_fit: ContactFit | None = None
        arc_fit: ArcStrengthFit | None = None
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

        compartments = _compartments_from_df(compartments_df, chr_set, region)
        accessibility = _signal_from_df(accessibility_df, chr_set, region)
        phasing = _signal_from_df(phasing_df, chr_set, region)
        compartments, accessibility = _finalize_tracks(
            compartments,
            accessibility,
            phasing,
            anchors,
            mode=accessibility_mode,
            percentile=accessibility_percentile,
        )

        if polymer_law:
            contact_fit = fit_contact_exponent(singletons)
            arc_fit = fit_arc_strength([a for al in arcs.values() for a in al])
        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            contact_fit=contact_fit,
            arc_fit=arc_fit,
            long_arcs=long_arcs,
            compartments=compartments,
            accessibility=accessibility,
        )


# Epigenomic track assembly, shared by both factories


def _finalize_tracks(
    compartments: CompartmentMap,
    accessibility: SignalMap,
    phasing: SignalMap,
    anchors: AnchorMap,
    mode: str = "log",
    percentile: float = 80.0,
) -> tuple[CompartmentMap, SignalMap]:
    """
    Phase the compartment track and rescale accessibility onto [0, 1].

    Phasing uses accessibility ahead of a dedicated phasing track, because A is
    the open compartment by definition, so that correlation is the one the model
    actually means.  It runs on the raw signal, before rescaling, since a
    monotone rescale cannot change the sign of the correlation.

    Both factories go through here so the file and frame paths cannot pick
    different rules.
    """
    if compartments:
        signal = accessibility if accessibility else phasing
        compartments = phase_compartments(compartments, signal or None, anchors)
    if accessibility:
        accessibility = normalize_signal_map(accessibility, mode=mode, percentile=percentile)
    return compartments, accessibility


def _compartments_from_df(df: Any, chr_set: set[str], region: BedRegion | None) -> CompartmentMap:
    """Build a CompartmentMap from a DataFrame with chr/start/end[/label][/value]."""
    out: CompartmentMap = {}
    if df is None:
        return out
    for _, row in df.iterrows():
        c = str(row["chr"])
        if c not in chr_set:
            continue
        st, en = int(row["start"]), int(row["end"])
        if region is not None and (st > region.end or en < region.start):
            continue
        if "label" in row.index:
            cls = compartment_from_label(str(row["label"]))
            score = 0.0
        else:
            cls = int(Compartment.NONE)
            score = float(row["value"])
            if score != score:
                continue
        out.setdefault(c, []).append(CompartmentInterval(c, st, en, cls, score))

    for lst in out.values():
        lst.sort(key=lambda iv: iv.start)
    return out


def _signal_from_df(df: Any, chr_set: set[str], region: BedRegion | None) -> SignalMap:
    """Build a SignalMap from a DataFrame with chr/start/end/value."""
    out: SignalMap = {}
    if df is None:
        return out
    for _, row in df.iterrows():
        c = str(row["chr"])
        if c not in chr_set:
            continue
        st, en = int(row["start"]), int(row["end"])
        if region is not None and (st > region.end or en < region.start):
            continue
        value = float(row["value"])
        if value != value:
            continue
        out.setdefault(c, []).append(SignalInterval(c, st, en, value))

    for lst in out.values():
        lst.sort(key=lambda iv: iv.start)
    return out


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
        tmp_arcs: dict[
            int, list[InteractionArc]
        ] = {}  # end_idx -> staged arcs for current start group
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
                        if j == len(arcs_group) or (
                            j > 0 and arcs_group[j].factor != arcs_group[j - 1].factor
                        ):
                            arcs_group[first_of_factor].score = factor_score
                            arcs_group[first_of_factor].eff_score = (
                                0 if multiple_factors else factor_score
                            )
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
        LOG.info("marked arcs %s: %d", chr_, len(result))

    return arcs


# keep only anchors that are endpoints of at least one arc


def remove_empty_anchors(
    anchors: AnchorMap,
    arcs: ArcMap,
) -> AnchorMap:
    """
    Remove anchors that are not endpoints of any arc.
    Mirrors Reference InteractionArcs::removeEmptyAnchors().

    Returns new anchors dict.
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
        LOG.info("removed empty anchors %s: %d", chr_, removed)

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
