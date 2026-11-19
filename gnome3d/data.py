"""The input data of a run.

`ContactData.from_files` reads the files of the settings. `ContactData.from_dataframes` takes
pandas DataFrames with these columns.

    anchors_df       chr, start, end[, orientation]
    arcs_df          chr_a, start_a, end_a, chr_b, start_b, end_b, score
    breakpoints_df   chr, pos
    singletons_df    chr1, pos1, chr2, pos2, score
    compartments_df  chr, start, end[, label][, value]
    phasing_df       chr, start, end, value
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

from gnome3d import anchor_contacts as ac
from gnome3d import log
from gnome3d.io import (
    compartment_from_label,
    filter_singletons,
    load_activity,
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
from gnome3d.tracks import phase_compartments
from gnome3d.types import *

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("data")


def _empty_anchor_contacts() -> ac.AnchorContactMap:
    return {}


def _load_anchor_contacts(path: str, anchors: AnchorMap) -> ac.AnchorContactMap:
    """Loads the anchor contact tables of the run's chromosomes and logs how many of the run's
    anchors they hold. An anchor a table lacks gets no contacts.
    """
    tables = ac.load(path, set(anchors))
    found = total = 0
    for chrom, lst in anchors.items():
        total += len(lst)
        t = tables.get(chrom)
        if t is None or not lst:
            continue
        st = np.array([a.start for a in lst], dtype=np.int64)
        en = np.array([a.end for a in lst], dtype=np.int64)
        found += int((ac.match(t, st, en) >= 0).sum())
    log.status(
        LOG,
        "anchor contacts: %s of %s run anchors found in %s, %d chromosomes",
        f"{found:,}",
        f"{total:,}",
        path,
        len(tables),
    )
    return tables


@dataclass
class ContactData:
    """The input data of a run.

    `anchors` are the anchors that at least one arc ends on. `arcs` are indexed by anchor.
    `breakpoints` are segment boundary positions. `singletons` build the segment heatmap.
    `compartments` are phased.
    """

    anchors: AnchorMap = field(default_factory=empty_anchor_map)
    arcs: ArcMap = field(default_factory=empty_arc_map)
    breakpoints: BreakpointMap = field(default_factory=empty_breakpoint_map)
    singletons: list[SingletonContact] = field(default_factory=empty_singleton_list)
    # Fitted on the whole loaded input, before the region filter.
    contact_fit: ContactFit | None = None
    arc_fit: ArcStrengthFit | None = None
    # One fit per factor, by the index of the cluster file. `arc_fit` is the fit of factor 0.
    arc_fits: dict[int, ArcStrengthFit] = field(default_factory=dict)
    # Arcs longer than `max_pet_length`. They are not mapped to anchors and go into the segment
    # heatmap.
    long_arcs: RawArcMap = field(default_factory=empty_raw_arc_map)
    compartments: CompartmentMap = field(default_factory=empty_compartment_map)
    activity: SignalMap = field(default_factory=empty_signal_map)
    anchor_contacts: ac.AnchorContactMap = field(default_factory=_empty_anchor_contacts)

    @classmethod
    def from_files(
        cls,
        settings: Settings,
        chrs: list[str],
        region: BedRegion | None = None,
    ) -> ContactData:
        """Loads every input from the files of `settings`.

        Parameters
        ----------
        settings
            The parsed config, with `data_dir` and the file names set.
        chrs
            Chromosomes to load.
        region
            Genomic window. None loads whole chromosomes.
        """
        chr_set = set(chrs)
        s = settings

        LOG.info("load anchors")
        anchors = load_anchors(s.data_path(s.data_anchors), chr_set, region)

        LOG.info("load arcs")
        raw_arcs: RawArcMap = {}
        long_arcs: RawArcMap = {}
        for path, factor, name in s.cluster_files():
            raw_f, long_f = load_arcs(path, chr_set, region, s.max_pet_length, factor)
            LOG.info("arcs of factor %d (%s): %d chromosomes", factor, name, len(raw_f))
            for c, lst in raw_f.items():
                raw_arcs.setdefault(c, []).extend(lst)
            for c, lst in long_f.items():
                long_arcs.setdefault(c, []).extend(lst)
        if len(s.cluster_files()) > 1:
            # The sort is stable. One file keeps the order it was read in.
            for lst in raw_arcs.values():
                lst.sort(key=lambda a: a.start)

        with log.step(LOG, "mark arcs"):
            arcs = mark_arcs(anchors, raw_arcs)

        with log.step(LOG, "remove empty anchors"):
            anchors = remove_empty_anchors(anchors, arcs)

        LOG.info("load breakpoints")
        breakpoints = load_breakpoints(s.data_path(s.data_segment_split), chrs)

        LOG.info("load singletons")
        # The exponent is fitted on the whole chromosome set. A region is too narrow a band.
        singletons = load_singletons(s.data_path(s.data_singletons), chr_set, None)
        contact_fit = fit_contact_exponent(singletons)
        arc_fits = fit_arc_strengths(arcs)
        singletons = filter_singletons(singletons, region)

        # Inter chromosomal singletons feed the chromosome level MC alone.
        if s.data_singletons_inter and len(chrs) > 1:
            inter_path = s.data_path(s.data_singletons_inter)
            LOG.info("load inter-chr singletons")
            inter = load_singletons(inter_path, chr_set, region)
            singletons.extend(inter)

        compartments: CompartmentMap = {}
        if s.data_compartments:
            LOG.info("load compartments")
            compartments = load_compartments(s.data_path(s.data_compartments), chr_set, region)

        activity: SignalMap = {}
        if s.data_anchor_activity:
            LOG.info("load anchor activity")
            activity = load_activity(s.data_path(s.data_anchor_activity), chr_set, region)

        phasing: SignalMap = {}
        if s.data_phasing_track:
            LOG.info("load phasing track")
            phasing = load_signal(s.data_path(s.data_phasing_track), chr_set, region)

        compartments = _finalize_tracks(compartments, phasing, anchors)

        anchor_contacts: ac.AnchorContactMap = {}
        if s.data_anchor_contacts:
            anchor_contacts = _load_anchor_contacts(s.data_path(s.data_anchor_contacts), anchors)

        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            contact_fit=contact_fit,
            arc_fit=arc_fits.get(0),
            arc_fits=arc_fits,
            long_arcs=long_arcs,
            compartments=compartments,
            activity=activity,
            anchor_contacts=anchor_contacts,
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
        phasing_df: Any | None = None,
    ) -> ContactData:
        """Builds ContactData from pandas DataFrames.

        Parameters
        ----------
        anchors_df
            Columns chr, start, end[, orientation].
        arcs_df
            Columns chr_a, start_a, end_a, chr_b, start_b, end_b, score.
        breakpoints_df
            Columns chr, pos.
        singletons_df
            Columns chr1, pos1, chr2, pos2, score.
        chrs
            Chromosomes to keep. None keeps every chromosome of `anchors_df`.
        region
            Genomic window.
        max_pet_length
            Arcs longer than this are discarded.
        compartments_df
            Columns chr, start, end[, label][, value]. `label` wins when both are present. A frame of
            values is phased as a file is.
        phasing_df
            Columns chr, start, end, value. Orients a compartment frame of values.
        """
        chr_set: set[str] = (
            set(chrs) if chrs is not None else {str(c) for c in anchors_df["chr"].unique()}
        )

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
            factor = int(row["factor"]) if "factor" in row.index else 0
            arc = RawArc(posa, posb, float(row["score"]), factor)
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

        breakpoints: BreakpointMap = {}
        if breakpoints_df is not None:
            for _, row in breakpoints_df.iterrows():
                c = str(row["chr"])
                if c not in chr_set:
                    continue
                breakpoints.setdefault(c, []).append(int(row["pos"]))

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

        compartments = _compartments_from_df(compartments_df, chr_set, region)
        phasing = _signal_from_df(phasing_df, chr_set, region)
        compartments = _finalize_tracks(compartments, phasing, anchors)

        contact_fit = fit_contact_exponent(singletons)
        arc_fits = fit_arc_strengths(arcs)
        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
            contact_fit=contact_fit,
            arc_fit=arc_fits.get(0),
            arc_fits=arc_fits,
            long_arcs=long_arcs,
            compartments=compartments,
        )


def _finalize_tracks(
    compartments: CompartmentMap,
    phasing: SignalMap,
    anchors: AnchorMap,
) -> CompartmentMap:
    """Phases the compartment track for both factories."""
    if compartments:
        compartments = phase_compartments(compartments, phasing or None, anchors)
    return compartments


def _compartments_from_df(df: Any, chr_set: set[str], region: BedRegion | None) -> CompartmentMap:
    """Builds a CompartmentMap from a DataFrame with chr, start, end[, label][, value]."""
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
    """Builds a SignalMap from a DataFrame with chr, start, end, value."""
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


def fit_arc_strengths(arcs: ArcMap) -> dict[int, ArcStrengthFit]:
    """Fits one loop strength per factor over a run's arcs."""
    by_factor: dict[int, list[InteractionArc]] = {}
    for lst in arcs.values():
        for a in lst:
            if a.factor >= 0:
                by_factor.setdefault(a.factor, []).append(a)
    return {f: fit_arc_strength(lst) for f, lst in sorted(by_factor.items())}


def mark_arcs(
    anchors: AnchorMap,
    raw_arcs: RawArcMap,
) -> ArcMap:
    """Maps arcs in genomic positions to arcs between anchor indices. `raw_arcs` are sorted by start."""
    import bisect

    arcs: ArcMap = {}

    for chr_ in anchors:
        chr_anchors = anchors[chr_]
        chr_raw = raw_arcs.get(chr_, [])

        # Anchors are sorted by start and do not overlap. The last anchor with start <= pos is
        # the only candidate.
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
        ] = {}  # end index -> arcs staged for the current start
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
                factor=int(raw.factor),
                genomic_start=raw.start,
                genomic_end=raw.end,
            )
            tmp_arcs.setdefault(end, []).append(arc)
        flush(result)

        arcs[chr_] = result
        LOG.info("marked arcs %s: %d", chr_, len(result))

    return arcs


def remove_empty_anchors(
    anchors: AnchorMap,
    arcs: ArcMap,
) -> AnchorMap:
    """Removes the anchors that no arc ends on.

    Returns the new anchors and rewrites the arc indices in place.
    """
    new_anchors: AnchorMap = {}
    index_maps: dict[str, dict[int, int]] = {}

    for chr_ in anchors:
        chr_anchors = anchors[chr_]
        chr_arcs = arcs.get(chr_, [])
        n = len(chr_anchors)

        used = [False] * n
        for arc in chr_arcs:
            if 0 <= arc.start < n:
                used[arc.start] = True
            if 0 <= arc.end < n:
                used[arc.end] = True

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
