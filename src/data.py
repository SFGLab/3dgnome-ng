"""
src/data.py - ContactData: the engine's input contract.

All data the solver needs lives here.  Two factory methods cover the two
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .io import (
    Anchor,
    BedRegion,
    RawArc,
    load_anchors,
    load_arcs,
    mark_arcs,
    remove_empty_anchors,
    load_breakpoints,
)

# A singleton contact in genomic coordinates.
# Stored as a plain tuple for memory efficiency.
# (chr1, pos1, chr2, pos2, score)
SingletonContact = tuple  # (str, int, str, int, int)


@dataclass
class ContactData:
    """
    anchors:     dict[chr -> list[Anchor]] - anchor beads (after empty-anchor removal)
    arcs:        dict[chr -> list[InteractionArc]] - mapped arcs (after mark_arcs)
    breakpoints: dict[chr -> list[int]] - segment boundary positions
    singletons:  list of (chr1, pos1, chr2, pos2, score) contacts
                 used to build the segment-level heatmap
    """
    anchors: dict[str, list] = field(default_factory=dict)
    arcs: dict[str, list] = field(default_factory=dict)
    breakpoints: dict[str, list] = field(default_factory=dict)
    singletons: list = field(default_factory=list)

    # -----------------------------------------------------------------------
    # File-based factory

    @classmethod
    def from_files(
        cls,
        settings,
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
        raw_arcs = load_arcs(
            s.data_path(s.data_pet_clusters), chr_set, region, s.max_pet_length
        )

        print("[data] mark arcs")
        arcs = mark_arcs(anchors, raw_arcs)

        print("[data] remove empty anchors")
        anchors = remove_empty_anchors(anchors, arcs)

        print("[data] load breakpoints")
        breakpoints = load_breakpoints(s.data_path(s.data_segment_split), chrs)

        print("[data] load singletons")
        singletons = _load_singletons(
            s.data_path(s.data_singletons), chr_set, region
        )

        return cls(
            anchors=anchors,
            arcs=arcs,
            breakpoints=breakpoints,
            singletons=singletons,
        )

    # -----------------------------------------------------------------------
    # DataFrame-based factory

    @classmethod
    def from_dataframes(
        cls,
        anchors_df,
        arcs_df,
        breakpoints_df=None,
        singletons_df=None,
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
        chr_set = set(chrs) if chrs is not None else set(anchors_df["chr"].unique())

        # -- anchors ----------------------------------------------------------
        anchors: dict[str, list] = {}
        for _, row in anchors_df.iterrows():
            c = str(row["chr"])
            if c not in chr_set:
                continue
            st, en = int(row["start"]), int(row["end"])
            if region is not None and not (region.contains(st) or region.contains(en)):
                continue
            ori = str(row["orientation"]) if "orientation" in row.index else "N"
            anchors.setdefault(c, []).append(Anchor(c, st, en, ori))

        # -- raw arcs -> mark -> remove empty ----------------------------------
        raw_arcs: dict[str, list] = {}
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
            if posb - posa > max_pet_length:
                continue
            arc = RawArc(posa, posb, float(row["score"]))
            lst = raw_arcs.setdefault(ca, [])
            p = len(lst)
            while p > 0 and lst[p - 1].start > arc.start:
                p -= 1
            lst.insert(p, arc)

        arcs = mark_arcs(anchors, raw_arcs)
        anchors = remove_empty_anchors(anchors, arcs)

        # -- breakpoints ------------------------------------------------------
        breakpoints: dict[str, list] = {}
        if breakpoints_df is not None:
            for _, row in breakpoints_df.iterrows():
                c = str(row["chr"])
                if c not in chr_set:
                    continue
                breakpoints.setdefault(c, []).append(int(row["pos"]))

        # -- singletons -------------------------------------------------------
        singletons: list = []
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
        )


# ---------------------------------------------------------------------------
# Internal helpers

def _load_singletons(
    path: str,
    chr_set: set,
    region: BedRegion | None,
) -> list:
    """
    Read a singletons BEDPE file into a list of (chr1, pos1, chr2, pos2, score).
    """
    import os
    contacts = []
    if not path or not os.path.exists(path):
        print(f"[data] singletons file not found: {path}")
        return contacts

    line_cnt = 0
    with open(path) as f:
        for line in f:
            line_cnt += 1
            if line_cnt % 1_000_000 == 0:
                print(f"  . ({line_cnt} lines)")
            parts = line.split()
            if len(parts) < 7:
                continue
            c1, c2 = parts[0], parts[3]
            if c1 not in chr_set or c2 not in chr_set:
                continue
            p1 = int(parts[1])
            p2 = int(parts[4])
            sc = int(parts[6])
            if region is not None and not (region.contains(p1) and region.contains(p2)):
                continue
            contacts.append((c1, p1, c2, p2, sc))

    print(f"  singletons loaded: {len(contacts)}")
    return contacts
