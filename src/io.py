"""
src/io.py - File loading for 3dgnome-ng.

Mirrors C++ InteractionArcs loading methods:
  - loadAnchorsData()      -> load_anchors()
  - loadPetClustersData()  -> load_arcs()
  - markArcs()             -> mark_arcs()
  - removeEmptyAnchors()   -> remove_empty_anchors()
  - createSingletonHeatmap() -> create_singleton_heatmap()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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


def parse_region(region_str: str) -> Optional[BedRegion]:
    """
    Parse 'chr:start-end' string.  Returns None on failure.
    C++ BedRegion::tryParse uses sscanf(str, "%30[^:]:%d-%d", ...) - dash separator.
    """
    try:
        chr_part, range_part = region_str.split(":", 1)
        st, en = range_part.split("-", 1)
        return BedRegion(chr=chr_part.strip(), start=int(st), end=int(en))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load anchors from BED file

def load_anchors(
    path: str,
    chr_set: set,
    region: Optional[BedRegion] = None,
) -> dict:
    """
    Load anchor BED file.  Format: chr start end [orientation]

    Returns dict[chr -> list[Anchor]] (only for chromosomes in chr_set).
    Anchors are included only if at least one end falls within `region`
    (if specified).
    """
    anchors: dict[str, list[Anchor]] = {}
    if not path or not os.path.exists(path):
        print(f"[io] anchors file not found: {path}")
        return anchors

    with open(path) as f:
        first_line = f.readline()
        has_orientation = len(first_line.split()) == 4
        f.seek(0)

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chr_ = parts[0]
            if chr_ not in chr_set:
                continue
            start, end = int(parts[1]), int(parts[2])
            orientation = parts[3] if has_orientation and len(parts) >= 4 else "N"

            if region is not None:
                if not (region.contains(start) or region.contains(end)):
                    continue

            anchors.setdefault(chr_, []).append(Anchor(chr_, start, end, orientation))

    for chr_, lst in anchors.items():
        print(f"  anchors loaded: {chr_}: {len(lst)}")
    return anchors


# ---------------------------------------------------------------------------
# Load PET cluster arcs from BEDPE file

def load_arcs(
    path: str,
    chr_set: set,
    region: Optional[BedRegion] = None,
    max_pet_length: int = 1_000_000,
) -> dict:
    """
    Load PET cluster BEDPE file.  Format: chr_a start_a end_a chr_b start_b end_b score

    Returns dict[chr -> list[RawArc]], sorted by start, intra only.
    Arcs longer than max_pet_length are excluded (they go to long_arcs).
    """
    raw: dict[str, list[RawArc]] = {}
    long_arcs: dict[str, list[RawArc]] = {}

    if not path or not os.path.exists(path):
        print(f"[io] arcs file not found: {path}")
        return raw

    added = 0
    long_cnt = 0

    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            chr_a, chr_b = parts[0], parts[3]
            if chr_a != chr_b:
                continue
            if chr_a not in chr_set:
                continue

            ast, aend = int(parts[1]), int(parts[2])
            bst, bend = int(parts[4]), int(parts[5])
            score = float(parts[6])

            posa = (ast + aend) // 2
            posb = (bst + bend) // 2
            if posa > posb:
                posa, posb = posb, posa

            if region is not None:
                if not (region.contains(posa) and region.contains(posb)):
                    continue

            arc = RawArc(posa, posb, score)

            if posb - posa > max_pet_length:
                long_cnt += 1
                long_arcs.setdefault(chr_a, []).append(arc)
                continue

            # Insert maintaining sort order by start
            lst = raw.setdefault(chr_a, [])
            p = len(lst)
            while p > 0 and lst[p - 1].start > arc.start:
                p -= 1
            lst.insert(p, arc)
            added += 1

    print(f"  arcs loaded: {added}, long arcs discarded: {long_cnt}")
    return raw


# ---------------------------------------------------------------------------
# markArcs: map RawArcs -> anchor-indexed InteractionArcs

def mark_arcs(
    anchors: dict,
    raw_arcs: dict,
) -> dict:
    """
    Map genomic-position arcs to anchor-index arcs.
    Mirrors C++ InteractionArcs::markArcs().

    anchors:  dict[chr -> list[Anchor]]
    raw_arcs: dict[chr -> list[RawArc]] (sorted by start)

    Returns dict[chr -> list[InteractionArc]].
    """
    import bisect

    arcs: dict[str, list[InteractionArc]] = {}

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

        result = []
        tmp_arcs: dict[int, list] = {}  # end_idx -> staged arcs for current start group
        last_start = -1

        def flush(target_list):
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

        cnt = len(chr_raw)
        for idx in range(cnt + 1):  # +1 to flush at end
            st = -1
            end = -1
            if idx < cnt:
                raw = chr_raw[idx]
                st = find_anchor(raw.start)
                end = find_anchor(raw.end)

                if st == -1 or end == -1 or st == end:
                    continue

            if st != last_start or idx == cnt:
                flush(result)
                last_start = st

            if idx == cnt:
                break

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

        arcs[chr_] = result
        print(f"  marked arcs {chr_}: {len(result)}")

    return arcs


# ---------------------------------------------------------------------------
# removeEmptyAnchors: keep only anchors that are endpoints of at least one arc

def remove_empty_anchors(
    anchors: dict,
    arcs: dict,
) -> dict:
    """
    Remove anchors that are not endpoints of any arc.
    Mirrors C++ InteractionArcs::removeEmptyAnchors().

    Returns new anchors dict (original is not modified).
    Also updates arc start/end indices to reflect removed anchors.
    """
    new_anchors = {}
    index_maps = {}  # old_index -> new_index per chr

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
        new_list = []
        idx_map = {}
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
        valid_arcs = []
        for arc in arcs[chr_]:
            ns = idx_map.get(arc.start, -1)
            ne = idx_map.get(arc.end, -1)
            if ns >= 0 and ne >= 0:
                arc.start = ns
                arc.end = ne
                valid_arcs.append(arc)
        arcs[chr_] = valid_arcs

    return new_anchors


# ---------------------------------------------------------------------------
# Load segment breakpoints

def load_breakpoints(path: str, chrs: list) -> dict:
    """
    Load segment breakpoint BED file.  Format: chr pos pos

    Returns dict[chr -> list[int]] of breakpoint positions.
    """
    bp: dict[str, list[int]] = {}
    if not path or not os.path.exists(path):
        return bp

    chr_set = set(chrs)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chr_ = parts[0]
            if chr_ not in chr_set:
                continue
            pos = int(parts[1])
            bp.setdefault(chr_, []).append(pos)

    return bp


# ---------------------------------------------------------------------------
# Create singleton heatmap from BEDPE file

def create_singleton_heatmap(
    path: str,
    bins: list,  # list of (chr, bin_starts) where bin_starts is a list of boundary positions
    start_ind: dict,  # chr -> starting column index in heatmap
    total_size: int,
    chr_set: set,
    region: Optional[BedRegion] = None,
    bin_lengths_mb: Optional[list] = None,
) -> list:
    """
    Read singletons BEDPE, bin into a contact frequency heatmap.
    Mirrors C++ createSingletonHeatmap().

    bins:           dict[chr -> list[int]] of bin boundary positions
    start_ind:      dict[chr -> int] mapping chr to starting heatmap column index
    total_size:     total number of bins across all chromosomes
    chr_set:        set of chromosomes to include
    bin_lengths_mb: flat list of bin genomic lengths in Mb (global bin index).
                    When provided, h[i][j] is divided by len_i * len_j after
                    binning, mirroring C++ createSingletonHeatmap() normalisation.

    Returns a 2D list h[i][j] = float contact frequency.
    """
    import bisect

    h = [[0.0] * total_size for _ in range(total_size)]

    if not path or not os.path.exists(path):
        print(f"[io] singletons file not found: {path}")
        return h

    is_region = region is not None

    line_cnt = 0
    ok_cnt = 0
    with open(path) as f:
        for line in f:
            line_cnt += 1
            if line_cnt % 1_000_000 == 0:
                print(f"  . ({line_cnt} lines)")
            parts = line.split()
            if len(parts) < 7:
                continue
            chr1, chr2 = parts[0], parts[3]
            if chr1 not in chr_set or chr2 not in chr_set:
                continue

            sta, stb = int(parts[1]), int(parts[2])
            enda, endb = int(parts[4]), int(parts[5])
            sc = int(parts[6])

            if is_region and not (region.contains(sta) and region.contains(endb)):
                continue

            # Find bin for (chr1, sta) and (chr2, enda)
            br1 = bins.get(chr1)
            br2 = bins.get(chr2)
            if br1 is None or br2 is None:
                continue

            st_bin = bisect.bisect_right(br1, sta) - 1
            end_bin = bisect.bisect_right(br2, enda) - 1

            if st_bin < 0 or end_bin < 0:
                continue
            if st_bin >= len(br1) - 1 or end_bin >= len(br2) - 1:
                continue

            si = start_ind[chr1] + st_bin
            ei = start_ind[chr2] + end_bin

            if si == ei:
                continue  # diagonal: skip

            if 0 <= si < total_size and 0 <= ei < total_size:
                h[si][ei] += sc
                h[ei][si] += sc
                ok_cnt += 1

    print(f"  singleton heatmap: {ok_cnt} arcs binned, size {total_size}x{total_size}")

    if bin_lengths_mb is not None:
        for i in range(total_size):
            for j in range(i + 1, total_size):
                denom = bin_lengths_mb[i] * bin_lengths_mb[j]
                if denom > 0.0:
                    v = h[i][j] / denom
                    h[i][j] = v
                    h[j][i] = v

    return h


# ---------------------------------------------------------------------------
# Create singleton heatmap from pre-loaded contacts

def create_singleton_heatmap_from_contacts(
    contacts: list,
    bins: dict,
    start_ind: dict,
    total_size: int,
    bin_lengths_mb: Optional[list] = None,
) -> list:
    """
    Build a contact frequency heatmap from a list of (chr1,pos1,chr2,pos2,score)
    tuples (as produced by ContactData.from_files / from_dataframes).

    Drop-in replacement for create_singleton_heatmap() when data is already
    in memory rather than in a file.

    bin_lengths_mb: flat list of bin genomic lengths in Mb (global bin index).
                    When provided, h[i][j] is divided by len_i * len_j after
                    binning, mirroring C++ createSingletonHeatmap() normalisation.
    """
    import bisect

    h = [[0.0] * total_size for _ in range(total_size)]
    ok_cnt = 0

    for c1, p1, c2, p2, sc in contacts:
        br1 = bins.get(c1)
        br2 = bins.get(c2)
        if br1 is None or br2 is None:
            continue

        si = start_ind.get(c1, -1) + bisect.bisect_right(br1, p1) - 1
        ei = start_ind.get(c2, -1) + bisect.bisect_right(br2, p2) - 1

        if si < 0 or ei < 0 or si >= total_size or ei >= total_size or si == ei:
            continue

        h[si][ei] += sc
        h[ei][si] += sc
        ok_cnt += 1

    print(f"  singleton heatmap: {ok_cnt} contacts binned, size {total_size}x{total_size}")

    if bin_lengths_mb is not None:
        for i in range(total_size):
            for j in range(i + 1, total_size):
                denom = bin_lengths_mb[i] * bin_lengths_mb[j]
                if denom > 0.0:
                    v = h[i][j] / denom
                    h[i][j] = v
                    h[j][i] = v

    return h


# ---------------------------------------------------------------------------
# CIF export

def write_cif(
    path: str,
    beads: list,
    entry_id: str = "3dgnome",
) -> None:
    """
    Write a single structure to an mmCIF file.

    beads : list of (midpoint_bp, x, y, z)
        One entry per anchor bead, as returned by run_region() for one structure.
    """
    header = f"""data_{entry_id}
#
_entry.id {entry_id}
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.296
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
"""
    with open(path, "w") as f:
        f.write(header)
        for i, (_, x, y, z) in enumerate(beads, start=1):
            f.write(f"ATOM {i} C CA . ALA A 1 {i} ? {x} {y} {z} 1.00 99.99 C\n")
