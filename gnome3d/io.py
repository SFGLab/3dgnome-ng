"""
File loading for 3dgnome-ng.
"""

from __future__ import annotations

import os

from .types import *


def parse_region(region_str: str) -> BedRegion | None:
    """
    Parse 'chr:start-end' string.  Returns None on failure.
    Reference BedRegion::tryParse uses sscanf(str, "%30[^:]:%d-%d", ...).
    """
    try:
        chr_part, range_part = region_str.split(":", 1)
        st, en = range_part.split("-", 1)
        return BedRegion(chr=chr_part.strip(), start=int(st), end=int(en))
    except Exception:
        return None


def _expand_chr_range(token: str) -> list[str]:
    """
    Expand 'chrN-chrM' into [chrN, chrN+1, ..., chrM] when both ends are
    numeric.  Anything else is returned as a single-element list.
    Examples:
      'chr1-chr3' -> ['chr1', 'chr2', 'chr3']
      'chrX'      -> ['chrX']
      'chr1-chrX' -> ['chr1-chrX']  (non-numeric end: not expanded)
    """
    if "-" not in token:
        return [token]
    lo, hi = token.split("-", 1)
    prefix_lo = "".join(c for c in lo if not c.isdigit())
    prefix_hi = "".join(c for c in hi if not c.isdigit())
    suffix_lo = lo[len(prefix_lo):]
    suffix_hi = hi[len(prefix_hi):]
    if prefix_lo != prefix_hi or not suffix_lo or not suffix_hi:
        return [token]
    try:
        a, b = int(suffix_lo), int(suffix_hi)
    except ValueError:
        return [token]
    if b < a:
        return [token]
    return [f"{prefix_lo}{i}" for i in range(a, b + 1)]


def parse_chrs_arg(arg: str) -> tuple[list[str], BedRegion | None]:
    """
    Parse the CLI chromosomes/region argument.  Mirrors the Reference -c flag.

    Accepts:
      'chr14:18288319-20307135' -> (['chr14'], BedRegion(...))    single region
      'chr14'                   -> (['chr14'], None)              single chromosome
      'chr1,chr3,chrX'          -> (['chr1','chr3','chrX'], None) comma list
      'chr1-chr22,chrX'         -> (['chr1',...,'chr22','chrX'], None)  range + extras
      ''                        -> default whole human genome (chr1..chr22, chrX)
    """
    arg = arg.strip()
    if not arg:
        return ([f"chr{i}" for i in range(1, 23)] + ["chrX"], None)

    region = parse_region(arg)

    if region is not None:
        return ([region.chr], region)

    chrs: list[str] = []

    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        chrs.extend(_expand_chr_range(token))

    # de-dup while preserving order
    seen: set[str] = set()
    unique: list[str] = []

    for c in chrs:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique, None


# Load anchors from BED file

def load_anchors(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
) -> AnchorMap:
    """
    Load anchor BED file.  Format: chr start end [orientation]

    Returns dict[chr -> list[Anchor]] (only for chromosomes in chr_set).
    Anchors are included only if at least one end falls within `region`
    (if specified).
    """
    anchors: AnchorMap = {}
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


# Load PET cluster arcs from BEDPE file

def load_arcs(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
    max_pet_length: int = 1_000_000,
) -> tuple[RawArcMap, RawArcMap]:
    """
    Load PET cluster BEDPE file.  Format: chr_a start_a end_a chr_b start_b end_b score

    Returns (raw, long_arcs) where:
      raw       : dict[chr -> list[RawArc]], sorted by start, intra only
      long_arcs : dict[chr -> list[RawArc]], arcs with (end-start) > max_pet_length

    Long arcs are not anchor-mapped (they would be too sparse) but are folded back
    into the segment-level heatmap by Solver, mirroring Reference LooperSolver.cpp:1069-1104.
    """
    raw: RawArcMap = {}
    long_arcs: RawArcMap = {}

    if not path or not os.path.exists(path):
        print(f"[io] arcs file not found: {path}")
        return raw, long_arcs

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

    print(f"  arcs loaded: {added}, long arcs separated: {long_cnt}")
    return raw, long_arcs


# Load segment breakpoints

def load_breakpoints(path: str, chrs: list[str]) -> BreakpointMap:
    """
    Load segment breakpoint BED file.  Format: chr pos pos

    Returns dict[chr -> list[int]] of breakpoint positions.
    """
    bp: BreakpointMap = {}
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


# Load singletons

def load_singletons(
    path: str,
    chr_set: set[str],
    region: BedRegion | None,
) -> list[SingletonContact]:
    """
    Read a singletons BEDPE file into a list of (chr1, pos1, chr2, pos2, score).
    """
    contacts: list[SingletonContact] = []
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


# Create singleton heatmap from pre-loaded contacts

def create_singleton_heatmap(
    contacts: list[SingletonContact],
    bins: dict[str, list[int]],
    start_ind: dict[str, int],
    total_size: int,
    bin_lengths_mb: list[float] | None = None,
) -> list[list[float]]:
    """
    Build a contact frequency heatmap from a list of (chr1,pos1,chr2,pos2,score)
    tuples (as produced by ContactData.from_files / from_dataframes).

    Drop-in replacement for create_singleton_heatmap() when data is already
    in memory rather than in a file.

    bin_lengths_mb: flat list of bin genomic lengths in Mb (global bin index).
                    When provided, h[i][j] is divided by len_i * len_j after
                    binning, mirroring Reference createSingletonHeatmap() normalisation.
    """
    import bisect

    h: list[list[float]] = [[0.0] * total_size for _ in range(total_size)]
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


# CIF export

def write_cif(
    path: str,
    beads: list[BeadOut],
    entry_id: str = "3dgnome",
) -> None:
    """
    Write a single structure to an mmCIF file.

    beads : list of BeadOut = (start_bp, end_bp, x, y, z)
        One entry per anchor bead, as returned by run_region() for one structure.

    Each bead carries its genomic region in two non-standard extension columns:
      _atom_site.gnome_region_start  - genomic start (bp)
      _atom_site.gnome_region_end    - genomic end   (bp)
    Standard mmCIF viewers ignore unknown _atom_site.* columns; downstream tools
    that want the per-bead genomic span can parse them directly.
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
_atom_site.gnome_region_start
_atom_site.gnome_region_end
"""
    with open(path, "w") as f:
        f.write(header)
        for i, (start, end, x, y, z) in enumerate(beads, start=1):
            f.write(
                f"ATOM {i} C CA . ALA A 1 {i} ? {x} {y} {z} 1.00 99.99 C "
                f"{start} {end}\n"
            )
