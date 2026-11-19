"""File readers and writers."""

from __future__ import annotations

import os

from gnome3d import log
from gnome3d.types import *

LOG = log.get("io")


def parse_region(region_str: str) -> BedRegion | None:
    """Parses 'chr:start-end'. Returns None when it does not parse."""
    try:
        chr_part, range_part = region_str.split(":", 1)
        st, en = range_part.split("-", 1)
        return BedRegion(chr=chr_part.strip(), start=int(st), end=int(en))
    except Exception:
        return None


def _expand_chr_range(token: str) -> list[str]:
    """Expands 'chrN-chrM' to the list from chrN to chrM when both ends are numbers.

    'chr1-chr3' gives ['chr1', 'chr2', 'chr3']. Any other token is returned as a list of one.
    """
    if "-" not in token:
        return [token]
    lo, hi = token.split("-", 1)
    prefix_lo = "".join(c for c in lo if not c.isdigit())
    prefix_hi = "".join(c for c in hi if not c.isdigit())
    suffix_lo = lo[len(prefix_lo) :]
    suffix_hi = hi[len(prefix_hi) :]
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
    """Parses the chromosomes or region argument of the CLI.

    'chr14:18288319-20307135' -> (['chr14'], BedRegion(...))
    'chr14'                   -> (['chr14'], None)
    'chr1,chr3,chrX'          -> (['chr1', 'chr3', 'chrX'], None)
    'chr1-chr22,chrX'         -> (['chr1', ..., 'chr22', 'chrX'], None)
    ''                        -> chr1 to chr22 and chrX
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

    seen: set[str] = set()
    unique: list[str] = []

    for c in chrs:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique, None


def load_anchors(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
) -> AnchorMap:
    """Loads an anchor BED file, `chr start end [orientation]`.

    Returns the anchors of the chromosomes in `chr_set`. With a region, an anchor is kept when at
    least one of its ends is inside.
    """
    anchors: AnchorMap = {}
    if not path or not os.path.exists(path):
        LOG.warning("anchors file not found: %s", path)
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
        LOG.info("anchors loaded: %s: %d", chr_, len(lst))

    return anchors


def load_arcs(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
    max_pet_length: int = 1_000_000,
    factor: int = 0,
) -> tuple[RawArcMap, RawArcMap]:
    """Loads a PET cluster BEDPE file, `chr_a start_a end_a chr_b start_b end_b score`.

    Every arc carries `factor`, the index of its cluster file. Returns `(raw, long_arcs)`. `raw`
    holds the intra chromosomal arcs sorted by start. `long_arcs` holds the arcs longer than
    `max_pet_length`, which are not mapped to anchors and go into the segment heatmap.
    """
    raw: RawArcMap = {}
    long_arcs: RawArcMap = {}

    if not path or not os.path.exists(path):
        LOG.warning("arcs file not found: %s", path)
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

            arc = RawArc(posa, posb, score, factor)

            if posb - posa > max_pet_length:
                long_cnt += 1
                long_arcs.setdefault(chr_a, []).append(arc)
                continue

            lst = raw.setdefault(chr_a, [])
            p = len(lst)
            while p > 0 and lst[p - 1].start > arc.start:
                p -= 1
            lst.insert(p, arc)
            added += 1

    LOG.info("arcs loaded: %d, long arcs separated: %d", added, long_cnt)
    return raw, long_arcs


def load_breakpoints(path: str, chrs: list[str]) -> BreakpointMap:
    """Loads a segment breakpoint BED file, `chr pos pos`. Returns the positions per chromosome."""
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


def compartment_from_label(label: str) -> int:
    """Maps a compartment label to a `Compartment` code.

    Accepts CALDER labels such as `A.1.1`, the compact `A1` and `B2` and the bare `A` and `B`.
    Only the first two levels are read. An unknown label gives `NONE`.
    """
    t = label.strip().upper()
    if t.startswith("A.1") or t.startswith("A1"):
        return int(Compartment.A1)
    if t.startswith("A"):
        return int(Compartment.A2)
    if t.startswith("B.2") or t.startswith("B2"):
        return int(Compartment.B2)
    if t.startswith("B"):
        return int(Compartment.B1)
    return int(Compartment.NONE)


def _overlaps(start: int, end: int, region: BedRegion | None) -> bool:
    if region is None:
        return True
    return start <= region.end and end >= region.start


def load_compartments(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
) -> CompartmentMap:
    """Loads compartment calls, `chr start end value_or_label`.

    The fourth column is a signed eigenvector value when it parses as a float and a label
    otherwise. A track of values keeps `cls` at `NONE` until it is phased. Returns the intervals
    of the chromosomes in `chr_set`, sorted by start.
    """
    out: CompartmentMap = {}
    if not path or not os.path.exists(path):
        LOG.warning("compartments file not found: %s", path)
        return out

    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            chr_ = parts[0]
            if chr_ not in chr_set:
                continue
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if not _overlaps(start, end, region):
                continue
            try:
                score = float(parts[3])
            except ValueError:
                cls = compartment_from_label(parts[3])
                score = 0.0
            else:
                if score != score:  # a NaN row is a gap of an eigenvector track
                    continue
                cls = int(Compartment.NONE)
            out.setdefault(chr_, []).append(CompartmentInterval(chr_, start, end, cls, score))

    for chr_, lst in out.items():
        lst.sort(key=lambda iv: iv.start)
        LOG.info("compartments loaded: %s: %d", chr_, len(lst))

    return out


def load_activity(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
) -> SignalMap:
    """Loads an anchor activity track.

    The value is the seventh column of a line with seven or more columns, the broadPeak signal,
    and the fourth column otherwise, as in a bedGraph. Returns the intervals of the chromosomes
    in `chr_set`, sorted by start.
    """
    out: SignalMap = {}
    if not path or not os.path.exists(path):
        LOG.warning("activity file not found: %s", path)
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            chr_ = parts[0]
            if chr_ not in chr_set:
                continue
            try:
                start, end = int(float(parts[1])), int(float(parts[2]))
                value = float(parts[6] if len(parts) >= 7 else parts[3])
            except ValueError:
                continue
            if value != value:
                continue
            if not _overlaps(start, end, region):
                continue
            out.setdefault(chr_, []).append(SignalInterval(chr_, start, end, value))
    for lst in out.values():
        lst.sort(key=lambda iv: iv.start)
    return out


def load_signal(
    path: str,
    chr_set: set[str],
    region: BedRegion | None = None,
) -> SignalMap:
    """Loads a continuous track such as ATAC-seq, `chr start end value`.

    Returns the intervals of the chromosomes in `chr_set`, sorted by start.
    """
    out: SignalMap = {}
    if not path or not os.path.exists(path):
        LOG.warning("signal file not found: %s", path)
        return out

    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            chr_ = parts[0]
            if chr_ not in chr_set:
                continue
            try:
                start, end = int(parts[1]), int(parts[2])
                value = float(parts[3])
            except ValueError:
                continue
            if value != value:
                continue
            if not _overlaps(start, end, region):
                continue
            out.setdefault(chr_, []).append(SignalInterval(chr_, start, end, value))

    for chr_, lst in out.items():
        lst.sort(key=lambda iv: iv.start)
        LOG.info("signal loaded: %s: %d", chr_, len(lst))

    return out


def load_singletons(
    path: str,
    chr_set: set[str],
    region: BedRegion | None,
) -> list[SingletonContact]:
    """Reads a singletons BEDPE file into a list of `(chr1, pos1, chr2, pos2, score)`."""
    contacts: list[SingletonContact] = []
    if not path or not os.path.exists(path):
        LOG.warning("singletons file not found: %s", path)
        return contacts

    line_cnt = 0
    with open(path) as f:
        for line in f:
            line_cnt += 1
            if line_cnt % 1_000_000 == 0:
                LOG.info(". (%d lines)", line_cnt)
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

    LOG.info("singletons loaded: %d", len(contacts))
    return contacts


def filter_singletons(
    contacts: list[SingletonContact], region: BedRegion | None
) -> list[SingletonContact]:
    """Returns the contacts with both ends inside `region`, or all of them without a region."""
    if region is None:
        return contacts
    return [c for c in contacts if region.contains(c[1]) and region.contains(c[3])]


def create_singleton_heatmap(
    contacts: list[SingletonContact],
    bins: dict[str, list[int]],
    start_ind: dict[str, int],
    total_size: int,
    bin_lengths_mb: list[float] | None = None,
) -> list[list[float]]:
    """Builds a contact frequency heatmap from `(chr1, pos1, chr2, pos2, score)` contacts.

    `bin_lengths_mb` holds the genomic length of each global bin in Mb. With it, cell (i, j) is
    divided by `len_i * len_j`.
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

    LOG.info("singleton heatmap: %d contacts binned, size %dx%d", ok_cnt, total_size, total_size)

    if bin_lengths_mb is not None:
        for i in range(total_size):
            for j in range(i + 1, total_size):
                denom = bin_lengths_mb[i] * bin_lengths_mb[j]
                if denom > 0.0:
                    v = h[i][j] / denom
                    h[i][j] = v
                    h[j][i] = v

    return h


def write_cif(
    path: str,
    beads: list[BeadOut],
    entry_id: str = "3dgnome",
) -> None:
    """Writes one structure to an mmCIF file.

    Each bead carries its genomic range and kind in three extension columns,
    `_atom_site.gnome_region_start`, `_atom_site.gnome_region_end` and
    `_atom_site.gnome_bead_kind`. Anchors have the residue name ALA and subanchors GLY, which
    viewers colour apart.
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
_atom_site.gnome_bead_kind
"""
    # A run that starts again skips a member whose file exists. The file gets its name only when
    # it is whole.
    part = path + ".part"
    with open(part, "w") as f:
        f.write(header)
        for i, bead in enumerate(beads, start=1):
            comp = "ALA" if bead.kind == "anchor" else "GLY"
            f.write(
                f"ATOM {i} C CA . {comp} A 1 {i} ? "
                f"{bead.x} {bead.y} {bead.z} 1.00 99.99 C "
                f"{bead.start} {bead.end} {bead.kind}\n"
            )
    os.replace(part, path)


def write_bead_table(path: str, chr_: str, beads: list[BeadOut]) -> None:
    """Writes one structure as a tab separated table, one row per bead.

    The columns are chromosome, genomic start and end in bp, x, y, z and the bead kind, after one
    header line.
    """
    with open(path, "w") as f:
        f.write("#chr\tstart\tend\tx\ty\tz\tkind\n")
        for b in beads:
            f.write(f"{chr_}\t{b.start}\t{b.end}\t{b.x}\t{b.y}\t{b.z}\t{b.kind}\n")
