"""The check of an upload, by its kind.

A check returns the one line summary of the dialog, or the line, what was found and the fix.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import expression


KINDS = ("loops", "anchors", "contacts", "enhancers", "vcf", "expression")
MAX_PET_LENGTH = 1_000_000


@dataclass
class Problem:
    line: int
    found: str
    fix: str


@dataclass
class Check:
    ok: bool
    summary: str = ""
    error: Problem | None = None
    samples: list[str] = field(default_factory=list)
    # An expression table with whole numbers alone, which read counts are.
    counts: bool = False


def _n(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def records(path: Path):  # noqa: ANN201
    """Yields (line number, columns) of a text file, skipping empty, # and track lines and
    reading Windows line ends."""
    with path.open(encoding="utf-8", errors="replace", newline=None) as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith(("#", "track", "browser")):
                continue
            yield i, s.split()


def whole(text: str) -> int | None:
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _pair(i: int, cols: list[str], what: str) -> Problem | None:
    if len(cols) == 10 and not cols[6].isdigit():
        return Problem(
            i,
            f"Line {i} has 10 columns and column 7 is the letter {cols[6]} where a count is expected.",
            "This is the loop format of 3D-GNOME 3.0. Move the orientations to the anchors file and keep one file per factor with seven columns.",
        )
    if len(cols) < 7:
        return Problem(i, f"Line {i} has {len(cols)} columns where 7 are expected.", "Give chr1 start1 end1 chr2 start2 end2 count on every line.")
    if any(whole(cols[k]) is None for k in (1, 2, 4, 5)):
        return Problem(i, f"Line {i} has a position that is not a whole number.", "Give the starts and ends in base pairs.")
    if whole(cols[6]) is None:
        return Problem(i, f"Line {i} has {cols[6]!r} in column 7.", f"Give the {what} as a whole number.")
    return None


def check_loops(path: Path, region: tuple[str, int, int] | None) -> Check:
    total = inside = inter = long = 0
    for i, cols in records(path):
        if p := _pair(i, cols, "number of PETs"):
            return Check(False, error=p)
        total += 1
        if cols[0] != cols[3]:
            inter += 1
            continue
        a, b = int(cols[1]), int(cols[5])
        if b - a > MAX_PET_LENGTH:
            long += 1
        if region and cols[0] == region[0] and a >= region[1] and b <= region[2]:
            inside += 1
    if total == 0:
        return Check(False, error=Problem(0, "The file has no loops.", "Give one loop per line, chr1 start1 end1 chr2 start2 end2 count."))
    parts = [f"{_n(total, 'loop', 'loops')} read."]
    if region:
        parts.append(f"{inside:,} {'is' if inside == 1 else 'are'} in the region.")
    tail = []
    if inter:
        tail.append(f"{_n(inter, 'loop', 'loops')} between chromosomes {'is' if inter == 1 else 'are'} skipped")
    if long:
        tail.append(f"{_n(long, 'loop', 'loops')} longer than 1 Mb {'goes' if long == 1 else 'go'} to the coarse contact map")
    if tail:
        parts.append(" and ".join(tail) + ".")
    return Check(True, " ".join(parts))


def check_anchors(path: Path) -> Check:
    total, first = 0, None
    for i, cols in records(path):
        if len(cols) < 3 or whole(cols[1]) is None or whole(cols[2]) is None:
            return Check(False, error=Problem(i, f"Line {i} is not chr start end.", "Give chr start end and an optional orientation L, R or N."))
        has = len(cols) >= 4
        if first is None:
            first = has
        elif has != first:
            found = f"Line {i} has {'an' if has else 'no'} orientation and the first line has {'none' if has else 'one'}."
            return Check(False, error=Problem(i, found, "Give the orientation on every line or on none."))
        if has and cols[3] not in ("L", "R", "N"):
            return Check(False, error=Problem(i, f"Line {i} has orientation {cols[3]!r}.", "Use L, R or N."))
        total += 1
    if total == 0:
        return Check(False, error=Problem(0, "The file has no anchors.", "Give one anchor per line, chr start end."))
    return Check(True, f"{_n(total, 'anchor', 'anchors')} read, " + ("with orientations." if first else "without orientations, read as N."))


def check_contacts(path: Path, region: tuple[str, int, int] | None) -> Check:
    total = inside = 0
    for i, cols in records(path):
        if p := _pair(i, cols, "contact count"):
            return Check(False, error=p)
        total += 1
        if region and cols[0] == region[0] == cols[3] and int(cols[1]) >= region[1] and int(cols[4]) <= region[2]:
            inside += 1
    if total == 0:
        return Check(False, error=Problem(0, "The file has no contact pairs.", "Give one pair per line, chr1 start1 end1 chr2 start2 end2 count."))
    return Check(True, f"{_n(total, 'contact pair', 'contact pairs')} read." + (f" {inside:,} {'is' if inside == 1 else 'are'} in the region." if region else ""))


def check_enhancers(path: Path, region: tuple[str, int, int] | None = None) -> Check:
    total = inside = 0
    for i, cols in records(path):
        if len(cols) < 3 or whole(cols[1]) is None or whole(cols[2]) is None:
            return Check(False, error=Problem(i, f"Line {i} is not chr start end.", "Give one enhancer per line, chr start end."))
        total += 1
        if region and cols[0] == region[0] and int(cols[1]) >= region[1] and int(cols[2]) <= region[2]:
            inside += 1
    if total == 0:
        return Check(False, error=Problem(0, "The file has no enhancers.", "Give one enhancer per line, chr start end."))
    return Check(True, f"{_n(total, 'enhancer', 'enhancers')} read." + (f" {inside:,} {'is' if inside == 1 else 'are'} in the region." if region else ""))


def check_vcf(path: Path) -> Check:
    with path.open("rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":
        return Check(False, error=Problem(0, "The file is not compressed.", "Compress it with bgzip and index it with tabix for large files."))
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                if not samples:
                    return Check(False, error=Problem(0, "The VCF has no sample columns.", "Give a VCF with genotypes of at least one sample."))
                return Check(True, f"VCF with {_n(len(samples), 'sample', 'samples')}.", samples=samples)
            if not line.startswith("##"):
                break
    return Check(False, error=Problem(0, "The file has no #CHROM header line.", "Give a VCF file compressed with bgzip."))


def check_expression(path: Path) -> Check:
    try:
        table = expression.read_table(path)
    except expression.ExpressionError as e:
        return Check(False, error=Problem(0, str(e), "Give a tab or comma separated table with the genes in the first column and one column for each sample."))
    samples = list(table.data.columns)
    whole = bool((table.data >= 0).all().all() and (table.data == table.data.round()).all().all())
    by = "Ensembl id" if table.key == "gene_id" else "name"
    values = "Its values are whole numbers, as read counts are." if whole else "Its values are not whole numbers and are read as normalised."
    return Check(True, f"{_n(len(table.data), 'gene', 'genes')} by {by} and {_n(len(samples), 'sample', 'samples')}. {values}", samples=samples, counts=whole)


def check(kind: str, path: Path, region: tuple[str, int, int] | None) -> Check:
    if kind == "expression":
        return check_expression(path)
    if kind == "loops":
        return check_loops(path, region)
    if kind == "anchors":
        return check_anchors(path)
    if kind == "contacts":
        return check_contacts(path, region)
    if kind == "enhancers":
        return check_enhancers(path, region)
    return check_vcf(path)
