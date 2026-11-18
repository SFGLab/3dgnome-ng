"""Uploads of a user's own data: stored under RESULTS_DIR/uploads/<id>/, checked on arrival,
read back when a job names them.

A check returns the one line summary of the dialog, or the line, what was found and the fix.
"""

from __future__ import annotations

import gzip
import json
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO

KINDS = ("loops", "anchors", "contacts", "enhancers", "vcf")
MAX_PET_LENGTH = 1_000_000
CHUNK = 1 << 20


class UploadTooLarge(ValueError):
    """An upload past the size cap."""


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


def _n(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def store(stream: BinaryIO, name: str, kind: str, root: Path, cap: int, extra: dict[str, str] | None = None) -> Path:
    """Copies an upload to a new directory under root in chunks and writes its meta.json.
    Returns the directory. Raises UploadTooLarge past cap bytes, leaving nothing behind."""
    uid = secrets.token_urlsafe(12)
    d = root / uid
    d.mkdir(parents=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name) or "upload"
    size = 0
    try:
        with (d / safe).open("wb") as out:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > cap:
                    raise UploadTooLarge(f"{name} is more than {cap // (1 << 20)} MB, the largest upload this server takes")
                out.write(chunk)
    except UploadTooLarge:
        shutil.rmtree(d)
        raise
    (d / "meta.json").write_text(json.dumps({"id": uid, "kind": kind, "name": safe, "size": size, "created": time.time()} | (extra or {})))
    return d


def meta(d: Path) -> dict[str, object]:
    return json.loads((d / "meta.json").read_text())


def data_file(d: Path) -> Path:
    return d / str(meta(d)["name"])


def records(path: Path):  # noqa: ANN201
    """Yields (line number, columns) of a text file, skipping empty, # and track lines and
    reading Windows line ends."""
    with path.open(encoding="utf-8", errors="replace", newline=None) as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith(("#", "track", "browser")):
                continue
            yield i, s.split()


def _int(text: str) -> int | None:
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
    if any(_int(cols[k]) is None for k in (1, 2, 4, 5)):
        return Problem(i, f"Line {i} has a position that is not a whole number.", "Give the starts and ends in base pairs.")
    if _int(cols[6]) is None:
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
        if len(cols) < 3 or _int(cols[1]) is None or _int(cols[2]) is None:
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
        if len(cols) < 3 or _int(cols[1]) is None or _int(cols[2]) is None:
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


def check(kind: str, path: Path, region: tuple[str, int, int] | None) -> Check:
    if kind == "loops":
        return check_loops(path, region)
    if kind == "anchors":
        return check_anchors(path)
    if kind == "contacts":
        return check_contacts(path, region)
    if kind == "enhancers":
        return check_enhancers(path, region)
    return check_vcf(path)


def record_check(d: Path, c: Check) -> None:
    """Writes the outcome of the check into the upload's meta.json."""
    m = meta(d) | {"ok": c.ok}
    (d / "meta.json").write_text(json.dumps(m))


def clean_copy(d: Path, dst: Path) -> Path:
    """Writes the upload's records tab separated, without header lines or Windows line ends."""
    with dst.open("w") as out:
        for _, cols in records(data_file(d)):
            out.write("\t".join(cols) + "\n")
    return dst


def result(d: Path, c: Check) -> dict[str, object]:
    m = meta(d)
    # A map read by link is shown by the remote file's name and size, not the window kept.
    name, size = m.get("link_name", m["name"]), int(m.get("link_size", m["size"]))  # type: ignore[arg-type]
    return {"id": m["id"], "kind": m["kind"], "name": name, "size": size, "ok": c.ok, "summary": c.summary, "error": asdict(c.error) if c.error else None, "samples": c.samples}


def preview(d: Path, region: tuple[str, int, int]) -> list[list[int]]:
    """The loops of a loops upload inside a region, as anchor midpoints and the count."""
    out = []
    for _, cols in records(data_file(d)):
        if len(cols) >= 7 and cols[0] == cols[3] == region[0] and _int(cols[6]) is not None:
            a, b = int(cols[1]), int(cols[5])
            if a >= region[1] and b <= region[2]:
                out.append([(a + int(cols[2])) // 2, (int(cols[4]) + b) // 2, int(cols[6])])
    return out


_NAMES = {"DEL": "deletion", "INS": "insertion", "INV": "inversion", "DUP": "duplication", "BND": "breakend", "CNV": "copy number change"}


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _change(kind: str, before: dict, after_loops: list, factor: str) -> str:
    """The words for what one variant did to the loops of one factor."""
    keyed = {(lp.ref_a, lp.ref_b): lp for lp in after_loops if lp.ref_a and lp.ref_b}
    removed = [k for k in before if k not in keyed]
    kept_anchors = {r for lp in after_loops for r in (lp.ref_a, lp.ref_b) if r}
    lost_anchors = {r for k in removed for r in k if r not in kept_anchors}
    new_loops = [lp for lp in after_loops if lp.ref_a is None or lp.ref_b is None]
    new_anchors = {lp.a if lp.ref_a is None else lp.b for lp in new_loops}
    changed = [k for k, lp in keyed.items() if k in before and (lp.b[1] - lp.a[0]) != (before[k].b[1] - before[k].a[0])]
    flipped = {r for k, lp in keyed.items() if k in before for r, o0, o1 in ((k[0], before[k].orient[0], lp.orient[0]), (k[1], before[k].orient[1], lp.orient[1])) if o0 != o1}
    if kind == "DEL":
        if lost_anchors:
            return f"removes {_plural(len(lost_anchors), f'{factor} anchor')} and {'its' if len(lost_anchors) == 1 else 'their'} {_plural(len(removed), 'loop')}"
        return f"shortens {_plural(len(changed), 'loop')}" if changed else "no anchor inside"
    if kind == "INS":
        if new_anchors:
            return f"adds {_plural(len(new_anchors), f'{factor} anchor')}, {_plural(len(new_loops), 'new loop')}"
        return f"lengthens {_plural(len(changed), 'loop')}" if changed else f"no {factor} motif above the cutoff"
    if kind == "DUP":
        # A copy keeps the reference interval of its original, so a copied loop has the key of a
        # loop that is there already and shows as one more loop than the keys count.
        copied = len(after_loops) - len(keyed) - len(new_loops)
        if copied > 0 or new_loops:
            return f"copies {_plural(max(copied, 0), 'loop')}, adds {_plural(len(new_loops), 'new loop')}" if new_loops else f"copies {_plural(copied, 'loop')}"
        return f"lengthens {_plural(len(changed), 'loop')}" if changed else "no anchor inside"
    if flipped or removed:
        return f"flips {_plural(len(flipped), 'anchor')}, replaces {_plural(len(removed) + len(new_loops), 'loop')}"
    return "no anchor inside"


def combine_changes(by_factor: list[tuple[str, str]]) -> str:
    """One sentence for the changes of several loop factors: one verb is summed with the split
    by factor, different changes are listed and a factor without change is left out."""
    real = [(f, w) for f, w in by_factor if not w.startswith("no ")]
    if not real:
        return by_factor[0][1] if by_factor else "no anchor inside"
    simple = [re.fullmatch(r"(\w+) (\d+) loops?", w) for _, w in real]
    if len(real) > 1 and all(simple) and len({m[1] for m in simple}) == 1:  # type: ignore[index]
        total = sum(int(m[2]) for m in simple)  # type: ignore[index]
        split = " and ".join(f"{m[2]} {f}" for (f, _), m in zip(real, simple, strict=True))  # type: ignore[index]
        return f"{simple[0][1]} {total} loop{'' if total == 1 else 's'}, {split}"  # type: ignore[index]
    return "; ".join(dict.fromkeys(w for _, w in real))


def variant_check(cell, region, factors: str, vcf: Path, sample: str, haplotype: int, own_loops: list[tuple[str, Path]] | None = None, own_anchors: Path | None = None) -> dict[str, object]:  # noqa: ANN001, PLR0913
    """What a sample's variants on one haplotype do to the loops of a region, one variant at a
    time, with the counts applied and skipped. Own loops replace the cell type's, with their
    anchors file or their merged ends as anchors, as the job builds them.

    @raises KeyError when the sample is not in the VCF
    """
    from personal3d import motifs, rewrite

    from .inputs import _clusters, _loop_end_anchors, _read
    from .svs import sample_variants

    records, skipped_all = sample_variants(vcf, sample, region)
    mine = [r for r in records if r["haps"][haplotype]]
    chrom = region.chrom
    if own_loops:
        anchors = _read(own_anchors, chrom, 4) if own_anchors else _loop_end_anchors([p for _, p in own_loops], chrom)
        if anchors.shape[1] < 4:
            anchors[3] = "N"
        sources = own_loops
    else:
        anchors = _read(cell.anchors_for(factors), chrom, 4)
        sources = cell.clusters_for(factors)
    anchors = anchors.sort_values(1).reset_index(drop=True)
    length = int(anchors[2].max()) + 10_000_000 if len(anchors) else region.end + 10_000_000
    nets = []
    for factor, path in sources:
        clusters = _clusters(_read(path, chrom, 7), anchors)
        segs = rewrite.domains(clusters)
        base = rewrite.apply(clusters, segs, {chrom: length}, [], None, None, motifs.THRESHOLD)
        nets.append((factor, clusters, segs, {(lp.ref_a, lp.ref_b): lp for lp in base}))
    rows = []
    applied: dict[str, int] = {}
    for r in mine:
        sv = r["sv"]
        words = []
        for factor, clusters, segs, before in nets:
            after = rewrite.apply(clusters, segs, {chrom: length}, [sv], None, None, motifs.THRESHOLD)
            words.append((factor, _change(str(r["type"]), before, after, factor)))
        applied[str(r["type"])] = applied.get(str(r["type"]), 0) + 1
        rows.append({"type": r["type"], "pos": r["pos"], "end": sv.end - 1, "length": r["length"], "genotype": r["genotype"], "change": combine_changes(words)})
    hap = "first" if haplotype == 0 else "second"
    span = f"{chrom}:{region.start:,}-{region.end:,}"
    n = len(rows)
    parts = [f"{_plural(n, 'variant')} of {sample} on the {hap} haplotype {'lies' if n == 1 else 'lie'} inside {span}."]
    if applied:
        parts.append(", ".join(_plural(c, _NAMES[k]) for k, c in applied.items()) + (" is" if n == 1 else " are") + " applied.")
    sk = sum(skipped_all.values())
    if sk:
        parts.append(f"{_plural(sk, 'record')} of other types {'is' if sk == 1 else 'are'} skipped, " + ", ".join(_plural(c, _NAMES.get(k, k.lower())) for k, c in skipped_all.items()) + ".")
    return {"summary": " ".join(parts), "applied": applied, "skipped": skipped_all, "variants": rows}
