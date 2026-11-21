"""The structural variants of one sample in a region, from any VCF with SVTYPE in its INFO.

Coordinates follow personal3d, 1 based, start the first affected base and end one past the
last. A duplication is read as tandem. Types other than deletion, insertion, inversion and
duplication are skipped and counted. A phased genotype puts each allele on its haplotype and
an unphased one puts the variant on the first. Only variants wholly inside the region are kept.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from personal3d.variants import SV
from typing_extensions import TypedDict

from ..regions import Region

KINDS = ("DEL", "INS", "INV", "DUP")


def _lines(vcf: Path, region: Region) -> Iterator[str]:
    """Returns the header line of the VCF and the records of the chromosome of the region, through
    tabix when the file is indexed."""
    if Path(f"{vcf}.tbi").is_file() and shutil.which("tabix"):
        yield subprocess.run(["tabix", "-H", str(vcf)], capture_output=True, text=True, check=True).stdout.splitlines()[-1]
        out = subprocess.run(["tabix", str(vcf), str(region)], capture_output=True, text=True, check=True).stdout
        yield from out.splitlines()
        return
    with gzip.open(vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#") or line.split("\t", 1)[0] == region.chrom:
                yield line.rstrip("\n")


def _alleles(gt: str) -> tuple[bool, bool]:
    phased = "|" in gt
    a = gt.replace("/", "|").split("|")
    if len(a) != 2:
        return a[0] not in ("0", "."), False
    first, second = a[0] not in ("0", "."), a[1] not in ("0", ".")
    if not phased and (first or second):
        return True, False
    return first, second


class UnknownSample(Exception):
    """The sample is not a column of the VCF."""


class Record(TypedDict):
    """A variant a sample carries."""

    sv: SV
    type: str
    pos: int
    length: int
    genotype: str
    haps: tuple[bool, bool]


def _info(text: str) -> dict[str, str]:
    """Returns the INFO column of a record by key. A flag has an empty value."""
    out = {}
    for part in text.split(";"):
        key, _, value = part.partition("=")
        out[key] = value
    return out


def sample_variants(vcf: Path, sample: str, region: Region) -> tuple[list[Record], dict[str, int]]:
    """Returns the records of one sample inside the region with their genotype and haplotypes, in
    the order of the file. Also returns the count of records of other types, by type.

    @raises UnknownSample when the sample is not a column of the VCF
    """
    records: list[Record] = []
    skipped: dict[str, int] = {}
    col = -1
    for line in _lines(vcf, region):
        x = line.split("\t")
        if line.startswith("#"):
            if sample not in x[9:]:
                raise UnknownSample(f"{sample!r} is not a sample of the VCF")
            col = 9 + x[9:].index(sample)
            continue
        info = _info(x[7])
        kind = info.get("SVTYPE", "")
        pos = int(x[1])
        gt = x[col].split(":", 1)[0]
        haps = _alleles(gt)
        if not any(haps) or x[0] != region.chrom:
            continue
        if kind not in KINDS:
            if region.start <= pos < region.end:
                skipped[kind or "unknown"] = skipped.get(kind or "unknown", 0) + 1
            continue
        if kind == "INS":
            alt = x[4]
            seq = alt[1:] if not alt.startswith("<") else "N" * abs(int(info.get("SVLEN", "0").split(",")[0]))
            sv = SV(x[0], pos + 1, pos + 1, "INS", seq)
            length = len(seq)
        else:
            end = int(info["END"]) if "END" in info else pos + abs(int(info["SVLEN"].split(",")[0]))
            sv = SV(x[0], pos + 1, end + 1, kind)
            length = end - pos
        if sv.start < region.start or sv.end > region.end:
            continue
        records.append({"sv": sv, "type": kind, "pos": pos, "length": length, "genotype": gt, "haps": haps})
    return records, skipped


def read_svs(vcf: Path, sample: str, region: Region) -> tuple[list[SV], list[SV], int]:
    """Returns the variants of the two haplotypes of the sample in the region and the number of
    records the sample carries there that are skipped for an unsupported type.

    @raises UnknownSample when the sample is not a column of the VCF
    """
    records, skipped = sample_variants(vcf, sample, region)
    first = [r["sv"] for r in records if r["haps"][0]]
    second = [r["sv"] for r in records if r["haps"][1]]
    return sorted(first, key=lambda s: s.start), sorted(second, key=lambda s: s.start), sum(skipped.values())
