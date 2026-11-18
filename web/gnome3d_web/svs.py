"""One sample's structural variants in a region, from any VCF with SVTYPE in its INFO.

Variants follow personal3d's convention, 1 based, start the first affected base and end one
past the last. A deletion, an inversion and a duplication run from the base after POS to END, an
insertion goes in after POS with the ALT sequence, or with N repeated SVLEN times when ALT is
symbolic. A duplication is read as tandem, the copy right after the original, which is how the
1000 Genomes panel calls them. Other types are skipped and counted. A phased genotype puts each allele on its haplotype; an
unphased one puts the variant on the first. Only variants wholly inside the region are kept,
the rule the rewiring rules take.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from personal3d.variants import SV

from .regions import Region

KINDS = ("DEL", "INS", "INV", "DUP")


def _lines(vcf: Path, region: Region) -> Iterator[str]:
    """The VCF's header line and the records of the region's chromosome, through tabix when
    the file is indexed."""
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


def read_svs(vcf: Path, sample: str, region: Region) -> tuple[list[SV], list[SV], int]:
    """The sample's two haplotypes' variants in the region and the number of records the sample
    carries there that are skipped for an unsupported type."""
    hap: tuple[list[SV], list[SV]] = ([], [])
    skipped = 0
    col = -1
    for line in _lines(vcf, region):
        x = line.split("\t")
        if line.startswith("#"):
            col = x.index(sample)
            continue
        info = dict(kv.split("=", 1) if "=" in kv else (kv, "") for kv in x[7].split(";"))
        kind = info.get("SVTYPE", "")
        pos = int(x[1])
        carried = _alleles(x[col].split(":", 1)[0])
        if not any(carried):
            continue
        if kind not in KINDS:
            if x[0] == region.chrom and region.start <= pos < region.end:
                skipped += 1
            continue
        if kind == "INS":
            alt = x[4]
            seq = alt[1:] if not alt.startswith("<") else "N" * abs(int(info.get("SVLEN", "0").split(",")[0]))
            sv = SV(x[0], pos + 1, pos + 1, "INS", seq)
        else:
            end = int(info["END"]) if "END" in info else pos + abs(int(info["SVLEN"].split(",")[0]))
            sv = SV(x[0], pos + 1, end + 1, kind)
        if not (sv.chrom == region.chrom and sv.start >= region.start and sv.end <= region.end):
            continue
        for h, on in enumerate(carried):
            if on:
                hap[h].append(sv)
    return sorted(hap[0], key=lambda s: s.start), sorted(hap[1], key=lambda s: s.start), skipped


def sample_variants(vcf: Path, sample: str, region: Region) -> tuple[list[dict[str, object]], dict[str, int]]:
    """The records of one sample inside the region with their genotype and the haplotypes that
    carry them and the count of records of other types the sample carries, by type.

    @raises KeyError when the sample is not a column of the VCF
    """
    records: list[dict[str, object]] = []
    skipped: dict[str, int] = {}
    col = -1
    for line in _lines(vcf, region):
        x = line.split("\t")
        if line.startswith("#"):
            if sample not in x[9:]:
                raise KeyError(sample)
            col = x.index(sample)
            continue
        info = dict(kv.split("=", 1) if "=" in kv else (kv, "") for kv in x[7].split(";"))
        kind = info.get("SVTYPE", "")
        pos = int(x[1])
        gt = x[col].split(":", 1)[0]
        haps = _alleles(gt)
        if not any(haps) or not (x[0] == region.chrom and region.start <= pos < region.end):
            continue
        if kind not in KINDS:
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
        if sv.end > region.end:
            continue
        records.append({"sv": sv, "type": kind, "pos": pos, "length": length, "genotype": gt, "haps": haps})
    return records, skipped
