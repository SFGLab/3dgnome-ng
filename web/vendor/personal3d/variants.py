"""A person's structural variants per haplotype, from the HGSVC freeze 4 calls.

The insertions and deletions come from `variants_freeze4_sv_insdel_alt.vcf.gz`, which carries
the sequences, so an inserted CTCF motif can be called. The genotypes are phased, and the
first and second alleles are the two haplotypes. The inversions come from
`variants_freeze4_sv_inv.tsv.gz`, which names the carriers but not their phase, so a carrier's
inversion is put on the first haplotype. The VCF names the YRI trio NA, the lab GM.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

INSDEL = "variants_freeze4_sv_insdel_alt.vcf.gz"
INV = "variants_freeze4_sv_inv.tsv.gz"
MIN_LEN = 50


def vcf_name(person: str) -> str:
    return "NA" + person[2:] if person.startswith("GM") else person


@dataclass(frozen=True)
class SV:
    """One variant on one haplotype, in reference coordinates, 1 based, start the first base
    and end one past the last. A deletion and an inversion cover start to end, an insertion
    goes in before start with its sequence."""

    chrom: str
    start: int
    end: int
    kind: str
    seq: str = ""

    @property
    def length(self) -> int:
        return len(self.seq) if self.kind == "INS" else self.end - self.start


def _alleles(gt: str) -> tuple[bool, bool]:
    a = gt.replace("/", "|").split("|")
    if len(a) != 2:
        return False, False
    return a[0] not in ("0", "."), a[1] not in ("0", ".")


def read_insdel(path: Path, people: list[str], chrom: str | None = None, min_len: int = MIN_LEN) -> dict[str, tuple[list[SV], list[SV]]]:
    """Each person's insertions and deletions of min_len or more, per haplotype."""
    out: dict[str, tuple[list[SV], list[SV]]] = {p: ([], []) for p in people}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            x = line.rstrip("\n").split("\t")
            if line.startswith("#"):
                col = {name: i for i, name in enumerate(x)}
                idx = {p: col[vcf_name(p)] for p in people}
                continue
            if chrom is not None and x[0] != chrom:
                continue
            info = dict(kv.split("=", 1) for kv in x[7].split(";") if "=" in kv)
            if abs(int(info["SVLEN"].split(",")[0])) < min_len:
                continue
            kind, pos = info["SVTYPE"], int(x[1])
            if kind == "DEL":
                sv = SV(x[0], pos + 1, pos + len(x[3]), "DEL")
            elif kind == "INS":
                sv = SV(x[0], pos + 1, pos + 1, "INS", x[4][1:])
            else:
                continue
            for p in people:
                for h, on in enumerate(_alleles(x[idx[p]])):
                    if on:
                        out[p][h].append(sv)
    return out


def read_inversions(path: Path, people: list[str], chrom: str | None = None) -> dict[str, list[SV]]:
    """Each person's inversions, unphased."""
    v = pd.read_csv(path, sep="\t", usecols=["#CHROM", "POS", "END", "MERGE_SAMPLES"])
    if chrom is not None:
        v = v[v["#CHROM"] == chrom]
    out: dict[str, list[SV]] = {p: [] for p in people}
    for r in v.itertuples(index=False):
        carriers = set(str(r.MERGE_SAMPLES).split(","))
        for p in people:
            if vcf_name(p) in carriers:
                # The table's POS is 0 based, its ID names the 1 based start.
                out[p].append(SV(r[0], int(r.POS) + 1, int(r.END) + 1, "INV"))
    return out


def haplotypes(freeze4: Path, people: list[str], chrom: str | None = None, min_len: int = MIN_LEN) -> dict[str, tuple[list[SV], list[SV]]]:
    """Each person's two haplotypes' variants, sorted by start, inversions on the first."""
    hap = read_insdel(freeze4 / INSDEL, people, chrom, min_len)
    inv = read_inversions(freeze4 / INV, people, chrom)
    return {p: (sorted(hap[p][0] + inv[p], key=lambda s: (s.chrom, s.start)), sorted(hap[p][1], key=lambda s: (s.chrom, s.start))) for p in people}


def in_region(svs: list[SV], chrom: str, start: int, end: int) -> list[SV]:
    """The variants lying wholly inside a region, as the rules take them."""
    return [s for s in svs if s.chrom == chrom and s.start >= start and s.end <= end]
