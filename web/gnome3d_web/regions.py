"""A region from the form, coordinates or a gene name."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

MAX_SPAN = 20_000_000
# For a request the site has accepted. The limit of the site may have changed since.
ANY_SPAN = 1 << 62
# HiCDiffusion predicts 2,097,152 bp of the haplotype. The margin leaves room for insertions.
HICDIFFUSION_SPAN = 2_000_000
GENE_WINDOW = 2_000_000


class RegionError(ValueError):
    """A region the site will not model, with the reason as its message."""


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"

    def shown(self) -> str:
        """The region for a reader, its positions with thousands separators."""
        return f"{self.chrom}:{self.start:,}-{self.end:,}"


def read_region(text: str) -> Region | None:
    """A region the site stated itself, `chr:start-end`, commas allowed. None for any other text.
    The chromosome of a job on a haplotype can have a name the form does not take."""
    m = re.fullmatch(r"(chr\w+):(\d+)-(\d+)", text.replace(",", ""))
    return Region(m[1], int(m[2]), int(m[3])) if m else None


def parse_region(text: str, genes: pd.DataFrame, max_span: int = MAX_SPAN) -> Region:
    """`chr:start-end`, commas allowed, or a gene name, which gives a 2 Mb window centred on its
    TSS. Spans over `max_span` bp are refused."""
    t = text.strip().replace(",", "")
    m = re.fullmatch(r"(chr[0-9XYM]+):(\d+)-(\d+)", t)
    if m:
        r = Region(m.group(1), int(m.group(2)), int(m.group(3)))
    else:
        hit = genes[genes["name"].str.upper() == t.upper()]
        if hit.empty:
            raise RegionError(f"{text!r} is neither chr:start-end nor a known gene name")
        g = hit.iloc[0]
        start = max(0, int(g["tss"]) - GENE_WINDOW // 2)
        r = Region(str(g["chrom"]), start, start + GENE_WINDOW)
    if r.end <= r.start:
        raise RegionError(f"{text!r} ends before it starts")
    if r.end - r.start > max_span:
        raise RegionError(f"{text!r} spans {r.end - r.start:,} bp, over the {max_span:,} bp limit")
    return r
