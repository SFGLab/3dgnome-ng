"""The genes of a cell type by name: the suggestions of the region field and the names of Ensembl ids."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .model import Catalog, CellType
from .shapes import GeneHit


@lru_cache(maxsize=None)
def _by_name(table: Path) -> list[tuple[str, str, str, int]]:
    """Each gene name of a gene table once: the name in capitals, the name, the chromosome and the TSS."""
    genes = pd.read_csv(table, sep="\t").drop_duplicates("name")
    return sorted((str(n).upper(), str(n), str(c), int(t)) for n, c, t in zip(genes["name"], genes["chrom"], genes["tss"], strict=True))


def find(cell: CellType, text: str, limit: int) -> list[GeneHit]:
    """Genes of a cell type whose name starts with the text, then those that hold it. Each has
    its name, chromosome and TSS. A text under two letters finds none."""
    text = text.strip().upper()
    if len(text) < 2:
        return []
    names = _by_name(cell.path("genes"))
    limit = max(1, min(limit, 25))
    starts = [g for g in names if g[0].startswith(text)]
    # The shortest names first. MYC comes before MYCBP.
    starts.sort(key=lambda g: (len(g[0]), g[0]))
    holds = [g for g in names if text in g[0] and not g[0].startswith(text)] if len(starts) < limit else []
    return [{"name": n, "chrom": c, "tss": t} for _, n, c, t in (starts + holds)[:limit]]


@lru_cache(maxsize=None)
def _names(table: Path) -> dict[str, str]:
    g = pd.read_csv(table, sep="\t")
    return dict(zip(g["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True), g["name"].astype(str), strict=True))


def names_by_id(catalog: Catalog) -> dict[str, str]:
    """The name of each gene by its Ensembl id without the version, from the gene table of the first cell type."""
    return _names(next(iter(catalog.cell_types.values())).path("genes"))
