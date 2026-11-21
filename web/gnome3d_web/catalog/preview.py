"""The hosted loops and genes of a region, for the drawing of the form."""

from __future__ import annotations

import pandas as pd

from ..regions import Region
from .model import CellType
from .shapes import Preview


def around(cell: CellType, r: Region) -> Preview:
    """The loops of each factor inside a region and its genes. A loop is the two anchor middles
    and the count, then the outer edges, which decide whether a region holds the loop."""
    loops: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for factor, name in cell.clusters.items():
        t = pd.read_csv(cell.root / name, sep=r"\s+", header=None, usecols=range(7), comment="#")
        t = t[(t[0] == r.chrom) & (t[3] == r.chrom) & (t[1] >= r.start) & (t[5] <= r.end)]
        loops[factor] = [(int((a + b) // 2), int((c + e) // 2), int(n), int(a), int(e)) for a, b, c, e, n in zip(t[1], t[2], t[4], t[5], t[6], strict=True)]
    g = cell.genes()
    g = g[(g["chrom"] == r.chrom) & (g["tss"] >= r.start) & (g["tss"] < r.end)]
    return {
        "loops": loops,
        "counts": {k: len(v) for k, v in loops.items()},
        "genes": [{"name": str(n), "tss": int(t)} for n, t in zip(g["name"], g["tss"], strict=True)],
    }
