"""The hosted loops and genes of a region, for the drawing of the form."""

from __future__ import annotations

import pandas as pd

from ..regions import Region
from .model import CellType
from .shapes import Preview


def around(cell: CellType, r: Region) -> Preview:
    """Returns the loops of each factor inside a region and its genes. A loop is the two anchor
    middles and the count, then the outer edges, which decide whether a region holds the loop."""
    loops: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for factor, name in cell.clusters.items():
        t = pd.read_csv(cell.root / name, sep=r"\s+", header=None, usecols=range(7), comment="#")
        t = t[(t[0] == r.chrom) & (t[3] == r.chrom) & (t[1] >= r.start) & (t[5] <= r.end)]
        loops[factor] = [(int((a + b) // 2), int((c + e) // 2), int(n), int(a), int(e)) for a, b, c, e, n in zip(t[1], t[2], t[4], t[5], t[6], strict=True)]
    # A gene is in the region when its span reaches into it.
    g = cell.genes()
    g = g[(g["chrom"] == r.chrom) & (g["end"] > r.start) & (g["start"] < r.end)]
    return {
        "loops": loops,
        "counts": {k: len(v) for k, v in loops.items()},
        "genes": [
            {"name": str(n), "tss": int(t), "start": int(s), "end": int(e), "strand": str(o)}
            for n, t, s, e, o in zip(g["name"], g["tss"], g["start"], g["end"], g["strand"], strict=True)
        ],
    }
