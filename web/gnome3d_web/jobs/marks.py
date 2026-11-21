"""Promoters and enhancers on the beads of a model and their distances.

A TSS or an enhancer centre takes the bead with the nearest genomic midpoint when it is within
5 kb. The enhancer distance of a gene is the ensemble mean 3D distance to the nearest marked
enhancer within 1 Mb of its TSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_GAP = 5_000
MAX_LINEAR = 1_000_000


def _no_marks() -> dict[str, int]:
    return {}


def _no_beads() -> list[int]:
    return []


@dataclass
class Marks:
    promoters: dict[str, int] = field(default_factory=_no_marks)
    enhancers: list[int] = field(default_factory=_no_beads)
    enhancer_centres: list[int] = field(default_factory=_no_beads)


def _nearest(mids: np.ndarray, p: int) -> int | None:
    k = int(np.argmin(np.abs(mids - p)))
    return k if abs(int(mids[k]) - p) <= MAX_GAP else None


def mark_beads(chrom: str, starts: np.ndarray, ends: np.ndarray, genes: pd.DataFrame, enhancers: pd.DataFrame) -> Marks:
    """The bead of each TSS and enhancer centre on `chrom` within 5 kb of a bead midpoint."""
    mids = (starts + ends) // 2
    lo, hi = int(starts.min()), int(ends.max())
    m = Marks()
    for g in genes[genes["chrom"] == chrom].itertuples(index=False):
        if lo <= g.tss <= hi and (k := _nearest(mids, int(g.tss))) is not None:
            m.promoters[str(g.gene_id)] = k
    e = enhancers[enhancers[0] == chrom]
    for c in ((e[1] + e[2]) // 2).astype(int):
        if lo <= c <= hi and (k := _nearest(mids, int(c))) is not None:
            m.enhancers.append(k)
            m.enhancer_centres.append(int(c))
    return m


def ep_table(models: list[np.ndarray], marks: Marks, genes: pd.DataFrame, starts: np.ndarray, ends: np.ndarray) -> pd.DataFrame:
    """Per marked gene its nearest enhancer by ensemble mean 3D distance within 1 Mb."""
    mids = (starts + ends) // 2
    tss = genes.set_index("gene_id")
    rows = []
    enh = np.array(marks.enhancers, dtype=int)
    cen = np.array(marks.enhancer_centres, dtype=int)
    for gid, b in marks.promoters.items():
        g = tss.loc[gid]
        ok = np.abs(cen - int(g["tss"])) <= MAX_LINEAR
        ok &= enh != b
        if not ok.any():
            continue
        cand = enh[ok]
        d = np.mean([np.linalg.norm(P[cand] - P[b], axis=1) for P in models], axis=0)
        k = int(np.argmin(d))
        rows.append(
            {
                "gene_id": gid,
                "name": g["name"],
                "tss": int(g["tss"]),
                "promoter_bead": b,
                "enhancer_bead": int(cand[k]),
                "enhancer_centre": int(cen[ok][k]),
                "linear_bp": abs(int(cen[ok][k]) - int(g["tss"])),
                "distance": float(d[k]),
                "bead_mid": int(mids[b]),
            }
        )
    return pd.DataFrame(rows, columns=["gene_id", "name", "tss", "promoter_bead", "enhancer_bead", "enhancer_centre", "linear_bp", "distance", "bead_mid"])
