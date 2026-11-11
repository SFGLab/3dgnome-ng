"""A panel's consensus loops, the reference network the rules rewrite for one person.

The anchors are the union of the panel people's anchors as the conf arm's models take them,
`<S>_conf_anchors_ctcf_rnapol2.bed`, merged where they overlap. A merged anchor is R or L when
every CTCF part of it agrees and N otherwise, as the rules read a conflict. Each person's
loops of one factor, the CTCF loops of `<S>_raw_clusters_3+.bedpe` and the RNAPOL2 loops of
`<S>_conf_rnapol2_clusters_3+.bedpe`, are placed on the merged anchors by their ends'
midpoints. A loop is in the panel when at least min_people of the panel have it, with the mean
of its PET counts over the panel, a person without it counting zero.

The same placement gives every person's measured loops on the merged anchors, which is what
the rules' predictions are read against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = {"CTCF": "{s}_raw_clusters_3+.bedpe", "RNAPOL2": "{s}_conf_rnapol2_clusters_3+.bedpe"}
ANCHORS = "{s}_conf_anchors_ctcf_rnapol2.bed"


@dataclass
class Panel:
    """Merged anchors, indexed from 0 within the frame, and per factor the loops of every
    person as anchor index pairs with PET counts, and the panel's consensus."""

    people: list[str]
    anchors: pd.DataFrame
    measured: dict[str, dict[str, pd.DataFrame]]

    def consensus(self, factor: str, members: list[str], min_people: int = 2) -> pd.DataFrame:
        """The loops at least min_people of members have, i j n weight."""
        x = pd.concat([self.measured[factor][s].assign(person=s) for s in members])
        g = x.groupby(["i", "j"]).agg(n=("person", "nunique"), pets=("pets", "sum")).reset_index()
        g["weight"] = g.pets / len(members)
        return g[g.n >= min_people].drop(columns="pets").reset_index(drop=True)

    def clusters(self, loops: pd.DataFrame) -> list[list]:
        """Loops as the rules' nine columns, rounded strengths, at least 1."""
        a = self.anchors
        i, j = loops.i.to_numpy(), loops.j.to_numpy()
        w = np.maximum(1, np.rint(loops.weight.to_numpy())).astype(int)
        return [[a.chrom.iat[x], int(a.start.iat[x]), int(a.end.iat[x]), a.chrom.iat[y], int(a.start.iat[y]), int(a.end.iat[y]), a.orient.iat[x], a.orient.iat[y], int(v)] for x, y, v in zip(i, j, w, strict=True)]

    def index_of(self, chrom: str, interval: tuple[int, int]) -> int:
        """The merged anchor with exactly this reference interval."""
        k = self._where.get((chrom, interval[0], interval[1]))
        if k is None:
            raise KeyError((chrom, interval))
        return k

    def __post_init__(self) -> None:
        a = self.anchors
        self._where = {(c, s, e): k for k, (c, s, e) in enumerate(zip(a.chrom, a.start, a.end, strict=True))}


def _merge(parts: pd.DataFrame) -> pd.DataFrame:
    out = []
    for chrom, g in parts.sort_values(["chrom", "start"]).groupby("chrom", sort=False):
        st = en = None
        ori: set = set()
        for r in g.itertuples(index=False):
            if st is not None and r.start <= en:
                en = max(en, r.end)
                ori.add(r.orient)
                continue
            if st is not None:
                out.append((chrom, st, en, ori))
            st, en, ori = r.start, r.end, {r.orient}
        out.append((chrom, st, en, ori))
    orient = [("L" if o - {"N"} == {"L"} else "R" if o - {"N"} == {"R"} else "N") for _, _, _, o in out]
    return pd.DataFrame({"chrom": [x[0] for x in out], "start": [x[1] for x in out], "end": [x[2] for x in out], "orient": orient})


def build(data_dir: Path, people: list[str], chrom: str | None = None) -> Panel:
    """The merged anchors of people and each person's loops on them, from the engine's input
    directories data_dir/<S>/, one chromosome or all."""
    parts = []
    for s in people:
        a = pd.read_csv(data_dir / s / ANCHORS.format(s=s), sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "orient"])
        parts.append(a if chrom is None else a[a.chrom == chrom])
    anchors = _merge(pd.concat(parts))
    measured: dict[str, dict[str, pd.DataFrame]] = {f: {} for f in FACTORS}
    starts = {c: g.start.to_numpy() for c, g in anchors.groupby("chrom")}
    ends = {c: g.end.to_numpy() for c, g in anchors.groupby("chrom")}
    first = {c: int(g.index[0]) for c, g in anchors.groupby("chrom")}

    def place(c: np.ndarray, mid: np.ndarray) -> np.ndarray:
        out = np.full(len(mid), -1)
        for chrom_, k in pd.Series(np.arange(len(mid))).groupby(c):
            if chrom_ not in starts:
                continue
            m = mid[k.to_numpy()]
            i = np.searchsorted(starts[chrom_], m, side="right") - 1
            ok = (i >= 0) & (m < ends[chrom_][np.clip(i, 0, None)])
            out[k.to_numpy()[ok]] = i[ok] + first[chrom_]
        return out

    for f, name in FACTORS.items():
        for s in people:
            b = pd.read_csv(data_dir / s / name.format(s=s), sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5, 6])
            if chrom is not None:
                b = b[b[0] == chrom]
            b = b[b[0] == b[3]]
            i = place(b[0].to_numpy(), ((b[1] + b[2]) // 2).to_numpy())
            j = place(b[3].to_numpy(), ((b[4] + b[5]) // 2).to_numpy())
            ok = (i >= 0) & (j >= 0) & (i != j)
            lo, hi = np.minimum(i, j)[ok], np.maximum(i, j)[ok]
            x = pd.DataFrame({"i": lo, "j": hi, "pets": b[6].to_numpy()[ok]})
            measured[f][s] = x.groupby(["i", "j"], as_index=False).pets.sum()
    return Panel(people, anchors.reset_index(drop=True), measured)
