"""Enhancer to promoter pairs held by RNAPII loops alone, and their 3D distance in the models.

Panel d of `design/2026-11-12-rnapii-figures.md`. A pair is held by a loop when the promoter,
the TSS plus or minus 1 kb, and the enhancer each lie within `pad` of opposite ends of that
loop, the rule of `enhancer3d.annotations.annotate_pairs`. The control pairs are held by no loop
of either factor and match each RNAPII-only pair's separation within 10 percent.

    python playground/rnapii/ep_pairs.py <LINE> <models_root> <ctcf.bedpe> <rnapii.bedpe> \
        <atlas.bed> <genes.tsv> <common_genes.tsv> <out.tsv>

`models_root/<LINE>_<ARM>/chr1/*.beads.tsv` are read for arms C and R. Writes line, set, arm,
gene_id, sep and distance, and prints the pair count per set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHROM = "chr1"
PROMOTER_BP = 1000


def read_loops(path: Path, chrom: str = CHROM) -> pd.DataFrame:
    """Intra chromosome loops of one chromosome, first end left of the second."""
    t = pd.read_csv(path, sep="\t", header=None, usecols=range(6))
    t.columns = ["c1", "s1", "e1", "c2", "s2", "e2"]
    t = t[(t.c1 == chrom) & (t.c2 == chrom)]
    swap = t.s1 > t.s2
    t.loc[swap, ["s1", "e1", "s2", "e2"]] = t.loc[swap, ["s2", "e2", "s1", "e1"]].to_numpy()
    return t[["s1", "e1", "s2", "e2"]].reset_index(drop=True)


def candidate_pairs(enh: pd.DataFrame, tss: pd.DataFrame, max_sep: int = 1_000_000) -> pd.DataFrame:
    """Every enhancer and promoter on the chromosome within `max_sep`, as gene_id, tss, enh_s,
    enh_e and sep."""
    mids = ((enh.start + enh.end) // 2).to_numpy()
    order = np.argsort(mids)
    es, ee, em = enh.start.to_numpy()[order], enh.end.to_numpy()[order], mids[order]
    rows = []
    for g, t in zip(tss.gene_id, tss.tss, strict=True):
        lo, hi = np.searchsorted(em, t - max_sep), np.searchsorted(em, t + max_sep)
        for k in range(lo, hi):
            if abs(em[k] - t) > 2 * PROMOTER_BP:
                rows.append((g, t, es[k], ee[k], abs(em[k] - t)))
    return pd.DataFrame(rows, columns=["gene_id", "tss", "enh_s", "enh_e", "sep"])


def held(pairs: pd.DataFrame, loops: pd.DataFrame, pad: int = 2500) -> np.ndarray:
    """Whether each pair is held directly by one of `loops`."""
    if pairs.empty or loops.empty:
        return np.zeros(len(pairs), bool)
    ps, pe = (pairs.tss - PROMOTER_BP).to_numpy()[:, None], (pairs.tss + PROMOTER_BP).to_numpy()[:, None]
    es, ee = pairs.enh_s.to_numpy()[:, None], pairs.enh_e.to_numpy()[:, None]
    out = np.zeros(len(pairs), bool)

    def near(a: np.ndarray, b: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
        return (a <= e + pad) & (b >= s - pad)

    for i0 in range(0, len(pairs), 2000):
        sl = slice(i0, i0 + 2000)
        s1, e1, s2, e2 = (loops[c].to_numpy()[None, :] for c in ("s1", "e1", "s2", "e2"))
        a = near(ps[sl], pe[sl], s1, e1) & near(es[sl], ee[sl], s2, e2)
        b = near(es[sl], ee[sl], s1, e1) & near(ps[sl], pe[sl], s2, e2)
        out[sl] = (a | b).any(1)
    return out


def matched_control(target: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator, per_pair: int = 5) -> pd.DataFrame:
    """Up to `per_pair` pool pairs per target pair with separation within 10 percent."""
    pool = pool.sort_values("sep").reset_index(drop=True)
    seps = pool.sep.to_numpy()
    take: list[int] = []
    for s in target.sep:
        lo, hi = np.searchsorted(seps, 0.9 * s), np.searchsorted(seps, 1.1 * s)
        if hi > lo:
            take.extend(rng.choice(np.arange(lo, hi), size=min(per_pair, hi - lo), replace=False))
    return pool.loc[sorted(set(take))].reset_index(drop=True)


def read_beads(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Bead midpoints and coordinates of one conformation."""
    t = pd.read_csv(path, sep="\t")
    mid = ((t["start"] + t["end"]) // 2).to_numpy()
    order = np.argsort(mid)
    return mid[order], t[["x", "y", "z"]].to_numpy(float)[order]


def _nearest(mid: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.clip(np.searchsorted(mid, pos), 1, len(mid) - 1)
    left = np.abs(pos - mid[j - 1]) <= np.abs(mid[j] - pos)
    k = np.where(left, j - 1, j)
    return k, np.abs(mid[k] - pos)


def pair_distance(model_dir: Path, pairs: pd.DataFrame, max_gap: int = 5000) -> np.ndarray:
    """Mean over the conformations in `model_dir` of the distance between the beads nearest the
    TSS and the enhancer midpoint, NaN where either is farther than `max_gap` from every bead."""
    files = sorted(model_dir.glob("*.beads.tsv"))
    if not files:
        raise ValueError(f"no beads files under {model_dir}")
    t = pairs.tss.to_numpy()
    c = ((pairs.enh_s + pairs.enh_e) // 2).to_numpy()
    acc = []
    for f in files:
        mid, xyz = read_beads(f)
        kt, gt = _nearest(mid, t)
        kc, gc = _nearest(mid, c)
        d = np.linalg.norm(xyz[kt] - xyz[kc], axis=1)
        d[(gt > max_gap) | (gc > max_gap)] = np.nan
        acc.append(d)
    return np.nanmean(np.vstack(acc), axis=0) if acc else np.full(len(pairs), np.nan)


def main() -> None:
    line, models, ctcf, rnapii, atlas, genes, common, out = sys.argv[1:9]
    rng = np.random.default_rng(20261112)
    g = pd.read_csv(genes, sep="\t")
    keep = set(pd.read_csv(common, sep="\t").gene_id)
    g = g[(g.chrom == CHROM) & g.gene_id.isin(keep)]
    tss = pd.DataFrame({"gene_id": g.gene_id, "tss": np.where(g.strand == "+", g.start, g.end)})
    e = pd.read_csv(atlas, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    e = e[e.chrom == CHROM].drop_duplicates()
    pairs = candidate_pairs(e, tss)
    on_r = held(pairs, read_loops(Path(rnapii)))
    on_c = held(pairs, read_loops(Path(ctcf)))
    only = pairs[on_r & ~on_c].reset_index(drop=True)
    ctrl = matched_control(only, pairs[~on_r & ~on_c], rng)
    rows = []
    for name, p in (("rnapii_only", only), ("control", ctrl)):
        print(f"{line} {name}: {len(p)} pairs")
        for arm in ("C", "R"):
            d = pair_distance(Path(models) / f"{line}_{arm}" / CHROM, p)
            rows.append(pd.DataFrame({"line": line, "set": name, "arm": arm, "gene_id": p.gene_id, "sep": p.sep, "distance": d}))
    pd.concat(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
