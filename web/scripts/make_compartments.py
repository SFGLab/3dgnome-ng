"""Write a cell type's A and B compartments from its Hi-C map.

    python web/scripts/make_compartments.py <map uri> <genes.tsv> <centromeres.bed> > compartments.bed

The map uri is a cooler at one resolution, `file.mcool::/resolutions/100000`. Each chromosome arm
is taken alone. Its balanced map is divided by the mean of each diagonal, clipped at the 99.9
percentile and reduced by one and the eigenvector of the largest eigenvalue is the compartment
signal. The sign of an eigenvector is arbitrary. It is set so the side with more gene starts is
A. An arm is left out when the signal does not follow gene density or jumps from bin to bin.
A call there would be noise. The engine reads the labels as they are and needs no phasing
track for them.

The output is `chr start end label` with neighbouring bins of one label joined. One line of JSON
per arm goes to standard error: the bins used, the correlation with gene density, the correlation
of neighbouring bins and the share of A. The site image has what this needs, numpy and cooler.

    docker exec -i <container> python - /data/maps/GM12878/4DNFIXP4QG5B.mcool::/resolutions/100000 \\
        /data/cell_types/GM12878/genes.tsv /data/cell_types/GM12878/centromeres.bed \\
        < web/scripts/make_compartments.py > compartments.bed 2> compartments.stats.jsonl
"""

from __future__ import annotations

import json
import sys

import cooler
import numpy as np

#: Diagonals left out of the signal, the bin itself and its neighbour.
IGNORE_DIAGS = 2
#: An arm with fewer mapped bins than this is left out.
MIN_BINS = 30
#: The least correlation of neighbouring bins for a signal that is not noise.
MIN_LAG1 = 0.3
#: The least correlation with gene density for a sign that can be trusted.
MIN_GENES = 0.1


def signal(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The compartment signal of one arm.

    @param m the arm's balanced map, NaN in the bins the balancing left out
    @return the signal over the mapped bins and the indices of those bins
    """
    n = len(m)
    idx = np.nonzero(np.isfinite(m).sum(axis=1) > n // 4)[0]
    if len(idx) < MIN_BINS:
        return np.zeros(len(idx)), idx
    oe = np.ones_like(m)
    for d in range(IGNORE_DIAGS, n):
        i = np.arange(n - d)
        v = m[i, i + d]
        mean = np.nanmean(v) if np.isfinite(v).any() else 0.0
        if mean > 0:
            oe[i, i + d] = oe[i + d, i] = v / mean
    oe = oe[np.ix_(idx, idx)]
    oe[~np.isfinite(oe)] = 1.0
    oe = np.minimum(oe, np.percentile(oe, 99.9)) - 1.0
    w, vec = np.linalg.eigh((oe + oe.T) / 2)
    return vec[:, int(np.argmax(w))], idx


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else 0.0


def main() -> None:
    uri, genes_path, cen_path = sys.argv[1:4]
    c = cooler.Cooler(uri)
    res = int(c.binsize)
    tss: dict[str, list[int]] = {}
    with open(genes_path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            tss.setdefault(p[2], []).append(int(p[3]))
    cen = {p[0]: (int(p[1]), int(p[2])) for p in (line.split() for line in open(cen_path)) if len(p) >= 3}
    for chrom in c.chromnames:
        if chrom not in cen or chrom not in tss:
            continue
        size = int(c.chromsizes[chrom])
        m = c.matrix(balance=True).fetch(chrom)
        density = np.bincount(np.array(tss[chrom]) // res, minlength=len(m))[: len(m)].astype(float)
        for arm, lo, hi in (("p", 0, cen[chrom][0] // res), ("q", -(-cen[chrom][1] // res), len(m))):
            stats: dict[str, object] = {"chrom": chrom, "arm": arm, "bins": 0, "kept": False}
            if hi - lo >= MIN_BINS:
                v, idx = signal(m[lo:hi, lo:hi])
                stats["bins"] = len(idx)
                if len(idx) >= MIN_BINS:
                    g = corr(v, density[lo + idx])
                    v = v if g >= 0 else -v
                    near = np.diff(idx) == 1
                    lag1 = corr(v[:-1][near], v[1:][near])
                    kept = abs(g) >= MIN_GENES and lag1 >= MIN_LAG1
                    stats |= {"genes": round(abs(g), 3), "lag1": round(lag1, 3), "share_a": round(float((v > 0).mean()), 3), "kept": bool(kept)}
                    if kept:
                        rows: list[list[int | str]] = []
                        for i, x in zip(lo + idx, v, strict=True):
                            label = "A" if x > 0 else "B"
                            start = int(i) * res
                            if rows and rows[-1][2] == start and rows[-1][3] == label:
                                rows[-1][2] = min(start + res, size)
                            else:
                                rows.append([chrom, start, min(start + res, size), label])
                        for r in rows:
                            print(*r, sep="\t")
            print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
