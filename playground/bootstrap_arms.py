"""How much of the gap between two arms survives resampling the genes.

With twenty odd genes per contrast a difference in Spearman can be noise. This resamples the
common gene set with replacement, recomputes every arm on the same resampled genes, and reports
how often each arm beats the others. Paired, so the resampling noise is shared.

    python bootstrap_arms.py <name>=<table.parquet> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path.home() / "enhancer3d"
sys.path.insert(0, str(REPO / "playground"))

import beyond_linear as BL  # noqa: E402

N_BOOT = 2000
arms = [a.split("=", 1) for a in sys.argv[1:]]
tables = {n: pd.read_parquet(p) for n, p in arms}
names = list(tables)

genes = pd.read_parquet(REPO / "playground" / "genome_distances" / "genes_all.parquet")[
    ["gene_id", "gene_chr", "gene_tss"]
]
v4 = pd.read_parquet(BL.V4_TABLE)
rng = np.random.default_rng(0)

pooled: dict[str, list[np.ndarray]] = {n: [] for n in names}
per_contrast = []
for (c1, c2), fname in BL.DESEQ.items():
    if not all({c1, c2} <= set(t["cell_line"].unique()) for t in tables.values()):
        continue
    deseq = pd.read_parquet(BL.DATA / "deseq" / fname)
    lin = BL.linear_distances(genes, [c1, c2])
    pinned = set(BL.build_pairs(v4, lin, genes, c1, c2, deseq, "min_dist", 2.0, 0.05).index)
    frames = {
        n: BL.build_pairs(t, lin, genes, c1, c2, deseq, "min_dist", 2.0, 0.05, pinned)
        for n, t in tables.items()
    }
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if len(common) < 10:
        continue
    cols = {n: frames[n].loc[common, ["d3_diff", "log2FoldChange"]] for n in names}
    boots = {n: np.empty(N_BOOT) for n in names}
    for b in range(N_BOOT):
        idx = rng.integers(0, len(common), len(common))
        for n in names:
            d = cols[n].to_numpy()[idx]
            boots[n][b] = abs(spearmanr(d[:, 0], d[:, 1]).statistic)
    for n in names:
        pooled[n].append(boots[n])
    row = {"contrast": f"{c1[:3]}/{c2[:3]}", "n": len(common)}
    for n in names:
        row[n] = float(np.mean(boots[n]))
    per_contrast.append(row)

print(f"bootstrap of |Spearman 3D|, {N_BOOT} resamples, paired across arms\n")
print(pd.DataFrame(per_contrast).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

mean = {n: np.mean(pooled[n], axis=0) for n in names}
print("\nmean over contrasts, per resample:")
for n in names:
    lo, hi = np.percentile(mean[n], [2.5, 97.5])
    print(f"  {n:>8s}: {mean[n].mean():.3f}   95 percent interval [{lo:.3f}, {hi:.3f}]")
print("\nhow often the first arm beats each other, over resamples:")
first = names[0]
for n in names[1:]:
    d = mean[first] - mean[n]
    print(f"  {first} > {n}: {100 * (d > 0).mean():5.1f} percent    median gap {np.median(d):+.3f}")
