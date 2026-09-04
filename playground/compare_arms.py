"""Compare enhancer promoter distance tables on one common gene set.

The tuned models cover chr1 only and every stored baseline covers the genome, so the per
contrast numbers are not comparable as they stand. This reuses `beyond_linear.build_pairs`,
already pinned to the v4 gene set, then keeps only the genes every arm has in common for that
contrast and recomputes the same two statistics: Spearman of the 3D distance change against the
expression change, and the partial that controls the linear genomic distance change.

    python compare_arms.py <name>=<table.parquet> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO = Path.home() / "enhancer3d"
sys.path.insert(0, str(REPO / "playground"))

import beyond_linear as BL  # noqa: E402

arms = [a.split("=", 1) for a in sys.argv[1:]]
tables: dict[str, pd.DataFrame] = {}
for name, path in arms:
    t = pd.read_parquet(path)
    tables[name] = t
    print(f"{name:>9s}: {len(t):7d} rows, {t['gene_id'].nunique():5d} genes, cells {sorted(t['cell_line'].unique())}")

genes = pd.read_parquet(REPO / "playground" / "genome_distances" / "genes_all.parquet")[
    ["gene_id", "gene_chr", "gene_tss"]
]
v4 = pd.read_parquet(BL.V4_TABLE)

rows = []
for (c1, c2), fname in BL.DESEQ.items():
    if c1 not in ("GM12878", "H1ESC") and c1 != "H1ESC":
        continue
    if not all({c1, c2} <= set(t["cell_line"].unique()) for t in tables.values()):
        continue
    deseq = pd.read_parquet(BL.DATA / "deseq" / fname)
    lin = BL.linear_distances(genes, [c1, c2])
    pinned = set(BL.build_pairs(v4, lin, genes, c1, c2, deseq, "min_dist", 2.0, 0.05).index)
    frames = {
        n: BL.build_pairs(t, lin, genes, c1, c2, deseq, "min_dist", 2.0, 0.05, pinned)
        for n, t in tables.items()
    }
    common = set.intersection(*(set(f.index) for f in frames.values()))
    for n, f in frames.items():
        g = f[f.index.isin(common)]
        if len(g) < 8:
            rows.append({"arm": n, "contrast": f"{c1[:3]}/{c2[:3]}", "n": len(g)})
            continue
        r3 = spearmanr(g["d3_diff"], g["log2FoldChange"])
        ry = BL.rank_residual(g["log2FoldChange"].to_numpy(), g["lin_diff"].to_numpy())
        rx = BL.rank_residual(g["d3_diff"].to_numpy(), g["lin_diff"].to_numpy())
        rp = spearmanr(rx, ry)
        rows.append(
            {
                "arm": n,
                "contrast": f"{c1[:3]}/{c2[:3]}",
                "n": len(g),
                "spearman_3d": float(r3.statistic),
                "p_3d": float(r3.pvalue),
                "partial": float(rp.statistic),
                "p_partial": float(rp.pvalue),
            }
        )

out = pd.DataFrame(rows)
if out.empty or "spearman_3d" not in out:
    print("\nno contrast had enough common genes")
    print(out.to_string(index=False))
    raise SystemExit(1)

print(f"\ncommon genes per contrast, every arm scored on the same set:\n")
print(out.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print("\nmean over contrasts:")
g = out.dropna(subset=["spearman_3d"]).groupby("arm").agg(
    mean_abs_3d=("spearman_3d", lambda s: s.abs().mean()),
    mean_abs_partial=("partial", lambda s: s.abs().mean()),
    contrasts=("contrast", "size"),
)
print(g.to_string(float_format=lambda v: f"{v:.3f}"))
