"""The significance tests the RNAPII figures mark with stars.

A test is made only where a panel compares two sets of many values. The three cell lines of the
dumbbell panels give one value a line, three values a comparison, and no test of three paired
values can reach 0.05, so those panels carry none.

Every p value is two sided. The pairs and genes of one chromosome are not independent draws,
and the nine trio individuals are three families, so the values are a guide to which
differences are larger than their spread and not a claim about a population.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

LEGEND = "* p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant, two sided"


def stars(p: float) -> str:
    """The star mark of a p value."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def mark(p: float) -> str:
    """The star mark with the p value written out, as it stands in a figure."""
    if p < 0.001:
        return f"{stars(p)}, p < 0.001"
    if p > 0.99:
        return f"{stars(p)}, p > 0.99"
    return f"{stars(p)}, p = {p:.3f}" if p < 0.1 else f"{stars(p)}, p = {p:.2f}"


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Wilcoxon signed rank test of two measurements of the same items.

    Returns the p value and the number of items measured in both. Exact for fewer than 26 items.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.all(a == b):
        return float("nan"), int(len(a))
    return float(wilcoxon(a, b, method="exact" if len(a) < 26 else "auto").pvalue), int(len(a))


def unpaired(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U test of two sets of values."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def ep_tests(ep: pd.DataFrame, line: str) -> dict[str, float]:
    """The enhancer to promoter distances of one line's RNAPII only pairs.

    `models` is the pairs with RNAPII loops against the same pairs without, paired. `reference`
    is the pairs with RNAPII loops against the reference pairs in the same models, unpaired.
    """
    e = ep[ep.line == line]
    only = e[e.set == "rnapii_only"]
    c, r = only[only.arm == "C"], only[only.arm == "R"]
    if not (c[["gene_id", "sep"]].to_numpy() == r[["gene_id", "sep"]].to_numpy()).all():
        raise ValueError(f"{line}: the pairs of the two model sets are not in one order")
    p, n = paired(c.distance.to_numpy(), r.distance.to_numpy())
    ctrl = e[(e.set == "control") & (e.arm == "R")]
    return {"models": p, "n": n, "reference": unpaired(r.distance.to_numpy(), ctrl.distance.to_numpy())}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def link_difference(e3d_root: Path, a: str, b: str, n: int = 2000, seed: int = 1) -> dict[str, tuple[float, float, int]]:
    """Whether the link to expression differs between two model sets, per line pair.

    The genes with a value in both sets are resampled in blocks over their cluster, the same
    resample for both sets, and the difference of the two correlations is taken on each draw.
    Returns per pair the difference of minus the correlation, set b less set a, on all genes,
    the p value as twice the smaller share of draws on either side of zero, never under one
    over the draws, and the number of genes.
    """
    out = {}
    for path in sorted((e3d_root / f"e3d_{a}").glob("genes_*_to_*.tsv")):
        x, y = path.stem.removeprefix("genes_").split("_to_")
        ga = pd.read_csv(path, sep="\t")
        gb = pd.read_csv(e3d_root / f"e3d_{b}" / path.name, sep="\t")
        g = ga.merge(gb[["gene_id", "total"]], on="gene_id", suffixes=("_a", "_b"))
        g = g[np.isfinite(g.total_a) & np.isfinite(g.total_b) & np.isfinite(g.log2fc)]
        ta, tb, fc = g.total_a.to_numpy(), g.total_b.to_numpy(), g.log2fc.to_numpy()
        cl = pd.factorize(g.cluster)[0]
        groups = [np.flatnonzero(cl == k) for k in range(cl.max() + 1)]
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n):
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            if fc[idx].std() > 0 and ta[idx].std() > 0 and tb[idx].std() > 0:
                diffs.append(_spearman(ta[idx], fc[idx]) - _spearman(tb[idx], fc[idx]))
        d = np.asarray(diffs)
        p = max(2 * min(float((d <= 0).mean()), float((d >= 0).mean())), 1 / len(d))
        out[f"{x} vs {y}"] = (_spearman(ta, fc) - _spearman(tb, fc), min(p, 1.0), int(len(g)))
    return out
