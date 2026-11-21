"""Two jobs compared: the distance of each gene to its enhancer and the expression."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .. import expression
from ..jobs import store
from .views import MAX_GAP, annotations, cached, genes, nearest, positions


def _pair_distances(job: Path) -> dict[str, list[float]]:
    """The enhancer to promoter distance of every gene of a job in each of its structures."""

    def make() -> dict[str, list[float]]:
        pos, _ = positions(job)
        return {str(r["gene"]): [round(float(np.linalg.norm(p[r["promoter_bead"]] - p[r["enhancer_bead"]])), 3) for p in pos] for r in genes(job)}

    return cached(job, "pair_distances", make)


def _rank_p(x: list[float], y: list[float]) -> float | None:
    """The two sided p of the Mann-Whitney U test of two sets of distances, None with fewer
    than two structures on a side."""
    if len(x) < 2 or len(y) < 2:
        return None
    p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    return float(f"{p:.3g}") if np.isfinite(p) else None


# Two separations of a gene from its enhancer that differ by more than this are two enhancers.
SAME_ENHANCER_BP = 500


def _pairs(a: Path, b: Path, shared: bool) -> tuple[dict[str, dict[str, object]], int | None]:
    """The enhancer of each gene of both jobs and its distance from the promoter in every
    structure of each job.

    By default a job measures a gene to its own nearest enhancer, the one of its gene table.
    Two cell types have two sets of enhancers and the two enhancers of a gene can differ. With
    `shared` a gene is measured in both jobs to its nearest enhancer among the enhancers the two
    jobs share, the ones that overlap in the two sets. Positions are compared as they are, which
    holds for two jobs without structural variants.

    Returns for each gene `linear`, the separation of the promoter and the enhancer in bp in a
    and in b, `beads`, the enhancer bead of a and of b, and `values`, the distances in a and in
    b. The second value is the number of shared enhancers, None by default.
    """
    ga = {r["gene"]: r for r in genes(a)}
    gb = {r["gene"]: r for r in genes(b)}
    out: dict[str, dict[str, object]] = {}
    if not shared:
        da, db = _pair_distances(a), _pair_distances(b)
        for name in ga.keys() & gb.keys():
            x, y = ga[name], gb[name]
            out[name] = {"linear": (x["linear_bp"], y["linear_bp"]), "beads": (x["enhancer_bead"], y["enhancer_bead"]), "values": (da.get(name, []), db.get(name, []))}
        return out, None
    ea, eb = annotations(a)["enhancers"], annotations(b)["enhancers"]
    centres = sorted({(max(s1, s2) + min(e1, e2)) // 2 for s1, e1 in ea for s2, e2 in eb if s1 < e2 and s2 < e1})  # type: ignore[union-attr]
    (pa, ma), (pb, mb) = positions(a), positions(b)
    for name in ga.keys() & gb.keys():
        x, y = ga[name], gb[name]
        for c in sorted(centres, key=lambda c: abs(c - int(x["tss"]))):  # type: ignore[call-overload]
            (ka, oa), (kb, ob) = nearest(ma, c), nearest(mb, c)
            if abs(oa) > MAX_GAP or abs(ob) > MAX_GAP:
                continue
            va = [round(float(np.linalg.norm(q[x["promoter_bead"]] - q[ka])), 3) for q in pa]
            vb = [round(float(np.linalg.norm(q[y["promoter_bead"]] - q[kb])), 3) for q in pb]
            out[name] = {"linear": (abs(c - int(x["tss"])), abs(c - int(y["tss"]))), "beads": (ka, kb), "values": (va, vb)}  # type: ignore[call-overload]
            break
    return out, len(centres)


def _spread(values: list[float]) -> dict[str, float]:
    v = np.array(values)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def compare(a: Path, b: Path, shared: bool = False) -> dict[str, object]:
    """The genes both jobs have, with each job's mean and SD of the enhancer to promoter distance,
    the ratio b over a and the p of a Mann-Whitney U test of the distances in the structures of
    the two jobs, the largest changes first.

    A row has the separation of the gene and its enhancer in each job and the enhancer bead of
    each job. `differ` counts the genes with another enhancer in b than in a. `shared` and
    `shared_enhancers` say which enhancers were measured, see `_pairs`.
    """
    ra, rb = store.status(a).get("region"), store.status(b).get("region")
    pairs, n_shared = _pairs(a, b, shared)
    rows = []
    for name, pair in pairs.items():
        (la, lb), (ka, kb), (va, vb) = pair["linear"], pair["beads"], pair["values"]  # type: ignore[misc]
        if not va or not vb:
            continue
        x, y = _spread(va), _spread(vb)
        if x["mean"] <= 0 or y["mean"] <= 0:
            continue
        rows.append(
            {"gene": name, "linear_bp": la, "linear_bp_b": lb, "enhancer_bead_a": ka, "enhancer_bead_b": kb, "a": x, "b": y, "ratio": y["mean"] / x["mean"], "p": _rank_p(va, vb)}
        )
    rows.sort(key=lambda r: (-max(r["ratio"], 1 / r["ratio"]), r["gene"]))
    differ = sum(1 for r in rows if abs(r["linear_bp"] - r["linear_bp_b"]) > SAME_ENHANCER_BP)
    return {"same_region": ra is not None and ra == rb, "rows": rows, "differ": differ, "shared": shared, "shared_enhancers": n_shared}


def compare_gene(a: Path, b: Path, gene: str, shared: bool = False) -> dict[str, object] | None:
    """The enhancer to promoter distance of one gene in each structure of job a and of job b,
    for a box plot, with the p of `compare`. None for a gene one of the jobs lacks."""
    pair = _pairs(a, b, shared)[0].get(gene)
    if pair is None:
        return None
    x, y = pair["values"]  # type: ignore[misc]
    return {"gene": gene, "a": x, "b": y, "p": _rank_p(x, y)}


# The volcano plot shows a fold change up to 12 in log2 and an adjusted p down to 1e-50. A gene
# past either limit is drawn on it.
VOLCANO_FOLD, VOLCANO_P = 12.0, 50.0


def _volcano(table: dict[str, list[float | None]]) -> dict[str, object] | None:
    """Every tested gene of a contrast as [log2 fold change, -log10 of the adjusted p], for the
    background of the volcano plot. The points are on a grid of 0.2 by 0.5 with each cell once.
    None for a contrast without p-values."""
    cells: set[tuple[float, float]] = set()
    for fold, padj, *_ in table.values():
        if fold is None or padj is None:
            continue
        x = max(-VOLCANO_FOLD, min(VOLCANO_FOLD, fold))
        y = VOLCANO_P if padj <= 0 else min(-float(np.log10(padj)), VOLCANO_P)
        cells.add((round(x * 5) / 5, round(y * 2) / 2))
    return {"fold": VOLCANO_FOLD, "p": VOLCANO_P, "points": sorted(cells)} if cells else None


def compare_expression(a: Path, b: Path, cache: Path, shared: bool = False, names: Callable[[], dict[str, str]] | None = None) -> dict[str, object]:
    """The expression of job b against job a for the genes both jobs have, beside the change of
    their enhancer to promoter distance.

    Each row has the gene, log2 of the distance of b over a, the log2 fold change of the
    expression, its adjusted p or None and the two levels. `rho` is the Spearman correlation of
    the two changes over the rows, None under five rows. `volcano` has every gene of the test for
    the background of the volcano plot, None without a test. The mode and its reason are those
    of `expression.contrast`.
    The distances are those of `compare` with the same `shared`.
    """
    c = expression.contrast(a, b, cache, names)
    table = c.pop("genes", None)
    if not table:
        return {**c, "rows": [], "rho": None, "volcano": None}
    by_id = c.get("key") == "gene_id"
    ids = {r["gene"]: r["gene_id"] for r in genes(a)}
    rows = []
    for d in compare(a, b, shared)["rows"]:  # type: ignore[union-attr]
        name = d["gene"]
        e = table.get(str(ids[name]).removesuffix(".copy2").split(".")[0] if by_id else name)
        if e is None:
            continue
        rows.append({"gene": name, "distance": round(float(np.log2(d["ratio"])), 4), "log2fc": e[0], "padj": e[1], "a": e[2], "b": e[3]})
    rows.sort(key=lambda r: (-abs(r["log2fc"]), r["gene"]))
    rho = None
    if len(rows) >= 5:
        d = pd.DataFrame(rows)
        rho = float(d["distance"].corr(d["log2fc"], method="spearman"))
        rho = None if not np.isfinite(rho) else round(rho, 3)
    return {**c, "rows": rows, "rho": rho, "volcano": _volcano(table)}
