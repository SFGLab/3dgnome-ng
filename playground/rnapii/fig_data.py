"""Readout tables of the redesigned RNAPII figures, `design/2026-11-12-rnapii-figures.md`.

Every function reads the readout files the RNAPII comparison wrote and returns a long table the
drawing module consumes. Arms are carried as letters here and spelled out only on the page, by
`ARM_NAME`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARM_NAME = {
    "E": "original models (cudaMMC)",
    "C": "gnome3d, CTCF loops",
    "R": "gnome3d, CTCF and RNAPII loops",
    "M": "gnome3d, RNAPII ends merged into CTCF anchors",
}
ARM_COLOUR = {"E": "#8c8c8c", "C": "#2a78d6", "R": "#eb6834", "M": "#1baf7a"}

_ARM = re.compile(r"^(?P<name>[A-Za-z0-9]+)_(?P<arm>[ECRM])_60mb$")


def _rows(path: Path) -> list[dict[str, object]]:
    """The arm rows of one readout JSON, the map's own row left out."""
    out = []
    for row in json.loads(path.read_text()):
        m = _ARM.match(str(row["arm"]))
        if m:
            out.append({**row, "name": m["name"], "letter": m["arm"]})
    return out


def agreement(readouts: Path) -> pd.DataFrame:
    """Agreement with contact maps per line and arm.

    Returns columns line, arm, measure and value. `hic` is the stratum adjusted correlation of
    `<LINE>_battery.json`, `microc` the 5 kb correlation of `<LINE>_microc.json`. A line with no
    Micro-C file has no microc rows.
    """
    rows = []
    for path in sorted(readouts.glob("*_battery.json")):
        for r in _rows(path):
            rows.append({"line": r["name"], "arm": r["letter"], "measure": "hic", "value": float(r["scc"])})
    for path in sorted(readouts.glob("*_microc.json")):
        for r in _rows(path):
            rows.append(
                {"line": r["name"], "arm": r["letter"], "measure": "microc", "value": float(r["r_fine"])}
            )
    return pd.DataFrame(rows, columns=["line", "arm", "measure", "value"])


def trio_paired(trio_dir: Path) -> pd.DataFrame:
    """Per person readouts against the person's own Hi-C.

    Returns columns person, arm, measure and value. `loop_pileup` is the model's pile-up at
    RNAPII loops over the map's, `hic` the stratum adjusted correlation.
    """
    rows = []
    for path in sorted(trio_dir.glob("*_pileup.json")):
        rec = json.loads(path.read_text())
        ref = [float(r["apa"]) for r in rec if r["arm"] == "map"]
        if len(ref) != 1:
            raise ValueError(f"{path}: expected one map row")
        for r in _rows(path):
            rows.append(
                {"person": r["name"], "arm": r["letter"], "measure": "loop_pileup", "value": float(r["apa"]) / ref[0]}
            )
    for path in sorted(trio_dir.glob("*_battery.json")):
        for r in _rows(path):
            rows.append({"person": r["name"], "arm": r["letter"], "measure": "hic", "value": float(r["scc"])})
    return pd.DataFrame(rows, columns=["person", "arm", "measure", "value"])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_draws(genes_tsv: Path, n: int = 2000, seed: int = 1) -> tuple[float, np.ndarray]:
    """Spearman of `total` against `log2fc` and its block bootstrap draws over `cluster`.

    Rows missing either value are left out. A draw whose resample is constant in either column
    is skipped, as `enhancer3d.stats.boot_r` does, so fewer than `n` draws may return.
    """
    g = pd.read_csv(genes_tsv, sep="\t")
    g = g[np.isfinite(g["total"]) & np.isfinite(g["log2fc"])]
    x, y = g["total"].to_numpy(), g["log2fc"].to_numpy()
    cl = pd.factorize(g["cluster"])[0]
    groups = [np.flatnonzero(cl == k) for k in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        a, b = x[idx], y[idx]
        if a.std() > 0 and b.std() > 0:
            draws.append(_spearman(a, b))
    return _spearman(x, y), np.asarray(draws)


def link_draws(e3d_root: Path, arms: tuple[str, ...], n: int = 2000, seed: int = 1) -> pd.DataFrame:
    """Bootstrap draws of the link to expression per line pair and arm, sign flipped.

    Reads `e3d_<A>/genes_<a>_to_<b>.tsv` and checks the point estimate against `r_total` of
    `e3d_<A>/summary.tsv`. Returns columns pair, arm, draw and value, where value is minus the
    correlation so that a stronger link is larger. Raises ValueError naming the gene table when
    the point estimate differs from the summary by more than 1e-3, the rounding of the written table.
    """
    rows = []
    for arm in arms:
        d = e3d_root / f"e3d_{arm}"
        summary = pd.read_csv(d / "summary.tsv", sep="\t").set_index("pair")["r_total"]
        for path in sorted(d.glob("genes_*_to_*.tsv")):
            a, b = path.stem.removeprefix("genes_").split("_to_")
            point, draws = boot_draws(path, n=n, seed=seed)
            want = float(summary[f"{a}->{b}"])
            if abs(point - want) > 1e-3:
                raise ValueError(f"{path}: point estimate {point:.6f} differs from summary r_total {want:.6f}")
            for i, v in enumerate(draws):
                rows.append({"pair": f"{a} vs {b}", "arm": arm, "draw": i, "value": -v})
    return pd.DataFrame(rows, columns=["pair", "arm", "draw", "value"])
