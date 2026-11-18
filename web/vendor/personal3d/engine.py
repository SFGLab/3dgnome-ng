"""3dgnome-ng inputs for one haplotype, and the run.

A haplotype's inputs are those of the conf arm, written in the haplotype's own coordinates.

  anchors and clusters, the rules' rewritten network per factor, CTCF first as the engine's
    orientation term expects, anchors with the rules' orientation
  singletons, a base map on the reference lifted to the haplotype, with the pixels of a
    predicted window put in its place at the base map's depth in that window. The engine fits
    the contact decay on the whole file, so the rest of the chromosome stays.
  segments and centromeres, the base files lifted

The config is the arm's own ini with only its [data] section replaced, so every engine setting
is the arm's. `run` starts gnome3d-ng on a region in the haplotype's coordinates.
"""
from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .lift import Lift
from .rewrite import Loop

FACTOR_ORDER = ("CTCF", "RNAPOL2")


def lift_bedpe(df: pd.DataFrame, lifts: dict[str, Lift]) -> pd.DataFrame:
    """A seven column BEDPE lifted end by end, rows with a deleted end dropped.

    A tandem duplication leaves two copies where the reference has one, and the lift puts a
    position on the first. A pair inside the duplication is a pair of each copy, so it is
    written twice. A pair from inside to beyond the duplication's end keeps its distance from
    the second copy, the one next to that end, so its inner end moves there. A pair from before
    the duplication keeps its distance to the first copy and stays. The two copies get no pair
    between them, since the reference has none to give."""
    out = df.copy()
    keep = np.ones(len(df), dtype=bool)
    for c_chrom, c_s, c_e in ((0, 1, 2), (3, 4, 5)):
        for chrom, L in lifts.items():
            m = (df[c_chrom] == chrom).to_numpy()
            if not m.any():
                continue
            s = L.many(df.loc[m, c_s].to_numpy())
            e = L.many(df.loc[m, c_e].to_numpy() - 1) + 1
            bad = (s < 0) | (e <= 0)
            out.loc[m, c_s] = np.minimum(s, e - 1)
            out.loc[m, c_e] = np.maximum(s + 1, e)
            keep[np.flatnonzero(m)[bad]] = False
    extra = []
    for chrom, L in lifts.items():
        same = ((df[0] == chrom) & (df[3] == chrom)).to_numpy() & keep
        for a, b, _ in L.duplications:
            first = same & (df[1].to_numpy() >= a) & (df[2].to_numpy() <= b)
            second = same & (df[4].to_numpy() >= a) & (df[5].to_numpy() <= b)
            both = first & second
            if both.any():
                twin = out[both].copy()
                twin[[1, 2, 4, 5]] += b - a
                extra.append(twin)
            for inner, cols, other in ((first & ~second, [1, 2], df[4].to_numpy()), (second & ~first, [4, 5], df[1].to_numpy())):
                move = inner & (other >= b)
                out.loc[move, cols] += b - a
    out = out[keep]
    return pd.concat([out, *extra], ignore_index=True) if extra else out


def splice_window(base: pd.DataFrame, chrom: str, start: int, cond: np.ndarray, pred_bp: int) -> pd.DataFrame:
    """The base singletons with a predicted window put in, haplotype coordinates.

    cond is the predicted map from start on the log of one plus the count at pred_bp. Its
    pixels are summed onto the base map's own grid by their centres, since the engine reads
    the grid step off the file, and the grid bins wholly inside the window replace the base
    map's, scaled to the base map's total over the same bins."""
    bb = int(np.bincount((base[2] - base[1]).to_numpy()).argmax())
    end = start + cond.shape[0] * pred_bp
    g0, g1 = -(-start // bb), end // bb
    n = g1 - g0
    if n <= 0:
        return base
    centre = (start + (np.arange(cond.shape[0]) + 0.5) * pred_bp) // bb - g0
    counts = np.expm1(np.clip(cond, 0, None))
    grid = np.zeros((n, n))
    ok = (centre >= 0) & (centre < n)
    ci = centre[ok].astype(int)
    np.add.at(grid, (ci[:, None], ci[None, :]), counts[np.ix_(ok, ok)])
    grid = np.triu(grid + grid.T - np.diag(np.diag(grid)))
    lo, hi = g0 * bb, g1 * bb
    inside = (base[0] == chrom) & (base[3] == chrom) & (base[1] >= lo) & (base[4] >= lo) & (base[2] <= hi) & (base[5] <= hi)
    total = float(base.loc[inside, 6].sum())
    if grid.sum() > 0 and total > 0:
        grid *= total / grid.sum()
    a, b = np.nonzero(np.rint(grid) > 0)
    new = pd.DataFrame({0: chrom, 1: lo + a * bb, 2: lo + (a + 1) * bb, 3: chrom, 4: lo + b * bb, 5: lo + (b + 1) * bb, 6: np.rint(grid[a, b]).astype(int)})
    return pd.concat([base[~inside], new], ignore_index=True).sort_values([0, 1, 4]).reset_index(drop=True)


def write_inputs(out: Path, name: str, loops: dict[str, list[Loop]], singletons: pd.DataFrame, segments: pd.DataFrame, centromeres: pd.DataFrame) -> dict[str, str]:
    """The haplotype's input files under out, named after name, and the [data] keys for them."""
    out.mkdir(parents=True, exist_ok=True)
    anchors = {}
    files = []
    for f in FACTOR_ORDER:
        rows = []
        for lp in loops.get(f, []):
            anchors[(lp.chrom, *lp.a)] = lp.orient[0]
            anchors[(lp.chrom, *lp.b)] = lp.orient[1]
            rows.append((lp.chrom, *lp.a, lp.chrom, *lp.b, max(1, int(round(lp.weight)))))
        path = out / f"{name}_{f.lower()}_clusters.bedpe"
        pd.DataFrame(rows).sort_values([0, 1, 4]).to_csv(path, sep="\t", header=False, index=False)
        files.append(path.name)
    a = pd.DataFrame([(c, s, e, o) for (c, s, e), o in anchors.items()]).sort_values([0, 1, 2])
    a.to_csv(out / f"{name}_anchors.bed", sep="\t", header=False, index=False)
    singletons.to_csv(out / f"{name}_singletons.bedpe", sep="\t", header=False, index=False)
    segments.to_csv(out / f"{name}_segments.bed", sep="\t", header=False, index=False)
    centromeres.to_csv(out / f"{name}_centromeres.bed", sep="\t", header=False, index=False)
    return {
        "data_dir": str(out.resolve()),
        "anchors": f"{name}_anchors.bed",
        "clusters": ",".join(files),
        "singletons": f"{name}_singletons.bedpe",
        "singletons_inter": "",
        "segment_split": f"{name}_segments.bed",
        "centromeres": f"{name}_centromeres.bed",
        "factors": ",".join(FACTOR_ORDER),
    }


def write_config(template: Path, data: dict[str, str], dst: Path, header: str = "") -> Path:
    """The template ini with its [data] section replaced, comments and order of the rest kept,
    and header, a comment saying where the file came from, put first."""
    text = Path(template).read_text().splitlines()
    out, skip = [f"# {line}" if line else "#" for line in header.splitlines()], False
    for line in text:
        if line.strip().startswith("["):
            skip = line.strip() == "[data]"
            if skip:
                out.append("[data]")
                out += [f"{k} = {v}" for k, v in data.items()]
                continue
        if not skip:
            out.append(line)
    if "[data]" not in out:
        out += ["", "[data]"] + [f"{k} = {v}" for k, v in data.items()]
    dst.write_text("\n".join(out) + "\n")
    configparser.ConfigParser().read(dst)
    return dst


def run(gnome3d: Path, config: Path, region: str, members: str, out: Path) -> None:
    """gnome3d-ng on one region, from the engine checkout's venv."""
    exe = Path(gnome3d) / ".venv" / "bin" / "gnome3d-ng"
    subprocess.run([str(exe), "--config", str(config), "--region", region, "--members", members, "--out", str(out)], cwd=gnome3d, check=True)
