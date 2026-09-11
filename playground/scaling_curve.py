"""At what genomic scale does one ensemble become more extended than another?

The trio structures are about five times the radius of gyration of the cell line structures at
the same bond length, and the inputs do not explain it. Arc targets match, 0.213 against 0.201
as a fraction of the chain bond scale. The segment heatmap asks the trio to be more compact,
not less, with a mean target of 363 against 723 and healthy decay in both, 4.0 against 5.8.
Bead count differs by 1.26x against a 5x radius, so size does not explain it either.

This measures the mean spatial distance between beads as a function of the genomic distance
separating them, log binned, which is the standard way to read a polymer's organisation and
needs no block or domain file. Two ensembles that agree at small separation and part at large
separation differ in how their blocks are placed. Two that part immediately differ in the chain
itself. The local slope is the scaling exponent, near 1/3 for a compact globule and near 0.6
for a swollen chain.

    python playground/scaling_curve.py --sets trio=<dir> gm=<dir> --limit 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def load_cif(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Bead coordinates and genomic midpoints from one CIF."""
    with path.open() as fh:
        rows = [ln.split() for ln in fh if ln.startswith("ATOM")]
    xyz = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float64)
    mid = np.array([(int(r[16]) + int(r[17])) // 2 for r in rows], dtype=np.int64)
    order = np.argsort(mid)
    return xyz[order], mid[order]


def curve(xyz: np.ndarray, mid: np.ndarray, edges: np.ndarray, n_pairs: int,
          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Median spatial distance per genomic separation bin, plus the pair count per bin.

    Pairs are drawn at random rather than enumerated, since a full pair set at 100k beads is
    5e9 entries. Sampling uniformly over index pairs already covers every bin, because the
    bins are log spaced and the small ones are the densely populated ones.
    """
    n = len(xyz)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    sep = np.abs(mid[i] - mid[j]).astype(np.float64)
    d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    idx = np.digitize(sep, edges) - 1
    med = np.full(len(edges) - 1, np.nan)
    cnt = np.zeros(len(edges) - 1, dtype=np.int64)
    for b in range(len(edges) - 1):
        sel = d[idx == b]
        cnt[b] = sel.size
        if sel.size >= 30:
            med[b] = float(np.median(sel))
    return med, cnt


def radius_of_gyration(xyz: np.ndarray) -> float:
    return float(np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="name=directory-of-cif")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    edges = np.logspace(3, 8.4, 23)  # 1 kb to ~250 Mb
    rng = np.random.default_rng(args.seed)

    names, curves = [], []
    for spec in args.sets:
        name, _, d = spec.partition("=")
        paths = sorted(Path(d).glob("*.cif"),
                       key=lambda p: int(re.search(r"_s(\d+)\.cif$", p.name).group(1)))[:args.limit]
        if not paths:
            raise SystemExit(f"no .cif under {d}")
        meds, rgs, bonds, nbeads = [], [], [], 0
        for p in paths:
            xyz, mid = load_cif(p)
            nbeads = len(xyz)
            rgs.append(radius_of_gyration(xyz))
            bonds.append(float(np.median(np.linalg.norm(np.diff(xyz, axis=0), axis=1))))
            m, cnt = curve(xyz, mid, edges, args.pairs, rng)
            meds.append(m)
        names.append(name)
        curves.append(np.nanmean(np.stack(meds), axis=0))
        print(f"[scal] {name:10s} n={len(paths)} beads={nbeads} "
              f"rg={np.mean(rgs):8.2f} bond={np.mean(bonds):6.3f}", flush=True)

    mids = np.sqrt(edges[:-1] * edges[1:])
    print(f"\n{'sep':>10s}" + "".join(f"{n:>12s}" for n in names) + f"{'ratio':>10s}")
    for b, sm in enumerate(mids):
        vals = [c[b] for c in curves]
        if all(np.isnan(v) for v in vals):
            continue
        sep = f"{sm / 1e6:.2f}Mb" if sm >= 1e6 else f"{sm / 1e3:.0f}kb"
        cells = "".join("         nan" if np.isnan(v) else f"{v:12.2f}" for v in vals)
        r = vals[0] / vals[1] if len(vals) > 1 and vals[1] else float("nan")
        print(f"{sep:>10s}{cells}{'' if np.isnan(r) else f'{r:10.2f}'}")
    print("\n[scal] ratio is set-1 over set-2. Flat near 1 then rising means the chain agrees "
          "locally and the blocks are placed further apart")


if __name__ == "__main__":
    main()
