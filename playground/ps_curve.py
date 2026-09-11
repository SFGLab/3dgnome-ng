"""Contact probability against genomic separation, from the cell line's own Hi-C.

The anchor placement problem needs an objective target for how spatial distance should grow with
genomic separation. Polymer physics supplies the link: if distance grows as s^nu, then two loci
meet with probability near s^(-3 nu), because the capture volume is fixed and the coil volume
grows as R^3. So a P(s) slope measured from Hi-C fixes nu without any appeal to another
implementation.

Reports the local slope of log P against log s, the slope fitted over the range our anchors
actually span, and the nu each implies. Compares that against the exponent already built into
`genomic_length_to_distance`, which is the shape option A would impose as its excluded volume
floor.
"""

import sys

import cooler
import cooltools
import numpy as np
import pandas as pd

CELLS = {
    "GM12878": "data/_hic/GM12878/4DNFIQ32RWCQ.mcool",
    "H1ESC": "data/_hic/H1ESC/hic.4DNFIHO3CXUQ.mcool",
    "HFFC6": "data/_hic/HFFC6/hic.4DNFIDKNBPC3.mcool",
}
CHROMS = ["chr1", "chr2", "chr5", "chr11", "chr17"]
RES = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

# the range our anchors span, and the range the block partition spans
FIT_RANGES = [(2e4, 1e6, "20kb-1Mb, within a block"),
              (1e6, 1e7, "1-10Mb, across blocks")]


def ps(path: str, res: int) -> pd.DataFrame:
    """Log binned P(s) for one cooler, averaged over CHROMS."""
    clr = cooler.Cooler(f"{path}::/resolutions/{res}")
    have = [c for c in CHROMS if c in clr.chromnames]
    view = pd.DataFrame({
        "chrom": have,
        "start": 0,
        "end": [clr.chromsizes[c] for c in have],
        "name": have,
    })
    balanced = "weight" in clr.bins().columns
    exp = cooltools.expected_cis(
        clr, view_df=view, smooth=False, aggregate_smoothed=False, nproc=4,
        clr_weight_name="weight" if balanced else None,
    )
    col = "balanced.avg" if balanced else "count.avg"
    exp = exp[exp["dist"] > 0].copy()
    exp["s"] = exp["dist"] * res

    # log bins, ten per decade, averaged across the chromosomes in the view
    edges = 10 ** np.arange(np.log10(res), 7.51, 0.1)
    exp["bin"] = np.searchsorted(edges, exp["s"], side="right") - 1
    g = exp.groupby("bin").agg(s=("s", "median"), p=(col, "mean"), n=(col, "size"))
    g = g[np.isfinite(g["p"]) & (g["p"] > 0)]
    g["p"] = g["p"] / g["p"].iloc[0]
    return g.reset_index(drop=True)


def slope(g: pd.DataFrame, lo: float, hi: float) -> tuple[float, int]:
    m = (g["s"] >= lo) & (g["s"] <= hi)
    if m.sum() < 4:
        return float("nan"), int(m.sum())
    a = np.polyfit(np.log10(g.loc[m, "s"]), np.log10(g.loc[m, "p"]), 1)[0]
    return float(a), int(m.sum())


curves = {}
for cell, path in CELLS.items():
    g = ps(path, RES)
    curves[cell] = g
    print(f"[ps] {cell}: {len(g)} log bins, {g['s'].min():,.0f} to {g['s'].max():,.0f} bp")

print(f"\n{'cell':>9s} {'range':>28s} {'alpha':>8s} {'nu=alpha/3':>11s} {'bins':>5s}")
nus = {}
for cell, g in curves.items():
    for lo, hi, lab in FIT_RANGES:
        a, n = slope(g, lo, hi)
        print(f"{cell:>9s} {lab:>28s} {-a:>8.3f} {-a/3:>11.3f} {n:>5d}")
        nus.setdefault(lab, []).append(-a / 3)

print("\n[ps] local slope of log P against log s, ten bins per decade")
print(f"{'separation':>12s} " + " ".join(f"{c:>9s}" for c in curves))
ref = next(iter(curves.values()))
for i in range(1, len(ref) - 1, 2):
    row = []
    for g in curves.values():
        if i + 1 >= len(g):
            row.append(float("nan")); continue
        d = ((np.log10(g["p"].iloc[i + 1]) - np.log10(g["p"].iloc[i - 1]))
             / (np.log10(g["s"].iloc[i + 1]) - np.log10(g["s"].iloc[i - 1])))
        row.append(-d)
    print(f"{ref['s'].iloc[i]:>12,.0f} " + " ".join(f"{v:>9.2f}" for v in row))

print("\n[ps] exponent comparison")
print(f"{'source':>34s} {'nu':>7s} {'implied alpha':>14s}")
for lab, v in nus.items():
    print(f"{'Hi-C, ' + lab:>34s} {np.mean(v):>7.3f} {3*np.mean(v):>14.3f}")
for lab, nu in [("genomic_length_to_distance", 0.75), ("self avoiding walk", 0.588),
                ("ideal chain", 0.5), ("fractal globule", 1/3)]:
    print(f"{lab:>34s} {nu:>7.3f} {3*nu:>14.3f}")

out = "playground/derived/ps_curve.csv"
pd.concat([g.assign(cell=c) for c, g in curves.items()]).to_csv(out, index=False)
print(f"\n[ps] wrote {out}")
