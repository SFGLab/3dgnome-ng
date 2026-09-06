"""The contact decay exponent from a singletons file, which every run already loads.

Fits log contact count against log separation over a band and reports `nu = -slope / 3`, the
exponent distance grows with. A Hi-C file binned at 25 kb has no separation under 25 kb, so
its band starts at 50 kb. A ChIA-PET singletons file is point resolution and can start at 20 kb.

    python playground/ps_from_singletons.py <bedpe> [<bedpe> ...]
"""

import sys

import numpy as np
import pandas as pd


def fit(path: str, lo: float, hi: float, nbins: int = 12) -> tuple[float, int]:
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    counts = np.zeros(nbins)
    n = 0
    for chunk in pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5], chunksize=2_000_000
    ):
        c = chunk[chunk[0] == chunk[3]]
        s = np.abs((c[4] + c[5]) / 2 - (c[1] + c[2]) / 2).to_numpy()
        h, _ = np.histogram(s, edges)
        counts += h
        n += len(c)
    centres = np.sqrt(edges[:-1] * edges[1:])
    dens = counts / np.diff(edges)  # per bp, so a log bin's width does not shape the slope
    k = counts > 20
    slope = float(np.polyfit(np.log(centres[k]), np.log(dens[k]), 1)[0])
    return slope, n


for p in sys.argv[1:]:
    hic = "hic_" in p
    lo, hi = (50_000, 1_000_000) if hic else (20_000, 1_000_000)
    slope, n = fit(p, lo, hi)
    print(
        f"{p.split('/')[-1]:44} intra {n:>11,}  band {lo // 1000:>3}kb-{hi // 1000_000}Mb"
        f"  P(s) slope {slope:+.3f}  nu {-slope / 3:.3f}",
        flush=True,
    )
