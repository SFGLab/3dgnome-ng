"""The validation battery for two arms of the same pipeline, on this project's own metrics.

Energy and Kolmogorov-Smirnov gates say two ensembles are interchangeable as distributions. They
do not say the structures agree with the data, and that is what should decide whether a change
is adopted. Three measures, all of which existing work in this repository already turns on.

Hi-C correlation, both the contact correlation for a single structure and the ensemble Pearson
that MultiMM actually reports, against the cell line's own contact map.

The realised distance exponent, how quickly distance grows with genomic separation. Contact
probability decays near s^-0.86 across three cell lines, so distance should grow near s^0.285,
and `anchor-placement.md` exists because our structures come out flatter than that.

And the radius of gyration, since the solver's structures are systematically a little more
expanded and expansion is the direction that work has been trying to move.

    python playground/validation_battery.py <mcool> <region> <binsize> <arm_dir> [<arm_dir> ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.metrics.hic import (  # noqa: E402
    hic_correlation,
    multimm_faithful_pearson,
    observed_hic,
)

NU_HIC = 0.285  # ps_curve.py, mean over three cell lines, 20 kb to 1 Mb


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float64)
    mid = np.array([int(r[16]) for r in rows], dtype=np.int64)
    o = np.argsort(mid)
    return pos[o], mid[o]


def exponent(pos: np.ndarray, mid: np.ndarray, lo: int = 20_000, hi: int = 1_000_000) -> float:
    """Slope of log distance against log genomic separation, over the band Hi-C is measured on."""
    n = len(mid)
    step = max(1, n // 700)  # bound the pair count on a whole region
    i = np.arange(0, n, step)
    a, b = np.meshgrid(i, i, indexing="ij")
    m = a < b
    a, b = a[m], b[m]
    sep = np.abs(mid[b] - mid[a]).astype(np.float64)
    d = np.linalg.norm(pos[b] - pos[a], axis=1)
    k = (sep >= lo) & (sep <= hi) & (d > 0)
    if k.sum() < 50:
        return float("nan")
    return float(np.polyfit(np.log(sep[k]), np.log(d[k]), 1)[0])


def main() -> None:
    mcool, region, binsize = sys.argv[1], sys.argv[2], int(sys.argv[3])
    c_obs, bin_starts = observed_hic(mcool, region, binsize, balance=True)
    print(f"observed Hi-C {region} at {binsize:,} bp: {c_obs.shape[0]} bins\n")
    print(
        f"  {'arm':>10s} {'n':>3s} {'pearson':>9s} {'spearman':>9s} {'SCC':>8s} "
        f"{'multimm':>9s} {'exponent':>9s} {'vs Hi-C':>8s} {'Rg':>8s}"
    )
    for d in sys.argv[4:]:
        cifs = sorted(Path(d).glob("*.cif"))
        if not cifs:
            print(f"  {Path(d).name:>10s}  no cif files")
            continue
        coords, mids = [], None
        pear, spear, scc, expo, rgs = [], [], [], [], []
        for c in cifs:
            p, m = load(c)
            coords.append(p)
            mids = m
            r = hic_correlation(p, m, mcool, region, binsize, contact_radius=2.0, balance=True)
            pear.append(r.get("pearson", float("nan")))
            spear.append(r.get("spearman", float("nan")))
            scc.append(r.get("scc", float("nan")))
            expo.append(exponent(p, m))
            rgs.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
        mm = multimm_faithful_pearson(coords, mids, c_obs, bin_starts, binsize)
        e = float(np.nanmean(expo))
        print(
            f"  {Path(d).name:>10s} {len(cifs):>3d} {np.nanmean(pear):>9.3f} "
            f"{np.nanmean(spear):>9.3f} {np.nanmean(scc):>8.3f} {mm:>9.3f} "
            f"{e:>9.3f} {e / NU_HIC:>7.2f}x {np.mean(rgs):>8.2f}",
            flush=True,
        )
    print(f"\n  exponent target is {NU_HIC} from the cell lines' own contact probability curves;")
    print("  the project's structures have measured flatter than that, so higher is better here.")


if __name__ == "__main__":
    main()
