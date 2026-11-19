"""A paired block bootstrap of the pooled Hi-C scores, the first arm against each other arm.

    python playground/ensemble_bootstrap.py MCOOL REGION ARM_DIR ARM_DIR [ARM_DIR ...]

`ensemble_score.py` gives one number per arm and region. This resamples the region's 25 kb bins
in blocks of ten, 250 kb, with replacement, a thousand times, and scores every arm on each
resample, so a difference between two arms comes with an interval. A resample gives each bin a
count, a pair of bins the product of its two counts as weight, and a pair keeps its genomic
separation, so the strata of the SCC stay what they are. Every arm sees the same resamples.

Four scores. Pearson on log counts, Spearman and SCC against raw Hi-C over every bin, and SCC
against balanced Hi-C over the bins Hi-C maps. Spearman ranks the pairs once on the whole
region and correlates the ranks under the weights. For each other arm it prints the first
arm's score, the other's, the difference with its 95 percent interval, and the share of
resamples the first arm leads in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402

RES = 25_000
BLOCK = 10
DRAWS = 1000
NAMES = ("raw Pearson", "raw Spearman", "raw SCC", "balanced SCC, mapped")


def pooled(d: Path, bins: np.ndarray) -> np.ndarray:
    """The contacts of every structure of an arm summed, at the battery's contact radius."""
    sim = np.zeros((len(bins), len(bins)))
    for c in sorted(d.glob("*.cif")):
        p, m, _, _ = load(c)
        r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        sim += simulated_contacts(p, m, bins, RES, r)
    return sim


def wcorr(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of x and y under weights w."""
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    cov = (w * (x - mx) * (y - my)).sum()
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def wscc(w: np.ndarray, sep: np.ndarray, n: int, x: np.ndarray, y: np.ndarray) -> float:
    """The stratum adjusted correlation under pair weights. Each separation's covariance and
    standard deviations are weighted, and its weight is its total weight times the two
    standard deviations, as `legacy.validation.metrics.hic._scc` has it at weight one."""

    def by(v: np.ndarray) -> np.ndarray:
        return np.bincount(sep, weights=v, minlength=n)

    sw = by(w)
    ok = sw > 0
    sx, sy = by(w * x), by(w * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / sw, sy / sw
        cov = by(w * x * y) / sw - mx * my
        vx, vy = by(w * x * x) / sw - mx * mx, by(w * y * y) / sw - my * my
    ok &= (vx > 1e-24) & (vy > 1e-24)
    den = (sw[ok] * np.sqrt(vx[ok] * vy[ok])).sum()
    return float((sw[ok] * cov[ok]).sum() / den) if den > 0 else float("nan")


def main() -> None:
    from scipy.stats import rankdata

    mcool, region = sys.argv[1], sys.argv[2]
    dirs = [Path(d) for d in sys.argv[3:]]
    raw, bins = observed_hic(mcool, region, RES, balance=False)
    bal, _ = observed_hic(mcool, region, RES, balance=True)
    n = len(bins)
    mapped = (bal.sum(axis=0) > 0).astype(float)
    iu = np.triu_indices(n, 1)
    sep = iu[1] - iu[0]
    sims = [pooled(d, bins)[iu] for d in dirs]
    o_raw, o_bal = raw[iu], bal[iu]
    lo_raw, rk_raw = np.log1p(o_raw), rankdata(o_raw)
    ls = [np.log1p(s) for s in sims]
    rs = [rankdata(s) for s in sims]
    rng = np.random.default_rng(0)
    nb = -(-n // BLOCK)
    block = np.arange(n) // BLOCK
    out = np.empty((DRAWS + 1, len(dirs), 4))
    for b in range(DRAWS + 1):
        # The first row is the region as it is, every bin once.
        cnt = np.ones(n) if b == 0 else np.bincount(rng.integers(0, nb, nb), minlength=nb)[block].astype(float)
        w = cnt[iu[0]] * cnt[iu[1]]
        wm = w * mapped[iu[0]] * mapped[iu[1]]
        for a in range(len(dirs)):
            out[b, a] = (
                wcorr(w, ls[a], lo_raw),
                wcorr(w, rs[a], rk_raw),
                wscc(w, sep, n, sims[a], o_raw),
                wscc(wm, sep, n, sims[a], o_bal),
            )
    print(f"{region}, {n} bins, {DRAWS} resamples of {BLOCK} bin blocks, first arm {dirs[0].name}")
    print(f"  {'score':>22s} {'against':>40s}   first   other    diff   95 percent interval   first leads")
    for a in range(1, len(dirs)):
        for k, name in enumerate(NAMES):
            d = out[1:, 0, k] - out[1:, a, k]
            d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(
                f"  {name:>22s} {dirs[a].name:>40s}  {out[0, 0, k]:6.3f}  {out[0, a, k]:6.3f}  "
                f"{out[0, 0, k] - out[0, a, k]:+6.3f}   {lo:+6.3f} to {hi:+6.3f}        {float((d > 0).mean()):5.3f}"
            )


if __name__ == "__main__":
    main()
