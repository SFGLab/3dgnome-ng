"""Hi-C agreement of each arm's pooled ensemble map rather than of each structure alone.

    python playground/ensemble_score.py MCOOL REGION ARM_DIR [ARM_DIR ...]

The battery scores every structure against Hi-C and averages, which rewards structures that each
look like the population average. Hi-C is that average, so this pools the contacts of every
structure of an arm into one map first, at the battery's contact radius, and prints Pearson on
log counts, Spearman and SCC against raw and balanced Hi-C at 25 kb, and SCC again over the
bins Hi-C maps, since an unmapped bin reads as zero and a model bead there counts against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from validation.metrics.hic import (  # noqa: E402
    _scc,
    observed_hic,
    simulated_contacts,
)

RES = 25_000


def scores(sim: np.ndarray, obs: np.ndarray, keep: np.ndarray) -> list[float]:
    """Pearson on log counts, Spearman and SCC over the off diagonal pairs of the kept bins."""
    from scipy.stats import spearmanr

    s, o = sim[np.ix_(keep, keep)], obs[np.ix_(keep, keep)]
    iu = np.triu_indices(s.shape[0], 1)
    a, b = s[iu], o[iu]
    return [
        float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]),
        float(spearmanr(a, b).statistic),
        float(_scc(s, o, s.shape[0] - 1)),
    ]


def stratum_spearman(sim: np.ndarray, obs: np.ndarray, keep: np.ndarray, max_bins: int = 80) -> float:
    """The Spearman correlation of each diagonal up to `max_bins` over the kept bins, averaged
    with the diagonal's length as weight. Ranks within a separation carry no distance decay and
    no scale, so a filled stretch is judged by its pattern and not by its level against zeros."""
    from scipy.stats import spearmanr

    s, o = sim[np.ix_(keep, keep)], obs[np.ix_(keep, keep)]
    num = den = 0.0
    for d in range(1, min(max_bins, s.shape[0] - 2) + 1):
        x, y = np.diagonal(s, d), np.diagonal(o, d)
        if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(spearmanr(x, y).statistic)
        if np.isfinite(r):
            num += x.size * r
            den += x.size
    return num / den if den > 0 else float("nan")


def main() -> None:
    mcool, region = sys.argv[1], sys.argv[2]
    raw, bins = observed_hic(mcool, region, RES, balance=False)
    bal, _ = observed_hic(mcool, region, RES, balance=True)
    mapped = bal.sum(axis=0) > 0
    every = np.ones(len(bins), dtype=bool)
    print(f"{region}, pooled ensemble against Hi-C at {RES} bp, {int((~mapped).sum())} unmapped bins")
    print(
        f"  {'arm':>44s} {'n':>3s}   raw P  raw Sp  raw SCC   bal P  bal Sp  bal SCC"
        "   mapped bins only: raw SCC  bal SCC  bal stratum Spearman"
    )
    for d in map(Path, sys.argv[3:]):
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        sim = np.zeros((len(bins), len(bins)))
        for c in cifs:
            p, m, _, _ = load(c)
            r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sim += simulated_contacts(p, m, bins, RES, r)
        out = scores(sim, raw, every) + scores(sim, bal, every)
        mo = [scores(sim, raw, mapped)[2], scores(sim, bal, mapped)[2], stratum_spearman(sim, bal, mapped)]
        print(
            f"  {d.name:>44s} {len(cifs):3d}  "
            + "  ".join(f"{v:6.3f}" for v in out)
            + "                    "
            + "   ".join(f"{v:6.3f}" for v in mo)
        )


if __name__ == "__main__":
    main()
