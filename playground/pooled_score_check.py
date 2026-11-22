"""Check of the pooled score against the bead count of a bin.

    python playground/pooled_score_check.py [--res BP] MCOOL REGION ARM_DIR [ARM_DIR ...]

The pooled map of `ensemble_score.py` counts the bead pairs within the contact radius and adds
them to the bins of the two bead starts. The count of a bin pair grows with the number of
beads in each of the two bins. A chain with beads of unequal width has bins with few beads and
bins with many. This script prints, per arm, the bead count of the bins and Pearson on log
counts, Spearman and SCC against raw and balanced Hi-C for four maps:

    counts      the pooled map as `ensemble_score.py` scores it
    per pair    the pooled map over the product of the bead counts of the two bins
    decay       the mean of each diagonal of the pooled map, a map of the separation alone
    beads       the product of the bead counts times the decay, a map without a 3D position

Each map is scored over every bin and over the bins that hold a bead in every arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import _scc, observed_hic, simulated_contacts  # noqa: E402


def scores(sim: np.ndarray, obs: np.ndarray, keep: np.ndarray) -> tuple[float, float, float]:
    from scipy.stats import spearmanr

    s, o = sim[np.ix_(keep, keep)], obs[np.ix_(keep, keep)]
    iu = np.triu_indices(s.shape[0], 1)
    a, b = s[iu], o[iu]
    return (
        float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]),
        float(spearmanr(a, b).statistic),
        float(_scc(s, o, s.shape[0] - 1)),
    )


def decay_of(sim: np.ndarray) -> np.ndarray:
    """Returns the map whose every cell is the mean of its diagonal in `sim`."""
    n = sim.shape[0]
    out = np.zeros_like(sim)
    i = np.arange(n)
    for d in range(1, n):
        m = float(np.diagonal(sim, d).mean())
        out[i[: n - d], i[d:]] = m
        out[i[d:], i[: n - d]] = m
    return out


def main() -> None:
    res = 25_000
    if sys.argv[1] == "--res":
        res = int(sys.argv[2])
        del sys.argv[1:3]
    mcool, region = sys.argv[1], sys.argv[2]
    raw, bins = observed_hic(mcool, region, res, balance=False)
    bal, _ = observed_hic(mcool, region, res, balance=True)
    nb = len(bins)
    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in map(Path, sys.argv[3:]):
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        sim = np.zeros((nb, nb))
        count = np.zeros(nb)
        for c in cifs:
            # ensemble_score.py bins a bead by its start.
            p, starts, _, _ = load(c)
            r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sim += simulated_contacts(p, starts, bins, res, r)
            idx = np.searchsorted(bins, starts, side="right") - 1
            ok = (idx >= 0) & (idx < nb) & (starts < int(bins[-1]) + res)
            count += np.bincount(idx[ok], minlength=nb)
        arms[d.name] = (sim, count / len(cifs))
    shared = np.ones(nb, dtype=bool)
    for _, count in arms.values():
        shared &= count > 0
    every = np.ones(nb, dtype=bool)
    cover = raw.sum(axis=0)
    print(f"{region} at {res} bp, {nb} bins, {int(shared.sum())} hold a bead in every arm")
    seps = [d for d in (1, 2, 4, 10, 20, 40, 80, 160) if d < nb]
    hic = "  ".join(f"{d}: {np.diagonal(raw, d).mean() / max(np.diagonal(raw, 1).mean(), 1e-12):.4f}" for d in seps)
    print(f"  raw Hi-C, mean count by separation in bins over that at 1   {hic}")
    print(f"  raw Hi-C, cells without a count {(raw[np.triu_indices(nb, 1)] == 0).mean():.3f}")
    for name, (sim, count) in arms.items():
        pair = np.outer(count, count)
        per_pair = np.divide(sim, pair, out=np.zeros_like(sim), where=pair > 0)
        decay = decay_of(sim)
        beads = pair * decay_of(per_pair)
        has = count > 0
        corr = float(np.corrcoef(count[has], cover[has])[0, 1]) if has.sum() > 2 else float("nan")
        print(
            f"  {name}: beads per bin min {count.min():.1f} median {np.median(count):.1f}"
            f" max {count.max():.1f}, bins without a bead {int((~has).sum())},"
            f" correlation of the bead count with the raw coverage of Hi-C {corr:.3f}"
        )
        iu = np.triu_indices(nb, 1)
        seps = [d for d in (1, 2, 4, 10, 20, 40, 80, 160) if d < nb]
        curve = "  ".join(f"{d}: {np.diagonal(sim, d).mean() / max(np.diagonal(sim, 1).mean(), 1e-12):.4f}" for d in seps)
        zero = "  ".join(f"{d}: {(np.diagonal(sim, d) == 0).mean():.2f}" for d in seps)
        print(f"    cells without a contact {(sim[iu] == 0).mean():.3f}")
        print(f"    mean count by separation in bins over that at 1   {curve}")
        print(f"    share of cells without a contact by separation   {zero}")
        print(f"    {'map':>9s} {'bins':>7s}   raw P  raw Sp raw SCC   bal P  bal Sp bal SCC")
        for label, m in (("counts", sim), ("per pair", per_pair), ("decay", decay), ("beads", beads)):
            for which, keep in (("every", every), ("shared", shared)):
                v = scores(m, raw, keep) + scores(m, bal, keep)
                print(f"    {label:>9s} {which:>7s}  " + "  ".join(f"{x:6.3f}" for x in v))


if __name__ == "__main__":
    main()
