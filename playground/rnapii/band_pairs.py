"""The pairs of bins of one band of separation by the loops at them, in Hi-C and in two arms.

    python playground/rnapii/band_pairs.py MCOOL CHROM CTCF.bedpe RNAPII.bedpe ARM_C ARM_R [LO_BP HI_BP]

`short_range_scc.py` gives the stratum adjusted correlation of a band, 100 to 500 kb unless given,
at 10 kb with the contacts per bead pair. This script splits the pairs of the band into classes by
the loops of the two factors and prints, per class, what the map and the two arms hold there:

- `RNAPII loop`: an RNAPII loop joins the two bins, an end of it within one bin of each.
- `CTCF loop`: a CTCF loop joins them and no RNAPII loop does.
- `two RNAPII anchors`: each bin holds an end of an RNAPII loop and no loop joins the two.
- `one RNAPII anchor` and `no RNAPII anchor`: the rest.

Per class the script prints the number of pairs, the mean contact over the mean of its separation
in the map and in each arm. It also prints the part of the correlation of each arm that the pairs
of the class carry. The
correlation of the band is the sum over its pairs of the products of the two deviations from the
mean of the separation, over the sum over separations of the count times the two standard
deviations, as `_scc` has it. The parts of the classes sum to it. The last column is the part of
the second arm minus that of the first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from short_range_scc import observed_band, simulated_band  # noqa: E402

RES = 10_000
CLASSES = ("RNAPII loop", "CTCF loop", "two RNAPII anchors", "one RNAPII anchor", "no RNAPII anchor")


def loops(path: str, chrom: str) -> list[tuple[int, int]]:
    """Returns the bins of the two ends of the loops of a chromosome, by the middles of their anchors."""
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] == chrom and f[3] == chrom:
            a, b = (int(f[1]) + int(f[2])) // 2 // RES, (int(f[4]) + int(f[5])) // 2 // RES
            out.append((min(a, b), max(a, b)))
    return out


def main() -> None:
    mcool, chrom, ctcf, rnapii, arm_c, arm_r = sys.argv[1:7]
    lo, hi = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (100_000, 500_000)
    d0, d1 = lo // RES + 1, hi // RES
    obs, _, _ = observed_band(mcool, chrom, RES, d1)
    n = obs.shape[0]
    arms = {}
    for name, folder in (("C", arm_c), ("R", arm_r)):
        _, arms[name] = simulated_band(sorted(Path(folder).glob("*.cif")), RES, n, d1)
    joined = {}
    for k, path in ((1, ctcf), (0, rnapii)):
        for a, b in loops(path, chrom):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if 0 <= a + da < b + db < n:
                        joined[(a + da, b + db)] = k if k == 0 else joined.get((a + da, b + db), 1)
    anchor = np.zeros(n, dtype=bool)
    for a, b in loops(rnapii, chrom):
        anchor[[a, b]] = True
    cls = np.full((n, d1 + 1), -1)
    for d in range(d0, d1 + 1):
        i = np.arange(n - d)
        both, one = anchor[i] & anchor[i + d], anchor[i] ^ anchor[i + d]
        cls[i, d] = np.where(both, 2, np.where(one, 3, 4))
    for (a, b), k in joined.items():
        if d0 <= b - a <= d1:
            cls[a, b - a] = k
    # The deviations from the mean of a separation, the mean of a separation and the weights of the correlation.
    dev = {}
    den = dict.fromkeys(arms, 0.0)
    ratio = {key: np.zeros_like(obs) for key in ("map", *arms)}
    dev_o = np.zeros_like(obs)
    for d in range(d0, d1 + 1):
        o = obs[: n - d, d]
        dev_o[: n - d, d] = o - o.mean()
        ratio["map"][: n - d, d] = o / o.mean() if o.mean() > 0 else 0
        for name, sim in arms.items():
            s = sim[: n - d, d]
            dev.setdefault(name, np.zeros_like(obs))[: n - d, d] = s - s.mean()
            ratio[name][: n - d, d] = s / s.mean() if s.mean() > 0 else 0
            den[name] += (n - d) * s.std() * o.std()
    print(f"{chrom}, pairs {lo // 1000} to {hi // 1000} kb apart at {RES // 1000} kb, {len(loops(ctcf, chrom))} CTCF and {len(loops(rnapii, chrom))} RNAPII loops, {int(anchor.sum())} bins with an RNAPII anchor of {n}")
    print(f"  {'class':>20s} {'pairs':>9s} {'share':>6s}   map  arm C  arm R   part of the correlation: C      R   change")
    total = int((cls >= 0).sum())
    sums = dict.fromkeys(arms, 0.0)
    for k, name in enumerate(CLASSES):
        m = cls == k
        if not m.any():
            continue
        part = {a: float((dev[a][m] * dev_o[m]).sum() / den[a]) if den[a] > 0 else float("nan") for a in arms}
        for a in arms:
            sums[a] += part[a]
        print(
            f"  {name:>20s} {int(m.sum()):9,d} {m.sum() / total:6.3f}  {ratio['map'][m].mean():5.2f}  {ratio['C'][m].mean():5.2f}  {ratio['R'][m].mean():5.2f}"
            f"                          {part['C']:+.3f} {part['R']:+.3f}   {part['R'] - part['C']:+.3f}"
        )
    print(f"  {'the band':>20s} {total:9,d}                                                  {sums['C']:+.3f} {sums['R']:+.3f}   {sums['R'] - sums['C']:+.3f}")


if __name__ == "__main__":
    main()
