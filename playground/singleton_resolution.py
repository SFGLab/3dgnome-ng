"""What the anchor contact map can hold at each singleton resolution and depth.

    python playground/singleton_resolution.py MCOOL REGION ARM_DIR THIN

The anchors are the anchor beads of the first structure in ARM_DIR. A singleton is a pixel of
the map written at its start, as `legacy.validation.studies.self_corr.hic_to_singleton_bedpe` writes
it, and it counts for an anchor pair when both starts fall inside the two anchors, as the
pipeline's anchor heatmap counts it. Counts are divided by the anchors' lengths, as there.

For each resolution, at full depth and binomially thinned by THIN (production's whole genome
file is about 1/230), prints the share of anchors that can receive a contact at all and, per
separation band, the share of anchor pairs held, observed over expected above one with the
expectation the mean at that separation, which is the cells the contact background enters.
Truth is the full depth 1 kb map: a pair is enriched when its count is three Poisson standard
deviations over its expectation. Precision is the share of held pairs that are enriched,
recall the share of enriched pairs held.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cooler
import numpy as np

BANDS = ((100_000, 500_000), (500_000, 2_000_000), (2_000_000, 10_000_000))
RESOLUTIONS = (1_000, 5_000, 10_000, 25_000)


def anchors(arm: Path) -> tuple[np.ndarray, np.ndarray]:
    f = sorted(arm.glob("*.cif"))[0]
    st, en = [], []
    for line in f.read_text().splitlines():
        if line.startswith("ATOM"):
            x = line.split()
            if x[18] == "anchor":
                st.append(int(x[16]))
                en.append(int(x[17]))
    o = np.argsort(st)
    return np.array(st)[o], np.array(en)[o]


def anchor_counts(c: cooler.Cooler, chrom: str, lo: int, hi: int, a_st: np.ndarray, a_en: np.ndarray) -> np.ndarray:
    """Raw counts between anchor pairs, each pixel at its start."""
    res = c.binsize
    bins = c.bins().fetch(f"{chrom}:{lo}-{hi}")
    starts = bins["start"].to_numpy()
    first = int(bins.index[0])
    # The anchor holding each bin's start, or -1.
    k = np.searchsorted(a_st, starts, side="right") - 1
    owner = np.where((k >= 0) & (starts < a_en[np.clip(k, 0, None)]), k, -1)
    n = len(a_st)
    out = np.zeros((n, n), dtype=np.float64)
    step = max(1, 2_000_000 // res)
    nb = len(starts)
    for r0 in range(0, nb, step):
        r1 = min(nb, r0 + step)
        px = c.matrix(balance=False, as_pixels=True).fetch(
            f"{chrom}:{int(starts[r0])}-{int(starts[r1 - 1]) + res}", f"{chrom}:{lo}-{hi}"
        )
        i = owner[px["bin1_id"].to_numpy() - first]
        j = owner[px["bin2_id"].to_numpy() - first]
        keep = (i >= 0) & (j >= 0) & (i != j)
        np.add.at(out, (i[keep], j[keep]), px["count"].to_numpy()[keep])
    return np.triu(out + out.T, 1)


def oe(counts: np.ndarray, mids: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed over expected of the length normalised map and the expected raw count, the
    expectation the mean over pairs in log spaced separation bins."""
    n = len(mids)
    i, j = np.triu_indices(n, 1)
    sep = mids[j] - mids[i]
    area = lens[i] * lens[j] / 1e6
    norm = counts[i, j] / area
    edges = np.geomspace(max(sep.min(), 1), sep.max() + 1, 60)
    b = np.clip(np.searchsorted(edges, sep) - 1, 0, len(edges) - 2)
    mean = np.bincount(b, norm, minlength=len(edges)) / np.maximum(np.bincount(b, minlength=len(edges)), 1)
    exp = mean[b]
    ratio = np.where(exp > 0, norm / np.where(exp > 0, exp, 1), 0)
    return ratio, exp * area


def main() -> None:
    mcool, region, arm, thin = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
    chrom, rest = region.split(":")
    lo, hi = (int(x) for x in rest.split("-"))
    a_st, a_en = anchors(arm)
    sel = (a_st >= lo) & (a_en <= hi)
    a_st, a_en = a_st[sel], a_en[sel]
    mids = (a_st + a_en) / 2
    lens = np.maximum(a_en - a_st, 1).astype(float)
    n = len(a_st)
    i, j = np.triu_indices(n, 1)
    sep = mids[j] - mids[i]
    print(f"{region}: {n} anchors, median width {np.median(lens) / 1e3:.1f} kb, thinning 1/{1 / thin:.0f}")
    rng = np.random.default_rng(0)
    truth = None
    for res in RESOLUTIONS:
        c = cooler.Cooler(f"{mcool}::/resolutions/{res}")
        full = anchor_counts(c, chrom, lo, hi, a_st, a_en)
        if truth is None:
            ratio, exp_count = oe(full, mids, lens)
            obs = full[i, j]
            truth = (obs - exp_count) > 3 * np.sqrt(np.maximum(exp_count, 1e-12))
            truth &= ratio > 1
        thinned = np.triu(rng.binomial(np.rint(full).astype(np.int64), thin).astype(np.float64), 1)
        for name, m in (("full", full), ("thinned", thinned)):
            row = (m + m.T).sum(1) > 0
            ratio, _ = oe(m, mids, lens)
            held = ratio > 1
            print(f"  {res // 1000:>2} kb {name:<7} contacts {m.sum():>13,.0f}  anchors receiving any {row.mean():.2f}")
            for b0, b1 in BANDS:
                band = (sep >= b0) & (sep < b1)
                h, t = held & band, truth & band
                prec = (h & t).sum() / max(h.sum(), 1)
                rec = (h & t).sum() / max(t.sum(), 1)
                print(
                    f"      {b0 / 1e6:g}-{b1 / 1e6:g} Mb: held {h.sum() / band.sum():.3f}  enriched in truth {t.sum() / band.sum():.3f}"
                    f"  precision {prec:.2f}  recall {rec:.2f}"
                )


if __name__ == "__main__":
    main()
