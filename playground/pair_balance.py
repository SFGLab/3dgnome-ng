"""How many anchor pairs a full depth map says sit closer and how many farther than expected.

    python playground/pair_balance.py MCOOL REGION ARM_DIR [RES]

Anchor pair counts come from the map at RES, 1 kb by default, each pixel shared between the
anchors it overlaps in proportion to the overlap, C = W^T M W with W the bins by anchors
overlap matrix. The expectation of a pair is the mean count per unit anchor area over pairs at
its separation, times its area. Per separation band prints the expected count's quartiles and
the share of pairs three Poisson standard deviations above and below expectation. A depleted
pair can be called only where the expectation is above nine, since a count of zero is then
three deviations under it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cooler
import numpy as np
import scipy.sparse as sp

BANDS = (
    (20_000, 100_000),
    (100_000, 500_000),
    (500_000, 2_000_000),
    (2_000_000, 10_000_000),
)


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


def overlap_weights(starts: np.ndarray, res: int, a_st: np.ndarray, a_en: np.ndarray) -> sp.csr_matrix:
    """Bins by anchors, the share of each bin inside each anchor."""
    rows, cols, vals = [], [], []
    for a, (s, e) in enumerate(zip(a_st, a_en, strict=True)):
        b0 = np.searchsorted(starts, s, side="right") - 1
        b1 = np.searchsorted(starts, e, side="left")
        for b in range(max(b0, 0), min(b1, len(starts))):
            ov = min(e, starts[b] + res) - max(s, starts[b])
            if ov > 0:
                rows.append(b)
                cols.append(a)
                vals.append(ov / res)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(starts), len(a_st)))


def main() -> None:
    mcool, region, arm = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    chrom, rest = region.split(":")
    lo, hi = (int(x) for x in rest.split("-"))
    a_st, a_en = anchors(arm)
    sel = (a_st >= lo) & (a_en <= hi)
    a_st, a_en = a_st[sel], a_en[sel]
    c = cooler.Cooler(f"{mcool}::/resolutions/{res}")
    m = c.matrix(balance=False, sparse=True).fetch(region).tocsr()
    starts = c.bins().fetch(region)["start"].to_numpy()
    w = overlap_weights(starts, res, a_st, a_en)
    keep = np.flatnonzero(np.asarray(w.sum(1)).ravel() > 0)
    m = m[keep][:, keep]
    w = w[keep]
    m = sp.triu(m, 1) + sp.triu(m, 1).T
    ca = (w.T @ m @ w).toarray()
    n = len(a_st)
    i, j = np.triu_indices(n, 1)
    mids = (a_st + a_en) / 2
    lens = np.maximum(a_en - a_st, 1).astype(float)
    sep = mids[j] - mids[i]
    area = lens[i] * lens[j]
    obs = ca[i, j]
    edges = np.geomspace(max(sep.min(), 1), sep.max() + 1, 80)
    b = np.clip(np.searchsorted(edges, sep) - 1, 0, len(edges) - 2)
    dens = np.bincount(b, obs, minlength=len(edges)) / np.maximum(np.bincount(b, area, minlength=len(edges)), 1)
    exp = dens[b] * area
    z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-12))
    print(f"{region} at {res // 1000} kb: {n} anchors, {obs.sum():,.0f} contacts between anchors")
    for b0, b1 in BANDS:
        band = (sep >= b0) & (sep < b1)
        q = np.percentile(exp[band], [25, 50, 75])
        callable_ = band & (exp > 9)
        print(
            f"  {b0 / 1e6:g}-{b1 / 1e6:g} Mb: {band.sum():>8,} pairs, expected count q25 {q[0]:.2f} median {q[1]:.2f} q75 {q[2]:.2f};"
            f" above +3 sd {np.mean(z[band] > 3):.3f}, below -3 sd {np.mean(z[band] < -3):.3f},"
            f" depletion callable {callable_.sum() / band.sum():.3f}"
        )


if __name__ == "__main__":
    main()
