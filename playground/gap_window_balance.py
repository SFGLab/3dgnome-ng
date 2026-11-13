"""Enriched and depleted pairs among windows of consecutive subanchors inside a gap.

    python playground/gap_window_balance.py MCOOL REGION ARM_DIR

Consecutive subanchors of one gap, in the first structure of ARM_DIR, are grouped into windows
of up to 5, 10 and 25 kb. Counts between windows come from the map at 1 kb at full depth, each
pixel shared by overlap as in `playground/pair_balance.py`, and the bands print the share of
window pairs three Poisson deviations above and below expectation. Single subanchor pairs hold
too few counts to call depletion at all; pooling is what makes both directions testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cooler
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_balance import BANDS, overlap_weights  # noqa: E402


def beads(arm: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = sorted(arm.glob("*.cif"))[0]
    st, en, kd = [], [], []
    for line in f.read_text().splitlines():
        if line.startswith("ATOM"):
            x = line.split()
            st.append(int(x[16]))
            en.append(int(x[17]))
            kd.append(x[18])
    o = np.argsort(st)
    return np.array(st)[o], np.array(en)[o], np.array(kd)[o]


def windows(st: np.ndarray, en: np.ndarray, kd: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    gap = np.cumsum(kd == "anchor")
    gs: list[int] = []
    ge: list[int] = []
    cur: list[int] | None = None
    for s, e, k, g in zip(st, en, kd, gap, strict=True):
        if k != "subanchor":
            if cur:
                gs.append(cur[0])
                ge.append(cur[1])
            cur = None
            continue
        if cur is None or cur[2] != g or e - cur[0] > width:
            if cur:
                gs.append(cur[0])
                ge.append(cur[1])
            cur = [int(s), int(e), int(g)]
        else:
            cur[1] = int(e)
    if cur:
        gs.append(cur[0])
        ge.append(cur[1])
    return np.array(gs), np.array(ge)


def main() -> None:
    mcool, region, arm = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    st, en, kd = beads(arm)
    c = cooler.Cooler(f"{mcool}::/resolutions/1000")
    m = c.matrix(balance=False, sparse=True).fetch(region).tocsr()
    bs = c.bins().fetch(region)["start"].to_numpy()
    for width in (5_000, 10_000, 25_000):
        gs, ge = windows(st, en, kd, width)
        w = overlap_weights(bs, 1000, gs, ge)
        keep = np.flatnonzero(np.asarray(w.sum(1)).ravel() > 0)
        mm = m[keep][:, keep]
        mm = sp.triu(mm, 1) + sp.triu(mm, 1).T
        ca = (w[keep].T @ mm @ w[keep]).toarray()
        n = len(gs)
        i, j = np.triu_indices(n, 1)
        mid = (gs + ge) / 2
        ln = np.maximum(ge - gs, 1).astype(float)
        sep = mid[j] - mid[i]
        area = ln[i] * ln[j]
        obs = ca[i, j]
        edges = np.geomspace(max(sep.min(), 1), sep.max() + 1, 80)
        b = np.clip(np.searchsorted(edges, sep) - 1, 0, 78)
        dens = np.bincount(b, obs, minlength=80) / np.maximum(np.bincount(b, area, minlength=80), 1)
        exp = dens[b] * area
        z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-12))
        print(f"windows of up to {width // 1000} kb inside gaps: {n}, median width {np.median(ln) / 1e3:.1f} kb")
        for b0, b1 in BANDS[:3]:
            bd = (sep >= b0) & (sep < b1)
            print(
                f"  {b0 / 1e6:g}-{b1 / 1e6:g} Mb: {bd.sum():,} pairs, expected median {np.median(exp[bd]):.1f};"
                f" above +3 sd {np.mean(z[bd] > 3):.3f}, below -3 sd {np.mean(z[bd] < -3):.3f}"
            )


if __name__ == "__main__":
    main()
