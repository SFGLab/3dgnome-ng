"""The contact map source for the contact background: a planted contact is held, the rest of
the far pairs are not, and the binned singletons path is untouched.

    python harness/test_contact_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.pipeline.coarse.build import add_contact_background, anchor_map_ratio  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def synthetic_map(nb: int = 200, seed: int = 0) -> np.ndarray:
    """Poisson counts decaying with separation, one planted contact far out."""
    rng = np.random.default_rng(seed)
    d = np.abs(np.arange(nb)[:, None] - np.arange(nb)[None, :])
    lam = 400.0 / (1.0 + d) ** 1.0
    m = rng.poisson(lam).astype(np.float64)
    m = np.triu(m) + np.triu(m, 1).T
    m[20, 150] = m[150, 20] = 60.0  # expected there is about 3
    return m


def test_ratio_and_significance() -> None:
    m = synthetic_map()
    bins = np.arange(0, 200, 5)  # forty anchors, one per five bins
    ratio, sig = anchor_map_ratio(m, bins, z=3.0, pool=0)
    i, j = 4, 30  # bins 20 and 150
    check(
        "the planted contact is significant",
        bool(sig[i, j]) and bool(sig[j, i]),
        f"ratio {ratio[i, j]:.1f}",
    )
    far = np.abs(bins[:, None] - bins[None, :]) > 40
    frac = float((sig & far).sum() / far.sum())
    check("few other far pairs pass at three sigma", frac < 0.05, f"{frac:.3f}")
    check(
        "ratios are one on average away from the plant",
        abs(float(np.median(ratio[far])) - 1.0) < 0.3,
        f"median {np.median(ratio[far]):.2f}",
    )
    ratio_p, sig_p = anchor_map_ratio(m, bins, z=3.0, pool=1)
    check("pooling keeps the planted contact", bool(sig_p[i, j]))


def test_background_from_map() -> None:
    s = Settings()
    s.use_contact_background = True
    s.background_weight = 0.1
    s.background_range_bp = 100_000
    s.polymer_exponent = 0.3
    n = 6
    mids = [int(x) for x in np.arange(n) * 500_000 + 1_000_000]
    mat = np.full((n, n), -0.5)
    np.fill_diagonal(mat, 0.0)
    mat[0, 1] = mat[1, 0] = 1.5  # an arc pair stays as it is
    ratio = np.ones((n, n))
    sig = np.zeros((n, n), dtype=bool)
    ratio[1, 4] = ratio[4, 1] = 8.0
    sig[1, 4] = sig[4, 1] = True
    ratio[2, 5] = ratio[5, 2] = 8.0  # over expected but not significant: left alone
    out = add_contact_background(mat, mids, None, s, (ratio, sig))
    law = s.polymer_law()
    sep = abs(mids[4] - mids[1])
    bg = max(1.0, (sep / law.s0_bp) ** law.nu)
    want = -max(1.0, bg * 8.0 ** (-1.0 / 3.0))
    check(
        "the significant pair is held under the background",
        abs(out[1, 4] - want) < 1e-9 and out[1, 4] == out[4, 1],
        f"{out[1, 4]:.3f} vs {want:.3f}",
    )
    check("an insignificant excess is left on the repulsion", out[2, 5] == -0.5)
    check("the arc pair is untouched", out[0, 1] == 1.5)
    check(
        "with no map and no heatmap the matrix is returned as is",
        add_contact_background(mat, mids, None, s) is mat,
    )


def main() -> int:
    test_ratio_and_significance()
    test_background_from_map()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
