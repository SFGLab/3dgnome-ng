"""Validate the vectorized coarse heatmap helpers == the original O(N^2) loops.

`CoarseModel._create_distance_heatmap` / `_get_diagonal_size` were rewritten from
Python list-of-list double loops (which blow up on large Hi-C matrices) to numpy.
This asserts byte/float-exact equality against a reference re-implementation of
the original loop logic, on symmetric AND asymmetric matrices (the reference uses
the upper triangle + mirror, so asymmetry is the interesting case), intra + inter.

    python playground/refactor/validate_heatmap_numpy.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from gnome3d import log  # noqa: E402
from gnome3d.coarse import CoarseModel  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def _ref_diag(h, n):  # original loop
    for w in range(n):
        for i in range(n - w):
            if h[i][i + w] > 1e-6:
                return w
    return 0


def _ref_distance(s, h, n, inter):  # original loop
    diag = _ref_diag(h, n)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = h[i][j]
            if val < 1e-6:
                dist[i][j] = 0.0
            elif abs(i - j) < diag:
                dist[i][j] = -1.0
            else:
                dist[i][j] = (
                    s.freq_to_dist_heatmap_inter(val) if inter else s.freq_to_dist_heatmap(val)
                )
            dist[j][i] = dist[i][j]
    vals = [dist[i][j] for i in range(n) for j in range(n) if dist[i][j] > 0]
    avg = sum(vals) / len(vals) if vals else 1.0
    max_d = avg * s.heatmap_distance_stretching
    for i in range(n):
        for j in range(n):
            if dist[i][j] > max_d:
                dist[i][j] = max_d
    return np.array(dist), avg


def _make(n, seed, symmetric):
    rng = np.random.default_rng(seed)
    h = rng.random((n, n))
    h[rng.random((n, n)) < 0.5] = 0.0  # sparsity
    if symmetric:
        h = np.triu(h) + np.triu(h, 1).T
    return h


def main() -> int:
    log.setup(0)
    s = Settings()
    s.load_ini("data/GM12878/config_dryrun.ini")
    cm = CoarseModel(s)

    ok = True
    cases = [(n, sym, inter) for n in (1, 2, 7, 33, 128) for sym in (True, False) for inter in (False, True)]
    worst = 0.0
    for n, sym, inter in cases:
        h = _make(n, seed=n * (2 if sym else 3) + int(inter), symmetric=sym)
        hl = h.tolist()

        d_new, avg_new = cm._create_distance_heatmap(hl, n, inter=inter)  # pyright: ignore[reportPrivateUsage]
        d_ref, avg_ref = _ref_distance(s, hl, n, inter)
        diag_new = cm._get_diagonal_size(hl, n)  # pyright: ignore[reportPrivateUsage]
        diag_ref = _ref_diag(hl, n)

        d_ok = np.allclose(d_new, d_ref, rtol=0, atol=1e-9, equal_nan=True)
        a_ok = abs(avg_new - avg_ref) <= 1e-9 * max(abs(avg_ref), 1.0)
        g_ok = diag_new == diag_ref
        worst = max(worst, float(np.max(np.abs(d_new - d_ref))) if d_new.size else 0.0)
        if not (d_ok and a_ok and g_ok):
            ok = False
            print(f"  FAIL n={n} sym={sym} inter={inter}: dist={d_ok} avg={a_ok}(({avg_new} vs {avg_ref})) diag={g_ok}({diag_new} vs {diag_ref})")

    print(f"  {len(cases)} cases (n in 1..128, sym/asym, intra/inter); max |dist diff| = {worst:.2e}")
    print("PASS (vectorized == loop, exact)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
