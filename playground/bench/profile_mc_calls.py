"""Summarise an MC profile log produced by GNOME3D_MC_PROFILE.

Reads the CSV written by gnome3d.mc when GNOME3D_MC_PROFILE was set during a
run, and reports:
  - call count and total wall time per level (heatmap / arcs / smooth / ib)
  - N distribution per level (quartiles + a coarse histogram)
  - K distribution per level
  - top contributors: which (level, N-bucket) ate the most wall time

The point of the report is one decision: at what N values does mc_heatmap
actually run in production, and what fraction of total MC time does heatmap
account for?  That decides whether a JAX/GPU heatmap backend is worth porting.

Usage:
    GNOME3D_MC_PROFILE=/tmp/mc.csv python -m gnome3d ...    # produce the log
    python playground/profile_mc_calls.py /tmp/mc.csv       # summarise
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["N"] = int(r["N"])
        r["K"] = int(r["K"])
        r["n_steps_per_batch"] = int(r["n_steps_per_batch"])
        r["wall_s"] = float(r["wall_s"])
        r["score"] = float(r["score"])
    return rows


def quantiles(xs: list[float], qs: list[float]) -> list[float]:
    xs = sorted(xs)
    if not xs:
        return [float("nan")] * len(qs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs]


def histogram(values: list[int], buckets: list[tuple[int, int, str]]) -> list[tuple[str, int]]:
    """Bucket values into (lo, hi, label) ranges."""
    out = [(label, 0) for _, _, label in buckets]
    for v in values:
        for i, (lo, hi, _label) in enumerate(buckets):
            if lo <= v < hi:
                out[i] = (out[i][0], out[i][1] + 1)
                break
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    rows = load(path)
    if not rows:
        print(f"empty log: {path}", file=sys.stderr)
        sys.exit(1)

    total_wall = sum(r["wall_s"] for r in rows)
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    # Per-level summary
    print(f"=== MC PROFILE SUMMARY ({path}) ===")
    print(f"total calls: {len(rows)}   total wall: {total_wall:.2f}s\n")

    print(f"{'level':>8}  {'calls':>6}  {'wall_s':>8}  {'wall_%':>6}  "
          f"{'N min':>6} {'N p25':>6} {'N med':>6} {'N p75':>6} {'N max':>6}  "
          f"{'K med':>6}")
    print("-" * 88)
    for level in sorted(by_level):
        sub = by_level[level]
        wall = sum(r["wall_s"] for r in sub)
        Ns = [r["N"] for r in sub]
        Ks = [r["K"] for r in sub]
        nq = quantiles([float(n) for n in Ns], [0.0, 0.25, 0.5, 0.75, 0.999])
        k_med = quantiles([float(k) for k in Ks], [0.5])[0]
        print(f"{level:>8}  {len(sub):>6}  {wall:>8.2f}  {wall / total_wall * 100:>5.1f}%  "
              f"{int(nq[0]):>6} {int(nq[1]):>6} {int(nq[2]):>6} {int(nq[3]):>6} {int(nq[4]):>6}  "
              f"{int(k_med):>6}")

    # N histogram per level (buckets aligned with the JAX-vs-numba crossover regimes)
    buckets = [
        (0,      32,    "<32"),
        (32,     128,   "32-127"),
        (128,    512,   "128-511"),
        (512,    2048,  "512-2047"),
        (2048,   8192,  "2048-8191"),
        (8192,   10**9, ">=8192"),
    ]
    print("\n=== N DISTRIBUTION PER LEVEL ===")
    print(f"{'level':>8}  " + "  ".join(f"{label:>9}" for _, _, label in buckets))
    print("-" * (10 + 11 * len(buckets)))
    for level in sorted(by_level):
        hist = histogram([r["N"] for r in by_level[level]], buckets)
        print(f"{level:>8}  " + "  ".join(f"{count:>9}" for _, count in hist))

    # Wall time per (level, N-bucket) — the load-bearing view
    print("\n=== WALL TIME PER (level, N-bucket) ===")
    print(f"{'level':>8}  " + "  ".join(f"{label:>9}" for _, _, label in buckets))
    print("-" * (10 + 11 * len(buckets)))
    for level in sorted(by_level):
        wall_per_bucket = [0.0] * len(buckets)
        for r in by_level[level]:
            for i, (lo, hi, _) in enumerate(buckets):
                if lo <= r["N"] < hi:
                    wall_per_bucket[i] += r["wall_s"]
                    break
        print(f"{level:>8}  " + "  ".join(f"{w:>9.2f}" for w in wall_per_bucket))

    # JAX-worthiness verdict
    print("\n=== JAX HEATMAP PORT DECISION ===")
    hm_rows = by_level.get("heatmap", [])
    if not hm_rows:
        print("  No heatmap calls in this log.")
    else:
        hm_wall = sum(r["wall_s"] for r in hm_rows)
        hm_wall_at_n_ge_1024 = sum(r["wall_s"] for r in hm_rows if r["N"] >= 1024)
        hm_wall_at_n_ge_256 = sum(r["wall_s"] for r in hm_rows if r["N"] >= 256)
        share_hm = hm_wall / total_wall * 100 if total_wall > 0 else 0
        share_1024 = hm_wall_at_n_ge_1024 / hm_wall * 100 if hm_wall > 0 else 0
        share_256 = hm_wall_at_n_ge_256 / hm_wall * 100 if hm_wall > 0 else 0
        print(f"  heatmap fraction of total MC time:     {share_hm:5.1f}%")
        print(f"  fraction of heatmap time at N >= 1024: {share_1024:5.1f}%  (JAX f32 ~4-10x faster here)")
        print(f"  fraction of heatmap time at N >= 256:  {share_256:5.1f}%   (JAX f32 ~1-4x faster here)")
        # Estimated total speedup if we ported JAX with the bench numbers:
        # be conservative — at N>=1024 assume 5x, at N>=256 (under 1024) assume
        # 1.5x, below 256 assume 0.5x (numba still wins).
        speedup = {}
        speedup["<256"] = 0.5
        speedup["256-1023"] = 1.5
        speedup[">=1024"] = 5.0
        saved = 0.0
        for r in hm_rows:
            if r["N"] >= 1024:
                f = speedup[">=1024"]
            elif r["N"] >= 256:
                f = speedup["256-1023"]
            else:
                f = speedup["<256"]
            # If f > 1, JAX saves time; if f < 1, JAX costs time.
            saved += r["wall_s"] - (r["wall_s"] / f)
        net_pct = saved / total_wall * 100 if total_wall > 0 else 0
        print(f"  estimated total-MC time saved by JAX port:  {saved:6.2f}s "
              f"({net_pct:+.1f}% of total)")
        print("  (uses bench speedups: 0.5x at N<256, 1.5x at 256<=N<1024, 5x at N>=1024)")


if __name__ == "__main__":
    main()
