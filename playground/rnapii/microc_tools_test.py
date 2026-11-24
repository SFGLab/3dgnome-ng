"""Tests of one tool against the others at the Micro-C loops of a region.

    python playground/rnapii/microc_tools_test.py <table.json> <first arm>

The table is that of `microc_loop_compare.py --json` with the key `moved`. Two tests per other arm,
both two sided.

- The share of loops placed above the moved loops. The loops and the moved loops are resampled
  with replacement 10,000 times, the same resample for every arm, and the share is taken on each.
  The p value is twice the smaller share of resamples on one side of zero and no less than one
  over the number of resamples.
- The share of loops with beads at both ends, the loops an arm scores. The exact sign test on the
  loops that one arm scores and the other does not.

A loop that an arm cannot score counts as not placed, as in the table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CLOSER = 0.8
DRAWS = 10_000


def main() -> None:
    table = json.loads(Path(sys.argv[1]).read_text())
    first = sys.argv[2]
    arms = [a for a in table["moved"] if a != first]
    real = {a: np.array(table["loops"][a], dtype=float) for a in [first, *arms]}
    moved = {a: np.array(table["moved"][a], dtype=float) for a in [first, *arms]}
    n, m = len(real[first]), len(moved[first])
    rng = np.random.default_rng(0)
    ri, mi = rng.integers(0, n, (DRAWS, n)), rng.integers(0, m, (DRAWS, m))

    def above(a: str) -> tuple[float, np.ndarray]:
        r, q = real[a] <= CLOSER, moved[a] <= CLOSER
        return float(r.mean() - q.mean()), r[ri].mean(axis=1) - q[mi].mean(axis=1)

    v0, d0 = above(first)
    print(f"{Path(sys.argv[1]).name}: {n} loops, {m} moved loops, {DRAWS} resamples, first arm {first}")
    for a in arms:
        v, d = above(a)
        diff = d0 - d
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = max(2 * min(float((diff <= 0).mean()), float((diff >= 0).mean())), 1 / DRAWS)
        s0, s = np.isfinite(real[first]), np.isfinite(real[a])
        only0, only = int((s0 & ~s).sum()), int((s & ~s0).sum())
        pb = float(binomtest(only0, only0 + only, 0.5).pvalue) if only0 + only else 1.0
        print(
            f"  against {a}: placed above moved {v0:.3f} and {v:.3f}, difference {v0 - v:+.3f}, 95 percent interval "
            f"{lo:+.3f} to {hi:+.3f}, p {p:.4f}; beads at both ends {s0.mean():.3f} and {s.mean():.3f}, "
            f"{only0} loops for the first arm alone and {only} for the other, p {pb:.2g}"
        )


if __name__ == "__main__":
    main()
