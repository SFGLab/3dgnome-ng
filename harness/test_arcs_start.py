"""Unit checks for the arcs stage's random walk start.

    python harness/test_arcs_start.py

Every anchor of a block starts at the block's centroid, and the solver descends from that
collapsed point to a compact minimum. A block of a few hundred anchors is bounded by its own
size; a chromosome solved as one block of four thousand folds to half the size the data
shows. The walk start places consecutive anchors at the law's distance for their gap along
random directions, so pairs no term acts on begin, and stay, near the law.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.ib.arcs import walk_start  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.util import seed_rng  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("arcs walk start checks")
    s = Settings()
    law = s.polymer_law()
    mids = np.array([0, 50_000, 150_000, 1_000_000, 1_020_000], dtype=np.int64)
    pos0 = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (5, 1))
    seed_rng(7)
    out = walk_start(pos0, mids, law)
    steps = np.linalg.norm(np.diff(out.astype(np.float64), axis=0), axis=1)
    want = np.array([law.background(int(b - a)) for a, b in zip(mids[:-1], mids[1:], strict=True)])
    check(
        "consecutive anchors sit at the law's distance for their gap",
        np.allclose(steps, want, rtol=1e-5),
        f"{steps.round(3)} vs {want.round(3)}",
    )
    check(
        "the walk keeps the block's centroid",
        np.allclose(out.mean(axis=0), [1, 2, 3], atol=1e-4),
        str(out.mean(axis=0).round(4)),
    )
    check(
        "it returns float32 of the input shape", out.dtype == np.float32 and out.shape == pos0.shape
    )
    seed_rng(7)
    again = walk_start(pos0, mids, law)
    check("the same seed gives the same walk", np.array_equal(out, again))
    seed_rng(8)
    other = walk_start(pos0, mids, law)
    check("another seed gives another walk", not np.array_equal(out, other))
    check("the start is the centroid unless asked", Settings().arcs_start == "centroid")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
