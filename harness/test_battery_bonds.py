"""The battery's overlap radius comes from a block's typical bond, not its mean.

    python harness/test_battery_bonds.py

The mean bond of a block is inflated by the bonds that touch an anchor, which sit at about 1.6
beads against 1.03 for a subanchor bond, and by the few sub kilobase bonds at 2.7. On a real
structure that put the counting radius 12 percent above the bead, and 24 percent above it on
an arm whose wall stretched those long bonds further, so an arm with fewer close pairs scored
more overlaps. The median is what a bead's own spacing is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.validation_battery import block_bonds  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("battery bond scale checks\n")
    # one block: eight bonds of 1.0 and two of 3.0 along x
    steps = np.array([1.0] * 4 + [3.0] + [1.0] * 4 + [3.0])
    pos = np.zeros((11, 3))
    pos[1:, 0] = np.cumsum(steps)
    owner = np.zeros(11, dtype=np.int64)
    got = block_bonds(pos, owner)
    check(
        "a block's bond scale is its typical bond, not its mean",
        np.isclose(got[0], 1.0),
        f"got {got[0]:.3f}, mean would be {steps.mean():.3f}",
    )
    # two blocks; the second all 2.0
    pos2 = np.zeros((6, 3))
    pos2[1:, 0] = np.cumsum([2.0] * 5)
    both = np.vstack([pos, pos2 + np.array([100.0, 0, 0])])
    owner2 = np.concatenate([np.zeros(11, dtype=np.int64), np.ones(6, dtype=np.int64)])
    got2 = block_bonds(both, owner2)
    check(
        "each block on its own scale",
        np.isclose(got2[0], 1.0) and np.isclose(got2[1], 2.0),
        f"{got2}",
    )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
