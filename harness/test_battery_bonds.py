"""The battery's overlap radius comes from the structure's subanchor bond, not a block mean.

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

from playground.validation_battery import bead_scale  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("battery bond scale checks\n")
    # a chain of ten bonds: subanchor bonds of 1.0, the two bonds that touch the anchor at bead
    # five at 3.0, and one subanchor bond at 2.0 so the median and the mean differ
    steps = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    pos = np.zeros((11, 3))
    pos[1:, 0] = np.cumsum(steps)
    anchor = np.zeros(11, dtype=np.bool_)
    anchor[5] = True
    got = bead_scale(pos, anchor)
    check(
        "the scale is the median subanchor bond, anchor bonds left out",
        np.isclose(got, 1.0),
        f"got {got:.3f}; the mean of all bonds is {steps.mean():.2f}",
    )
    got = bead_scale(pos, np.ones(11, dtype=np.bool_))
    check("with no subanchor bonds it falls back to the median of all", np.isclose(got, 1.0))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
