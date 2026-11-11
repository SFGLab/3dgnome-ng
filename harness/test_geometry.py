"""Unit checks for the chain smoothing shared by the figures and the website,
`gnome3d/geometry.py`.

    python harness/test_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.geometry import smooth_chain  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def main() -> int:
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(20, 3)), axis=0)
    sm = smooth_chain(pos, per_bond=4)
    check("four points per bond and the last bead", sm.shape == (19 * 4 + 1, 3), str(sm.shape))
    check("the curve passes through every bead", np.allclose(sm[::4], pos))
    line = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], axis=1)
    check("a straight chain stays straight", np.allclose(smooth_chain(line, 5)[:, 1:], 0.0))
    check("two beads are returned as they are", np.array_equal(smooth_chain(pos[:2], 4), pos[:2]))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
