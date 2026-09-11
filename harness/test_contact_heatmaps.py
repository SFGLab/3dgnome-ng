"""Unit checks for the singleton contact heatmaps.

    python harness/test_contact_heatmaps.py

The anchor level heatmap scales the arc targets and carries the contact background, and
production reads it with the subanchor heat term off. It used to be cut out of the full
subanchor matrix, an N by N float64 held per block whether or not anything read it, which is
15 GB on a 43,000 bead block. Built directly at anchor resolution it must be the same matrix
to the byte, and no subanchor matrix is returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse import build_state  # noqa: E402
from gnome3d.pipeline.coarse.build import build_contact_heatmaps  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("contact heatmap checks")
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    s.data_dir = "data/GM12878"
    chrs, region = parse_chrs_arg("chr1:18288319-20307135")
    data = ContactData.from_files(s, chrs, region)
    state = build_state(s, data, chrs, region)
    ibs = [c for c in state.clusters if int(c.level) == 3 and len(c.children) > 1]
    check("the region has blocks to test on", len(ibs) > 0, f"{len(ibs)} blocks")
    for k, ib in enumerate(ibs[:3]):
        active = list(ib.children)
        full_anchor, sub = build_contact_heatmaps(state, active, "chr1")
        direct_anchor, none = build_contact_heatmaps(state, active, "chr1", with_subanchor=False)
        check(
            f"block {k}: the direct anchor heatmap is the cut out one, exactly",
            full_anchor.shape == direct_anchor.shape and np.array_equal(full_anchor, direct_anchor),
            f"{full_anchor.shape}, max abs diff {np.abs(full_anchor - direct_anchor).max():.3g}"
            if full_anchor.shape == direct_anchor.shape
            else f"{full_anchor.shape} vs {direct_anchor.shape}",
        )
        check(
            f"block {k}: no subanchor matrix without the heat term",
            none is None and sub is not None,
        )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
