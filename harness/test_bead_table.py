"""Unit checks for the per bead coordinate table written beside every CIF.

    python harness/test_bead_table.py

The table is one row per bead, chromosome, genomic start and end, x, y, z and the bead kind,
tab separated with one header line, so a reader that cannot parse mmCIF still gets the model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.io import write_bead_table  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("bead table checks")
    beads = [
        BeadOut(100, 200, 1.5, -2.0, 0.25, "anchor"),
        BeadOut(200, 300, 3.0, 4.0, 5.0, "subanchor"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.beads.tsv"
        write_bead_table(str(path), "chr10", beads)
        lines = path.read_text().splitlines()
        check("one header line and one row per bead", len(lines) == 3, str(len(lines)))
        check(
            "the header names the columns",
            lines[0] == "#chr\tstart\tend\tx\ty\tz\tkind",
            lines[0],
        )
        check(
            "a row carries the chromosome, span, coordinates and kind",
            lines[1] == "chr10\t100\t200\t1.5\t-2.0\t0.25\tanchor",
            lines[1],
        )
        check("the second bead keeps its kind", lines[2].endswith("\tsubanchor"), lines[2])
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
