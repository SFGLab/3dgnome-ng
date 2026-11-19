"""Unit checks for the per bead coordinate table written beside every CIF."""

from __future__ import annotations

from pathlib import Path

from gnome3d.io import write_bead_table
from gnome3d.types import BeadOut


def test_one_header_line_and_one_row_per_bead(tmp_path: Path) -> None:
    beads = [
        BeadOut(100, 200, 1.5, -2.0, 0.25, "anchor"),
        BeadOut(200, 300, 3.0, 4.0, 5.0, "subanchor"),
    ]
    tmp = str(tmp_path)
    path = Path(tmp) / "m.beads.tsv"
    write_bead_table(str(path), "chr10", beads)
    lines = path.read_text().splitlines()
    assert len(lines) == 3, "one header line and one row per bead: " + str(len(lines))
    assert lines[0] == "#chr\tstart\tend\tx\ty\tz\tkind", "the header names the columns: " + str(
        lines[0]
    )
    assert lines[1] == "chr10\t100\t200\t1.5\t-2.0\t0.25\tanchor", (
        "a row carries the chromosome, span, coordinates and kind: " + str(lines[1])
    )
    assert lines[2].endswith("\tsubanchor"), "the second bead keeps its kind: " + str(lines[2])
