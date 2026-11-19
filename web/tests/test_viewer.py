"""The viewer's file: the smoothed chain and the boxes around promoters and enhancers.

    python web/tests/test_viewer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from _check import check, done

from gnome3d_web.viewer import viewer_payload


def write_chain(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gnome3d.io import write_cif
    from gnome3d.types import BeadOut

    beads = [BeadOut(s, s + 2000, s / 2000.0, 0.0, 0.0, "subanchor") for s in range(0, 100_000, 2000)]
    write_cif(str(path), beads[::-1])  # file order need not be genomic order


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        cif = Path(d) / "m.cif"
        write_chain(cif)
        elements = [
            {"kind": "promoter", "name": "A", "start": 20_000, "end": 22_000},
            {"kind": "enhancer", "name": "", "start": 60_500, "end": 61_500},
            {"kind": "enhancer", "name": "", "start": 500_000, "end": 501_000},
        ]
        p = viewer_payload(cif, elements, per_bond=4, pad=0.1)
        chain = np.array(p["chain"])
        check("the chain is smoothed in genomic order", len(chain) == 49 * 4 + 1 and chain[0][0] < chain[-1][0], str(len(chain)))
        boxes = p["boxes"]
        check("an element off the chain gets no box", len(boxes) == 2, str(len(boxes)))
        a = boxes[0]
        # bead midpoints are at s + 1000, x = s / 2000; the promoter spans x 9.5 to 10.5 plus the pad
        check(
            "a promoter's box spans the chain over its interval, padded",
            a["kind"] == "promoter" and abs(a["min"][0] - 9.4) < 0.3 and abs(a["max"][0] - 10.6) < 0.3,
            str((a["min"], a["max"])),
        )
        e = boxes[1]
        check("a short enhancer still gets a box at its place", 29.5 < (e["min"][0] + e["max"][0]) / 2 < 30.5, str((e["min"], e["max"])))
    return done()


if __name__ == "__main__":
    sys.exit(main())
