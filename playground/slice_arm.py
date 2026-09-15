"""Cut every structure of an arm to one genomic window, as a new arm directory.

    python playground/slice_arm.py ARM_DIR OUT_DIR chr1:850000-3500000

Keeps every bead whose start lies inside the window and everything else of the cif as it is,
so the battery reads the slice the way it reads a whole structure. A model of a wide region
can then be scored on the window a narrower model covers, against that window's Hi-C alone.
"""

from __future__ import annotations

import sys
from pathlib import Path


def slice_cif(src: Path, dst: Path, lo: int, hi: int) -> int:
    kept = 0
    out: list[str] = []
    for line in src.read_text().splitlines():
        if line.startswith("ATOM"):
            start = int(line.split()[16])
            if not (lo <= start <= hi):
                continue
            kept += 1
        out.append(line)
    dst.write_text("\n".join(out) + "\n")
    return kept


def main() -> None:
    arm, out, region = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    span = region.split(":")[1]
    lo, hi = (int(v) for v in span.split("-"))
    out.mkdir(parents=True, exist_ok=True)
    for cif in sorted(arm.glob("*.cif")):
        n = slice_cif(cif, out / cif.name, lo, hi)
        print(f"{cif.name}: {n} beads in {region}")


if __name__ == "__main__":
    main()
