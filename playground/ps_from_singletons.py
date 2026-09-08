"""Fit the contact decay exponent on singletons files by hand, with the run's own fit.

    python playground/ps_from_singletons.py data/GM12878/GM12878_hic_25kb_singletons.bedpe ...

Routes through `gnome3d.polymer.fit_contact_exponent` so this can never disagree with what a
run measures at load. Prints the slope, `nu`, the band, the pairs in it and the refusal
reason when there is one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.polymer import fit_contact_exponent  # noqa: E402
from gnome3d.types import SingletonContact  # noqa: E402


def read(path: Path) -> list[SingletonContact]:
    """Seven column BEDPE rows as the loader returns them, midpoints and score."""
    out: list[SingletonContact] = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 7 or p[0].startswith("#"):
                continue
            out.append((p[0], (int(p[1]) + int(p[2])) // 2, p[3], (int(p[4]) + int(p[5])) // 2, int(float(p[6]))))
    return out


for arg in sys.argv[1:]:
    path = Path(arg)
    f = fit_contact_exponent(read(path))
    tail = "" if f.ok else f"  REFUSED: {f.reason}"
    print(
        f"{path.name:<48s} pairs in band {f.n_pairs:>11,}  {f.lo // 1000}-{f.hi // 1000}kb  "
        f"slope {f.slope:+.3f}  nu {f.nu:.3f}{tail}"
    )
