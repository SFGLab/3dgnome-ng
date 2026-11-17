"""Which contacts of the hole beads are overlaps.

    python playground/hole_contact_overlap.py HOLE_ARM ARM

Holes are taken from the first structure of HOLE_ARM, which should carry no hole beads. For the
first structure of ARM every bead pair within the battery's contact radius is classed as inside
one hole, across two holes, hole with block or block with block, and by genomic separation. Per
class and band the contact count per thousand beads of the class is printed, with the share of
those contacts closer than 0.7 of the median bond, the battery's overlap distance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from hole_readout import holes, read_spans  # noqa: E402
from law_spring_readout import read_models  # noqa: E402
from validation_battery import CONTACT_BEADS  # noqa: E402

BANDS = ((0, 25_000), (25_000, 100_000), (100_000, 400_000), (400_000, 2_000_000), (2_000_000, 10**10))
OVERLAP = 0.7


def main() -> None:
    hole_arm, arm = Path(sys.argv[1]), Path(sys.argv[2])
    _, mids0, kinds0 = read_models(hole_arm)
    st, en = read_spans(hole_arm)
    h = holes(mids0, st, en, kinds0)
    structs, mids, _ = read_models(arm)
    p = structs[0]
    hole_id = np.full(len(mids), -1)
    for k, (a, b) in enumerate(h):
        hole_id[(mids > a) & (mids < b)] = k
    bond = float(np.median(np.linalg.norm(np.diff(p[np.argsort(mids, kind="stable")], axis=0), axis=1)))
    pairs = cKDTree(p).query_pairs(CONTACT_BEADS * bond, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    sep = np.abs(mids[i] - mids[j])
    keep = sep > 0
    i, j, sep = i[keep], j[keep], sep[keep]
    d = np.linalg.norm(p[i] - p[j], axis=1) / bond
    hi, hj = hole_id[i] >= 0, hole_id[j] >= 0
    cls = {
        "one hole": hi & hj & (hole_id[i] == hole_id[j]),
        "two holes": hi & hj & (hole_id[i] != hole_id[j]),
        "hole with block": hi ^ hj,
        "block with block": ~hi & ~hj,
    }
    n_hole = int((hole_id >= 0).sum())
    n_block = len(mids) - n_hole
    print(f"{arm.name} structure 1: {len(mids)} beads, {n_hole} in {len(h)} holes, median bond {bond:.3f}")
    print("  contacts per thousand beads of the class, share closer than 0.7 bond")
    head = "  ".join(f"{lo // 1000}-{up // 1000 if up < 10**9 else 'inf'} kb".rjust(16) for lo, up in BANDS)
    print(f"  {'':>18s}  {head}")
    for name, m in cls.items():
        per = n_hole if name != "block with block" else n_block
        cells = []
        for lo, up in BANDS:
            sel = m & (sep >= lo) & (sep < up)
            c = int(sel.sum())
            share = float((d[sel] < OVERLAP).mean()) if c else float("nan")
            cells.append(f"{1000 * c / max(per, 1):7.1f} {share:6.2f}".rjust(16))
        print(f"  {name:>18s}  {'  '.join(cells)}")


if __name__ == "__main__":
    main()
