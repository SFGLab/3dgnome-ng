"""Which bead pairs overlap inside a block.

    python playground/overlap_classes.py ARM [ARM ...]

For the first structure of each arm, every non neighbour pair of subanchors closer than 0.7 of
the median bond is classed as inside one gap, the stretch between two consecutive anchors, or
across two gaps, and binned by genomic separation. Per class the count per thousand beads is
printed, with, for pairs across gaps, the median distance between the anchors that start the two
gaps over the law, which says whether those anchors sit closer than the law puts them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground"))

from law_spring_readout import nu_from_log, read_models  # noqa: E402

BANDS = ((0, 10_000), (10_000, 50_000), (50_000, 200_000), (200_000, 1_000_000), (1_000_000, 10**10))


def main() -> None:
    for arm in map(Path, sys.argv[1:]):
        structs, mids, kinds = read_models(arm)
        order = np.argsort(mids, kind="stable")
        p, m, k = structs[0][order], mids[order], kinds[order]
        anc = k == "anchor"
        gap = np.cumsum(anc)
        bond = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        pairs = cKDTree(p).query_pairs(0.7 * bond, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        keep = (np.abs(i - j) > 1) & ~anc[i] & ~anc[j]
        i, j = i[keep], j[keep]
        sep = np.abs(m[i] - m[j])
        same = gap[i] == gap[j]
        nu = nu_from_log(arm.with_suffix(".log"))
        apos = np.flatnonzero(anc)
        n = len(m)
        print(f"{arm.name}: {n} beads, {int(anc.sum())} anchors, {len(i)} overlapping subanchor pairs")
        for name, cls in (("one gap", same), ("two gaps", ~same)):
            cells = []
            for lo, up in BANDS:
                sel = cls & (sep >= lo) & (sep < up)
                cells.append(f"{1000 * sel.sum() / n:7.1f}")
            print(f"  {name:>9s} per thousand by separation " + " ".join(cells))
        two = ~same
        if two.any():
            ga, gb = gap[i[two]], gap[j[two]]
            ok = (ga >= 1) & (gb >= 1)
            a1, a2 = apos[ga[ok] - 1], apos[gb[ok] - 1]
            d = np.linalg.norm(p[a1] - p[a2], axis=1) / bond
            law = np.maximum(1.0, (np.abs(m[a1] - m[a2]) / 1000.0) ** nu)
            r = d / law
            print(f"  anchors starting two overlapping gaps: distance over law median {np.median(r):.2f}, 10% {np.quantile(r, 0.1):.2f}")


if __name__ == "__main__":
    main()
