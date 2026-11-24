"""The 3D distance of the two ends of the loops of a band of separation, in the structures of arms.

    python playground/rnapii/loop_end_distances.py CHROM CTCF.bedpe RNAPII.bedpe ARM_DIR [ARM_DIR ...]

For each arm and each of three sets of pairs 100 to 500 kb apart, the ends of the RNAPII loops, the
ends of the CTCF loops and random pairs of the separations of the RNAPII loops: the median distance
in bonds over the structures, its 5th percentile and the share of the pairs within 1.33 bonds, the
contact radius of the scores, within 2 and within 3. An end is the bead nearest the middle of its
anchor, within 3 kb. The last column is the number of pairs of beads more than three beads apart
along the chain and within half a bond, per 1,000 beads, the overlaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import load  # noqa: E402

LO, HI = 100_000, 500_000


def ends(path: str, chrom: str) -> np.ndarray:
    """Returns the middles of the two anchors of the loops of the band on a chromosome."""
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] == chrom and f[3] == chrom:
            a, b = (int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2
            if LO <= abs(b - a) <= HI:
                out.append((min(a, b), max(a, b)))
    return np.array(out)


def main() -> None:
    chrom, ctcf, rnapii = sys.argv[1:4]
    sets = {"RNAPII loop ends": ends(rnapii, chrom), "CTCF loop ends": ends(ctcf, chrom)}
    for folder in map(Path, sys.argv[4:]):
        rng = np.random.default_rng(0)
        got: dict[str, list[np.ndarray]] = {k: [] for k in (*sets, "random pairs")}
        close = []
        for cif in sorted(folder.glob("*.cif")):
            pos, mid, _, _ = load(cif)
            bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

            def distance(pairs: np.ndarray) -> np.ndarray:
                a = np.searchsorted(mid, pairs[:, 0]).clip(0, len(mid) - 1)
                b = np.searchsorted(mid, pairs[:, 1]).clip(0, len(mid) - 1)
                ok = (np.abs(mid[a] - pairs[:, 0]) < 3000) & (np.abs(mid[b] - pairs[:, 1]) < 3000)
                return np.linalg.norm(pos[a[ok]] - pos[b[ok]], axis=1) / bond

            for name, pairs in sets.items():
                got[name].append(distance(pairs))
            span = sets["RNAPII loop ends"][:, 1] - sets["RNAPII loop ends"][:, 0]
            start = rng.integers(int(mid.min()), int(mid.max()) - HI - 1, len(span))
            got["random pairs"].append(distance(np.c_[start, start + span]))
            near = cKDTree(pos).query_pairs(0.5 * bond, output_type="ndarray")
            close.append(1000 * int((np.abs(near[:, 0] - near[:, 1]) > 3).sum()) / len(pos))
        print(f"{folder.parent.name}/{folder.name}: {len(close)} structures, {np.mean(close):.1f} overlaps per 1,000 beads")
        for name, parts in got.items():
            d = np.concatenate(parts)
            print(
                f"  {name:>18s}  pairs {len(d):7,d}  median {np.median(d):5.2f}  5th percentile {np.percentile(d, 5):5.2f}"
                f"  within 1.33 {np.mean(d <= 1.33):.3f}  within 2 {np.mean(d <= 2):.3f}  within 3 {np.mean(d <= 3):.3f}"
            )


if __name__ == "__main__":
    main()
