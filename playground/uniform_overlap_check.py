"""Overlapping pairs of the arms by a count that does not read the bead kind.

    python playground/uniform_overlap_check.py OUT_DIR ARM [ARM ...]

OUT_DIR is the output of `slurm/ensemble/arms_trial.sh`. Per cell, region and arm it prints the
pairs of beads closer than 0.7 model units per thousand beads, over every pair and over the pairs
that are not next to each other in the file sorted by start, the median bond and the bead count.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

out = Path(sys.argv[1])
arms = sys.argv[2:]
print("pairs under 0.7 model units per thousand beads: every pair | pairs not next to each other in the file | bond median | beads")
for cell in ("GM12878", "H1ESC", "HFFC6"):
    for region in ("chr1_10000000_15000000", "chr1_20000000_25000000"):
        line = f"{cell} {region[5:]}"
        for arm in arms:
            every, apart, bond, beads = [], [], [], []
            for cif in sorted((out / f"{cell}_{region}_{arm}").glob("*.cif")):
                rows = [ln.split() for ln in cif.read_text().splitlines() if ln.startswith("ATOM")]
                pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows])
                start = np.array([int(r[16]) for r in rows])
                pos = pos[np.argsort(start, kind="stable")]
                q = cKDTree(pos).query_pairs(0.7, output_type="ndarray")
                k = 1000.0 / len(pos)
                every.append(k * len(q))
                apart.append(k * int((np.abs(q[:, 0] - q[:, 1]) > 1).sum()) if len(q) else 0.0)
                bond.append(float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
                beads.append(len(pos))
            line += f" | {arm}: {np.mean(every):6.1f} {np.mean(apart):6.1f} {np.mean(bond):.2f} {int(np.mean(beads))}"
        print(line)
