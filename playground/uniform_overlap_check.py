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
# A row is a cell and a region that the first arm has structures for.
rows = sorted(d.name[: -len(arms[0]) - 1] for d in out.glob(f"*_chr*_{arms[0]}") if d.is_dir())
print(
    "pairs under 0.7 model units per thousand beads: every pair | pairs not next to each other"
    " in the file | bond median | beads"
)
totals = {arm: [[], []] for arm in arms}
for row in rows:
    line = row
    for arm in arms:
        every, apart, bond, beads = [], [], [], []
        for cif in sorted((out / f"{row}_{arm}").glob("*.cif")):
            atoms = [ln.split() for ln in cif.read_text().splitlines() if ln.startswith("ATOM")]
            pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in atoms])
            start = np.array([int(r[16]) for r in atoms])
            pos = pos[np.argsort(start, kind="stable")]
            q = cKDTree(pos).query_pairs(0.7, output_type="ndarray")
            k = 1000.0 / len(pos)
            every.append(k * len(q))
            apart.append(k * int((np.abs(q[:, 0] - q[:, 1]) > 1).sum()) if len(q) else 0.0)
            bond.append(float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
            beads.append(len(pos))
        if not every:
            line += f" | {arm}: no structures"
            continue
        totals[arm][0].append(float(np.mean(every)))
        totals[arm][1].append(float(np.mean(apart)))
        line += (
            f" | {arm}: {np.mean(every):6.1f} {np.mean(apart):6.1f} {np.mean(bond):.2f}"
            f" {int(np.mean(beads))}"
        )
    print(line)
print("mean over the rows")
for arm in arms:
    if totals[arm][0]:
        print(f"  {arm:28s} {np.mean(totals[arm][0]):6.1f} {np.mean(totals[arm][1]):6.1f}")
