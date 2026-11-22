"""Overlap of the two ends of a region in the structures of the arms.

    python playground/uniform_ends_check.py OUT_DIR ARM [ARM ...]

OUT_DIR is the output of `slurm/ensemble/arms_trial.sh`. Per cell, region and arm it prints the
mean and the smallest distance between the centroids of the first megabase and the last megabase
of the region, in median bonds, and the count of structures with more than 300 bead pairs of the
two megabases within the contact radius of the battery. Two megabases of a region of 5 Mb that
share their space raise the pooled contacts at the largest separations, which the Spearman
correlation of the pooled map reads and SCC does not.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

END_BP = 1_000_000
CONTACT_BEADS = 1.33
PAIRS = 300

out = Path(sys.argv[1])
arms = sys.argv[2:]
rows = sorted(d.name[: -len(arms[0]) - 1] for d in out.glob(f"*_chr*_{arms[0]}") if d.is_dir())
print(f"mean distance of the end megabases in bonds | smallest | structures over {PAIRS} pairs between them")
totals = {arm: [[], 0, 0] for arm in arms}
for row in rows:
    line = row
    for arm in arms:
        apart, over, n = [], 0, 0
        for cif in sorted((out / f"{row}_{arm}").glob("*.cif")):
            atoms = [ln.split() for ln in cif.read_text().splitlines() if ln.startswith("ATOM")]
            pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in atoms])
            start = np.array([int(r[16]) for r in atoms])
            order = np.argsort(start, kind="stable")
            pos, start = pos[order], start[order]
            pos = pos / float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            first = pos[start < start[0] + END_BP]
            last = pos[start >= start[-1] - END_BP]
            apart.append(float(np.linalg.norm(first.mean(axis=0) - last.mean(axis=0))))
            hits = cKDTree(first).query_ball_tree(cKDTree(last), CONTACT_BEADS)
            over += int(sum(len(h) for h in hits) > PAIRS)
            n += 1
        if not n:
            line += f" | {arm}: no structures"
            continue
        totals[arm][0] += apart
        totals[arm][1] += over
        totals[arm][2] += n
        line += f" | {arm}: {np.mean(apart):5.1f} {min(apart):5.1f} {over}/{n}"
    print(line)
print("over the rows")
for arm in arms:
    t = totals[arm]
    if t[2]:
        print(f"  {arm:28s} {np.mean(t[0]):5.1f} {min(t[0]):5.1f} {t[1]}/{t[2]}")
