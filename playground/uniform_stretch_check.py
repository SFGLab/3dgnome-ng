"""Stretched pieces of the structures of the arms.

    python playground/uniform_stretch_check.py OUT_DIR ARM [ARM ...]

OUT_DIR is the output of `slurm/ensemble/arms_trial.sh`. Per cell, region and arm it prints the
radius of gyration in median bonds, the bonds over ten median bonds per structure and the count of
structures with a stretched piece. A structure has a stretched piece when beads are further from
the median position than four times the median of that distance. A chain that starts on a line
beyond its last fixed bead shows here and not in the pooled scores, since a piece far from the
rest makes no contact.
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
arms = sys.argv[2:]
rows = sorted(d.name[: -len(arms[0]) - 1] for d in out.glob(f"*_chr*_{arms[0]}") if d.is_dir())
print("radius of gyration in bonds | bonds over ten per structure | structures with a stretched piece")
totals = {arm: [[], [], 0, 0] for arm in arms}
for row in rows:
    line = row
    for arm in arms:
        rg, long, stretched, n = [], [], 0, 0
        for cif in sorted((out / f"{row}_{arm}").glob("*.cif")):
            atoms = [ln.split() for ln in cif.read_text().splitlines() if ln.startswith("ATOM")]
            pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in atoms])
            start = np.array([int(r[16]) for r in atoms])
            pos = pos[np.argsort(start, kind="stable")]
            bonds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            bond = float(np.median(bonds))
            far = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
            rg.append(float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean())) / bond)
            long.append(int((bonds > 10 * bond).sum()))
            stretched += int((far > 4 * np.median(far)).any())
            n += 1
        if not n:
            line += f" | {arm}: no structures"
            continue
        totals[arm][0].append(float(np.mean(rg)))
        totals[arm][1].append(float(np.mean(long)))
        totals[arm][2] += stretched
        totals[arm][3] += n
        line += f" | {arm}: {np.mean(rg):6.1f} {np.mean(long):5.1f} {stretched}/{n}"
    print(line)
print("mean over the rows")
for arm in arms:
    if totals[arm][3]:
        t = totals[arm]
        print(f"  {arm:28s} {np.mean(t[0]):6.1f} {np.mean(t[1]):5.1f} {t[2]}/{t[3]}")
