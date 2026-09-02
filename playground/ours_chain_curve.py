"""Our full bead chain, binned by consecutive genomic gap, for exact comparison with v4.

v4's level-3 export is the whole leaf chain, not an anchor backbone, so the comparable
measurement on our side is every bead rather than anchors only. Same bins, same target function.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "3dgnome-ng"))
from gnome3d.settings import Settings  # noqa: E402

cif, cfg = sys.argv[1], sys.argv[2]
s = Settings()
if not s.load_ini(cfg):
    raise SystemExit("cannot load config")

rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
mid = np.array([(int(r[16]) + int(r[17])) // 2 for r in rows], np.int64)
xyz = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], np.float64)
o = np.argsort(mid)
mid, xyz = mid[o], xyz[o]
gap = np.diff(mid)
d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

print(f"OURS {Path(cif).name}: {len(mid)} beads, consecutive gap median={np.median(gap):,.0f} bp")
print(f"{'gap band':>14s} {'n':>7s} {'median dist':>12s} {'target':>8s} {'ratio':>7s}")
edges = [0, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (gap >= lo) & (gap < hi)
    if m.sum() < 10:
        continue
    lab = f"{lo/1e3:.0f}-{hi/1e3:.0f}kb" if hi <= 1e6 else f">{lo/1e3:.0f}kb"
    t = s.genomic_length_to_distance(int(np.median(gap[m])))
    print(f"{lab:>14s} {m.sum():>7d} {np.median(d[m]):>12.3f} {t:>8.2f} {np.median(d[m]) / t:>7.3f}")
