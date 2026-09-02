"""Do the structures reproduce their own distance targets?

`genomic_length_to_distance` is what the chain term asks for: 1.50 at 1 kb rising to 10.40 at
50 kb, a 6.9-fold spread. The realised curve rises about 1.2-fold over the same span, so either
the chain term is not being met or something else dominates it.

Compares, per consecutive bead pair, the realised separation against the target implied by that
pair's genomic gap, and does the same for pairs k beads apart so the comparison extends past the
bond itself.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "3dgnome-ng"))
from gnome3d.settings import Settings  # noqa: E402

model_dir = Path(sys.argv[1])
cfg = sys.argv[2]

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")

paths = sorted(model_dir.glob("*.cif"),
               key=lambda p: int(re.search(r"_s(\d+)\.cif$", p.name).group(1)))[:3]

acc: dict[int, list[float]] = {}
for p in paths:
    rows = [ln.split() for ln in open(p) if ln.startswith("ATOM")]
    xyz = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], np.float64)
    st = np.array([int(r[16]) for r in rows], np.int64)
    en = np.array([int(r[17]) for r in rows], np.int64)
    mid = (st + en) // 2
    for k in (1, 2, 5, 10, 25, 50):
        gap = np.abs(mid[k:] - mid[:-k]).astype(np.float64)
        d = np.linalg.norm(xyz[k:] - xyz[:-k], axis=1)
        # only pairs whose genomic gap is a plausible chain span, so centromere jumps are out
        m = (gap > 0) & (gap < 500_000)
        tgt = np.array([s.genomic_length_to_distance(int(g)) for g in gap[m]])
        acc.setdefault(k, []).extend((d[m] / tgt).tolist())

print(f"[tgt] {model_dir.parent.name}/{model_dir.name}, {len(paths)} models")
print(f"[tgt] {'k':>4s} {'n':>9s} {'median realised/target':>24s} {'p10':>8s} {'p90':>8s}")
for k, v in acc.items():
    a = np.array(v)
    print(f"[tgt] {k:>4d} {len(a):>9d} {np.median(a):>24.3f} "
          f"{np.percentile(a, 10):>8.3f} {np.percentile(a, 90):>8.3f}")
print("[tgt] 1.0 means the structure sits exactly where the chain term asks")
