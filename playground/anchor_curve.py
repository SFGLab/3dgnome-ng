"""Are anchors really collapsed, or did rare overlapping pairs fake it?

The claim that the anchor backbone is flat rests on a binned distance curve. Adjacent anchors
average about 35 kb apart, so any bin below that can only hold overlapping or near-duplicate
anchors, which sit near zero whatever the model does. Those bins would manufacture flatness.

This prints the pair count per bin so thin bins are visible, and separately reports consecutive
anchors only, where the genomic gap and the spatial distance are both unambiguous.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "3dgnome-ng"))
from gnome3d.settings import Settings  # noqa: E402

cif = Path(sys.argv[1])
cfg = sys.argv[2] if len(sys.argv) > 2 else None

rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM") and ln.split()[5] == "ALA"]
mid = np.array([(int(r[16]) + int(r[17])) // 2 for r in rows], np.int64)
xyz = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], np.float64)
o = np.argsort(mid)
mid, xyz = mid[o], xyz[o]
print(f"[anc] {cif.name}: {len(mid)} anchors, span {mid.min()/1e6:.1f}-{mid.max()/1e6:.1f} Mb")

s = None
if cfg:
    s = Settings()
    if not s.load_ini(cfg):
        s = None

gap = np.diff(mid)
d1 = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
print(f"[anc] consecutive anchor gap: median={np.median(gap):,.0f} bp  "
      f"p10={np.percentile(gap, 10):,.0f}  p90={np.percentile(gap, 90):,.0f}")
print(f"[anc] consecutive anchor distance: median={np.median(d1):.3f}\n")

print("--- consecutive anchors only, binned by their own genomic gap")
print(f"{'gap band':>14s} {'n':>7s} {'median dist':>12s}" + ("  target  ratio" if s else ""))
edges = [0, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (gap >= lo) & (gap < hi)
    if m.sum() < 10:
        continue
    lab = f"{lo/1e3:.0f}-{hi/1e3:.0f}kb" if hi <= 1e6 else f">{lo/1e3:.0f}kb"
    line = f"{lab:>14s} {m.sum():>7d} {np.median(d1[m]):>12.3f}"
    if s:
        t = s.genomic_length_to_distance(int(np.median(gap[m])))
        line += f"  {t:6.2f}  {np.median(d1[m])/t:5.3f}"
    print(line)

print("\n--- all anchor pairs, with counts so thin bins are visible")
rng = np.random.default_rng(0)
i = rng.integers(0, len(mid), 3_000_000)
j = rng.integers(0, len(mid), 3_000_000)
k = i != j
sep = np.abs(mid[i[k]] - mid[j[k]]).astype(np.float64)
dd = np.linalg.norm(xyz[i[k]] - xyz[j[k]], axis=1)
edges2 = np.logspace(3, 8.2, 18)
print(f"{'sep':>10s} {'n':>9s} {'median dist':>12s}")
for lo, hi in zip(edges2[:-1], edges2[1:]):
    m = (sep >= lo) & (sep < hi)
    if m.sum() < 10:
        continue
    c = np.sqrt(lo * hi)
    lab = f"{c/1e6:.2f}Mb" if c >= 1e6 else f"{c/1e3:.0f}kb"
    print(f"{lab:>10s} {m.sum():>9d} {np.median(dd[m]):>12.3f}")
