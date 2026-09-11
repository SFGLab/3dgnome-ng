"""Where do block centroids actually sit, and what decides it.

Cross block anchor distance came out flat, near 75 model units across a 45 fold range of genomic
separation. Two things could produce that. Either the blocks carry no separation information
because nothing couples them, which is the case option B addresses, or they are packed inside a
confinement sphere whose radius sets the distance and overrides whatever the chain bonds ask for.

This separates them. It reduces each block to its centroid, measures how centroid distance grows
with genomic separation, and compares the observed spread against the confinement radius the
settings imply at that level.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

cfg, data_dir, cif, chrom = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
s.data_dir = data_dir

chrs, bed = parse_chrs_arg(chrom)
state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
cl = state.clusters
block_of = {cl[i].genomic_pos: cl[i].parent
            for i, c in enumerate(cl) if c.level == Level.ANCHOR}
seg_of = {b: cl[b].parent for b in set(block_of.values())}

acc: dict[int, list[tuple[np.ndarray, int]]] = {}
for ln in open(cif):
    if not ln.startswith("ATOM"):
        continue
    r = ln.split()
    if r[5] != "ALA":
        continue
    mid = (int(r[16]) + int(r[17])) // 2
    b = block_of.get(mid)
    if b is None:
        continue
    acc.setdefault(b, []).append((np.array([float(r[10]), float(r[11]), float(r[12])]), mid))

blocks = sorted(acc, key=lambda b: np.median([m for _, m in acc[b]]))
cen = np.array([np.mean([p for p, _ in acc[b]], axis=0) for b in blocks])
gmid = np.array([np.median([m for _, m in acc[b]]) for b in blocks])
seg = np.array([seg_of[b] for b in blocks])
print(f"[bl] {len(blocks)} blocks in {len(set(seg.tolist()))} segments, "
      f"{sum(len(v) for v in acc.values())} anchors\n")

# centroid distance against genomic separation
iu = np.triu_indices(len(blocks), k=1)
sep = np.abs(gmid[iu[0]] - gmid[iu[1]])
d = np.linalg.norm(cen[iu[0]] - cen[iu[1]], axis=1)
sameseg = seg[iu[0]] == seg[iu[1]]
edges = 10 ** np.arange(5.5, 8.51, 0.35)
print(f"{'separation':>22s} {'pairs':>7s} {'median d':>10s} {'same segment':>14s}")
pts = []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (sep >= lo) & (sep < hi)
    if k.sum() < 8:
        continue
    pts.append((float(np.median(sep[k])), float(np.median(d[k]))))
    ss = np.median(d[k & sameseg]) if (k & sameseg).sum() >= 4 else np.nan
    print(f"{lo:>10,.0f}-{hi:<11,.0f}"[:22].rjust(22)
          + f" {k.sum():>7,} {np.median(d[k]):>10.2f} {ss:>14.2f}")
if len(pts) >= 4:
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    print(f"\n[bl] centroid distance exponent {np.polyfit(np.log10(x), np.log10(y), 1)[0]:.3f}"
          f"   Hi-C {0.285:.3f}, gld 0.704")

# is that spread the confinement radius
print(f"\n[bl] per segment: does the block cloud fill a sphere of the confined radius")
print(f"{'segment':>9s} {'blocks':>7s} {'span Mb':>9s} {'mean dtn':>9s} "
      f"{'R auto':>8s} {'R real':>8s} {'ratio':>7s}")
tot = []
for sg in sorted(set(seg.tolist())):
    k = seg == sg
    if k.sum() < 3:
        continue
    c = cen[k]
    g = np.sort(gmid[k])
    dtn = np.array([s.genomic_length_to_distance(int(x)) for x in np.diff(g)])
    n = int(k.sum())
    r_auto = s.confinement_packing_factor_ib * float(dtn.mean()) * n ** (1 / 3)
    r_real = float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)))
    tot.append((r_auto, r_real))
    print(f"{sg:>9d} {n:>7d} {(g[-1]-g[0])/1e6:>9.1f} {dtn.mean():>9.1f} "
          f"{r_auto:>8.1f} {r_real:>8.1f} {r_real/r_auto:>7.2f}")
if tot:
    ra = np.array([t[0] for t in tot]); rr = np.array([t[1] for t in tot])
    print(f"\n[bl] median realised over auto radius: {np.median(rr/ra):.2f}")
    print("[bl] near 1 means confinement is binding and sets the block layout; "
          "well below 1 means it never binds and the flatness comes from somewhere else")
