"""Are anchors that straddle a block boundary placed worse than anchors inside a block?

The arcs MC runs per block, so a consecutive anchor pair split by a boundary has no term coupling
it at all. Both anchors are positioned only through their own block's centroid, and IB placement
sets those centroids from a chain bond between block midpoints. So two anchors a few kb apart can
end up as far apart as their blocks are.

Compares consecutive anchor pairs inside one block against consecutive pairs that cross a
boundary, both against `genomic_length_to_distance` of their own gap. If the crossing pairs are
much worse, stitching block edges together is worth doing and is a narrower change than widening
the arcs MC, which is what use_segment_arcs tried and lost on.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path.home() / "3dgnome-ng"))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

cfg, chrom, cif = sys.argv[1], sys.argv[2], sys.argv[3]
data_dir = sys.argv[4] if len(sys.argv) > 4 else None

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
if data_dir:
    s.data_dir = data_dir

chrs, bed = parse_chrs_arg(chrom)
state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
cl = state.clusters
anchors = [i for i, c in enumerate(cl) if c.level == Level.ANCHOR]
anchors.sort(key=lambda i: cl[i].genomic_pos)
block_of = {i: cl[i].parent for i in anchors}
print(f"[bnd] {len(anchors)} anchors in {len(set(block_of.values()))} blocks")

# realised positions, matched to the tree by genomic midpoint
rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM") and ln.split()[5] == "ALA"]
pos = {(int(r[16]) + int(r[17])) // 2: np.array([float(r[10]), float(r[11]), float(r[12])])
       for r in rows}
print(f"[bnd] {len(pos)} anchor beads in the cif\n")

same, cross = [], []
for a, b in zip(anchors[:-1], anchors[1:]):
    ma, mb = cl[a].genomic_pos, cl[b].genomic_pos
    if ma not in pos or mb not in pos:
        continue
    gap = abs(mb - ma)
    if gap < 1 or gap > 5_000_000:
        continue
    d = float(np.linalg.norm(pos[ma] - pos[mb]))
    t = s.genomic_length_to_distance(int(gap))
    rec = (gap, d, t, d / t if t else np.nan)
    (same if block_of[a] == block_of[b] else cross).append(rec)

print(f"{'pairs':>22s} {'n':>6s} {'median gap':>11s} {'median dist':>12s} "
      f"{'target':>8s} {'realised/target':>16s}")
for name, rec in [("inside one block", same), ("crossing a boundary", cross)]:
    if not rec:
        continue
    g, d, t, r = (np.array([x[k] for x in rec]) for k in range(4))
    print(f"{name:>22s} {len(rec):>6d} {np.median(g):>11,.0f} {np.median(d):>12.3f} "
          f"{np.median(t):>8.3f} {np.median(r):>16.3f}")

if same and cross:
    gs = np.array([x[0] for x in same])
    gc = np.array([x[0] for x in cross])
    # compare only where the genomic gaps overlap, so the split is not just gap size
    lo, hi = np.percentile(gc, [25, 75])
    ms = [x for x in same if lo <= x[0] <= hi]
    mc = [x for x in cross if lo <= x[0] <= hi]
    if len(ms) > 20 and len(mc) > 20:
        rs = np.median([x[3] for x in ms])
        rc = np.median([x[3] for x in mc])
        print(f"\n[bnd] matched on gap ({lo:,.0f}-{hi:,.0f} bp): inside {rs:.3f}, "
              f"crossing {rc:.3f}, ratio {rc / rs:.2f}")
        print(f"[bnd] n inside {len(ms)}, n crossing {len(mc)}")
        print("[bnd] a ratio well above 1 means the boundary itself is what hurts, "
              "not the genomic gap")
