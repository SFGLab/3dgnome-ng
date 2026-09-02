"""Do the arcs stage and the chain agree on how far apart two anchors should be?

Anchors are placed by the arcs MC against freq_to_distance(arc score). The subanchor chain
between them is then built against genomic_length_to_distance of each 1 kb step, and the anchors
are held fixed while it relaxes. Nothing makes those two scales agree.

If the arc target is far below the chain's own target for the same genomic separation, the chain
is handed far more contour than the gap it has to span and balloons into a free coil, which is
what the structures show: on one H1ESC chr11 file, subanchor beads 13 kb apart sit 8.5 apart
while anchors 18 kb apart sit 1.44 apart.

For every arc, prints the two targets and the realised distance, so the disagreement is visible
per pair rather than inferred from aggregates.
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

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")

chrs, bed = parse_chrs_arg(chrom)
data = ContactData.from_files(s, chrs, bed)
state = build_state(s, data, chrs, bed)
cl = state.clusters
arcs = state.arcs.get(chrs[0], [])
anchor_set = {i for i, c in enumerate(cl) if c.level == Level.ANCHOR}

# realised anchor positions, keyed by genomic midpoint so the CIF and the tree can be matched
rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM") and ln.split()[5] == "ALA"]
real = {(int(r[16]) + int(r[17])) // 2: np.array([float(r[10]), float(r[11]), float(r[12])])
        for r in rows}
print(f"[clash] {len(anchor_set)} anchors in tree, {len(real)} anchors in cif, {len(arcs)} arcs")

arc_t, chain_t, dist, seps = [], [], [], []
for a in arcs:
    if a.start not in anchor_set or a.end not in anchor_set:
        continue
    ma, mb = cl[a.start].genomic_pos, cl[a.end].genomic_pos
    if ma not in real or mb not in real:
        continue
    sep = abs(mb - ma)
    if sep < 1:
        continue
    arc_t.append(s.freq_to_distance(a.score))
    chain_t.append(s.genomic_length_to_distance(sep))
    dist.append(float(np.linalg.norm(real[ma] - real[mb])))
    seps.append(sep)

arc_t = np.array(arc_t); chain_t = np.array(chain_t)
dist = np.array(dist); seps = np.array(seps)
print(f"[clash] {len(arc_t)} arcs with both anchors placed\n")

print(f"{'sep band':>14s} {'n':>6s} {'arc target':>11s} {'chain target':>13s} "
      f"{'realised':>10s} {'chain/arc':>10s} {'real/chain':>11s}")
edges = [0, 5e3, 2e4, 5e4, 1e5, 3e5, 1e6, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (seps >= lo) & (seps < hi)
    if m.sum() < 20:
        continue
    lab = f"{lo/1e3:.0f}-{hi/1e3:.0f}kb" if hi <= 1e6 else f">{lo/1e3:.0f}kb"
    a_, c_, d_ = np.median(arc_t[m]), np.median(chain_t[m]), np.median(dist[m])
    print(f"{lab:>14s} {m.sum():>6d} {a_:>11.3f} {c_:>13.3f} {d_:>10.3f} "
          f"{c_/a_:>10.1f} {d_/c_:>11.3f}")
print("\n[clash] chain/arc is how much more room the chain wants than the arcs stage gives;"
      "\n[clash] real/chain near 1 would mean anchors sit where the chain expects them")
