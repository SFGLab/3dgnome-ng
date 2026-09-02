"""Can Hi-C constrain the anchor pairs that have no arc?

The arcs stage gives a pair either an arc target, saturated near the 0.2 floor and independent of
genomic separation, or a scale-free 1/d repulsion. With about 1.2 arcs per anchor almost every
pair takes the second branch, so the equilibrium is a uniform ball and the backbone carries no
separation structure.

`calc_anchor_expected_distances` already receives a Hi-C anchor heatmap but only uses it to shrink
targets that already exist, skipping any pair whose target is not positive. The proposed fix is to
let it create targets for arcless pairs instead. That only works if the Hi-C actually covers those
pairs, so this measures coverage and what target the data would imply, binned by separation:

  arc          pairs an arc already constrains
  hic only     arcless pairs with non-zero Hi-C, which the fix would newly constrain
  neither      pairs left to the repulsion, which would need the genomic-scaled soft core instead

and for the middle group, the Hi-C implied distance against the chain's own expectation, since a
target that is also flat in separation would fix nothing.
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
from gnome3d.pipeline.coarse.build import (  # noqa: E402
    build_contact_heatmaps,
    build_state,
    calc_anchor_expected_distances,
)
from gnome3d.settings import Settings  # noqa: E402

cfg, chrom = sys.argv[1], sys.argv[2]
data_dir = sys.argv[3] if len(sys.argv) > 3 else None
max_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 12

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
if data_dir:
    s.data_dir = data_dir

chrs, bed = parse_chrs_arg(chrom)
data = ContactData.from_files(s, chrs, bed)
state = build_state(s, data, chrs, bed)
cl = state.clusters
blocks = [i for i, c in enumerate(cl) if c.level == Level.INTERACTION_BLOCK]
blocks.sort(key=lambda b: -len(cl[b].children))
blocks = blocks[:max_blocks]
print(f"[hic] {len(blocks)} largest blocks, "
      f"{sum(len(cl[b].children) for b in blocks)} anchors total\n")

sep_all, kind_all, hicd_all, chaind_all = [], [], [], []
for b in blocks:
    region = list(cl[b].children)
    n = len(region)
    if n < 20:
        continue
    heat, _ = build_contact_heatmaps(state, region, chrom)
    mat = calc_anchor_expected_distances(state, region, chrom, None)   # arcs only, no heat scaling
    pos = np.array([cl[c].genomic_pos for c in region], dtype=np.int64)

    iu = np.triu_indices(n, k=1)
    sep = np.abs(pos[iu[0]] - pos[iu[1]]).astype(float)
    arc = mat[iu] > 0.0
    hic = heat[iu] > 0.0
    kind = np.where(arc, 0, np.where(hic, 1, 2))          # 0 arc, 1 hic only, 2 neither
    hd = np.array([s.freq_to_dist_heatmap(float(v)) if v > 0 else np.nan for v in heat[iu]])
    cd = np.array([s.genomic_length_to_distance(int(x)) for x in sep])
    sep_all.append(sep); kind_all.append(kind); hicd_all.append(hd); chaind_all.append(cd)

sep = np.concatenate(sep_all)
kind = np.concatenate(kind_all)
hicd = np.concatenate(hicd_all)
chaind = np.concatenate(chaind_all)
print(f"[hic] {len(sep):,} anchor pairs\n")

edges = [0, 5e3, 2e4, 5e4, 1e5, 3e5, 1e6, 1e9]
print(f"{'separation':>14s} {'pairs':>10s} {'arc':>7s} {'hic only':>9s} {'neither':>8s}"
      f" {'hic dist':>9s} {'chain dist':>11s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sep >= lo) & (sep < hi)
    if m.sum() < 50:
        continue
    lab = f"{lo/1e3:.0f}-{hi/1e3:.0f}kb" if hi <= 1e6 else f">{lo/1e3:.0f}kb"
    k = kind[m]
    ho = np.nanmedian(hicd[m & (kind == 1)]) if (m & (kind == 1)).sum() else np.nan
    print(f"{lab:>14s} {m.sum():>10,} {100*np.mean(k==0):>6.1f}% {100*np.mean(k==1):>8.1f}%"
          f" {100*np.mean(k==2):>7.1f}% {ho:>9.3f} {np.median(chaind[m]):>11.3f}")

hs = kind == 1
print(f"\n[hic] arcless pairs with Hi-C support: {100*np.mean(kind==1):.1f}% of all pairs")
if hs.sum() > 100:
    lo_m = hs & (sep < 5e4)
    hi_m = hs & (sep >= 3e5)
    if lo_m.sum() > 20 and hi_m.sum() > 20:
        print(f"[hic] Hi-C implied distance, <50kb {np.nanmedian(hicd[lo_m]):.3f} "
              f"vs >300kb {np.nanmedian(hicd[hi_m]):.3f}  "
              f"ratio {np.nanmedian(hicd[hi_m])/np.nanmedian(hicd[lo_m]):.2f}")
        print("[hic] a ratio near 1 means the Hi-C target is as flat as the arc target and "
              "would fix nothing")
