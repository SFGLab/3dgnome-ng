"""How much does a block boundary cost, measured over enough pairs to trust.

`boundary_gap.py` compared consecutive anchor pairs and found a factor near 27, but on only 35
crossing pairs, and the two groups it compared barely overlap in genomic gap: consecutive pairs
inside a block sit about 500 bp apart while consecutive pairs that cross a boundary sit about
400 kb apart. Matching on gap therefore threw away almost all the data.

This drops the consecutive restriction. Any anchor pair at a given genomic separation is a valid
probe of whether a boundary between the two anchors costs anything, and there are millions of
those. Pairs are binned by separation, then split by whether both anchors belong to the same
block, and each group is scored against `genomic_length_to_distance` of its own separation.

Pools over whatever chromosomes are given. Reports a bootstrap interval on the ratio so the
result carries its own error bar.
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

cfg, data_dir, cif_dir = sys.argv[1], sys.argv[2], sys.argv[3]
chroms = sys.argv[4].split(",")
MAXPAIR = 3_000_000

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
s.data_dir = data_dir

rng = np.random.default_rng(0)
sep_all, rat_all, same_all, chr_all, dist_all = [], [], [], [], []

for chrom in chroms:
    cif = Path(cif_dir) / f"{chrom}_s1.cif"
    if not cif.exists():
        print(f"[bw] {chrom}: no cif, skipped")
        continue
    chrs, bed = parse_chrs_arg(chrom)
    state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
    cl = state.clusters
    anch = [i for i, c in enumerate(cl) if c.level == Level.ANCHOR]
    block = {cl[i].genomic_pos: cl[i].parent for i in anch}

    pos, gp, blk = [], [], []
    for ln in open(cif):
        if not ln.startswith("ATOM"):
            continue
        r = ln.split()
        if r[5] != "ALA":
            continue
        mid = (int(r[16]) + int(r[17])) // 2
        if mid not in block:
            continue
        pos.append((float(r[10]), float(r[11]), float(r[12])))
        gp.append(mid)
        blk.append(block[mid])
    pos, gp, blk = np.asarray(pos), np.asarray(gp), np.asarray(blk)
    n = len(pos)
    if n < 100:
        print(f"[bw] {chrom}: {n} matched anchors, skipped")
        continue

    i = rng.integers(0, n, MAXPAIR)
    j = rng.integers(0, n, MAXPAIR)
    m = i != j
    i, j = i[m], j[m]
    sep = np.abs(gp[i] - gp[j]).astype(float)
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    tgt = 1.0 + 0.5 * (sep / 1000.0) ** 0.75
    sep_all.append(sep); rat_all.append(d / tgt); dist_all.append(d)
    same_all.append(blk[i] == blk[j]); chr_all.append(np.full(len(sep), chrom))
    nb = len(set(blk.tolist()))
    print(f"[bw] {chrom}: {n} anchors, {nb} blocks, {len(sep):,} pairs")

sep = np.concatenate(sep_all); rat = np.concatenate(rat_all)
same = np.concatenate(same_all); dist = np.concatenate(dist_all)
cache = Path("playground/derived/boundary_pairs.npz")
cache.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(cache, sep=sep.astype(np.float32), dist=dist.astype(np.float32),
                    same=same)
print(f"[bw] cached pairs to {cache}")
print(f"\n[bw] pooled {len(sep):,} pairs over {len(sep_all)} chromosomes, "
      f"{100*same.mean():.1f}% within a block\n")

edges = 10 ** np.arange(4.0, 7.51, 0.25)
print(f"{'separation':>22s} {'n same':>10s} {'n cross':>10s} "
      f"{'same d/gld':>11s} {'cross d/gld':>12s} {'ratio':>8s}")
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (sep >= lo) & (sep < hi)
    a, b = k & same, k & ~same
    if a.sum() < 200 or b.sum() < 200:
        continue
    ra, rb = float(np.median(rat[a])), float(np.median(rat[b]))
    rows.append((lo, hi, a.sum(), b.sum(), ra, rb))
    print(f"{lo:>10,.0f}-{hi:<11,.0f}"[:22].rjust(22)
          + f" {a.sum():>10,} {b.sum():>10,} {ra:>11.3f} {rb:>12.3f} {rb/ra:>8.2f}")

# one number: pooled over the separations where both groups are well populated
if rows:
    lo = min(r[0] for r in rows); hi = max(r[1] for r in rows)
    k = (sep >= lo) & (sep < hi)
    a, b = k & same, k & ~same
    boot = []
    ia, ib = np.flatnonzero(a), np.flatnonzero(b)
    for _ in range(400):
        boot.append(np.median(rat[rng.choice(ib, len(ib))])
                    / np.median(rat[rng.choice(ia, len(ia))]))
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
    print(f"\n[bw] over {lo:,.0f}-{hi:,.0f} bp: same {np.median(rat[a]):.3f} (n={a.sum():,}), "
          f"cross {np.median(rat[b]):.3f} (n={b.sum():,})")
    print(f"[bw] boundary penalty {np.median(rat[b])/np.median(rat[a]):.2f}x  "
          f"95% CI [{lo_ci:.2f}, {hi_ci:.2f}]")
    print("[bw] the pooled number mixes separations and is not the headline; the per bin "
          "ratios are, and they fall with separation")

# the same split against the Hi-C shape rather than gld, since gld's own exponent is wrong
NU = 0.285
print(f"\n[bw] raw distance against separation, per group, and the exponent each follows")
print(f"{'separation':>22s} {'d same':>9s} {'d cross':>9s} {'all':>9s}")
pts = {"same": [], "cross": [], "all": []}
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (sep >= lo) & (sep < hi)
    a, b = k & same, k & ~same
    if a.sum() < 200 or b.sum() < 200:
        continue
    ms = float(np.median(sep[k]))
    da, db, dall = (float(np.median(dist[x])) for x in (a, b, k))
    pts["same"].append((ms, da)); pts["cross"].append((ms, db)); pts["all"].append((ms, dall))
    print(f"{lo:>10,.0f}-{hi:<11,.0f}"[:22].rjust(22)
          + f" {da:>9.2f} {db:>9.2f} {dall:>9.2f}")
print(f"\n{'group':>8s} {'exponent':>10s}   target: Hi-C {NU:.3f}, gld 0.704")
for g, v in pts.items():
    x = np.array([p[0] for p in v]); y = np.array([p[1] for p in v])
    if len(x) >= 4:
        print(f"{g:>8s} {np.polyfit(np.log10(x), np.log10(y), 1)[0]:>10.3f}")
