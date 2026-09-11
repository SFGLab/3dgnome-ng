"""Does IB confinement flatten the block layout, and what happens if it is relaxed.

Block centroids are set by the coarse stage, so this drives that stage alone under several
confinement settings and measures how centroid distance grows with genomic separation. The Hi-C
curve says the exponent should be near 0.29. A flat result means the layout carries no
separation information.

First checks that the coarse centroids agree with the finished structure's, since the ablation is
only meaningful if the later stages leave the layout alone.
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
import gnome3d.pipeline.coarse.build as cb  # noqa: E402
from gnome3d.hierarchy import set_level  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def place(s: Settings, chrs, bed):
    """Run the coarse chain only, which is what fixes the block layout.

    Mirrors SegmentStage followed by IBPositionStage. The per-IB arcs and smooth passes come
    after and are skipped here, so this costs minutes rather than hours."""
    state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
    cb.seed_global_rng(0)
    lvl = set_level(Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs)
    if len(state.chrs) > 1:
        cb.reconstruct_chromosome_level(state)
    if s.random_walk:
        cb.random_walk_segment_level(state, lvl)
    elif sum(len(v) for v in lvl.values()) > 1:
        cb.reconstruct_segment_level(state, lvl)
    else:
        cb.place_single_segment(state, lvl)
    for chr_ in state.chrs:
        segs = lvl.get(chr_, [])
        if segs:
            cb.position_interaction_blocks(state, segs, chr_)
    return state

data_dir, chrom, cif = sys.argv[1], sys.argv[2], sys.argv[3]
configs = sys.argv[4:]
EDGES = 10 ** np.arange(5.5, 8.51, 0.35)


def centroids(state) -> tuple[np.ndarray, np.ndarray]:
    cl = state.clusters
    ibs = [i for i, c in enumerate(cl) if c.level == Level.INTERACTION_BLOCK]
    ibs = [b for b in ibs if cl[b].children]
    ibs.sort(key=lambda b: cl[b].genomic_pos)
    cen = np.array([np.asarray(cl[b].pos, dtype=float) for b in ibs])
    gm = np.array([float(np.median([cl[c].genomic_pos for c in cl[b].children])) for b in ibs])
    return cen, gm


def curve(cen: np.ndarray, gm: np.ndarray) -> tuple[float, list[tuple[float, float, int]]]:
    iu = np.triu_indices(len(cen), k=1)
    sep = np.abs(gm[iu[0]] - gm[iu[1]])
    d = np.linalg.norm(cen[iu[0]] - cen[iu[1]], axis=1)
    pts = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (sep >= lo) & (sep < hi)
        if k.sum() >= 8:
            pts.append((float(np.median(sep[k])), float(np.median(d[k])), int(k.sum())))
    if len(pts) < 4:
        return float("nan"), pts
    x = np.log10([p[0] for p in pts]); y = np.log10([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0]), pts


chrs, bed = parse_chrs_arg(chrom)
results = {}
for cfg in configs:
    s = Settings()
    if not s.load_ini(cfg):
        raise SystemExit(f"cannot load {cfg}")
    s.data_dir = data_dir
    state = place(s, chrs, bed)
    cen, gm = centroids(state)
    e, pts = curve(cen, gm)
    rg = float(np.sqrt(np.mean(np.sum((cen - cen.mean(axis=0)) ** 2, axis=1))))
    results[Path(cfg).stem] = (e, pts, rg, cen, gm, state)
    print(f"[abl] {Path(cfg).stem:>10s}: {len(cen)} blocks, exponent {e:>6.3f}, Rg {rg:>8.1f}")

# does the coarse layout survive the rest of the pipeline
first = next(iter(results.values()))
state = first[5]
cl = state.clusters
block_of = {cl[i].genomic_pos: cl[i].parent
            for i, c in enumerate(cl) if c.level == Level.ANCHOR}
acc: dict[int, list[np.ndarray]] = {}
for ln in open(cif):
    if not ln.startswith("ATOM"):
        continue
    r = ln.split()
    if r[5] != "ALA":
        continue
    b = block_of.get((int(r[16]) + int(r[17])) // 2)
    if b is not None:
        acc.setdefault(b, []).append(np.array([float(r[10]), float(r[11]), float(r[12])]))
ibs = [i for i, c in enumerate(cl) if c.level == Level.INTERACTION_BLOCK and c.children]
ibs.sort(key=lambda b: cl[b].genomic_pos)
pair = [(np.asarray(cl[b].pos, dtype=float), np.mean(acc[b], axis=0)) for b in ibs if b in acc]
if len(pair) >= 5:
    a = np.array([p[0] for p in pair]); c = np.array([p[1] for p in pair])
    iu = np.triu_indices(len(a), k=1)
    da = np.linalg.norm(a[iu[0]] - a[iu[1]], axis=1)
    dc = np.linalg.norm(c[iu[0]] - c[iu[1]], axis=1)
    r = float(np.corrcoef(da, dc)[0, 1])
    print(f"\n[abl] coarse vs finished centroid distances: r={r:.4f}, "
          f"median ratio {np.median(dc/da):.3f}  (n={len(pair)} blocks)")
    print("[abl] a high r means the later stages keep the layout and this ablation is valid")

print(f"\n{'separation':>22s} " + " ".join(f"{k:>12s}" for k in results))
ref = first[1]
for i, (ms, _, n) in enumerate(ref):
    row = []
    for _, pts, *_ in results.values():
        row.append(pts[i][1] if i < len(pts) else float("nan"))
    print(f"{ms:>22,.0f} " + " ".join(f"{v:>12.1f}" for v in row))
print(f"\n{'exponent':>22s} " + " ".join(f"{v[0]:>12.3f}" for v in results.values()))
print(f"{'Rg':>22s} " + " ".join(f"{v[2]:>12.1f}" for v in results.values()))
print(f"\n[abl] Hi-C exponent 0.285, gld 0.704")
