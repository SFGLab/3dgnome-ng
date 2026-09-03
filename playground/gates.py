"""Every gate the anchor placement work is judged by, for one finished structure, in one table.

    python playground/gates.py <config> <data_dir> <chrom> <mcool> <name>=<cif> [<name>=<cif> ...]

Per structure: cross block bead contacts within one bond and the beads they touch, straight
subanchor strands, boundary pairs realised over the stitch target, block pairs overlapping by
their own radii, Rg, the within block distance exponent over 20 kb to 1 Mb, and agreement with
Hi-C at 25 kb on the anchors, SCC, Pearson and the MultiMM Pearson. See
design/anchor-placement.md for what each number should do.
"""

import bisect
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.pipeline.stitch import within_block_curve  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402
from validation.metrics import hic as H  # noqa: E402

cfg, data_dir, chrom, mcool = sys.argv[1:5]
arms = [a.split("=", 1) for a in sys.argv[5:]]
BIN, RADIUS = 25_000, 5.0

s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
s.data_dir = data_dir
chrs, bed = parse_chrs_arg(chrom)
state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
cl = state.clusters
anchors = sorted((i for i, c in enumerate(cl) if c.level == Level.ANCHOR), key=lambda i: cl[i].start)
a_start = [cl[i].start for i in anchors]
a_block = [cl[i].parent for i in anchors]
c_obs, bin_starts = H.observed_hic(mcool, chrom, BIN, balance=True)


def load(path: str) -> list[BeadOut]:
    rows = [ln.split() for ln in open(path) if ln.startswith("ATOM")]
    beads = [BeadOut(int(r[16]), int(r[17]), float(r[10]), float(r[11]), float(r[12]), r[18]) for r in rows]  # type: ignore[arg-type]
    return sorted(beads, key=lambda b: b.start)


print(
    f"{'arm':>14s} {'contacts':>9s} {'touched':>8s} {'strands':>7s} {'bnd r/t':>8s} {'bnd max':>7s} "
    f"{'ovl':>4s} {'Rg':>6s} {'exp':>6s} {'scc':>6s} {'pears':>6s} {'multimm':>7s}"
)
for name, path in arms:
    beads = load(path)
    P = np.array([[b.x, b.y, b.z] for b in beads])
    S = np.array([b.start for b in beads])
    K = np.array([b.kind == "anchor" for b in beads])
    blk = np.array([a_block[max(bisect.bisect_right(a_start, x) - 1, 0)] for x in S])
    bond = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    pairs = KDTree(P).query_pairs(bond, output_type="ndarray")
    cross = pairs[blk[pairs[:, 0]] != blk[pairs[:, 1]]] if pairs.size else pairs
    touched = int(np.unique(cross.ravel()).size) if cross.size else 0
    ai = np.flatnonzero(K)
    straight = 0
    for a, b in zip(ai[:-1], ai[1:]):
        if b - a < 2:
            continue
        seg = P[a : b + 1]
        contour = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
        if contour > 0 and np.linalg.norm(P[b] - P[a]) / contour > 0.9:
            straight += 1
    groups: dict[int, list[BeadOut]] = {}
    for b, k in zip(beads, blk):
        groups.setdefault(int(k), []).append(b)
    blocks = sorted(groups.values(), key=lambda v: v[0].start)
    curve = within_block_curve(blocks)
    ratios = []
    for k in range(len(blocks) - 1):
        A = [b for b in blocks[k] if b.kind == "anchor"][-1]
        B = [b for b in blocks[k + 1] if b.kind == "anchor"][0]
        d = float(np.linalg.norm(np.array([A.x, A.y, A.z]) - np.array([B.x, B.y, B.z])))
        t = curve(B.midpoint - A.midpoint) if curve else float("nan")
        ratios.append(d / t)
    ids = sorted(groups, key=lambda k: groups[k][0].start)
    cen = np.array([P[(blk == k) & K].mean(0) for k in ids])
    rg = np.array([np.sqrt(np.mean(np.sum((P[blk == k] - P[blk == k].mean(0)) ** 2, 1))) for k in ids])
    iu = np.triu_indices(len(ids), 1)
    ovl = int((np.linalg.norm(cen[iu[0]] - cen[iu[1]], axis=1) < rg[iu[0]] + rg[iu[1]]).sum())
    Rg = float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, 1))))
    # within block exponent, anchors, 20 kb to 1 Mb
    A_ = P[K]
    m_ = np.array([b.midpoint for b in beads if b.kind == "anchor"])
    bk = blk[K]
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(A_), 2_000_000)
    j = rng.integers(0, len(A_), 2_000_000)
    ok = (i != j) & (bk[i] == bk[j])
    i, j = i[ok], j[ok]
    sep = np.abs(m_[i] - m_[j]).astype(float)
    d = np.linalg.norm(A_[i] - A_[j], axis=1)
    edges = 10 ** np.arange(4.3, 6.01, 0.25)
    pts = [(np.median(sep[(sep >= lo) & (sep < hi)]), np.median(d[(sep >= lo) & (sep < hi)])) for lo, hi in zip(edges[:-1], edges[1:]) if ((sep >= lo) & (sep < hi)).sum() > 200]
    exp = float(np.polyfit(np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1)[0]) if len(pts) >= 3 else float("nan")
    hc = H.hic_correlation(A_, m_, mcool, chrom, BIN, RADIUS, balance=True)
    mm = H.multimm_faithful_pearson([A_], m_, c_obs, bin_starts, BIN)
    print(
        f"{name:>14s} {len(cross):>9,d} {100 * touched / len(P):>7.1f}% {straight:>7d} {np.median(ratios):>8.2f} {max(ratios):>7.2f} "
        f"{ovl:>4d} {Rg:>6.1f} {exp:>6.3f} {hc['scc']:>6.3f} {hc['pearson']:>6.3f} {mm:>7.3f}",
        flush=True,
    )
