"""How many far arcless anchor pairs a contact map can hold, by source and pooling."""
import sys
import numpy as np
import cooler
mcool, bed, sing, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
B = 25000
anc = np.array([[int(f[1]), int(f[2])] for f in (l.split() for l in open(bed)) if f[0] == "chr1" and int(f[1]) >= lo and int(f[2]) <= hi])
mid = (anc[:, 0] + anc[:, 1]) // 2
n = len(mid)
print(f"{n} anchors, mean width {np.mean(anc[:,1]-anc[:,0])/1e3:.1f} kb")
clr = cooler.Cooler(f"{mcool}::/resolutions/{B}")
M = clr.matrix(balance=False).fetch(f"chr1:{lo}-{hi}").astype(np.float64)
M = np.nan_to_num(M)
nb = M.shape[0]
print(f"map {nb} bins, {M.sum()/2:.3g} contacts in the window")
b = (mid - lo) // B
sep = np.abs(mid[:, None] - mid[None, :])
far = np.triu(sep > 100_000, 1)
# expected per bin separation from the map itself
d = np.abs(np.arange(nb)[:, None] - np.arange(nb)[None, :])
exp_by_d = np.array([M[d == k].mean() if (d == k).any() else 0.0 for k in range(nb)])
def held(counts):
    e = exp_by_d[np.abs(b[:, None] - b[None, :])]
    return float(((counts > e) & far).sum() / far.sum()), float(((counts > 0) & far).sum() / far.sum())
pix = M[b[:, None], b[None, :]]
h1, nz1 = held(pix)
# 3x3 pooled: sum over bins b-1..b+1 on both sides
P = np.zeros_like(M)
for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
        P += np.roll(np.roll(M, di, 0), dj, 1)
exp3 = np.array([P[d == k].mean() if (d == k).any() else 0.0 for k in range(nb)])
pooled = P[b[:, None], b[None, :]]
e3 = exp3[np.abs(b[:, None] - b[None, :])]
h3 = float(((pooled > e3) & far).sum() / far.sum()); nz3 = float(((pooled > 0) & far).sum() / far.sum())
# the thinned singletons as the run bins them: both bin centres inside anchors
S = np.zeros((n, n))
starts = anc[:, 0]; ends = anc[:, 1]
order = np.argsort(starts)
def anchor_of(p):
    k = np.searchsorted(starts[order], p, side="right") - 1
    ok = (k >= 0)
    kk = order[np.maximum(k, 0)]
    return np.where(ok & (p < ends[kk]), kk, -1)
rows = [l.split() for l in open(sing)]
p1 = np.array([(int(r[1]) + int(r[2])) // 2 for r in rows if r[0] == "chr1" and r[3] == "chr1"])
p2 = np.array([(int(r[4]) + int(r[5])) // 2 for r in rows if r[0] == "chr1" and r[3] == "chr1"])
sc = np.array([float(r[6]) for r in rows if r[0] == "chr1" and r[3] == "chr1"])
inw = (p1 >= lo) & (p1 <= hi) & (p2 >= lo) & (p2 <= hi)
a1, a2 = anchor_of(p1[inw]), anchor_of(p2[inw])
ok = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
np.add.at(S, (a1[ok], a2[ok]), sc[inw][ok]); S = S + S.T
sep_b = np.abs(b[:, None] - b[None, :])
# expected for the singletons map by separation over anchor pairs
es = np.zeros(nb)
for k in range(nb):
    m = (sep_b == k) & (np.triu(np.ones((n, n), bool), 1))
    if m.any(): es[k] = S[m].mean()
hs = float(((S > es[sep_b]) & far).sum() / far.sum()); nzs = float(((S > 0) & far).sum() / far.sum())
print(f"far arcless-or-not anchor pairs (> 100 kb): {far.sum():,}")
print(f"  thinned singletons, both ends in anchors:  held {hs:.3f}  nonzero {nzs:.3f}")
print(f"  full map, one pixel per pair:              held {h1:.3f}  nonzero {nz1:.3f}")
print(f"  full map, 3x3 pixels pooled:               held {h3:.3f}  nonzero {nz3:.3f}")
e = exp_by_d[sep_b]
print("by separation band: held = obs > exp; sig = obs > exp + 3 sqrt(exp), Poisson")
for s_lo, s_hi in ((100_000, 500_000), (500_000, 2_000_000), (2_000_000, 10_000_000), (10_000_000, 60_000_000)):
    band = far & (sep >= s_lo) & (sep < s_hi)
    nb_ = band.sum()
    print(f"  {s_lo/1e6:>4.1f}-{s_hi/1e6:<4.0f} Mb: pairs {nb_:>9,}  exp/pixel {e[band].mean():6.2f}  singletons held {((S > es[sep_b]) & band).sum()/nb_:.3f}"
          f"  pixel held {((pix > e) & band).sum()/nb_:.3f} sig {((pix > e + 3*np.sqrt(e)) & band).sum()/nb_:.3f}"
          f"  pooled held {((pooled > e3) & band).sum()/nb_:.3f} sig {((pooled > e3 + 3*np.sqrt(e3)) & band).sum()/nb_:.3f}")
