"""Measure the realised distance curve and pick the excluded volume floor that matches Hi-C.

`ps_curve.py` fixes the exponent: contact probability decays near s^-0.86 across three cell
lines, so distance grows near s^0.29. This measures what our structures actually do, then solves
for the floor r0(s) = beta * (s/1000)^nu that would carry them onto that curve.

The floor only pushes apart, so it cannot shrink anything. Beta is therefore anchored at the
separation where the structures are already widest relative to the target, and the exponent does
the rest at shorter range where the collapse is.

Also reports contact probability computed from the beads themselves, which is the gate a change
has to move. Its slope is comparable to Hi-C directly, and the capture radius shifts only the
prefactor.
"""

import sys

import numpy as np

NU_HIC = 0.285          # from ps_curve.py, mean over three cell lines, 20kb-1Mb
KIND = sys.argv[2] if len(sys.argv) > 2 else "ALA"


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    p, s = [], []
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        r = ln.split()
        if KIND != "all" and r[5] != KIND:
            continue
        p.append((float(r[10]), float(r[11]), float(r[12])))
        s.append((int(r[16]) + int(r[17])) // 2)
    o = np.argsort(s)
    return np.asarray(p)[o], np.asarray(s)[o]


pos, gpos = load(sys.argv[1])
print(f"[cal] {len(pos)} beads ({KIND}), {gpos[0]:,} to {gpos[-1]:,} bp")

# sample pairs rather than forming the full N^2 matrix
rng = np.random.default_rng(0)
n = len(pos)
i = rng.integers(0, n, 4_000_000)
j = rng.integers(0, n, 4_000_000)
m = i != j
i, j = i[m], j[m]
sep = np.abs(gpos[i] - gpos[j]).astype(float)
dist = np.linalg.norm(pos[i] - pos[j], axis=1)
gld = 1.0 + 0.5 * (sep / 1000.0) ** 0.75

edges = 10 ** np.arange(3.5, 7.6, 0.25)
print(f"\n{'separation':>14s} {'pairs':>9s} {'median d':>10s} {'gld':>9s} {'d/gld':>8s}")
mids, meds = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (sep >= lo) & (sep < hi)
    if k.sum() < 200:
        continue
    md = float(np.median(dist[k]))
    mids.append(float(np.median(sep[k])))
    meds.append(md)
    print(f"{lo:>9,.0f}-{hi:<11,.0f}"[:14].rjust(14)
          + f" {k.sum():>9,} {md:>10.2f} {np.median(gld[k]):>9.2f} "
            f"{md/np.median(gld[k]):>8.3f}")

mids, meds = np.asarray(mids), np.asarray(meds)
fit = (mids >= 2e4) & (mids <= 1e6)
nu_ours = np.polyfit(np.log10(mids[fit]), np.log10(meds[fit]), 1)[0]
print(f"\n[cal] realised exponent, 20kb-1Mb: {nu_ours:.3f}   Hi-C: {NU_HIC:.3f}   gld: 0.704")

# beta: anchor the floor where the structures are already most extended relative to the
# target shape, so the floor binds inward of that point and nowhere pushes an existing
# distance further out than it already is
target_shape = (mids / 1000.0) ** NU_HIC
ratio = meds / target_shape
pivot = int(np.argmax(ratio[fit]) + np.flatnonzero(fit)[0])
beta = float(ratio[pivot])
print(f"[cal] pivot at {mids[pivot]:,.0f} bp   beta = {beta:.3f}")
print(f"[cal] floor r0(s) = {beta:.3f} * (s/1000)^{NU_HIC}")

print(f"\n{'separation':>12s} {'realised':>10s} {'floor':>9s} {'binds':>7s} {'inflation':>10s}")
for s_, d_ in zip(mids, meds):
    if s_ < 3e3 or s_ > 2e7:
        continue
    r0 = beta * (s_ / 1000.0) ** NU_HIC
    print(f"{s_:>12,.0f} {d_:>10.2f} {r0:>9.2f} {'yes' if r0 > d_ else 'no':>7s} "
          f"{max(r0/d_, 1.0):>10.2f}x")

# contact probability from the beads themselves
print("\n[cal] simulated contact probability, capture radius = median bond length")
bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
for cap in (bond, 2 * bond):
    pts, slopes = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (sep >= lo) & (sep < hi)
        if k.sum() < 2000:
            continue
        pts.append((float(np.median(sep[k])), float(np.mean(dist[k] < cap))))
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    ok = (y > 0) & (x >= 2e4) & (x <= 1e6)
    if ok.sum() >= 4:
        a = np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0]
        print(f"[cal]   radius {cap:6.2f}: alpha = {-a:.3f}  (Hi-C 0.855)  "
              f"nu = {-a/3:.3f}  from {ok.sum()} bins")
