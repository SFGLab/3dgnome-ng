"""Where the within block overlaps of a finished structure are, before anyone touches a weight.

    python playground/overlap_profile.py STRUCTURE.cif [--target-bp 1000] [--ev-factor 0.7]

The battery counts a pair as overlapping when it is more than one bead apart along the chain
and closer than `ev_factor` times its block's mean bond. This splits that count by chain
separation, by bead kind, by how deep inside the radius the pair sits, and by block size, and
puts an ideal chain with no excluded volume at all beside it, which says whether the term is
doing anything. It also prices the overlaps in the smooth stage's own energy at a given weight
against the chain springs, since a term that costs less than the springs it competes with is
not a term the annealer can feel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.validation_battery import _flag, bead_scale, block_owner, load  # noqa: E402

SEP_BANDS = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 30), (31, 100), (101, 1000), (1001, 10**9)]


def overlapping_pairs(pos: np.ndarray, rad: float, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = KDTree(pos).query_pairs(float(rad), output_type="ndarray")
    q = q[np.abs(q[:, 0] - q[:, 1]) > 1]
    i, j = q[:, 0], q[:, 1]
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    keep = owner[i] == owner[j]
    return q[keep], d[keep]


def ideal_chain(n: int, bond: float, seed: int = 0) -> np.ndarray:
    """A freely jointed chain of `n` beads at the given bond, no excluded volume."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n - 1, 3))
    v *= bond / np.linalg.norm(v, axis=1)[:, None]
    return np.vstack([np.zeros(3), np.cumsum(v, axis=0)])


def profile(label: str, pos: np.ndarray, anchor: np.ndarray, owner: np.ndarray, ev_factor: float, weight: float) -> None:
    n = len(pos)
    bead = bead_scale(pos, anchor)
    rad = ev_factor * bead
    q, d = overlapping_pairs(pos, rad, owner)
    i, j = q[:, 0], q[:, 1]
    sep = np.abs(i - j)
    r0 = rad
    frac = d / r0
    print(f"\n{label}: {n:,} beads, {len(q):,} overlapping pairs, {1000 * len(q) / n:.0f} per thousand beads")
    print("  by chain separation (pairs per thousand beads, share of overlaps):")
    for lo, hi in SEP_BANDS:
        m = (sep >= lo) & (sep <= hi)
        if m.any():
            print(f"    {lo:>5}-{min(hi, sep.max()):<6} {1000 * m.sum() / n:>8.1f}  {100 * m.sum() / len(q):>5.1f}%")
    aa = anchor[i] & anchor[j]
    sa = anchor[i] ^ anchor[j]
    ss = ~anchor[i] & ~anchor[j]
    print(f"  by kind: anchor-anchor {100 * aa.mean():.1f}%  anchor-subanchor {100 * sa.mean():.1f}%  subanchor-subanchor {100 * ss.mean():.1f}%")
    print("  depth, distance over radius:")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        m = (frac >= lo) & (frac < hi)
        print(f"    {lo:.2f}-{hi:.2f}  {100 * m.mean():>5.1f}%")
    print(f"  median pair distance {np.median(d):.3f} against a subanchor bond of {bead:.3f}")
    # energy the overlaps cost at the smooth stage's soft quadratic, double counted like the kernel
    ev = weight * ((r0 - d) / r0) ** 2
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    same = owner[:-1] == owner[1:]
    target = np.full(int(same.sum()), bead)  # the bead stands in for the target
    chain = np.abs(step[same] - target) / target  # the smooth stage's bond term at dist_weight 1
    print(
        f"  excluded volume energy of the overlaps at weight {weight}: {2 * ev.sum():,.1f} total,"
        f" {2 * ev.sum() / n:.4f} per bead; the bond term at weight 1 on the realised bonds: {chain.sum():,.1f}"
    )
    print(f"  a pair at half the radius costs {2 * weight * 0.25:.4f} counted both ways; one bond stretched 5 percent costs {0.05:.3f}")
    # per block: rate against block size
    sizes = np.bincount(owner)
    per_block = np.bincount(owner[i], minlength=len(sizes))
    print("  by block size (beads): rate per thousand, blocks")
    for lo, hi in ((1, 200), (201, 1000), (1001, 5000), (5001, 20000), (20001, 10**9)):
        m = (sizes >= lo) & (sizes <= hi)
        if m.any():
            print(f"    {lo:>6}-{min(hi, sizes.max()):<6} {1000 * per_block[m].sum() / max(sizes[m].sum(), 1):>8.1f}  {m.sum():>4}")


def main() -> None:
    target_bp = int(_flag("--target-bp", 1000))
    ev_factor = _flag("--ev-factor", 0.7)
    weight = _flag("--weight", 0.1)
    pos, _, mid, anchor = load(Path(sys.argv[1]))
    owner = block_owner(mid, anchor, target_bp)
    profile(Path(sys.argv[1]).name, pos, anchor, owner, ev_factor, weight)
    bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    n = len(pos)
    ideal = ideal_chain(n, bond)
    profile("ideal chain, same bead count and bond, no excluded volume", ideal, np.zeros(n, bool), np.zeros(n, np.int64), ev_factor, weight)


if __name__ == "__main__":
    main()
