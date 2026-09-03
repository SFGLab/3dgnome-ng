"""Rigid block stitching across interaction block boundaries.

The per block chains place anchors only through their own block, so the last anchor of one
block and the first anchor of the next have no term coupling them and end up as far apart as
their block centroids happen to be. This pass moves each block as a rigid body so that every
such boundary pair sits at the distance an interior pair of the same genomic separation
realises in the same structure. Rigid means every distance inside a block is preserved, so
the arcs and smooth results are untouched.

The pass runs after all chains of a chromosome are done, on the calling thread, with no RNG.
Two terms are minimised over one rotation and one translation per block. A two sided spring
per boundary, and a soft excluded volume between block centroids with a radius per pair of
the two blocks' radii of gyration added, so closing a boundary cannot fold a block onto a
neighbour it shares no edge with. `exclusion_radius_ib`, when positive, replaces that with one
constant radius for every pair. See [[project_ib_packing_factor]] for why there is no chain bond and no
confinement here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from gnome3d.settings import Settings
from gnome3d.types import BeadOut, F64Array, I64Array

_MIN_BIN_COUNT = 5


def _mid(b: BeadOut) -> int:
    return (b.start + b.end) // 2


def _positions(block: list[BeadOut]) -> F64Array:
    return np.array([[b.x, b.y, b.z] for b in block], dtype=np.float64)


def _anchor_index(block: list[BeadOut]) -> I64Array:
    return np.array([i for i, b in enumerate(block) if b.kind == "anchor"], dtype=np.int64)


def within_block_curve(
    blocks: list[list[BeadOut]], max_pairs: int = 2_000_000, bins_per_decade: int = 8
) -> Callable[[int], float] | None:
    """Median anchor distance against genomic separation, from the structure's own interior
    pairs. Returns a function of the separation in bp, clamped to the outermost sampled bins,
    or None when no block holds enough anchor pairs to say anything.

    Pairs are taken over every block separately so no boundary is ever inside one. Above
    `max_pairs` the pairs are thinned by a fixed stride, which keeps the pass deterministic.
    """
    seps: list[F64Array] = []
    dists: list[F64Array] = []
    for block in blocks:
        idx = _anchor_index(block)
        if idx.size < 2:
            continue
        pos = _positions(block)[idx]
        mid = np.array([_mid(block[i]) for i in idx], dtype=np.float64)
        iu = np.triu_indices(idx.size, k=1)
        seps.append(np.abs(mid[iu[0]] - mid[iu[1]]))
        dists.append(np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1))
    if not seps:
        return None
    sep = np.concatenate(seps)
    dist = np.concatenate(dists)
    keep = sep > 0
    sep, dist = sep[keep], dist[keep]
    if sep.size == 0:
        return None
    if sep.size > max_pairs:
        stride = np.linspace(0, sep.size - 1, max_pairs).astype(np.int64)
        sep, dist = sep[stride], dist[stride]

    ls = np.log10(sep)
    edges = np.arange(np.floor(ls.min() * bins_per_decade), np.ceil(ls.max() * bins_per_decade) + 1)
    edges = edges / bins_per_decade
    which = np.clip(np.searchsorted(edges, ls, side="right") - 1, 0, edges.size - 2)
    xs: list[float] = []
    ys: list[float] = []
    for b in range(edges.size - 1):
        m = which == b
        if m.sum() < _MIN_BIN_COUNT:
            continue
        xs.append(float(np.median(ls[m])))
        ys.append(float(np.log10(np.median(dist[m]))))
    if not xs:
        return None
    xa = np.array(xs)
    ya = np.array(ys)

    def curve(gap_bp: int) -> float:
        return float(10.0 ** np.interp(np.log10(max(gap_bp, 1)), xa, ya))

    return curve


def stitch_blocks(blocks: list[list[BeadOut]], s: Settings) -> list[list[BeadOut]]:
    """Return the blocks with each one moved rigidly so boundary pairs sit on the interior
    curve. Blocks without anchors pass through untouched and do not take part in the chain.
    With fewer than two blocks holding anchors, or no usable curve, the input is returned as
    is.

    Parameters
    ----------
    blocks
        One chromosome's per block bead lists, in any order. Chain order is taken from each
        block's first anchor midpoint.
    s
        Settings. Reads the `boundary_stitch_*` weights and `exclusion_radius_ib`.
    """
    active = [k for k, b in enumerate(blocks) if _anchor_index(b).size > 0]
    if len(active) < 2:
        return blocks
    curve = within_block_curve(blocks)
    if curve is None:
        return blocks

    pos = [_positions(blocks[k]) for k in active]
    aidx = [_anchor_index(blocks[k]) for k in active]
    order = sorted(range(len(active)), key=lambda j: _mid(blocks[active[j]][aidx[j][0]]))
    pos = [pos[j] for j in order]
    aidx = [aidx[j] for j in order]
    active = [active[j] for j in order]
    n = len(active)

    cen = np.array([p[a].mean(axis=0) for p, a in zip(pos, aidx, strict=True)])
    first = np.array([p[a[0]] - c for p, a, c in zip(pos, aidx, cen, strict=True)])
    last = np.array([p[a[-1]] - c for p, a, c in zip(pos, aidx, cen, strict=True)])
    gaps = [
        _mid(blocks[active[j + 1]][aidx[j + 1][0]]) - _mid(blocks[active[j]][aidx[j][-1]])
        for j in range(n - 1)
    ]
    target = np.array([curve(int(abs(g))) for g in gaps], dtype=np.float64)
    w_spring = float(s.boundary_stitch_spring_weight)
    w_ev = float(s.boundary_stitch_ev_weight)
    iu = np.triu_indices(n, k=1)
    rg = np.array([float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1)))) for p in pos])
    r0 = (
        np.full(iu[0].size, float(s.exclusion_radius_ib))
        if s.exclusion_radius_ib > 0.0
        else rg[iu[0]] + rg[iu[1]]
    )
    ev_pairs = r0 > 0.0

    def energy(x: F64Array) -> float:
        rv = x[: 3 * n].reshape(n, 3)
        t = x[3 * n :].reshape(n, 3)
        rot = Rotation.from_rotvec(rv)
        c = cen + t
        a_last = c + rot.apply(last)
        a_first = c + rot.apply(first)
        d = np.linalg.norm(a_last[:-1] - a_first[1:], axis=1)
        e = w_spring * float(np.sum(((d - target) / target) ** 2))
        if w_ev > 0.0 and ev_pairs.any():
            dc = np.linalg.norm(c[iu[0]] - c[iu[1]], axis=1)[ev_pairs]
            over = np.clip(r0[ev_pairs] - dc, 0.0, None)
            e += w_ev * float(np.sum((over / r0[ev_pairs]) ** 2))
        return e

    res = minimize(
        energy,
        np.zeros(6 * n),
        method="L-BFGS-B",
        options={"maxiter": int(s.boundary_stitch_max_iter), "ftol": 1e-14, "gtol": 1e-10},
    )
    rv = res.x[: 3 * n].reshape(n, 3)
    t = res.x[3 * n :].reshape(n, 3)

    out = list(blocks)
    for j, k in enumerate(active):
        moved = Rotation.from_rotvec(rv[j]).apply(pos[j] - cen[j]) + cen[j] + t[j]
        out[k] = [
            BeadOut(b.start, b.end, float(p[0]), float(p[1]), float(p[2]), b.kind)
            for b, p in zip(blocks[k], moved, strict=True)
        ]
    return out
