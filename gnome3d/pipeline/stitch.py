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

The energy carries its own gradient. A chromosome is a thousand or more blocks, so six
variables per block puts the problem in the thousands of dimensions, where a finite difference
gradient costs one evaluation per variable and the solver runs out of its evaluation budget
after one step. See [[project_boundary_stitch]].
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BoolArray, F64Array, I64Array

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


def _hat(v: F64Array) -> F64Array:
    """The skew symmetric matrices of a stack of vectors, shape (..., 3) to (..., 3, 3)."""
    z = np.zeros(v.shape[:-1], dtype=np.float64)
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack(
        [
            np.stack([z, -w, y], axis=-1),
            np.stack([w, z, -x], axis=-1),
            np.stack([-y, x, z], axis=-1),
        ],
        axis=-2,
    )


def _rot_and_jac(rv: F64Array) -> tuple[F64Array, F64Array]:
    """The rotation matrices of a stack of rotation vectors and their derivatives.

    Returns `R` of shape (n, 3, 3) and `dR` of shape (n, 3, 3, 3) where `dR[:, i]` is the
    derivative of `R` with respect to the i-th component of the rotation vector. The formula is
    Gallego and Yezzi's, with the small angle limit taken as the generator itself.

    Parameters
    ----------
    rv
        Rotation vectors, shape (n, 3).
    """
    n = rv.shape[0]
    rot: F64Array = Rotation.from_rotvec(rv).as_matrix().reshape(n, 3, 3)
    rx = _hat(rv)
    th2 = np.sum(rv * rv, axis=1)
    small = th2 < 1e-16
    gen = _hat(np.eye(3))
    a = np.eye(3)[None, :, :] - rot
    out = np.empty((n, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        m = rv[:, i, None, None] * rx + _hat(np.cross(rv, a[:, :, i]))
        out[:, i] = np.where(
            small[:, None, None],
            gen[i][None, :, :],
            m @ rot / np.where(small, 1.0, th2)[:, None, None],
        )
    return rot, out


def _energy_grad(
    x: F64Array,
    cen: F64Array,
    first: F64Array,
    last: F64Array,
    target: F64Array,
    w_spring: float,
    iu0: I64Array,
    iu1: I64Array,
    r0: F64Array,
    w_ev: float,
) -> tuple[float, F64Array]:
    """The stitch energy and its gradient at one rotation and one translation per block.

    The variable vector is the n rotation vectors followed by the n translations, both
    flattened. Excluded volume pairs are expected to be pre filtered to a positive radius.

    Parameters
    ----------
    x
        The 6n variables.
    cen
        Per block anchor centroids, shape (n, 3).
    first, last
        The first and last anchor of each block relative to its centroid, shape (n, 3).
    target
        The boundary distance each consecutive block pair is held to, shape (n - 1,).
    w_spring
        Weight on the boundary springs.
    iu0, iu1
        Block index pairs carrying the centroid excluded volume.
    r0
        The excluded volume radius of each of those pairs.
    w_ev
        Weight on the excluded volume.
    """
    n = cen.shape[0]
    rv = x[: 3 * n].reshape(n, 3)
    t = x[3 * n :].reshape(n, 3)
    rot, drot = _rot_and_jac(rv)
    c = cen + t
    wl = np.einsum("nij,nj->ni", rot, last)
    wf = np.einsum("nij,nj->ni", rot, first)

    v = (c + wl)[:-1] - (c + wf)[1:]
    d = np.linalg.norm(v, axis=1)
    e = w_spring * float(np.sum(((d - target) / target) ** 2))
    gv = (2.0 * w_spring * (d - target) / target**2 / np.maximum(d, 1e-30))[:, None] * v

    gc = np.zeros((n, 3), dtype=np.float64)
    gc[:-1] += gv
    gc[1:] -= gv
    gwl = np.zeros((n, 3), dtype=np.float64)
    gwf = np.zeros((n, 3), dtype=np.float64)
    gwl[:-1] = gv
    gwf[1:] = -gv

    if w_ev > 0.0 and iu0.size:
        u = c[iu0] - c[iu1]
        dc = np.linalg.norm(u, axis=1)
        over = np.clip(r0 - dc, 0.0, None)
        e += w_ev * float(np.sum((over / r0) ** 2))
        m = over > 0.0
        if m.any():
            g = (-2.0 * w_ev * over[m] / r0[m] ** 2 / np.maximum(dc[m], 1e-30))[:, None] * u[m]
            np.add.at(gc, iu0[m], g)
            np.add.at(gc, iu1[m], -g)

    grv = np.einsum("ni,nkij,nj->nk", gwl, drot, last) + np.einsum(
        "ni,nkij,nj->nk", gwf, drot, first
    )
    return e, np.concatenate([grv.reshape(-1), gc.reshape(-1)])


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
    ev_pairs: BoolArray = r0 > 0.0
    iu0 = iu[0][ev_pairs]
    iu1 = iu[1][ev_pairs]
    r0 = r0[ev_pairs]

    res = minimize(
        _energy_grad,
        np.zeros(6 * n),
        args=(cen, first, last, target, w_spring, iu0, iu1, r0, w_ev),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(s.boundary_stitch_max_iter),
            # The evaluation budget must not bind before the iteration count does, which is what
            # scipy's own default of 15000 did once a chromosome brought thousands of variables.
            "maxfun": 100 * int(s.boundary_stitch_max_iter) + 1000,
            "ftol": 1e-14,
            "gtol": 1e-10,
        },
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
