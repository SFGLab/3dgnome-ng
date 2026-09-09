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

A third term, off unless `boundary_stitch_compartment_weight` is positive, is a compartment
affinity between blocks. Compartments are a pattern over many blocks, and the per block
chains cannot build it, while this pass decides where blocks sit relative to one another and
would otherwise undo whatever the block placement stage arranged. Each block carries one site
per compartment, the centroid of its A beads and of its B beads, each with the fraction of
the block's beads it holds, and two sites of the same class on different blocks attract
through the well `1 - exp(-(d - R)^2 / 2 R^2)` beyond touching, with R the two blocks' radii
of gyration added, and zero inside it, so the term draws two blocks together until they touch
and no further. The well is flat beyond a few R, so only neighbouring blocks feel it. See
[[project_epigenome_terms]].

The energy carries its own gradient. A chromosome is a thousand or more blocks, so six
variables per block puts the problem in the thousands of dimensions, where a finite difference
gradient costs one evaluation per variable and the solver runs out of its evaluation budget
after one step. See [[project_boundary_stitch]].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BoolArray, F64Array, I8Array, I64Array

_MIN_BIN_COUNT = 5


class CompartmentSites(NamedTuple):
    """The compartment affinity's inputs. One A site and one B site per block, as local
    vectors from the block's pivot, with the fraction of the block's beads each holds, and
    the block pairs it acts over with a well radius per pair."""

    sites: F64Array
    mass: F64Array
    strength: tuple[float, float]
    pairs0: I64Array
    pairs1: I64Array
    radius: F64Array
    weight: float


def compartment_sites(
    pos: list[F64Array],
    cen: F64Array,
    classes: list[I8Array],
    rg: F64Array,
    strength: tuple[float, float],
    weight: float,
) -> CompartmentSites:
    """Build the affinity's sites from each block's bead classes, positive for A and negative
    for B. The well radius of a pair is the two blocks' radii of gyration added, so the term
    reaches as far as the blocks are large whatever the excluded volume is set to."""
    n = len(pos)
    sites = np.zeros((n, 2, 3), dtype=np.float64)
    mass = np.zeros((n, 2), dtype=np.float64)
    for k, (p, c) in enumerate(zip(pos, classes, strict=True)):
        for slot, m in enumerate((c > 0, c < 0)):
            if m.any():
                sites[k, slot] = p[m].mean(axis=0) - cen[k]
                mass[k, slot] = float(m.sum()) / p.shape[0]
    iu = np.triu_indices(n, k=1)
    return CompartmentSites(sites, mass, strength, iu[0], iu[1], rg[iu[0]] + rg[iu[1]], weight)


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
    comp: CompartmentSites | None = None,
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
    comp
        The compartment affinity's sites, or None for no such term.
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

    if comp is not None and comp.weight > 0.0 and comp.pairs0.size:
        ws = np.einsum("nij,nsj->nsi", rot, comp.sites)
        world = c[:, None, :] + ws
        gsite = np.zeros_like(ws)
        for slot, g_cls in enumerate(comp.strength):
            i, j = comp.pairs0, comp.pairs1
            mm = comp.mass[i, slot] * comp.mass[j, slot]
            live = mm > 0.0
            if not live.any():
                continue
            i, j, mm, rr = i[live], j[live], mm[live], comp.radius[live]
            u = world[i, slot] - world[j, slot]
            d = np.sqrt(np.sum(u * u, axis=1))
            # The well starts at touching, so the term pulls two blocks together and no
            # further. A well centred on zero kept pulling until the sites coincided, which
            # drove blocks into each other and left the excluded volume holding them apart,
            # and every bead then touched another block.
            over = np.clip(d - rr, 0.0, None)
            ex = np.exp(-(over * over) / (2.0 * rr * rr))
            e += comp.weight * g_cls * float(np.sum(mm * (1.0 - ex)))
            gu = (comp.weight * g_cls * mm * ex * over / (rr * rr * np.maximum(d, 1e-30)))[
                :, None
            ] * u
            np.add.at(gsite[:, slot], i, gu)
            np.add.at(gsite[:, slot], j, -gu)
        gc += gsite.sum(axis=1)
        grv += np.einsum("nsi,nkij,nsj->nk", gsite, drot, comp.sites)
    return e, np.concatenate([grv.reshape(-1), gc.reshape(-1)])


def stitch_blocks(
    blocks: list[list[BeadOut]], s: Settings, compartments: list[I8Array] | None = None
) -> list[list[BeadOut]]:
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
    compartments
        One int8 class array per block, aligned with the block's beads, positive for A and
        negative for B. Read only when `boundary_stitch_compartment_weight` is positive.
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
    comp: CompartmentSites | None = None
    w_comp = float(s.boundary_stitch_compartment_weight)
    if w_comp > 0.0 and compartments is not None:
        comp = compartment_sites(
            pos,
            cen,
            [compartments[k] for k in active],
            rg,
            (float(s.compartment_energy_a), float(s.compartment_energy_b)),
            w_comp,
        )

    res = minimize(
        _energy_grad,
        np.zeros(6 * n),
        args=(cen, first, last, target, w_spring, iu0, iu1, r0, w_ev, comp),
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
