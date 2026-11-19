"""The uniform chain of a block and its anchors as spans of beads, `[main] uniform_chain`.

The chain has one bead per `target_bp_per_subanchor` from the start of the first anchor of the
block to the end of its last. Every base of the block is in one bead. The span of an anchor is
the run of beads whose midpoints are in its range. Anchors that overlap share beads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gnome3d.tracks import bin_compartments
from gnome3d.types import AnchorSpans

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import (
        BoolArray,
        CompartmentInterval,
        F32Array,
        F64Array,
        I8Array,
        I32Array,
        I64Array,
        SingletonContact,
        StrArray,
    )

# (pos, fixed, starts, ends, dtn, step_size_smooth, compartment, spans)
UniformResult = tuple[
    "F32Array",
    "BoolArray",
    list[int],
    list[int],
    "F32Array",
    float,
    "I8Array | None",
    AnchorSpans,
]

# The orientation of a site is the direction from the bead before it to the bead after it. Two
# sites closer than this many beads would read one bead in common.
SITE_SPACING = 3


def grid_edges(anchor_genomic: list[tuple[int, int, int]], target_bp: int) -> I64Array:
    """Returns the edges of the beads of a block, `n + 1` positions for `n` beads of equal
    width. `n` is the span of the block over `target_bp`, rounded and at least one.
    """
    lo = min(a[0] for a in anchor_genomic)
    hi = max(a[1] for a in anchor_genomic)
    span = max(hi - lo, 1)
    n = max(1, round(span / max(target_bp, 1)))
    edges: I64Array = lo + (np.arange(n + 1, dtype=np.int64) * span) // n
    return edges


def anchor_spans(
    anchor_genomic: list[tuple[int, int, int]], edges: I64Array
) -> tuple[I32Array, I32Array, I32Array]:
    """Returns `(lo, hi, knot)` per anchor. The span holds the beads whose midpoints are in the
    range of the anchor and always the knot, the bead at the midpoint of the anchor.
    """
    n = int(edges.shape[0]) - 1
    mids = (edges[:-1] + edges[1:]) / 2.0
    start = np.array([a[0] for a in anchor_genomic], dtype=np.int64)
    end = np.array([a[1] for a in anchor_genomic], dtype=np.int64)
    centre = np.array([a[2] for a in anchor_genomic], dtype=np.int64)
    knot = np.clip(np.searchsorted(edges, centre, side="right") - 1, 0, n - 1)
    lo = np.minimum(np.searchsorted(mids, start, side="left"), knot)
    hi = np.maximum(np.searchsorted(mids, end, side="left"), knot + 1)
    return lo.astype(np.int32), hi.astype(np.int32), knot.astype(np.int32)


def _start_chain(knot: I32Array, home: F64Array, n: int, bond: float) -> F64Array:
    """Returns the start of the chain. A knot is at the mean home of its anchors. The beads
    between two knots are on the line between them. The beads before the first knot and after
    the last continue the line at one bond per bead.
    """
    beads = np.unique(knot)
    at = np.zeros((beads.shape[0], 3), dtype=np.float64)
    for j, b in enumerate(beads):
        at[j] = home[knot == b].mean(axis=0)
    idx = np.arange(n, dtype=np.float64)
    pos = np.stack([np.interp(idx, beads, at[:, c]) for c in range(3)], axis=1)
    first, last = int(beads[0]), int(beads[-1])
    out = np.array([1.0, 0.0, 0.0])
    if beads.shape[0] > 1:
        d = at[0] - at[-1]
        nd = float(np.linalg.norm(d))
        if nd > 1e-9:
            out = d / nd
    for i in range(first):
        pos[i] = at[0] + out * bond * (first - i)
    for i in range(last + 1, n):
        pos[i] = at[-1] - out * bond * (i - last)
    return pos


def densify_uniform(
    anchor_pos: F32Array,
    anchor_genomic: list[tuple[int, int, int]],
    s: Settings,
    track_compartments: list[CompartmentInterval] | None = None,
) -> UniformResult:
    """Builds the uniform chain of a block with its anchors as spans of beads. No bead is fixed."""
    edges = grid_edges(anchor_genomic, int(s.target_bp_per_subanchor))
    n = int(edges.shape[0]) - 1
    lo, hi, knot = anchor_spans(anchor_genomic, edges)
    home: F64Array = np.asarray(anchor_pos, dtype=np.float64).copy()

    starts = [int(e) for e in edges[:-1]]
    ends = [int(e) for e in edges[1:]]
    mids = (edges[:-1] + edges[1:]) // 2
    dtn: F32Array = np.array(
        [s.genomic_length_to_distance(int(g)) for g in np.diff(mids)], dtype=np.float32
    )
    bond = float(dtn.mean()) if dtn.size else 1.0
    pos: F32Array = _start_chain(knot, home, n, bond).astype(np.float32)
    fixed: BoolArray = np.zeros(n, dtype=np.bool_)

    compartment: I8Array | None = None
    if track_compartments:
        compartment, _score = bin_compartments(track_compartments, starts, ends)

    step_size_smooth = bond * s.noise_smooth
    return (
        pos,
        fixed,
        starts,
        ends,
        dtn,
        step_size_smooth,
        compartment,
        AnchorSpans(lo=lo, hi=hi, knot=knot, home=home),
    )


def uniform_heatmap(singletons: list[SingletonContact], chrom: str, edges: I64Array) -> F32Array:
    """Returns the singleton contact heatmap of a block on the beads of its uniform chain.

    The counts are divided by their mean and by the areas of the bins in kb squared, as the
    subanchor heatmap of the default chain is. The diagonal is zero.
    """
    n = int(edges.shape[0]) - 1
    lo, hi = int(edges[0]), int(edges[-1])
    rows = [
        (p1, p2, sc)
        for c1, p1, c2, p2, sc in singletons
        if c1 == chrom and c2 == chrom and lo <= p1 <= hi and lo <= p2 <= hi
    ]
    heat: F64Array = np.zeros((n, n), dtype=np.float64)
    if rows:
        arr = np.array(rows, dtype=np.int64)
        i = np.searchsorted(edges, arr[:, 0], side="right") - 1
        j = np.searchsorted(edges, arr[:, 1], side="right") - 1
        keep = (i < n) & (j < n) & (i != j)
        np.add.at(heat, (i[keep], j[keep]), arr[keep, 2])
        np.add.at(heat, (j[keep], i[keep]), arr[keep, 2])
    mean = float(heat.mean())
    if mean > 1e-6:
        kb = np.maximum(np.diff(edges).astype(np.float64), 1.0) / 1000.0
        heat /= mean * np.outer(kb, kb)
    return heat.astype(np.float32)


def merge_nodes(
    exp_dist: F64Array, arc_w: F32Array | None, node: I64Array
) -> tuple[F64Array, F32Array | None]:
    """Returns the arcs target matrix and the loop weights with the anchors of one node as one row.

    `node[k]` is the node of anchor k. A pair of nodes takes the shortest loop target of its
    anchor pairs, else their shortest background, else the repulsion mark of any of them. A
    pair of anchors inside one node is dropped. The weight is the largest of the loops.
    """
    order = np.argsort(node, kind="stable")
    first = np.flatnonzero(np.diff(node[order], prepend=-1))
    ep = exp_dist[np.ix_(order, order)]

    def shortest(values: F64Array) -> F64Array:
        rows: F64Array = np.minimum.reduceat(values, first, axis=0)
        out: F64Array = np.minimum.reduceat(rows, first, axis=1)
        return out

    loop = shortest(np.where(ep > 1e-6, ep, np.inf))
    background = shortest(np.where(ep <= -0.75, -ep, np.inf))
    mark = (ep < 0.0) & (ep > -0.75)
    repel = np.logical_or.reduceat(np.logical_or.reduceat(mark, first, axis=0), first, axis=1)
    merged: F64Array = np.zeros(loop.shape, dtype=np.float64)
    merged[repel] = -0.5
    has_bg = np.isfinite(background)
    merged[has_bg] = -background[has_bg]
    has_loop = np.isfinite(loop)
    merged[has_loop] = loop[has_loop]
    np.fill_diagonal(merged, 0.0)
    if arc_w is None:
        return merged, None
    wp = np.where(ep > 1e-6, arc_w[np.ix_(order, order)], 0.0)
    w_rows = np.maximum.reduceat(wp, first, axis=0)
    weights: F32Array = np.maximum.reduceat(w_rows, first, axis=1).astype(np.float32)
    return merged, weights


def knot_mask(spans: AnchorSpans, n: int) -> BoolArray:
    """Returns the beads that are the knot of an anchor."""
    mask: BoolArray = np.zeros(n, dtype=np.bool_)
    mask[spans.knot] = True
    return mask


def covered_mask(spans: AnchorSpans, n: int) -> BoolArray:
    """Returns the beads that an anchor holds."""
    mask: BoolArray = np.zeros(n, dtype=np.bool_)
    for a, b in zip(spans.lo, spans.hi, strict=True):
        mask[int(a) : int(b)] = True
    return mask


def motif_sites(
    spans: AnchorSpans,
    orientations: I8Array,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    n: int,
    code_to_char: dict[int, str],
) -> tuple[BoolArray, StrArray, dict[int, list[int]], dict[int, list[float]]]:
    """Returns the orientation sites of a uniform chain as `(site beads, orientation per bead,
    neighbours per site, weights per site)`. Sites are numbered in bead order.

    An anchor with a CTCF loop makes a site at its knot. A site is `SITE_SPACING` beads or more
    from every other and one bead or more from each end of the chain. An anchor too near a
    site of more loop weight joins that site and its loops go to it. The site has the
    orientation of its heaviest anchor.
    """
    site_beads: BoolArray = np.zeros(n, dtype=np.bool_)
    chars: StrArray = np.array(["N"] * n, dtype="<U1")
    if n < SITE_SPACING:
        return site_beads, chars, {}, {}
    total = {k: float(sum(weights.get(k, []))) for k in neighbors if neighbors[k]}
    bead_of = {k: int(np.clip(int(spans.knot[k]), 1, n - 2)) for k in total}
    kept: list[int] = []
    owner: dict[int, int] = {}
    for k in sorted(total, key=lambda a: (-total[a], bead_of[a], a)):
        near = [j for j in kept if abs(bead_of[j] - bead_of[k]) < SITE_SPACING]
        if near:
            owner[k] = min(near, key=lambda j: (abs(bead_of[j] - bead_of[k]), bead_of[j]))
        else:
            kept.append(k)
            owner[k] = k
    kept.sort(key=lambda a: bead_of[a])
    site_of = {k: i for i, k in enumerate(kept)}
    for k in kept:
        site_beads[bead_of[k]] = True
        chars[bead_of[k]] = code_to_char[int(orientations[k])]

    pair_w: list[dict[int, float]] = [{} for _ in kept]
    for k, own in owner.items():
        i = site_of[own]
        for j, w in zip(neighbors[k], weights[k], strict=True):
            if j not in owner:
                continue
            t = site_of[owner[j]]
            if t != i:
                pair_w[i][t] = pair_w[i].get(t, 0.0) + float(w)
    site_nbrs = {i: sorted(pair_w[i]) for i in range(len(kept))}
    site_w = {i: [pair_w[i][t] for t in site_nbrs[i]] for i in range(len(kept))}
    return site_beads, chars, site_nbrs, site_w
