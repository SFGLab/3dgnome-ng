"""The uniform chain of a block and the spans of its anchors, `[main] uniform_chain`.

The chain has one bead per `target_bp_per_subanchor` from the start of the first anchor of the
block to the end of its last. Every base of the block is in one bead. The span of an anchor is
the run of beads whose midpoints are in its range. Anchors that overlap share beads. The rules
of the model are in `design/uniform-chain.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gnome3d.tracks import bin_compartments

if TYPE_CHECKING:
    from gnome3d.pipeline.state import AnchorMapEntry
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
    )

# (pos, fixed, starts, ends, dtn, anchor_map, step_size_smooth, compartment, covered)
UniformResult = tuple[
    "F32Array",
    "BoolArray",
    list[int],
    list[int],
    "F32Array",
    "list[AnchorMapEntry]",
    float,
    "I8Array | None",
    "BoolArray",
]


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
    """Returns `(lo, hi, knot)` per anchor. The span holds the beads `lo` to `hi - 1`, the beads
    whose midpoints are in the range of the anchor and always the knot, the bead at the midpoint
    of the anchor.
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


def covered_mask(lo: I32Array, hi: I32Array, n: int) -> BoolArray:
    """Returns the beads that the span of an anchor holds."""
    mask: BoolArray = np.zeros(n, dtype=np.bool_)
    for a, b in zip(lo, hi, strict=True):
        mask[int(a) : int(b)] = True
    return mask


def start_chain(
    node_bead: I32Array, node_pos: F64Array, n: int, bond: float, centre: F64Array
) -> F64Array:
    """Returns the start of the chain of a block.

    A node is at its position from the arcs stage. The beads between two nodes are on the line
    between them. The beads before the first node and after the last continue the line at one
    bond per bead. A block without a node is a straight line through `centre`.
    """
    idx = np.arange(n, dtype=np.float64)
    axis = np.array([1.0, 0.0, 0.0])
    if node_bead.shape[0] == 0:
        line: F64Array = centre[None, :] + (idx - (n - 1) / 2.0)[:, None] * bond * axis[None, :]
        return line
    pos = np.stack([np.interp(idx, node_bead, node_pos[:, c]) for c in range(3)], axis=1)
    first, last = int(node_bead[0]), int(node_bead[-1])
    if node_bead.shape[0] > 1:
        d = node_pos[0] - node_pos[-1]
        nd = float(np.linalg.norm(d))
        if nd > 1e-9:
            axis = d / nd
    for i in range(first):
        pos[i] = node_pos[0] + axis * bond * (first - i)
    for i in range(last + 1, n):
        pos[i] = node_pos[-1] - axis * bond * (i - last)
    return pos


def densify_uniform(
    node_bead: I32Array,
    node_pos: F32Array,
    centre: F32Array,
    anchor_genomic: list[tuple[int, int, int]],
    s: Settings,
    track_compartments: list[CompartmentInterval] | None = None,
) -> UniformResult:
    """Builds the uniform chain of a block. The nodes are the fixed beads.

    Parameters
    ----------
    node_bead
        The bead of each node of the block, in bead order.
    node_pos
        The position of each node from the arcs stage.
    centre
        The centroid of the block, the place of a chain without a node.
    anchor_genomic
        The anchors of the block as (start, end, midpoint) triples.
    """
    edges = grid_edges(anchor_genomic, int(s.target_bp_per_subanchor))
    n = int(edges.shape[0]) - 1
    lo, hi, _knot = anchor_spans(anchor_genomic, edges)

    starts = [int(e) for e in edges[:-1]]
    ends = [int(e) for e in edges[1:]]
    mids = (edges[:-1] + edges[1:]) // 2
    dtn: F32Array = np.array(
        [s.genomic_length_to_distance(int(g)) for g in np.diff(mids)], dtype=np.float32
    )
    bond = float(dtn.mean()) if dtn.size else 1.0
    pos: F32Array = start_chain(
        node_bead, node_pos.astype(np.float64), n, bond, centre.astype(np.float64)
    ).astype(np.float32)
    fixed: BoolArray = np.zeros(n, dtype=np.bool_)
    fixed[node_bead] = True
    anchor_map: list[AnchorMapEntry] = [(int(b), k) for k, b in enumerate(node_bead)]

    compartment: I8Array | None = None
    if track_compartments:
        compartment, _score = bin_compartments(track_compartments, starts, ends)

    return (
        pos,
        fixed,
        starts,
        ends,
        dtn,
        anchor_map,
        bond * s.noise_smooth,
        compartment,
        covered_mask(lo, hi, n),
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
