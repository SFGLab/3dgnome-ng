"""The contact beads of a conformation on the uniform chain, `design/uniform-chain.md`.

A loop acts on one bead of the span of each of its two anchors. Each loop end draws its bead by
itself. The distinct contact beads of a chromosome are the nodes of the arcs stage. The strengths
of the loops on one pair of nodes add.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.settings import Settings
    from gnome3d.types import F32Array, F64Array, I32Array, I64Array


class ContactNodes(NamedTuple):
    """The nodes of one conformation of a chromosome and the loops between them.

    A node is a bead of the chain of a block. Nodes are in the order of the chromosome. Each
    pair with a loop is listed once with the lower node first.
    """

    block: I32Array  # (M,) block of each node
    bead: I32Array  # (M,) bead of the node in the chain of its block
    mid: I64Array  # (M,) genomic midpoint of the bead
    pair_i: I32Array  # (P,)
    pair_j: I32Array  # (P,)
    strength: F64Array  # (P,) summed strength of the loops of the pair
    loop_node: I32Array  # (L, 2) node of each loop end, -1 for a loop left out


def draw_ends(lo: I32Array, hi: I32Array, ends: I32Array, rng: np.random.Generator) -> I64Array:
    """Returns the contact bead of each loop end, in the chain of the block of its anchor.

    `ends` is (L, 2) anchors. Each end draws a bead of the span of its anchor with equal chance.
    """
    width = (hi - lo)[ends].astype(np.float64)
    drawn: I64Array = lo[ends] + np.floor(rng.random(ends.shape) * width).astype(np.int64)
    return drawn


def contact_nodes(
    edges: list[I64Array],
    anchor_block: I32Array,
    ends: I32Array,
    beads: I64Array,
    strength: F64Array,
    extra: I64Array | None = None,
) -> ContactNodes:
    """Builds the nodes of a conformation from the contact beads of its loops.

    Parameters
    ----------
    edges
        The bead edges of the chain of each block, `grid_edges`.
    anchor_block
        The block of each anchor.
    ends
        (L, 2) the two anchors of each kept loop.
    beads
        (L, 2) the contact bead of each loop end in the chain of its block, `draw_ends`.
    strength
        (L,) the strength of each loop by the law.
    extra
        Beads that are nodes without a loop, counted along the chromosome, `locus_beads`.
    """
    sizes = np.array([e.shape[0] - 1 for e in edges], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(sizes)])
    along = offset[anchor_block[ends]] + beads  # bead counted along the chromosome
    kept = along[:, 0] != along[:, 1]
    of_loops = along[kept].ravel()
    nodes = np.unique(of_loops if extra is None else np.concatenate([of_loops, extra]))
    inverse = np.searchsorted(nodes, of_loops)
    loop_node: I32Array = np.full(along.shape, -1, dtype=np.int32)
    loop_node[kept] = inverse.reshape(-1, 2)
    pair = np.sort(inverse.reshape(-1, 2), axis=1)
    m = int(nodes.shape[0])
    keys, which = np.unique(pair[:, 0] * max(m, 1) + pair[:, 1], return_inverse=True)
    counts = np.bincount(which, weights=strength[kept], minlength=keys.shape[0])
    summed: F64Array = counts.astype(np.float64)
    block = (np.searchsorted(offset, nodes, side="right") - 1).astype(np.int32)
    bead = (nodes - offset[block]).astype(np.int32)
    mid: I64Array = np.zeros(m, dtype=np.int64)
    for k in range(m):
        e = edges[int(block[k])]
        mid[k] = (int(e[bead[k]]) + int(e[bead[k] + 1])) // 2
    return ContactNodes(
        block=block,
        bead=bead,
        mid=mid,
        pair_i=(keys // max(m, 1)).astype(np.int32),
        pair_j=(keys % max(m, 1)).astype(np.int32),
        strength=summed,
        loop_node=loop_node,
    )


def locus_beads(edges: list[I64Array], size: int) -> tuple[I64Array, I64Array]:
    """Returns the locus bead of every locus that the chains of the blocks cover, counted along the
    chromosome, and the locus of each.

    A locus is a stretch of `size` bp on a grid from the start of the chromosome. Its locus bead
    is the bead of a chain nearest to the middle of the locus.
    """
    sizes = np.array([e.shape[0] - 1 for e in edges], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(sizes)])
    parts: list[I64Array] = [
        (int(offset[b]) + np.arange(int(sizes[b]), dtype=np.int64)) for b in range(len(edges))
    ]
    along: I64Array = np.concatenate(parts)
    mid = np.concatenate([(e[:-1] + e[1:]) // 2 for e in edges])
    locus = mid // int(size)
    off = np.abs(mid - (locus * int(size) + int(size) // 2))
    order = np.lexsort((off, locus))
    first = np.ones(order.shape[0], dtype=np.bool_)
    first[1:] = locus[order][1:] != locus[order][:-1]
    beads: I64Array = along[order][first].astype(np.int64)
    loci: I64Array = locus[order][first].astype(np.int64)
    return beads, loci


def mean_loop_target(
    mids: list[int],
    arcs: list[tuple[int, int, int] | tuple[int, int, int, int]],
    law: PolymerLaw,
    s: Settings,
) -> float:
    """Returns the mean target of the loops of a chromosome, each at the midpoints of its two
    anchors, or 1 without a loop. `arcs` are `(anchor, anchor, score[, factor])`.
    """
    if not arcs:
        return 1.0
    if s.loop_contact:
        return 1.0
    total = 0.0
    for arc in arcs:
        sep = abs(int(mids[arc[0]]) - int(mids[arc[1]]))
        q = law.arc_strength(int(arc[2]), sep, int(arc[3]) if len(arc) > 3 else 0)
        total += float(law.contact_distance(sep, float(q)))
    return total / len(arcs)


def chain_share(mid: I64Array, start: int, end: int, count: int) -> F64Array:
    """Returns the chain each node stands for over the mean chain of a node.

    A node stands for the chain from the midpoint to its neighbour before it to the midpoint to
    its neighbour after it. The first node starts at `start` and the last one ends at `end`, the
    two ends of the chain of the chromosome. The mean chain of a node is the whole chain over
    `count`, the number of nodes of a conformation that keeps every loop. The shares sum to
    `count`.
    """
    m = np.asarray(mid, dtype=np.float64)
    bounds = np.concatenate([[float(start)], 0.5 * (m[:-1] + m[1:]), [float(end)]])
    share: F64Array = np.diff(bounds) * (max(count, 1) / max(float(end - start), 1.0))
    return share


def node_weights(nodes: ContactNodes, s: Settings) -> F32Array | None:
    """Returns the spring weight of each pair of nodes, the summed strength of its loops to
    `arc_weight_exponent` and one elsewhere. Returns None at exponent zero.
    """
    alpha = float(s.arc_weight_exponent)
    if alpha <= 0.0:
        return None
    m = int(nodes.mid.shape[0])
    w: F32Array = np.ones((m, m), dtype=np.float32)
    for i, j, q in zip(nodes.pair_i, nodes.pair_j, nodes.strength, strict=True):
        w[i, j] = w[j, i] = max(float(q), 1e-6) ** alpha
    return w


def reach_pairs(nodes: ContactNodes, bond: float) -> tuple[I32Array, I32Array, F64Array]:
    """Returns the consecutive nodes of each block and the length of chain between them, the
    bonds between the two beads times `bond`.
    """
    same = nodes.block[1:] == nodes.block[:-1]
    first = np.flatnonzero(same).astype(np.int32)
    second: I32Array = first + 1
    length: F64Array = (nodes.bead[second] - nodes.bead[first]).astype(np.float64) * bond
    return first, second, length


def node_targets(nodes: ContactNodes, law: PolymerLaw, s: Settings) -> F64Array:
    """Returns the arcs target matrix of the nodes, with the marks of `arc_expected_matrix`.

    A pair with a loop holds the contact distance of the law for its summed strength and the
    separation of its two beads, or one bead under `loop_contact`. A pair without a loop closer
    than `background_range_bp` holds minus the background of its separation when
    `background_weight` is on. Any other pair holds -0.5. The diagonal is 0.
    """
    m = int(nodes.mid.shape[0])
    mat: F64Array = np.full((m, m), -0.5, dtype=np.float64)
    mid = nodes.mid.astype(np.float64)
    sep = np.abs(mid[:, None] - mid[None, :])
    if float(s.background_weight) > 0.0:
        near = (sep > 0.0) & (sep < float(s.background_range_bp))
        mat[near] = -np.maximum(1.0, (sep[near] / max(int(law.s0_bp), 1)) ** law.nu)
    np.fill_diagonal(mat, 0.0)
    for i, j, q in zip(nodes.pair_i, nodes.pair_j, nodes.strength, strict=True):
        target = 1.0 if s.loop_contact else law.contact_distance(int(sep[i, j]), float(q))
        mat[i, j] = target
        mat[j, i] = target
    return mat
