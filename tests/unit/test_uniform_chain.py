"""Unit checks for the uniform chain of a block, `[main] uniform_chain`."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.pipeline.ib.densify import densify
from gnome3d.pipeline.ib.spans import (
    anchor_spans,
    covered_mask,
    densify_uniform,
    grid_edges,
    start_chain,
    uniform_heatmap,
)
from gnome3d.pipeline.relax import relax_blocks
from gnome3d.settings import Settings
from gnome3d.types import BeadOut

# Anchors of one block: two that overlap, one nested in a wide one and one alone.
ANCHORS = [
    (10_000, 11_400, 10_700),
    (10_900, 12_100, 11_500),
    (20_000, 32_000, 26_000),
    (24_000, 25_200, 24_600),
    (39_200, 40_000, 39_600),
]


def chain_settings(**kw: object) -> Settings:
    s = Settings()
    s.uniform_chain = True
    s.target_bp_per_subanchor = 1000
    s.noise_smooth = 1.0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_grid_covers_the_block_once() -> None:
    edges = grid_edges(ANCHORS, 1000)
    assert edges[0] == 10_000 and edges[-1] == 40_000, "from the first start to the last end"
    assert edges.shape[0] - 1 == 30, "one bead per kilobase"
    assert (np.diff(edges) == 1000).all(), "beads of equal width with no gap and no overlap"


def test_grid_rounds_a_span_that_is_not_a_multiple() -> None:
    edges = grid_edges([(0, 700, 350), (2000, 2600, 2300)], 1000)
    widths = np.diff(edges)
    assert edges[0] == 0 and edges[-1] == 2600 and widths.shape[0] == 3
    assert widths.min() >= 866 and widths.max() <= 867


def test_spans_follow_the_width() -> None:
    edges = grid_edges(ANCHORS, 1000)
    lo, hi, knot = anchor_spans(ANCHORS, edges)
    assert list(knot) == [0, 1, 16, 14, 29], "the bead at the midpoint of each anchor"
    assert list(hi - lo) == [1, 1, 12, 1, 1], "about one bead per kilobase of anchor"
    assert lo[2] <= lo[3] and hi[3] <= hi[2], "a nested anchor holds beads of the wide one"
    assert ((lo <= knot) & (knot < hi)).all(), "the span holds the knot"
    assert covered_mask(lo, hi, 30).sum() == 15


def test_start_chain_puts_nodes_at_their_positions() -> None:
    node_bead = np.array([2, 5, 6], dtype=np.int32)
    node_pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [3.0, 2, 0]])
    pos = start_chain(node_bead, node_pos, 9, 1.0, np.zeros(3))
    assert np.allclose(pos[node_bead], node_pos)
    assert np.allclose(pos[3], [1.0, 0, 0]) and np.allclose(pos[4], [2.0, 0, 0]), "on the line"
    tail = np.linalg.norm(np.diff(pos[[0, 1, 2]], axis=0), axis=1)
    head = np.linalg.norm(np.diff(pos[[6, 7, 8]], axis=0), axis=1)
    assert np.allclose(tail, 1.0) and np.allclose(head, 1.0), "one bond per bead beyond the nodes"


def test_start_chain_without_a_node_is_a_line_through_the_centre() -> None:
    centre = np.array([4.0, 5.0, 6.0])
    pos = start_chain(np.zeros(0, dtype=np.int32), np.zeros((0, 3)), 5, 1.0, centre)
    assert np.allclose(pos.mean(axis=0), centre)
    assert np.allclose(np.linalg.norm(np.diff(pos, axis=0), axis=1), 1.0)


def test_densify_uniform_fixes_the_nodes() -> None:
    s = chain_settings()
    node_bead = np.array([0, 14, 29], dtype=np.int32)
    node_pos = np.array([[0.0, 0, 0], [5.0, 3, 0], [9.0, 0, 2]], dtype=np.float32)
    centre = np.zeros(3, dtype=np.float32)
    pos, fixed, starts, ends, dtn, anchor_map, step, comp, covered = densify_uniform(
        node_bead, node_pos, centre, ANCHORS, s
    )
    assert pos.shape == (30, 3) and list(np.flatnonzero(fixed)) == [0, 14, 29]
    assert anchor_map == [(0, 0), (14, 1), (29, 2)], "the bead of each node and its number"
    assert starts[1:] == ends[:-1], "consecutive beads share an edge"
    assert np.allclose(dtn, s.genomic_length_to_distance(1000)), "every bond is one bead"
    assert step == pytest.approx(float(dtn.mean())) and comp is None
    assert np.allclose(pos[node_bead], node_pos, atol=1e-6), "a node is at its arcs position"
    assert covered.sum() == 15, "the beads under an anchor"


def test_densify_returns_the_uniform_chain_under_the_flag() -> None:
    home = np.zeros((5, 3), dtype=np.float32)
    home[:, 0] = np.arange(5)
    off = densify(home, ANCHORS, chain_settings(uniform_chain=False, use_dynamic_loop_density=True))
    assert off[8] is None and off[1].sum() == 5, "off, the anchors are the fixed beads"
    node_bead = np.array([1, 16], dtype=np.int32)
    node_pos = np.array([[0.0, 0, 0], [4.0, 0, 0]], dtype=np.float32)
    on = densify(home, ANCHORS, chain_settings(), None, node_bead, node_pos)
    assert on[8] is not None and len(on[2]) == 30 and on[1].sum() == 2


def test_densify_needs_the_nodes_of_the_joint_solve() -> None:
    with pytest.raises(ValueError, match="scope to chromosome"):
        densify(np.zeros((5, 3), dtype=np.float32), ANCHORS, chain_settings())


def test_uniform_heatmap_in_closed_form() -> None:
    """Three beads of 1 kb. Two contacts between beads 0 and 2, one between 1 and 2, one inside
    bead 0 and one on another chromosome.
    """
    edges = np.array([0, 1000, 2000, 3000], dtype=np.int64)
    singletons = [
        ("chr1", 100, "chr1", 2500, 2),
        ("chr1", 1500, "chr1", 2100, 1),
        ("chr1", 100, "chr1", 900, 7),
        ("chr1", 100, "chr2", 2500, 5),
    ]
    heat = uniform_heatmap(singletons, "chr1", edges)
    raw = np.array([[0.0, 0, 2], [0, 0, 1], [2, 1, 0]])
    assert heat.dtype == np.float32
    assert np.allclose(heat, raw / raw.mean()), "counts over their mean, bins of 1 kb squared"
    assert np.allclose(heat, heat.T) and (np.diag(heat) == 0).all()


def test_uniform_heatmap_divides_by_the_bin_areas() -> None:
    edges = np.array([0, 2000, 4000], dtype=np.int64)
    heat = uniform_heatmap([("chr1", 10, "chr1", 3000, 3)], "chr1", edges)
    assert heat[0, 1] == pytest.approx(3.0 / 1.5 / 4.0), "mean count 1.5 and 2 kb by 2 kb"


def test_uniform_heatmap_without_contacts_is_zero() -> None:
    edges = np.array([0, 1000, 2000], dtype=np.int64)
    assert not uniform_heatmap([], "chr1", edges).any()


def test_relaxation_holds_the_given_beads_not_the_kind() -> None:
    """Two blocks cross. Every bead has the kind anchor and only the given beads are held."""
    s = Settings()
    s.use_cross_block_relax = True
    s.relax_local_window = -1
    a = [BeadOut(i * 1000, (i + 1) * 1000, float(i), 0.0, 0.0, "anchor") for i in range(8)]
    b = [
        BeadOut(20_000 + i * 1000, 21_000 + i * 1000, 3.5, float(i) - 3.5, 0.05, "anchor")
        for i in range(8)
    ]
    held = [np.zeros(8, dtype=np.bool_), np.zeros(8, dtype=np.bool_)]
    held[0][[0, 7]] = True
    held[1][[0, 7]] = True
    out = relax_blocks([a, b], s, held)
    before = np.array([(x.x, x.y, x.z) for x in a + b])
    after = np.array([(x.x, x.y, x.z) for blk in out for x in blk])
    moved = np.linalg.norm(after - before, axis=1) > 1e-9
    assert not moved[[0, 7, 8, 15]].any(), "the held beads keep their place"
    assert moved.any(), "a bead of kind anchor that is not held moves"
