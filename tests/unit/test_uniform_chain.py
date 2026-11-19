"""Unit checks for the grid chain and the anchor span term, `[main] uniform_chain`."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc import numba as mc_numba
from gnome3d.mc.numba.smooth import beads_to_anchors
from gnome3d.mc.numba.terms import init_spans_nb, local_spans_nb, shift_spans_nb
from gnome3d.pipeline.ib.densify import densify
from gnome3d.pipeline.ib.spans import (
    SITE_SPACING,
    anchor_spans,
    covered_mask,
    densify_uniform,
    grid_edges,
    merge_nodes,
    motif_sites,
    uniform_heatmap,
)
from gnome3d.settings import Settings
from gnome3d.types import AnchorSpans, F64Array, I32Array

CHARS = {0: "N", 1: "L", 2: "R"}

# Anchors of one block: two that overlap, one nested in a wide one and one alone.
ANCHORS = [
    (10_000, 11_400, 10_700),
    (10_900, 12_100, 11_500),
    (20_000, 32_000, 26_000),
    (24_000, 25_200, 24_600),
    (39_200, 40_000, 39_600),
]


def grid_settings(**kw: object) -> Settings:
    s = Settings()
    s.uniform_chain = True
    s.target_bp_per_subanchor = 1000
    s.use_subanchor_heatmap = False
    s.noise_smooth = 1.0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def runs(lo: list[int], hi: list[int]) -> tuple[I32Array, I32Array]:
    return np.array(lo, dtype=np.int32), np.array(hi, dtype=np.int32)


@pytest.fixture
def held() -> tuple[F64Array, I32Array, I32Array, F64Array]:
    """Six beads on the x axis under two anchors that share bead 2."""
    pos = np.array([[float(i), 0.0, 0.0] for i in range(6)], dtype=np.float64)
    lo, hi = runs([0, 2], [3, 6])
    home = np.array([[1.0, 2.0, 0.0], [3.5, 0.0, -1.0]], dtype=np.float64)
    return pos, lo, hi, home


def test_spans_score_in_closed_form(held: tuple[F64Array, I32Array, I32Array, F64Array]) -> None:
    pos, lo, hi, home = held
    cen = np.zeros_like(home)
    got = init_spans_nb(pos, lo, hi, home, 0.5, cen)
    # Centroids (1, 0, 0) and (3.5, 0, 0), offsets of length 2 and 1.
    assert np.allclose(cen, [[1.0, 0.0, 0.0], [3.5, 0.0, 0.0]]), "centroids are the bead means"
    assert got == pytest.approx(0.5 * (4.0 + 1.0)), "weight times the squared offsets"


def test_spans_score_is_zero_at_home(held: tuple[F64Array, I32Array, I32Array, F64Array]) -> None:
    pos, lo, hi, _home = held
    cen = np.zeros((2, 3))
    at = np.array([pos[0:3].mean(axis=0), pos[2:6].mean(axis=0)])
    assert init_spans_nb(pos, lo, hi, at, 1.0, cen) == pytest.approx(0.0, abs=1e-12)


def test_spans_score_is_non_negative() -> None:
    rng = np.random.default_rng(3)
    lo, hi = runs([0, 4, 5, 20], [6, 5, 30, 21])
    cen = np.zeros((4, 3))
    for _ in range(200):
        pos = rng.normal(size=(30, 3)) * rng.uniform(0.01, 50)
        home = rng.normal(size=(4, 3)) * rng.uniform(0.01, 50)
        assert init_spans_nb(pos, lo, hi, home, 1.0, cen) >= 0.0, "a sum of squares"


def test_local_spans_scores_sum_to_the_full_score() -> None:
    """Each anchor is counted once per bead it holds. A share of one over its bead count per
    bead sums to the full score.
    """
    rng = np.random.default_rng(5)
    n = 30
    lo, hi = runs([0, 4, 5, 20], [6, 5, 30, 21])
    pos = rng.normal(size=(n, 3)) * 3.0
    home = rng.normal(size=(4, 3)) * 3.0
    cen = np.zeros((4, 3))
    full = init_spans_nb(pos, lo, hi, home, 0.7, cen)
    inv_n = 1.0 / (hi - lo).astype(np.float64)
    off, anc = beads_to_anchors(lo, hi, n)
    per_anchor = np.zeros(4)
    for p in range(n):
        for q in range(off[p], off[p + 1]):
            k = int(anc[q])
            one_off, one_anc = np.array([0, 1], dtype=np.int32), np.array([k], dtype=np.int32)
            per_anchor[k] += inv_n[k] * local_spans_nb(
                cen, home, inv_n, one_off, one_anc, 0, 0.0, 0.0, 0.0, 0.7
            )
    assert per_anchor.sum() == pytest.approx(full, rel=1e-12), "the shares sum to the full score"
    local = sum(local_spans_nb(cen, home, inv_n, off, anc, p, 0.0, 0.0, 0.0, 0.7) for p in range(n))
    counted = sum((hi[k] - lo[k]) * 0.7 * float(((cen[k] - home[k]) ** 2).sum()) for k in range(4))
    assert local == pytest.approx(counted, rel=1e-12), "a bead sums the anchors that hold it"


def test_local_delta_matches_the_full_score_after_a_move() -> None:
    rng = np.random.default_rng(9)
    n = 30
    lo, hi = runs([0, 4, 5, 20], [6, 5, 30, 21])
    pos = rng.normal(size=(n, 3)) * 3.0
    home = rng.normal(size=(4, 3)) * 3.0
    cen = np.zeros((4, 3))
    score = init_spans_nb(pos, lo, hi, home, 0.7, cen)
    inv_n = 1.0 / (hi - lo).astype(np.float64)
    off, anc = beads_to_anchors(lo, hi, n)
    for _ in range(300):
        p = int(rng.integers(0, n))
        dx, dy, dz = (float(v) for v in rng.uniform(-1, 1, 3))
        prev = local_spans_nb(cen, home, inv_n, off, anc, p, 0.0, 0.0, 0.0, 0.7)
        curr = local_spans_nb(cen, home, inv_n, off, anc, p, dx, dy, dz, 0.7)
        pos[p] += (dx, dy, dz)
        shift_spans_nb(cen, inv_n, off, anc, p, dx, dy, dz)
        score += curr - prev
    fresh = np.zeros((4, 3))
    assert score == pytest.approx(init_spans_nb(pos, lo, hi, home, 0.7, fresh), rel=1e-9), (
        "the running score is the full score"
    )
    assert np.allclose(cen, fresh), "the running centroids are the bead means"


def test_beads_to_anchors_lists_every_holder() -> None:
    lo, hi = runs([0, 2, 2], [3, 6, 3])
    off, anc = beads_to_anchors(lo, hi, 7)
    held_by = [sorted(int(k) for k in anc[off[p] : off[p + 1]]) for p in range(7)]
    assert held_by == [[0], [0], [0, 1, 2], [1], [1], [1], []]


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
    assert ((lo <= knot) & (knot < hi)).all(), "the run holds the knot"


def test_densify_uniform_fixes_no_bead_and_starts_knots_at_home() -> None:
    s = grid_settings()
    home = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [5.0, 3, 0], [5.0, 1, 0], [9.0, 0, 2]], dtype=np.float32
    )
    pos, fixed, starts, ends, dtn, step, comp, spans = densify_uniform(home, ANCHORS, s)
    assert pos.shape == (30, 3) and not fixed.any()
    assert starts[1:] == ends[:-1], "consecutive beads share an edge"
    assert np.allclose(dtn, s.genomic_length_to_distance(1000)), "every bond is one bead"
    assert step == pytest.approx(float(dtn.mean())) and comp is None
    assert np.allclose(pos[spans.knot], home, atol=1e-6), "a knot starts at its anchor"
    assert np.isfinite(pos).all()
    assert covered_mask(spans, 30).sum() == 15


def test_densify_returns_the_grid_under_the_flag() -> None:
    home = np.zeros((5, 3), dtype=np.float32)
    home[:, 0] = np.arange(5)
    off = densify(home, ANCHORS, grid_settings(uniform_chain=False, use_dynamic_loop_density=True))
    on = densify(home, ANCHORS, grid_settings())
    assert off[8] is None and off[1].sum() == 5, "off, the anchors are fixed beads"
    assert on[8] is not None and len(on[2]) == 30 and not on[1].any()


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


def test_merge_nodes_in_closed_form() -> None:
    """Anchors 0 and 2 are one node, anchors 1 and 3 are nodes of their own."""
    node = np.array([0, 1, 0, 2], dtype=np.int64)
    exp = np.array(
        [
            [0.0, -1.5, -1.0, 2.0],
            [-1.5, 0.0, -1.2, -0.5],
            [-1.0, -1.2, 0.0, 1.4],
            [2.0, -0.5, 1.4, 0.0],
        ]
    )
    w = np.ones((4, 4), dtype=np.float32)
    w[2, 3] = w[3, 2] = 3.0
    merged, weights = merge_nodes(exp, w, node)
    want = np.array([[0.0, -1.2, 1.4], [-1.2, 0.0, -0.5], [1.4, -0.5, 0.0]])
    assert np.allclose(merged, want), "shortest loop, else shortest background, else the mark"
    assert weights is not None and weights[0, 2] == 3.0 and weights[0, 1] == 0.0
    assert merge_nodes(exp, None, node)[1] is None


def test_merge_nodes_with_one_anchor_per_node_changes_nothing() -> None:
    rng = np.random.default_rng(0)
    exp = rng.choice([0.0, -0.5, -1.3, 1.7], size=(6, 6))
    exp = np.triu(exp, 1) + np.triu(exp, 1).T
    merged, _ = merge_nodes(exp, None, np.arange(6, dtype=np.int64))
    assert np.array_equal(merged, exp)


def spans_at(knots: list[int]) -> AnchorSpans:
    k = np.array(knots, dtype=np.int32)
    return AnchorSpans(lo=k, hi=(k + 1).astype(np.int32), knot=k, home=np.zeros((len(knots), 3)))


def test_motif_sites_merge_anchors_inside_the_spacing() -> None:
    """Anchors 0 and 1 are one bead apart. Anchor 1 has more loop weight and keeps the site."""
    spans = spans_at([10, 11, 20, 30])
    orn = np.array([1, 2, 1, 2], dtype=np.int8)
    nbrs = {0: [2], 1: [2, 3], 2: [0, 1], 3: [1]}
    w = {0: [1.0], 1: [2.0, 3.0], 2: [1.0, 2.0], 3: [3.0]}
    beads, chars, site_n, site_w = motif_sites(spans, orn, nbrs, w, 40, CHARS)
    assert list(np.flatnonzero(beads)) == [11, 20, 30]
    assert chars[11] == "R" and chars[20] == "L" and chars[30] == "R"
    assert site_n == {0: [1, 2], 1: [0], 2: [0]}
    assert site_w == {0: [3.0, 3.0], 1: [3.0], 2: [3.0]}, "the loops of a joined anchor are added"
    for i, js in site_n.items():
        for j, wij in zip(js, site_w[i], strict=True):
            assert site_w[j][site_n[j].index(i)] == wij, "the weights are symmetric"


def test_motif_sites_keep_their_spacing_and_the_chain_ends() -> None:
    spans = spans_at([0, 2, 5, 39])
    orn = np.array([2, 2, 2, 1], dtype=np.int8)
    nbrs = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
    w = {0: [4.0], 1: [1.0], 2: [2.0], 3: [4.0, 1.0, 2.0]}
    beads, _chars, site_n, _site_w = motif_sites(spans, orn, nbrs, w, 40, CHARS)
    at = np.flatnonzero(beads)
    assert at[0] >= 1 and at[-1] <= 38, "a site reads the bead before it and the bead after it"
    assert (np.diff(at) >= SITE_SPACING).all()
    assert all(i not in js for i, js in site_n.items()), "no site is its own neighbour"


def test_motif_sites_skip_anchors_without_a_loop() -> None:
    spans = spans_at([5, 15])
    beads, chars, site_n, site_w = motif_sites(
        spans, np.array([1, 2], dtype=np.int8), {0: [], 1: []}, {0: [], 1: []}, 20, CHARS
    )
    assert not beads.any() and (chars == "N").all() and site_n == {} and site_w == {}


def smooth_settings() -> Settings:
    s = grid_settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 0.1
    s.exclusion_radius_smooth = 0.7
    s.use_confinement = False
    s.mc_smooth_chains = 1
    s.max_temp_smooth = 5.0
    s.dt_temp_smooth = 0.9999
    s.jump_scale_smooth = 50.0
    s.jump_coef_smooth = 20.0
    s.mc_stop_steps_smooth = 5000
    s.mc_stop_improvement_smooth = 0.999
    s.mc_stop_successes_smooth = 50
    s.spring_stretch = 0.1
    s.spring_squeeze = 0.1
    s.spring_angular = 0.1
    s.smooth_dist_weight = 1.0
    s.smooth_angle_weight = 1.0
    s.smooth_hard_wall = True
    return s


def test_smooth_brings_the_centroids_home() -> None:
    """A chain of 60 beads with three anchors starts with every centroid two beads off."""
    rng = np.random.default_rng(1)
    n = 60
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.6, axis=0).astype(np.float32)
    dtn = np.ones(n - 1, dtype=np.float32)
    lo, hi = runs([5, 28, 30], [8, 40, 31])
    knot = ((lo + hi) // 2).astype(np.int32)
    shift = np.array([2.0, 0.0, 0.0])
    home = np.array([pos[a:b].mean(axis=0) + shift for a, b in zip(lo, hi, strict=True)])
    spans = AnchorSpans(lo=lo, hi=hi, knot=knot, home=home.astype(np.float64))
    before = np.linalg.norm(
        home - [pos[a:b].mean(axis=0) for a, b in zip(lo, hi, strict=True)], axis=1
    )
    mc_numba.seed_numba(11)
    fixed = np.zeros(n, dtype=np.bool_)
    score = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, smooth_settings(), spans=spans)
    after = np.linalg.norm(
        home - [pos[a:b].mean(axis=0) for a, b in zip(lo, hi, strict=True)], axis=1
    )
    assert score >= 0.0 and np.isfinite(pos).all()
    assert before.min() == pytest.approx(2.0, abs=1e-5)
    assert after.max() < 0.5, f"centroids within half a bead of home, got {after.round(2)}"
    bonds = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1)
    assert np.median(bonds) < 1.5, "the chain keeps its bonds"


def test_smooth_without_spans_ignores_the_weight() -> None:
    """The span term is off without spans. Two weights give one result."""
    rng = np.random.default_rng(2)
    n = 40
    start = np.cumsum(rng.normal(size=(n, 3)) * 0.6, axis=0).astype(np.float32)
    dtn = np.ones(n - 1, dtype=np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[0, 20, 39]] = True
    out = []
    for weight in (1.0, 50.0):
        s = smooth_settings()
        s.anchor_span_weight = weight
        pos = start.copy()
        mc_numba.seed_numba(4)
        mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s)
        out.append(pos)
    assert np.array_equal(out[0], out[1])
