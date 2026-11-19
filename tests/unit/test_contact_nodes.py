"""Unit checks for the contact beads and the nodes of the uniform chain."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.pipeline.ib.nodes import ContactNodes, contact_nodes, draw_ends, node_targets
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings
from gnome3d.types import I32Array, I64Array

LAW = PolymerLaw(nu=0.5, s0_bp=1000, q_half=1.0)


def i32(values: list[int] | list[list[int]]) -> I32Array:
    return np.array(values, dtype=np.int32)


def edges_of(start: int, beads: int) -> I64Array:
    return start + 1000 * np.arange(beads + 1, dtype=np.int64)


def settings(**kw: object) -> Settings:
    s = Settings()
    s.background_weight = 0.1
    s.background_range_bp = 100_000
    s.loop_contact = False
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_uniform_draw_stays_inside_each_span_and_covers_it() -> None:
    lo, hi, knot = i32([0, 10]), i32([1, 18]), i32([0, 14])
    ends = np.tile(i32([[0, 1]]), (4000, 1))
    beads = draw_ends(lo, hi, knot, ends, np.random.default_rng(0))
    assert (beads[:, 0] == 0).all(), "a span of one bead has one choice"
    assert beads[:, 1].min() == 10 and beads[:, 1].max() == 17, "every bead of the span is drawn"
    counts = np.bincount(beads[:, 1] - 10, minlength=8)
    assert counts.min() > 400 and counts.max() < 600, f"with equal chance, got {counts}"


def test_each_loop_end_draws_by_itself() -> None:
    lo, hi, knot = i32([0, 10]), i32([8, 18]), i32([4, 14])
    ends = np.tile(i32([[0, 1]]), (50, 1))
    beads = draw_ends(lo, hi, knot, ends, np.random.default_rng(1))
    assert len(set(beads[:, 0].tolist())) > 1, "two loops of one anchor can have two contact beads"


def test_midpoint_draw_takes_the_knot() -> None:
    lo, hi, knot = i32([0, 10]), i32([8, 18]), i32([4, 14])
    beads = draw_ends(lo, hi, knot, i32([[0, 1], [1, 0]]), np.random.default_rng(0), "midpoint")
    assert beads.tolist() == [[4, 14], [14, 4]]


def test_draw_refuses_an_unknown_rule() -> None:
    with pytest.raises(ValueError, match="uniform or midpoint"):
        draw_ends(i32([0]), i32([1]), i32([0]), i32([[0, 0]]), np.random.default_rng(0), "centre")


def test_strengths_of_loops_on_one_bead_pair_add() -> None:
    """Anchors 0 and 1 are copies of one site in block 0. Both loop to anchor 2 in block 1."""
    edges = [edges_of(10_000, 5), edges_of(50_000, 4)]
    nodes = contact_nodes(
        edges,
        anchor_block=i32([0, 0, 1]),
        ends=i32([[0, 2], [1, 2], [2, 0]]),
        beads=np.array([[2, 1], [2, 1], [1, 2]], dtype=np.int64),
        strength=np.array([1.5, 0.5, 2.0]),
    )
    assert nodes.block.tolist() == [0, 1] and nodes.bead.tolist() == [2, 1]
    assert nodes.mid.tolist() == [12_500, 51_500], "the midpoints of the two beads"
    assert nodes.pair_i.tolist() == [0] and nodes.pair_j.tolist() == [1]
    assert nodes.strength.tolist() == [4.0], "three loops on one bead pair are one loop of the sum"
    assert nodes.dropped == 0


def test_nodes_are_distinct_beads_in_chromosome_order() -> None:
    edges = [edges_of(10_000, 5), edges_of(50_000, 4)]
    nodes = contact_nodes(
        edges,
        anchor_block=i32([0, 0, 1, 1]),
        ends=i32([[3, 0], [1, 2], [0, 1]]),
        beads=np.array([[3, 0], [4, 0], [0, 4]], dtype=np.int64),
        strength=np.array([1.0, 2.0, 3.0]),
    )
    assert list(zip(nodes.block.tolist(), nodes.bead.tolist(), strict=True)) == [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 3),
    ]
    keys = zip(nodes.pair_i.tolist(), nodes.pair_j.tolist(), strict=True)
    pairs = dict(zip(keys, nodes.strength.tolist(), strict=True))
    assert pairs == {(0, 3): 1.0, (1, 2): 2.0, (0, 1): 3.0}, "the lower node is first"


def test_a_loop_with_both_ends_on_one_bead_is_left_out() -> None:
    nodes = contact_nodes(
        [edges_of(0, 6)],
        anchor_block=i32([0, 0, 0]),
        ends=i32([[0, 1], [0, 2]]),
        beads=np.array([[3, 3], [3, 5]], dtype=np.int64),
        strength=np.array([9.0, 1.0]),
    )
    assert nodes.dropped == 1 and nodes.strength.tolist() == [1.0]
    assert nodes.bead.tolist() == [3, 5]


def test_no_loops_give_no_nodes() -> None:
    nodes = contact_nodes(
        [edges_of(0, 6)],
        anchor_block=i32([0]),
        ends=np.zeros((0, 2), dtype=np.int32),
        beads=np.zeros((0, 2), dtype=np.int64),
        strength=np.zeros(0),
    )
    assert nodes.bead.shape == (0,) and nodes.strength.shape == (0,)
    assert node_targets(nodes, LAW, settings()).shape == (0, 0)


def three_nodes() -> ContactNodes:
    """Beads at 500, 16,500 and 400,500 bp. A loop of strength 3 between the first and the last."""
    return contact_nodes(
        [edges_of(0, 20), edges_of(400_000, 2)],
        anchor_block=i32([0, 0, 1]),
        ends=i32([[0, 2], [1, 2]]),
        beads=np.array([[0, 0], [16, 0]], dtype=np.int64),
        strength=np.array([3.0, 0.0]),
    )


def test_node_targets_in_closed_form() -> None:
    nodes = three_nodes()
    mat = node_targets(nodes, LAW, settings())
    bg = 400.0**0.5
    assert mat[0, 2] == pytest.approx(1.0 + (bg - 1.0) / 4.0), "the law at strength 3, q_half 1"
    assert mat[1, 2] == pytest.approx(384.0**0.5), "a loop of strength 0 is at the background"
    assert mat[0, 1] == pytest.approx(-4.0), "no loop, 16 kb apart: minus the background"
    assert np.array_equal(mat, mat.T) and (np.diag(mat) == 0).all()


def test_node_targets_mark_far_pairs_and_follow_the_settings() -> None:
    nodes = three_nodes()
    far = node_targets(nodes, LAW, settings(background_range_bp=10_000))
    assert far[0, 1] == -0.5, "beyond the range the pair has the repulsion mark"
    off = node_targets(nodes, LAW, settings(background_weight=0.0))
    assert off[0, 1] == -0.5, "with the background spring off as well"
    held = node_targets(nodes, LAW, settings(loop_contact=True))
    assert held[0, 2] == 1.0 and held[1, 2] == 1.0, "a kept loop at one bead under loop_contact"


def reach_pairs() -> tuple[I32Array, I32Array, np.ndarray]:
    return i32([0, 1]), i32([1, 2]), np.array([2.0, 3.0])


def test_reach_energy_in_closed_form() -> None:
    """Nodes on the x axis at 0, 5 and 6. The first pair has 2 bonds of chain, the second 3."""
    from gnome3d.mc.numba.arcs_solver import reach_energy_grad

    x = np.array([[0.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]]).reshape(-1)
    pi, pj, length = reach_pairs()
    energy, grad = reach_energy_grad(x, pi, pj, length, 0.5)
    assert energy == pytest.approx(0.5 * 1.5**2), "the first pair is 1.5 lengths over its reach"
    g = grad.reshape(3, 3)
    assert g[0, 0] == pytest.approx(-0.75) and g[1, 0] == pytest.approx(0.75)
    assert (g[2] == 0).all(), "a pair inside its reach has no energy and no force"


def test_reach_gradient_matches_a_finite_difference() -> None:
    from gnome3d.mc.numba.arcs_solver import reach_energy_grad

    rng = np.random.default_rng(4)
    x = (rng.normal(size=(3, 3)) * 4.0).reshape(-1)
    pi, pj, length = reach_pairs()
    _e, grad = reach_energy_grad(x, pi, pj, length, 1.3)
    for k in range(9):
        step = np.zeros(9)
        step[k] = 1e-6
        up, _ = reach_energy_grad(x + step, pi, pj, length, 1.3)
        down, _ = reach_energy_grad(x - step, pi, pj, length, 1.3)
        assert grad[k] == pytest.approx((up - down) / 2e-6, abs=1e-5)


def test_solver_brings_a_pair_inside_its_reach() -> None:
    """Two nodes with one bond of chain between them and no other term start 6 beads apart."""
    from gnome3d.mc.numba.arcs_solver import solve_arcs

    s = Settings()
    s.use_excluded_volume = False
    s.use_confinement = False
    s.gyration_weight = 0.0
    s.background_weight = 0.0
    s.spring_stretch_arcs = 1.0
    s.arcs_solver_iters = 500
    s.arcs_solver_tol = 1e-12
    pos = np.array([[0.0, 0, 0], [6.0, 0, 0]], dtype=np.float32)
    exp = np.zeros((2, 2))
    reach = (i32([0]), i32([1]), np.array([1.0]))
    _e0, held = solve_arcs(pos, exp, s, reach=reach)
    _e1, free = solve_arcs(pos, exp, s)
    assert np.linalg.norm(held[0] - held[1]) == pytest.approx(1.0, abs=1e-2)
    assert np.linalg.norm(free[0] - free[1]) == pytest.approx(6.0, abs=1e-5), "no term, no move"
