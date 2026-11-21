"""The excluded volume radius of a pair in the arcs solver, its cap and its rule from the chain."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc.numba.arcs_solver import (
    arcs_energy_grad,
    pair_radii,
    pair_radius_energy_grad,
    solve_arcs,
)
from gnome3d.pipeline.ib.nodes import chain_share, mean_loop_target
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings
from gnome3d.types import F64Array

FULL = 3.0
WEIGHT = 0.1


def settings(**kw: object) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.exclusion_radius_arcs = FULL
    s.exclusion_weight = WEIGHT
    s.exclusion_skip_neighbors = 1
    s.use_confinement = False
    s.gyration_weight = 0.0
    s.arcs_repulsion_cutoff_factor = 0.0
    s.target_bp_per_subanchor = 1000
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def stage_energy(x: F64Array, n: int) -> tuple[float, F64Array]:
    """The energy of `arcs_energy_grad` with no spring, the excluded volume at the radius of the stage."""
    exp = np.zeros((n, n))
    return arcs_energy_grad(x, exp, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, FULL, WEIGHT, 1)


def by_hand(pos: F64Array, radius: F64Array) -> float:
    """Twice `WEIGHT * ((r - d) / r)^2` over the pairs two or more apart in order and closer than r."""
    total = 0.0
    for i in range(len(pos)):
        for j in range(i + 2, len(pos)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            r = float(radius[i, j])
            if d < r:
                total += 2.0 * WEIGHT * ((r - d) / r) ** 2
    return total


@pytest.fixture
def cloud() -> F64Array:
    return np.random.default_rng(3).normal(scale=1.5, size=(12, 3))


def test_change_of_one_pair_in_closed_form() -> None:
    x = np.array([0.0, 0, 0, 9.0, 0, 0, 1.0, 0, 0])
    pair_i, pair_j = np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)
    energy, grad = pair_radius_energy_grad(x, pair_i, pair_j, np.array([2.0]), FULL, WEIGHT)
    want = 2.0 * WEIGHT * (0.5**2 - (2.0 / 3.0) ** 2)
    assert energy == pytest.approx(want), (
        "at d of 1: the energy at radius 2 minus the energy at radius 3"
    )
    assert grad[3:6] == pytest.approx(0.0), "a bead outside the pair has no gradient"
    assert grad[0] == pytest.approx(-grad[6]), "the two beads of the pair have opposite gradients"


def test_stage_energy_plus_the_change_is_the_energy_at_the_radius_of_each_pair(
    cloud: F64Array,
) -> None:
    n = len(cloud)
    exp = np.zeros((n, n))
    exp[0, 5] = exp[5, 0] = 1.2
    exp[2, 9] = exp[9, 2] = -2.0
    exp[3, 4] = exp[4, 3] = 1.5
    pair_i, pair_j, radius = pair_radii(exp, None, settings(exclusion_arcs_cap="target"), FULL, 1)
    assert sorted(zip(pair_i.tolist(), pair_j.tolist(), radius.tolist(), strict=True)) == [
        (0, 5, 1.2),
        (2, 9, 2.0),
    ], "a loop and a background pair under the radius, and not the neighbours 3 and 4"
    x = cloud.reshape(-1)
    base, _ = stage_energy(x, n)
    change, _ = pair_radius_energy_grad(x, pair_i, pair_j, radius, FULL, WEIGHT)
    per_pair = np.full((n, n), FULL)
    per_pair[0, 5] = 1.2
    per_pair[2, 9] = 2.0
    assert base == pytest.approx(by_hand(cloud, np.full((n, n), FULL))), (
        "the stage scores every pair at its radius"
    )
    assert base + change == pytest.approx(by_hand(cloud, per_pair)), (
        "with the change, each pair at its own"
    )
    assert base + change >= 0.0, "the excluded volume is not negative"


def test_gradient_of_the_change_matches_a_finite_difference(cloud: F64Array) -> None:
    pair_i = np.array([0, 2, 1], dtype=np.int32)
    pair_j = np.array([5, 9, 7], dtype=np.int32)
    radius = np.array([1.2, 2.0, 2.5])
    x = cloud.reshape(-1).copy()
    _, grad = pair_radius_energy_grad(x, pair_i, pair_j, radius, FULL, WEIGHT)
    eps = 1e-6
    for k in (0, 1, 7, 15, 17, 21, 29):
        up, down = x.copy(), x.copy()
        up[k] += eps
        down[k] -= eps
        e_up, _ = pair_radius_energy_grad(up, pair_i, pair_j, radius, FULL, WEIGHT)
        e_down, _ = pair_radius_energy_grad(down, pair_i, pair_j, radius, FULL, WEIGHT)
        assert grad[k] == pytest.approx((e_up - e_down) / (2 * eps), abs=1e-6), f"coordinate {k}"


def test_solver_lets_a_capped_pair_stay_at_its_target() -> None:
    """Three beads in a row. The outer two have a loop at 1.5 and the radius of the stage is 3."""
    exp = np.zeros((3, 3))
    exp[0, 2] = exp[2, 0] = 1.5
    start = np.array([[0.0, 0, 0], [0.7, 4.0, 0], [1.5, 0, 0]], dtype=np.float32)
    apart = {}
    for cap in ("none", "target"):
        s = settings(exclusion_arcs_cap=cap, exclusion_weight=1.0)
        s.spring_stretch_arcs = s.spring_squeeze_arcs = 0.1
        _, pos = solve_arcs(start, exp, s, iters=200)
        apart[cap] = float(np.linalg.norm(pos[0] - pos[2]))
    assert apart["none"] > 1.9, (
        f"at the radius of the stage the pair is pushed out, got {apart['none']:.2f}"
    )
    assert apart["target"] == pytest.approx(1.5, abs=0.02), "with the cap the pair is at its target"


def test_loader_refuses_an_unknown_cap() -> None:
    with pytest.raises(ValueError, match="none or target"):
        Settings.from_dict({"excluded_volume": {"arcs_cap": "law"}})


def test_change_of_one_pair_with_a_radius_above_the_stage() -> None:
    pair_i, pair_j = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    for d, want in (
        (4.0, 2.0 * WEIGHT * (2.0 / 6.0) ** 2),
        (1.5, 2.0 * WEIGHT * ((4.5 / 6.0) ** 2 - 0.5**2)),
        (7.0, 0.0),
    ):
        x = np.array([0.0, 0, 0, d, 0, 0])
        energy, grad = pair_radius_energy_grad(x, pair_i, pair_j, np.array([6.0]), FULL, WEIGHT)
        assert energy == pytest.approx(want), f"at d of {d} with a radius of 6 and a stage of 3"
        assert grad[0] == pytest.approx(-grad[3]), "the two beads have opposite gradients"
        if d >= FULL:
            assert grad[3] <= 0.0, "beyond the radius of the stage the energy falls with d"


def test_chain_share_is_the_chain_between_the_midpoints_to_the_neighbours() -> None:
    mid = np.array([1000, 3000, 4000, 9000], dtype=np.int64)
    share = chain_share(mid, 0, 10_000, 4)
    assert share == pytest.approx(np.array([2000, 1500, 3000, 3500]) / 2500), (
        "the chain of each node over the mean chain of four nodes on 10 kb"
    )
    assert chain_share(mid, 0, 10_000, 8).sum() == pytest.approx(8.0), (
        "the shares sum to the count of nodes of a conformation with every loop"
    )


@pytest.mark.parametrize("rule", ["product", "sum"])
def test_stage_energy_plus_the_change_is_the_energy_at_the_radius_from_the_chain(
    cloud: F64Array, rule: str
) -> None:
    n = len(cloud)
    share = np.random.default_rng(5).uniform(0.2, 4.0, size=n)
    f = np.cbrt(share)
    want = FULL * (
        f[:, None] * f[None, :] if rule == "product" else 0.5 * (f[:, None] + f[None, :])
    )
    exp = np.zeros((n, n))
    exp[0, 5] = exp[5, 0] = 1.2
    for cap in ("none", "target"):
        radius = want.copy()
        if cap == "target":
            radius[0, 5] = radius[5, 0] = min(radius[0, 5], 1.2)
        s = settings(exclusion_arcs_chain_radius=rule, exclusion_arcs_cap=cap)
        pair_i, pair_j, own = pair_radii(exp, share, s, FULL, 1)
        assert own == pytest.approx(radius[pair_i, pair_j]), f"the radius of each pair under {cap}"
        x = cloud.reshape(-1).copy()
        base, _ = stage_energy(x, n)
        change, _ = pair_radius_energy_grad(x, pair_i, pair_j, own, FULL, WEIGHT)
        assert base + change == pytest.approx(by_hand(cloud, radius)), (
            f"the excluded volume at the radius of each pair under {rule} and cap {cap}"
        )
        assert base + change >= 0.0, "the excluded volume is not negative"


def test_gradient_with_radii_on_both_sides_of_the_stage_matches_a_finite_difference(
    cloud: F64Array,
) -> None:
    pair_i = np.array([0, 2, 1, 3], dtype=np.int32)
    pair_j = np.array([5, 9, 7, 11], dtype=np.int32)
    radius = np.array([1.2, 5.0, 7.5, 3.4])
    x = cloud.reshape(-1).copy()
    _, grad = pair_radius_energy_grad(x, pair_i, pair_j, radius, FULL, WEIGHT)
    eps = 1e-6
    for k in (0, 3, 7, 10, 15, 21, 27, 33):
        up, down = x.copy(), x.copy()
        up[k] += eps
        down[k] -= eps
        e_up, _ = pair_radius_energy_grad(up, pair_i, pair_j, radius, FULL, WEIGHT)
        e_down, _ = pair_radius_energy_grad(down, pair_i, pair_j, radius, FULL, WEIGHT)
        assert grad[k] == pytest.approx((e_up - e_down) / (2 * eps), abs=1e-6), f"coordinate {k}"


def test_solver_pushes_a_pair_out_to_the_radius_from_its_chain() -> None:
    """Two nodes with a third between them in order. No spring acts on the outer pair."""
    exp = np.zeros((3, 3))
    start = np.array([[0.0, 0, 0], [0.5, 9.0, 0], [3.5, 0, 0]], dtype=np.float32)
    apart = {}
    for rule, share in (("none", None), ("product", np.array([8.0, 1.0, 8.0]))):
        s = settings(exclusion_arcs_chain_radius=rule, exclusion_weight=1.0)
        _, pos = solve_arcs(start, exp, s, iters=200, share=share)
        apart[rule] = float(np.linalg.norm(pos[0] - pos[2]))
    assert apart["none"] == pytest.approx(3.5, abs=0.01), (
        "beyond the radius of the stage, not moved"
    )
    assert apart["product"] > 11.9, (
        f"a pair of nodes with eight times the chain is out at four radii, got {apart['product']:.2f}"
    )


def test_radius_from_the_chain_needs_the_chain() -> None:
    s = settings(exclusion_arcs_chain_radius="product")
    with pytest.raises(ValueError, match="needs the chain of each node"):
        pair_radii(np.zeros((4, 4)), None, s, FULL, 1)


def test_loader_refuses_an_unknown_rule_and_a_rule_without_the_uniform_chain() -> None:
    with pytest.raises(ValueError, match="none, product or sum"):
        Settings.from_dict({"excluded_volume": {"arcs_chain_radius": "cube"}})
    with pytest.raises(ValueError, match="uniform_chain is off"):
        Settings.from_dict({"excluded_volume": {"arcs_chain_radius": "sum"}})


def test_mean_loop_target_is_the_mean_of_the_contact_distances_of_every_loop() -> None:
    law = PolymerLaw(nu=0.5, s0_bp=1000, q_half=1.0)
    mids = [0, 16_000, 64_000]
    arcs = [(0, 1, 1), (0, 2, 3), (1, 2, 1, 0)]
    want = np.mean(
        [
            law.contact_distance(16_000, law.arc_strength(1, 16_000, 0)),
            law.contact_distance(64_000, law.arc_strength(3, 64_000, 0)),
            law.contact_distance(48_000, law.arc_strength(1, 48_000, 0)),
        ]
    )
    s = settings()
    assert mean_loop_target(mids, arcs, law, s) == pytest.approx(want)
    assert mean_loop_target(mids, [], law, s) == 1.0, "a chromosome without a loop"
