"""The cap of the excluded volume radius of a pair in the arcs solver."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc.numba.arcs_solver import (
    arcs_energy_grad,
    capped_exclusion_energy_grad,
    capped_pairs,
    solve_arcs,
)
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
    energy, grad = capped_exclusion_energy_grad(x, pair_i, pair_j, np.array([2.0]), FULL, WEIGHT)
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
    pair_i, pair_j, radius = capped_pairs(exp, None, settings(exclusion_arcs_cap="target"), FULL, 1)
    assert sorted(zip(pair_i.tolist(), pair_j.tolist(), radius.tolist(), strict=True)) == [
        (0, 5, 1.2),
        (2, 9, 2.0),
    ], "a loop and a background pair under the radius, and not the neighbours 3 and 4"
    x = cloud.reshape(-1)
    base, _ = stage_energy(x, n)
    change, _ = capped_exclusion_energy_grad(x, pair_i, pair_j, radius, FULL, WEIGHT)
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
    _, grad = capped_exclusion_energy_grad(x, pair_i, pair_j, radius, FULL, WEIGHT)
    eps = 1e-6
    for k in (0, 1, 7, 15, 17, 21, 29):
        up, down = x.copy(), x.copy()
        up[k] += eps
        down[k] -= eps
        e_up, _ = capped_exclusion_energy_grad(up, pair_i, pair_j, radius, FULL, WEIGHT)
        e_down, _ = capped_exclusion_energy_grad(down, pair_i, pair_j, radius, FULL, WEIGHT)
        assert grad[k] == pytest.approx((e_up - e_down) / (2 * eps), abs=1e-6), f"coordinate {k}"


def test_chain_limit_is_the_bonds_between_two_beads() -> None:
    mids = np.array([500, 2500, 3500, 60500], dtype=np.int64)
    s = settings(exclusion_arcs_cap="chain")
    bond = float(s.polymer_law().background(1000))
    pair_i, pair_j, radius = capped_pairs(np.zeros((4, 4)), mids, s, FULL, 0)
    got = sorted(zip(pair_i.tolist(), pair_j.tolist(), radius.tolist(), strict=True))
    assert [(i, j) for i, j, _ in got] == [(0, 1), (1, 2)], (
        "the pairs with fewer than three bonds between them"
    )
    assert [r for _, _, r in got] == pytest.approx([2 * bond, bond])
    with pytest.raises(ValueError, match="needs the genomic midpoints"):
        capped_pairs(np.zeros((4, 4)), None, s, FULL, 0)


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
    with pytest.raises(ValueError, match="none, target or chain"):
        Settings.from_dict({"excluded_volume": {"arcs_cap": "law"}})
