"""The excluded volume radius of a pair of nodes from the chain the nodes stand for."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc.numba.arcs_solver import arcs_energy_grad, solve_arcs
from gnome3d.mc.numba.terms import NO_W
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


ARGS = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, FULL, WEIGHT, 1)


def energy(x: F64Array, n: int, node_f: F64Array | None = None) -> tuple[float, F64Array]:
    """The energy of `arcs_energy_grad` with no spring: the excluded volume alone."""
    exp = np.zeros((n, n))
    if node_f is None:
        return arcs_energy_grad(x, exp, *ARGS)
    return arcs_energy_grad(x, exp, *ARGS, False, NO_W, True, node_f)


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


def test_one_pair_in_closed_form() -> None:
    """Three beads in a row. The outer two are 4 apart and stand for eight times the chain."""
    x = np.array([0.0, 0, 0, 2.0, 30.0, 0, 4.0, 0, 0])
    f = np.array([2.0, 1.0, 2.0])
    plain, _ = energy(x, 3)
    assert plain == 0.0, "4 apart is beyond the radius of the stage, 3"
    got, grad = energy(x, 3, f)
    want = 2.0 * WEIGHT * ((6.0 - 4.0) / 6.0) ** 2
    assert got == pytest.approx(want), "at a radius of 3 times (2 + 2) / 2, a third of it inside"
    assert grad[0] == pytest.approx(-grad[6]) and grad[6] < 0.0, "the pair is pushed apart"
    assert grad[3:6] == pytest.approx(0.0), "the bead between them has no gradient"


def test_chain_share_is_the_chain_between_the_midpoints_to_the_neighbours() -> None:
    mid = np.array([1000, 3000, 4000, 9000], dtype=np.int64)
    share = chain_share(mid, 0, 10_000, 4)
    assert share == pytest.approx(np.array([2000, 1500, 3000, 3500]) / 2500), (
        "the chain of each node over the mean chain of four nodes on 10 kb"
    )
    assert chain_share(mid, 0, 10_000, 8).sum() == pytest.approx(8.0), (
        "the shares sum to the count of nodes of a conformation with every loop"
    )


def test_energy_is_the_excluded_volume_at_the_radius_of_each_pair(cloud: F64Array) -> None:
    n = len(cloud)
    f = np.cbrt(np.random.default_rng(5).uniform(0.2, 4.0, size=n))
    radius = FULL * 0.5 * (f[:, None] + f[None, :])
    x = cloud.reshape(-1).copy()
    plain, _ = energy(x, n)
    assert plain == pytest.approx(by_hand(cloud, np.full((n, n), FULL))), (
        "without the factors every pair is at the radius of the stage"
    )
    got, _ = energy(x, n, f)
    assert got == pytest.approx(by_hand(cloud, radius)), "each pair at the radius from its chain"
    assert got >= 0.0, "the excluded volume is not negative"
    ones, _ = energy(x, n, np.ones(n))
    assert ones == pytest.approx(plain), "factors of one are the radius of the stage"


def test_gradient_matches_a_finite_difference(cloud: F64Array) -> None:
    n = len(cloud)
    f = np.cbrt(np.random.default_rng(6).uniform(0.2, 4.0, size=n))
    x = cloud.reshape(-1).copy()
    _, grad = energy(x, n, f)
    eps = 1e-6
    for k in (0, 3, 7, 10, 15, 21, 27, 33):
        up, down = x.copy(), x.copy()
        up[k] += eps
        down[k] -= eps
        assert grad[k] == pytest.approx(
            (energy(up, n, f)[0] - energy(down, n, f)[0]) / (2 * eps), abs=1e-6
        ), f"coordinate {k}"


def test_the_device_energy_is_the_cpu_energy(cloud: F64Array) -> None:
    pytest.importorskip("jax")
    from gnome3d.mc.jax.arcs_energy import DeviceArcsEnergy  # noqa: PLC0415

    n = len(cloud)
    f = np.cbrt(np.random.default_rng(7).uniform(0.2, 4.0, size=n))
    x = cloud.reshape(-1).copy()
    exp = np.zeros((n, n))
    want, grad = energy(x, n, f)
    got, got_grad = DeviceArcsEnergy(exp, ARGS, None, f)(x)
    assert got == pytest.approx(want, rel=1e-4), "the energy on the device, in float32"
    assert got_grad == pytest.approx(grad, abs=2e-4), "and its gradient"
    plain, _ = DeviceArcsEnergy(exp, ARGS)(x)
    assert plain == pytest.approx(energy(x, n)[0], rel=1e-4), "without the factors"


def test_solver_pushes_a_pair_out_to_the_radius_from_its_chain() -> None:
    """Two nodes with a third between them in order. No spring acts on the outer pair."""
    exp = np.zeros((3, 3))
    start = np.array([[0.0, 0, 0], [0.5, 9.0, 0], [3.5, 0, 0]], dtype=np.float32)
    apart = {}
    for rule, share in (("none", None), ("sum", np.array([64.0, 1.0, 64.0]))):
        s = settings(exclusion_arcs_chain_radius=rule, exclusion_weight=1.0)
        _, pos = solve_arcs(start, exp, s, iters=200, share=share)
        apart[rule] = float(np.linalg.norm(pos[0] - pos[2]))
    assert apart["none"] == pytest.approx(3.5, abs=0.01), (
        "beyond the radius of the stage, not moved"
    )
    assert apart["sum"] > 11.9, (
        f"a pair of nodes with 64 times the chain is out at four radii, got {apart['sum']:.2f}"
    )


def test_radius_from_the_chain_needs_the_chain() -> None:
    s = settings(exclusion_arcs_chain_radius="sum", exclusion_weight=1.0)
    start = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="needs the chain of each node"):
        solve_arcs(start, np.zeros((3, 3)), s, iters=1)


def test_loader_refuses_an_unknown_rule_and_a_rule_without_the_uniform_chain() -> None:
    with pytest.raises(ValueError, match="none or sum"):
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
