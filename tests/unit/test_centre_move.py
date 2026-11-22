"""Unit checks for the move of a node with its two neighbours that keeps their centre of mass,
in the numba kernel and in the JAX kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc import numba as mc_numba
from gnome3d.mc.numba.terms import init_confine_nb, init_excl_nb, init_smooth_nb
from gnome3d.pipeline.ib.smooth import centre_beads
from gnome3d.settings import Settings


def settings(wall: bool = True) -> Settings:
    s = Settings()
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
    s.smooth_hard_wall = wall
    return s


def chain(n: int = 60, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A walk with nodes at beads 10, 30 and 31. Bead 10 may move with its neighbours."""
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[10, 30, 31]] = True
    centre = np.zeros(n, dtype=np.bool_)
    centre[10] = True
    return pos, np.ones(n - 1, dtype=np.float32), fixed, centre


@pytest.mark.parametrize("wall", [False, True])
def test_the_centre_of_mass_of_a_node_and_its_neighbours_stays(wall: bool) -> None:
    pos, dtn, fixed, centre = chain()
    start = pos.astype(np.float64).copy()
    mc_numba.seed_numba(5)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, settings(wall), centre=centre)
    assert np.allclose(pos[9:12].mean(axis=0), start[9:12].mean(axis=0), atol=1e-4)
    assert np.linalg.norm(pos[10] - start[10]) > 1e-3, "the node moved"
    assert np.allclose(pos[[30, 31]], start[[30, 31]]), "a node without the rule stays"


def test_running_score_is_the_full_energy_with_the_three_bead_move() -> None:
    pos, dtn, fixed, centre = chain(seed=3)
    s = settings(wall=False)
    s.use_confinement = True
    s.confinement_apply_to_smooth = True
    s.confinement_weight = 0.1
    s.confinement_radius_smooth = 4.0
    cx, cy, cz = (float(v) for v in pos.astype(np.float64).mean(axis=0))
    mc_numba.seed_numba(8)
    score = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s, centre=centre)
    p64 = pos.astype(np.float64)
    full = (
        init_smooth_nb(p64, dtn.astype(np.float64), 0.1, 0.1, 0.1, 1.0, 1.0)
        + init_excl_nb(p64, 0.7, 0.1, int(s.exclusion_skip_neighbors))
        + init_confine_nb(p64, cx, cy, cz, 4.0, 0.1)
    )
    assert score == pytest.approx(full, rel=1e-4), "the three single moves sum to the whole move"


def test_without_the_rule_a_node_does_not_move() -> None:
    pos, dtn, fixed, _centre = chain()
    start = pos.copy()
    mc_numba.seed_numba(5)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, settings())
    assert np.array_equal(pos[[10, 30, 31]], start[[10, 30, 31]])


def test_orientation_score_follows_the_three_bead_move() -> None:
    """Sites at beads 8 and 12 read beads 9 and 11, which move with the node at bead 10."""
    from gnome3d.mc.numba.terms import score_orientation_full_nb
    from gnome3d.util import calc_orientation

    rng = np.random.default_rng(2)
    n = 40
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[8, 10, 12, 30]] = True
    centre = np.zeros(n, dtype=np.bool_)
    centre[10] = True
    chars = np.array(["N"] * n, dtype="<U1")
    chars[[8, 10, 12, 30]] = ["R", "L", "R", "L"]
    nbrs = {0: [2, 3], 1: [3], 2: [0], 3: [0, 1]}
    w = {0: [1.0, 2.0], 1: [1.5], 2: [1.0], 3: [2.0, 1.5]}
    s = settings(wall=False)
    s.use_excluded_volume = False
    s.use_ctcf_motif = True
    s.motif_weight = 5.0
    dtn = np.ones(n - 1, dtype=np.float32)
    mc_numba.seed_numba(6)
    score = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s, chars, nbrs, w, centre=centre)
    p64 = pos.astype(np.float64)
    sites = [8, 10, 12, 30]
    orn = np.array([calc_orientation(p64, b, n, chars[b]) for b in sites])
    offsets = np.array([0, 2, 3, 4, 6], dtype=np.int32)
    indices = np.array([2, 3, 3, 0, 0, 1], dtype=np.int32)
    weights = np.array([1.0, 2.0, 1.5, 1.0, 2.0, 1.5])
    full = init_smooth_nb(p64, np.ones(n - 1), 0.1, 0.1, 0.1, 1.0, 1.0) + score_orientation_full_nb(
        orn, offsets, indices, weights, 5.0, bool(s.motifs_symmetric)
    )
    assert score == pytest.approx(full, rel=1e-4)


def test_centre_beads_are_apart_and_have_free_neighbours() -> None:
    fixed = np.zeros(20, dtype=np.bool_)
    fixed[[0, 3, 5, 9, 10, 14, 19]] = True
    got = np.flatnonzero(centre_beads(fixed)).tolist()
    assert got == [3, 14], "not an end, not beside a fixed bead, three beads from the one before"


def test_a_neighbour_of_such_a_bead_moves_only_with_the_other_neighbour() -> None:
    """Beads 9 and 11 are free. Their sum stays when the node at bead 10 is put back."""
    pos, dtn, fixed, centre = chain(seed=7)
    start = pos.astype(np.float64).copy()
    mc_numba.seed_numba(9)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, settings(), centre=centre)
    shift = pos.astype(np.float64) - start
    assert np.allclose(shift[9] + shift[10] + shift[11], 0.0, atol=1e-4)
    assert np.linalg.norm(shift[9] - shift[11]) > 1e-3, (
        "the neighbours also move against each other"
    )


# The JAX kernel. It is float32 and its checks have a wider tolerance.

# The grid copies its table per move on the CPU backend. Its checks are marked slow.
VARIANTS = [
    (1, False),
    (8, False),
    pytest.param(1, True, marks=pytest.mark.slow),
    pytest.param(8, True, marks=pytest.mark.slow),
]


def jax_settings(wall: bool, prefetch: int = 1, grid: bool = False) -> Settings:
    s = settings(wall)
    s.smooth_prefetch = prefetch
    s.smooth_jax_grid = grid
    s.smooth_jax_grid_min_beads = 1
    return s


def full_energy(pos: np.ndarray, s: Settings, conf: tuple[float, float, float] | None) -> float:
    p64 = pos.astype(np.float64)
    full = init_smooth_nb(p64, np.ones(p64.shape[0] - 1), 0.1, 0.1, 0.1, 1.0, 1.0)
    if s.use_excluded_volume:
        full += init_excl_nb(p64, 0.7, 0.1, int(s.exclusion_skip_neighbors))
    if conf is not None:
        full += init_confine_nb(p64, *conf, 4.0, 0.1)
    return float(full)


@pytest.mark.parametrize(("prefetch", "grid"), VARIANTS)
@pytest.mark.parametrize("wall", [False, True])
def test_jax_keeps_the_centre_of_mass_and_the_full_energy(
    wall: bool, prefetch: int, grid: bool
) -> None:
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    pos, dtn, fixed, centre = chain(seed=3)
    start = pos.astype(np.float64).copy()
    s = jax_settings(wall, prefetch, grid)
    s.use_confinement = True
    s.confinement_apply_to_smooth = True
    s.confinement_weight = 0.1
    s.confinement_radius_smooth = 4.0
    conf = (float(start[:, 0].mean()), float(start[:, 1].mean()), float(start[:, 2].mean()))
    score = mc_jax.mc_smooth_jax(pos, dtn, fixed, 1.0, s, centre=centre)
    assert np.allclose(pos[9:12].mean(axis=0), start[9:12].mean(axis=0), atol=1e-3)
    assert np.linalg.norm(pos[10] - start[10]) > 1e-3, "the node moved"
    assert np.allclose(pos[[30, 31]], start[[30, 31]]), "a node without the rule stays"
    assert score == pytest.approx(full_energy(pos, s, conf), rel=2e-3), (
        "the running score is the energy of the final chain"
    )


def test_jax_without_the_rule_a_node_does_not_move() -> None:
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    pos, dtn, fixed, _centre = chain()
    start = pos.copy()
    mc_jax.mc_smooth_jax(pos, dtn, fixed, 1.0, jax_settings(True))
    assert np.array_equal(pos[[10, 30, 31]], start[[10, 30, 31]])


@pytest.mark.parametrize("prefetch", [1, 8])
def test_jax_orientation_score_follows_the_three_bead_move(prefetch: int) -> None:
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    from gnome3d.mc.numba.terms import score_orientation_full_nb
    from gnome3d.util import calc_orientation

    rng = np.random.default_rng(2)
    n = 40
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    start = pos.copy()
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[8, 10, 12, 30]] = True
    centre = np.zeros(n, dtype=np.bool_)
    centre[10] = True
    chars = np.array(["N"] * n, dtype="<U1")
    chars[[8, 10, 12, 30]] = ["R", "L", "R", "L"]
    nbrs = {0: [2, 3], 1: [3], 2: [0], 3: [0, 1]}
    w = {0: [1.0, 2.0], 1: [1.5], 2: [1.0], 3: [2.0, 1.5]}
    s = jax_settings(False, prefetch)
    s.use_excluded_volume = False
    s.use_ctcf_motif = True
    s.motif_weight = 5.0
    dtn = np.ones(n - 1, dtype=np.float32)
    score = mc_jax.mc_smooth_jax(pos, dtn, fixed, 1.0, s, chars, nbrs, w, centre=centre)
    p64 = pos.astype(np.float64)
    sites = [8, 10, 12, 30]
    orn = np.array([calc_orientation(p64, b, n, chars[b]) for b in sites])
    offsets = np.array([0, 2, 3, 4, 6], dtype=np.int32)
    indices = np.array([2, 3, 3, 0, 0, 1], dtype=np.int32)
    weights = np.array([1.0, 2.0, 1.5, 1.0, 2.0, 1.5])
    full = init_smooth_nb(p64, np.ones(n - 1), 0.1, 0.1, 0.1, 1.0, 1.0) + score_orientation_full_nb(
        orn, offsets, indices, weights, 5.0, bool(s.motifs_symmetric)
    )
    assert np.linalg.norm(pos[10] - start[10]) > 1e-3, "the node moved"
    assert score == pytest.approx(full, rel=2e-3)


def batch_problem(seed: int, s: Settings, n: int = 60) -> dict[str, object]:
    pos, dtn, fixed, centre = chain(n=n, seed=seed)
    return {
        "pos": pos, "dtn": dtn, "fixed": fixed, "step_size": 1.0, "settings": s, "seed": seed,
        "char_orientations": None, "anchor_neighbors": None, "anchor_neighbor_weights": None,
        "heat_dist": None, "compartment": None, "centre": centre,
    }  # fmt: skip


@pytest.mark.parametrize(("prefetch", "grid"), [VARIANTS[0], VARIANTS[1], VARIANTS[3]])
def test_jax_batch_of_blocks_keeps_the_centre_of_mass_and_the_full_energy(
    prefetch: int, grid: bool
) -> None:
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    s = jax_settings(True, prefetch, grid)
    problems = [batch_problem(3, s), batch_problem(4, s, n=45), batch_problem(5, s)]
    out = mc_jax.mc_smooth_jax_batch(problems, s)
    for prob, (score, final) in zip(problems, out, strict=True):
        start = np.asarray(prob["pos"], dtype=np.float64)
        assert np.allclose(final[9:12].mean(axis=0), start[9:12].mean(axis=0), atol=1e-3)
        assert np.linalg.norm(final[10] - start[10]) > 1e-3, "the node moved"
        assert np.allclose(final[[30, 31]], start[[30, 31]]), "a node without the rule stays"
        assert score == pytest.approx(full_energy(final, s, None), rel=2e-3)


def test_jax_and_numba_anneal_to_the_same_energy_over_seeds() -> None:
    """The two kernels draw from different streams. The medians over seeds agree."""
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    s = jax_settings(True)
    seeds = range(20, 32)
    numba_scores = []
    for seed in seeds:
        pos, dtn, fixed, centre = chain(seed=seed)
        mc_numba.seed_numba(seed)
        numba_scores.append(mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s, centre=centre))
    out = mc_jax.mc_smooth_jax_batch([batch_problem(seed, s) for seed in seeds], s)
    a, b = float(np.median(numba_scores)), float(np.median([o[0] for o in out]))
    assert abs(a - b) <= 0.3 * max(a, b), f"median final energy: numba {a:.4f}, jax {b:.4f}"
