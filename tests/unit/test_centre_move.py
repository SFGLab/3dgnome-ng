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


def orientation_case() -> tuple[np.ndarray, ...]:
    """Sites at beads 8, 10, 12 and 30 with loops between them. The node at bead 10 may move with
    its neighbours, which the sites at beads 8 and 12 read.
    """
    rng = np.random.default_rng(2)
    n = 40
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[8, 10, 12, 30]] = True
    centre = np.zeros(n, dtype=np.bool_)
    centre[10] = True
    chars = np.array(["N"] * n, dtype="<U1")
    chars[[8, 10, 12, 30]] = ["R", "L", "R", "L"]
    return pos, np.ones(n - 1, dtype=np.float32), fixed, centre, chars


ORN_NBRS = {0: [2, 3], 1: [3], 2: [0], 3: [0, 1]}
ORN_W = {0: [1.0, 2.0], 1: [1.5], 2: [1.0], 3: [2.0, 1.5]}


def orientation_energy(pos: np.ndarray, fixed: np.ndarray, chars: np.ndarray, s: Settings) -> float:
    """The term of a chain from scratch, as the numba kernel prepares it."""
    from gnome3d.mc.numba.common import orientation_weight, prepare_orientation

    return float(
        prepare_orientation(
            pos.astype(np.float64), fixed, chars, ORN_NBRS, ORN_W, orientation_weight(s),
            bool(s.motifs_symmetric),
        )[-1]
    )  # fmt: skip


def test_orientation_score_follows_the_three_bead_move() -> None:
    pos, dtn, fixed, centre, chars = orientation_case()
    s = settings(wall=False)
    s.use_excluded_volume = False
    s.use_ctcf_motif = True
    mc_numba.seed_numba(6)
    score = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s, chars, ORN_NBRS, ORN_W, centre=centre)
    p64 = pos.astype(np.float64)
    full = init_smooth_nb(p64, np.ones(39), 0.1, 0.1, 0.1, 1.0, 1.0) + orientation_energy(
        pos, fixed, chars, s
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


@pytest.mark.parametrize("rule", ["fixed", "centre"])
@pytest.mark.parametrize("prefetch", [1, 8])
def test_jax_orientation_score_follows_the_moves(prefetch: int, rule: str) -> None:
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    pos, dtn, fixed, centre, chars = orientation_case()
    start = pos.copy()
    s = jax_settings(False, prefetch)
    s.use_excluded_volume = False
    s.use_ctcf_motif = True
    before = orientation_energy(pos, fixed, chars, s)
    score = mc_jax.mc_smooth_jax(
        pos, dtn, fixed, 1.0, s, chars, ORN_NBRS, ORN_W,
        centre=centre if rule == "centre" else None,
    )  # fmt: skip
    p64 = pos.astype(np.float64)
    term = orientation_energy(pos, fixed, chars, s)
    full = init_smooth_nb(p64, np.ones(39), 0.1, 0.1, 0.1, 1.0, 1.0) + term
    assert (np.linalg.norm(pos[10] - start[10]) > 1e-3) == (rule == "centre"), (
        "the node moves by the rule"
    )
    assert score == pytest.approx(full, rel=2e-3), "the running score is the energy of the chain"
    assert term < before, f"the term falls: {before:.4f} to {term:.4f}"


def test_jax_batch_of_blocks_scores_the_orientation_of_each_block() -> None:
    """Two blocks of different length in one launch, one with the rule and both with loops."""
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    s = jax_settings(False, 8)
    s.use_excluded_volume = False
    s.use_ctcf_motif = True
    problems = []
    for seed, cut in ((2, 40), (2, 34)):
        pos, dtn, fixed, centre, chars = orientation_case()
        problems.append(
            {
                "pos": pos[:cut], "dtn": dtn[: cut - 1], "fixed": fixed[:cut], "step_size": 1.0,
                "settings": s, "seed": seed + cut, "char_orientations": chars[:cut],
                "anchor_neighbors": ORN_NBRS, "anchor_neighbor_weights": ORN_W,
                "heat_dist": None, "compartment": None, "centre": centre[:cut],
            }
        )  # fmt: skip
    out = mc_jax.mc_smooth_jax_batch(problems, s)
    for prob, (score, final) in zip(problems, out, strict=True):
        cut = final.shape[0]
        term = orientation_energy(final, prob["fixed"], prob["char_orientations"], s)
        full = init_smooth_nb(final.astype(np.float64), np.ones(cut - 1), 0.1, 0.1, 0.1, 1.0, 1.0)
        assert score == pytest.approx(full + term, rel=2e-3), f"block of {cut} beads"
        assert term < orientation_energy(prob["pos"], prob["fixed"], prob["char_orientations"], s)


@pytest.mark.parametrize("prefetch", [1, 8])
def test_jax_heat_and_compartment_scores_follow_the_three_bead_move(prefetch: int) -> None:
    """Two nodes with the rule, a heat target for every pair and a compartment for every bead."""
    mc_jax = pytest.importorskip("gnome3d.mc.jax")
    from gnome3d.mc.numba.common import affinity_params, init_affinity_scores
    from gnome3d.mc.numba.terms import init_heat_nb

    rng = np.random.default_rng(9)
    n = 48
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    start = pos.astype(np.float64).copy()
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[10, 20, 30]] = True
    centre = np.zeros(n, dtype=np.bool_)
    centre[[10, 20]] = True
    dtn = np.ones(n - 1, dtype=np.float32)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.float64)
    heat = np.where(sep > 0, np.maximum(sep, 1.0) ** 0.5, 0.0)
    comp = np.where((np.arange(n) // 6) % 2 == 0, 1, -1).astype(np.int8)
    s = jax_settings(False, prefetch)
    s.subanchor_heatmap_dist_weight = 0.05
    s.use_compartments = True
    s.compartment_weight = 1.5
    score = mc_jax.mc_smooth_jax(
        pos, dtn, fixed, 1.0, s, heat_dist=heat, compartment=comp, centre=centre
    )
    p64 = pos.astype(np.float64)
    aff = affinity_params(s, "smooth", 1.0, comp)
    full = (
        full_energy(pos, s, None) + init_heat_nb(p64, heat, 0.05) + init_affinity_scores(p64, aff)
    )
    for c in (10, 20):
        assert np.allclose(
            p64[c - 1 : c + 2].mean(axis=0), start[c - 1 : c + 2].mean(axis=0), atol=1e-3
        )
        assert np.linalg.norm(p64[c] - start[c]) > 1e-3, "the node moved"
    assert score == pytest.approx(full, rel=2e-3), "the running score is the energy of the chain"


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
