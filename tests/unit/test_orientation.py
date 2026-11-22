"""Unit checks for the orientation term with the vector of an anchor over a stretch of chain,
`[motif_orientation] form = stretch`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc import numba as mc_numba
from gnome3d.mc.numba.common import (
    ORIENTATION_STRETCH,
    loop_shares,
    prepare_orientation,
    sites_of_moves,
)
from gnome3d.mc.numba.terms import (
    calc_orientation_stretch_nb,
    init_smooth_nb,
    score_orientation_full_nb,
)
from gnome3d.pipeline.ib.smooth import _batch_run, centre_beads
from gnome3d.settings import Settings
from gnome3d.types import F64Array

K = ORIENTATION_STRETCH


def stretch_settings() -> Settings:
    s = Settings()
    s.use_ctcf_motif = True
    s.motif_form = "stretch"
    s.use_excluded_volume = False
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
    return s


def test_vector_of_a_bent_chain_in_closed_form() -> None:
    """Three beads along x, the anchor, three beads along y. The centroids are (1, 0, 0) and
    (3, 2, 0).
    """
    pos = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0]],
        dtype=np.float64,
    )
    got = np.array(calc_orientation_stretch_nb(pos, 3, 7, False, K))
    assert np.allclose(got, np.array([2.0, 2.0, 0.0]) / np.sqrt(8.0))
    assert np.allclose(np.array(calc_orientation_stretch_nb(pos, 3, 7, True, K)), -got), (
        "the motif L reverses it"
    )


def test_vector_at_the_ends_and_near_them() -> None:
    pos = np.zeros((6, 3))
    pos[:, 0] = np.arange(6)
    pos[5, 1] = 3.0
    first = np.array(calc_orientation_stretch_nb(pos, 0, 6, False, K))
    assert np.allclose(first, [1.0, 0.0, 0.0]), "no bead before: from the anchor to the centroid"
    last = np.array(calc_orientation_stretch_nb(pos, 5, 6, False, K))
    want = pos[5] - pos[2:5].mean(axis=0)
    assert np.allclose(last, want / np.linalg.norm(want)), "no bead after"
    near = np.array(calc_orientation_stretch_nb(pos, 1, 6, False, K))
    want = pos[2:5].mean(axis=0) - pos[0]
    assert np.allclose(near, want / np.linalg.norm(want)), "one bead before: the bead there is"


def test_with_one_bead_a_side_it_is_the_vector_between_the_neighbours() -> None:
    from gnome3d.util import calc_orientation

    rng = np.random.default_rng(1)
    pos = rng.normal(size=(9, 3))
    for bead, char in ((0, "R"), (4, "L"), (8, "R")):
        got = np.array(calc_orientation_stretch_nb(pos, bead, 9, char == "L", 1))
        assert np.allclose(got, calc_orientation(pos, bead, 9, char))


def test_one_bead_moved_by_one_bond_turns_the_vector_by_under_five_degrees() -> None:
    pos = np.zeros((9, 3))
    pos[:, 0] = np.arange(9)
    before = np.array(calc_orientation_stretch_nb(pos, 4, 9, False, K))
    pos[5, 1] += 1.0
    after = np.array(calc_orientation_stretch_nb(pos, 4, 9, False, K))
    angle = float(np.degrees(np.arccos(np.clip(before @ after, -1.0, 1.0))))
    assert 4.0 < angle < 5.0, f"{angle:.2f} degrees"


def test_the_loops_of_a_site_sum_to_one_at_most_and_a_loop_has_one_weight() -> None:
    # site 0: loops to 1 (weight 3) and 2 (weight 1). site 1: to 0. site 2: to 0 and 3 (weight 7).
    offsets = np.array([0, 2, 3, 5, 6], dtype=np.int32)
    indices = np.array([1, 2, 0, 0, 3, 2], dtype=np.int32)
    weights = np.array([3.0, 1.0, 3.0, 1.0, 7.0, 7.0])
    got = loop_shares(offsets, indices, weights)
    assert got.tolist() == pytest.approx([3 / 4, 1 / 8, 3 / 4, 1 / 8, 7 / 8, 7 / 8])
    sums = [got[offsets[k] : offsets[k + 1]].sum() for k in range(4)]
    assert max(sums) <= 1.0 + 1e-12, "a site at its worst has the weight of the term"


def test_a_move_reads_every_site_within_the_stretch() -> None:
    n = 30
    anchor_ar = np.array([5, 7, 20, 26], dtype=np.int32)
    offsets = np.array([0, 1, 2, 3, 3], dtype=np.int32)  # the site at bead 26 has no loop
    table = sites_of_moves(n, anchor_ar, offsets, None, K)
    assert [k for k in table[4] if k >= 0] == [0, 1], "bead 4 is within 3 of beads 5 and 7"
    assert [k for k in table[10] if k >= 0] == [1], "bead 10 is 3 from bead 7 and 5 from bead 5"
    assert [k for k in table[14] if k >= 0] == [], "no site within 3 beads"
    assert [k for k in table[25] if k >= 0] == [], "a site without a loop is left out"
    centre_of = np.full(n, -1, dtype=np.int32)
    centre_of[19:22] = 20
    moved = sites_of_moves(n, anchor_ar, offsets, centre_of, K)
    assert [k for k in moved[19] if k >= 0] == [2]
    centre_of[8:11] = 9
    moved = sites_of_moves(n, anchor_ar, offsets, centre_of, K)
    assert [k for k in moved[10] if k >= 0] == [0, 1], (
        "beads 8 to 10 move together: bead 8 is 3 from bead 5 and bead 10 alone is 5 from it"
    )


def chain_with_sites(seed: int = 3) -> tuple[np.ndarray, ...]:
    """A walk of 50 beads with sites at beads 8, 10, 12, 30 and 31, two of them two beads from
    another, and loops between them.
    """
    rng = np.random.default_rng(seed)
    n = 50
    pos = np.cumsum(rng.normal(size=(n, 3)) * 0.8, axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[8, 10, 12, 30, 31]] = True
    chars = np.array(["N"] * n, dtype="<U1")
    chars[[8, 10, 12, 30, 31]] = ["R", "L", "R", "L", "N"]
    return pos, np.ones(n - 1, dtype=np.float32), fixed, chars


NBRS = {0: [2, 3], 1: [3, 4], 2: [0], 3: [0, 1], 4: [1]}
NBR_W = {0: [1.0, 2.0], 1: [1.5, 0.5], 2: [1.0], 3: [2.0, 1.5], 4: [0.5]}


def full_orientation(pos: F64Array, fixed: np.ndarray, chars: np.ndarray, s: Settings) -> float:
    """The term of a chain from scratch, as the kernel prepares it."""
    weight = 2.0 * K * float(s.spring_stretch) * float(s.smooth_dist_weight)
    _ar, offsets, indices, weights, _is_l, _b2a, orn, score = prepare_orientation(
        pos, fixed, chars, NBRS, NBR_W, weight, bool(s.motifs_symmetric), K
    )
    again = score_orientation_full_nb(
        orn, offsets, indices, weights, weight, bool(s.motifs_symmetric)
    )
    assert score == pytest.approx(again)
    return float(score)


@pytest.mark.parametrize("rule", ["fixed", "centre"])
def test_running_score_is_the_full_energy_with_sites_two_beads_apart(rule: str) -> None:
    pos, dtn, fixed, chars = chain_with_sites()
    s = stretch_settings()
    centre = centre_beads(fixed) if rule == "centre" else None
    start = full_orientation(pos.astype(np.float64), fixed, chars, s)
    mc_numba.seed_numba(4)
    score = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1.0, s, chars, NBRS, NBR_W, centre=centre)
    p64 = pos.astype(np.float64)
    term = full_orientation(p64, fixed, chars, s)
    full = init_smooth_nb(p64, np.ones(49), 0.1, 0.1, 0.1, 1.0, 1.0) + term
    assert score == pytest.approx(full, rel=1e-6), "every site that reads a moved bead is scored"
    assert 0.0 <= term < start, f"the term is non negative and falls: {start:.4f} to {term:.4f}"


def test_the_term_is_bounded_by_the_bonds_of_the_stretches() -> None:
    pos, _dtn, fixed, chars = chain_with_sites(seed=5)
    s = stretch_settings()
    term = full_orientation(pos.astype(np.float64), fixed, chars, s)
    assert term <= 5 * 2.0 * K * 0.1 + 1e-9, "five sites, each at most the weight of the term"


def test_the_loader_knows_two_forms_and_the_batch_executor_refuses_stretch() -> None:
    with pytest.raises(ValueError, match="neighbours or stretch"):
        Settings.from_dict({"motif_orientation": {"form": "some"}})
    with pytest.raises(ValueError, match="threaded or serial"):
        _batch_run([{"settings": stretch_settings()}])


def test_the_multiple_under_trial_scales_the_term() -> None:
    """The same chain and the same moves refused: with no steps the score is that of the start."""
    scores = []
    for multiple in (1.0, 10.0):
        pos, dtn, fixed, chars = chain_with_sites()
        s = stretch_settings()
        s.motif_stretch_weight = multiple
        s.max_temp_smooth = 0.0
        s.mc_stop_steps_smooth = 1
        s.mc_stop_successes_smooth = 10
        start = pos.astype(np.float64).copy()
        mc_numba.seed_numba(1)
        total = mc_numba.mc_smooth_numba(pos, dtn, fixed, 1e-9, s, chars, NBRS, NBR_W)
        bonds = init_smooth_nb(pos.astype(np.float64), np.ones(49), 0.1, 0.1, 0.1, 1.0, 1.0)
        assert np.allclose(pos, start, atol=1e-6)
        scores.append(total - bonds)
    assert scores[1] == pytest.approx(10.0 * scores[0], rel=1e-4)
    with pytest.raises(ValueError, match="above zero"):
        Settings.from_dict({"motif_orientation": {"stretch_weight": "0"}})
