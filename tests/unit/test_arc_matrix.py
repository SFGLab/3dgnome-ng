"""Unit checks for the target matrix of the arcs stage."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gnome3d.mc.numba.terms import (
    _local_arcs_nb,
    init_arcs_nb,
)
from gnome3d.pipeline.coarse.build import (
    _active_arcs,
    add_contact_background,
    arc_expected_matrix,
    repulsion_range,
)
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings


def settings() -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    return s


def test_matrix() -> None:
    """Arc pairs carry the law's contact distance, the rest nothing."""
    s = settings()
    mids = [0, 10_000, 50_000, 200_000]
    arcs = [(0, 2, 4), (1, 3, 40)]
    m = arc_expected_matrix(s, mids, arcs)
    assert float(np.abs(np.diag(m)).max()) == 0.0, "diagonal is zero"
    assert m[0, 1] == -0.5 and m[2, 3] == -0.5, (
        "an arcless pair is the repulsion marker when the short range background is off"
    )
    assert abs(m[0, 2] - s.arc_expected_distance(4, 50_000)) < 1e-12, (
        "an arc pair carries the law's distance"
    )
    assert m[0, 2] == m[2, 0] and m[1, 3] == m[3, 1], "and is symmetric"
    assert m[1, 3] < s.polymer.background(190_000), (
        f"a stronger arc is closer than a weaker one at a longer span: {m[1, 3]:.3f} against background {s.polymer.background(190_000):.3f}"
    )
    assert float(m[m > 0].min()) >= 1.0 - 1e-12, "no arc target is under one bead"


def test_two_factors() -> None:
    """A pair joined by a CTCF and an RNAPII loop is held by the closer."""
    s = settings()
    mids = [0, 10_000, 50_000]
    weak, strong = s.arc_expected_distance(3, 50_000, 0), s.arc_expected_distance(40, 50_000, 1)
    for order in ([(0, 2, 3, 0), (0, 2, 40, 1)], [(0, 2, 40, 1), (0, 2, 3, 0)]):
        m = arc_expected_matrix(s, mids, order)
        assert abs(m[0, 2] - min(weak, strong)) < 1e-12 and m[0, 2] == m[2, 0], (
            f"the pair takes the closer target in either order: {m[0, 2]:.3f} against {weak:.3f} and {strong:.3f}"
        )
    assert min(weak, strong) < s.polymer.background(50_000), "which is under the background"

    # The summary arc that data.py adds after the arcs of a pair has score 0 and factor -1.
    arc = lambda a, b, score, factor: SimpleNamespace(start=a, end=b, score=score, factor=factor)  # noqa: E731
    clusters = [
        SimpleNamespace(arcs=[0, 1, 2], genomic_pos=0),
        SimpleNamespace(arcs=[], genomic_pos=10_000),
        SimpleNamespace(arcs=[0, 1, 2], genomic_pos=50_000),
    ]
    state = SimpleNamespace(
        clusters=clusters, arcs={"chr1": [arc(0, 2, 3, 0), arc(0, 2, 40, 1), arc(0, 2, 0, -1)]}, s=s
    )
    got_mids, got = _active_arcs(state, [0, 1, 2], "chr1")
    assert got == [(0, 2, 3, 0), (0, 2, 40, 1)], (
        "the summary arc is left out of the active arcs: " + str(got)
    )
    m = arc_expected_matrix(s, got_mids, got)
    assert m[0, 2] < s.polymer.background(50_000) - 1e-9, (
        f"so the pair is not held at the background: {m[0, 2]:.3f}"
    )


def test_loop_contact() -> None:
    """A kept loop is held at contact, one bead, whatever its strength."""
    s = settings()
    mids = [0, 10_000, 50_000, 200_000]
    arcs = [(0, 2, 3, 0), (1, 3, 40, 1)]
    off = arc_expected_matrix(s, mids, arcs)
    s.loop_contact = True
    on = arc_expected_matrix(s, mids, arcs)
    assert off[0, 2] > 1.0 + 1e-6 and off[1, 3] > 1.0 + 1e-6, (
        f"off, the law's part way targets: {off[0, 2]:.3f} {off[1, 3]:.3f}"
    )
    assert on[0, 2] == 1.0 and on[1, 3] == 1.0 and on[2, 0] == 1.0, (
        "on, every loop pair at one bead"
    )
    assert np.array_equal(on[off <= 0], off[off <= 0]), "and the arcless pairs as they were"


def test_short_range_background() -> None:
    """Arcless pairs inside the range sit on the background."""
    s = settings()
    s.background_weight = 0.3
    s.background_range_bp = 100_000
    mids = [0, 10_000, 50_000, 200_000]
    m = arc_expected_matrix(s, mids, [(0, 2, 4)])
    assert (
        abs(m[0, 1] + s.polymer.background(10_000)) < 1e-12
        and abs(m[1, 2] + s.polymer.background(40_000)) < 1e-12
    ), "a pair inside the range carries minus its background"
    assert m[0, 3] == -0.5 and m[1, 3] == -0.5, "a pair beyond the range keeps the repulsion marker"
    assert m[0, 2] > 0.0, "the arc pair is untouched"
    assert all(v <= -1.0 for v in m[m < -0.5]), (
        "every background is at least one bead, so the two kinds cannot be confused"
    )

    pos = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    exp = np.zeros((3, 3))
    exp[0, 1] = exp[1, 0] = -3.0  # background 3, realised 6, a spring
    exp[0, 2] = exp[2, 0] = -0.5  # repulsion marker, realised 2
    exp[1, 2] = exp[2, 1] = 2.0  # arc target 2, realised sqrt(40)
    w, cut = 0.3, 0.1
    full = float(init_arcs_nb(pos, exp, 1.0, 1.0, cut, w))
    want = (
        w * ((6.0 - 3.0) / 3.0) ** 2
        + max(0.0, 1.0 / 2.0 - cut)
        + ((np.sqrt(40.0) - 2.0) / 2.0) ** 2
    )
    assert abs(full - want) < 1e-12, (
        f"the full score is spring plus repulsion plus arc: {full:.6f} against {want:.6f}"
    )
    loc = sum(float(_local_arcs_nb(pos, exp, p, 1.0, 1.0, cut, w)) for p in range(3))
    assert abs(loc - 2.0 * full) < 1e-12, "local scores sum to twice the full score"
    squeezed = pos.copy()
    squeezed[1, 0] = 0.5
    assert float(init_arcs_nb(squeezed, exp, 1.0, 1.0, cut, w)) > full + 1.0, (
        "crushing a background pair to a sixth of its distance costs far more"
    )
    off = float(init_arcs_nb(pos, exp, 1.0, 1.0, cut, 0.0))
    assert abs(off - (max(0.0, 1.0 / 2.0 - cut) + ((np.sqrt(40.0) - 2.0) / 2.0) ** 2)) < 1e-12, (
        "weight zero scores the background pair as nothing, so production is untouched"
    )


def test_contact_background() -> None:
    """Beyond the short range, an arcless pair whose contact cell puts it under the background
    holds minus the contact distance of the law. A pair at or under its expected contact keeps
    the repulsion. Pairs inside the range and arc pairs are unchanged.
    """
    print("\n[contact background] enriched arcless pairs beyond the range get the law's distance")
    s = settings()
    s.background_weight = 0.3
    s.background_range_bp = 100_000
    s.use_contact_background = True
    mids = [0, 50_000, 500_000, 1_000_000, 1_500_000, 2_000_000]
    n = len(mids)
    m0 = arc_expected_matrix(s, mids, [(0, 3, 4)])
    heat = np.zeros((n, n))
    # Three pairs at 500 kb, one enriched, one at the mean and one below.
    heat[1, 2] = heat[2, 1] = 8.0
    heat[2, 3] = heat[3, 2] = 1.0
    heat[3, 4] = heat[4, 3] = 0.5
    heat[0, 1] = heat[1, 0] = 50.0  # inside the range, unchanged
    heat[0, 3] = heat[3, 0] = 50.0  # the arc pair, unchanged
    m = add_contact_background(m0, mids, heat, s)
    bg = s.polymer.background(450_000)
    assert m[1, 2] <= -1.0 and -m[1, 2] < bg, (
        f"the enriched pair carries minus a distance under its background: {m[1, 2]:.2f} vs bg {bg:.2f}"
    )
    assert m[2, 3] == -0.5, "the pair at its expected contact keeps the repulsion"
    assert m[3, 4] == -0.5, "the pair below its expected contact keeps the repulsion"
    assert m[1, 4] == -0.5 and m[2, 5] == -0.5, "a pair with no contact keeps the repulsion"
    assert m[0, 1] == m0[0, 1], "inside the range the short range entry stands"
    assert m[0, 3] == m0[0, 3] and m[0, 3] > 0.0, "the arc pair is untouched"
    assert np.allclose(m, m.T), "symmetric"
    assert np.array_equal(m0, arc_expected_matrix(s, mids, [(0, 3, 4)])), "the input is left alone"
    s.use_contact_background = False
    assert add_contact_background(m0, mids, heat, s) is m0, "off, the matrix is returned as is"
    s.use_contact_background = True
    s.background_weight = 0.0
    assert add_contact_background(m0, mids, heat, s) is m0, (
        "without the spring weight there is nothing to hold it, so off"
    )


def test_jax_matches_numba() -> None:
    """The batched kernel's initial energy is numba's."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp

    from gnome3d.mc.jax.arcs import _build_arcs_kernel

    rng = np.random.default_rng(3)
    n = 40
    pos = rng.normal(0.0, 2.0, (n, 3))
    exp = np.full((n, n), -0.5)
    for i in range(n):
        for j in range(n):
            if 0 < abs(i - j) <= 6:
                exp[i, j] = -float(1.0 + 0.4 * abs(i - j))  # a short range background
    for _ in range(25):
        i, j = rng.integers(0, n, 2)
        if i != j:
            exp[i, j] = exp[j, i] = float(rng.uniform(0.5, 3.0))
    np.fill_diagonal(exp, 0.0)
    rep_inv, bg_w = 0.3, 0.3
    want = float(init_arcs_nb(pos, exp, 1.0, 1.0, rep_inv, bg_w))
    _, init_arcs, _, _, _ = _build_arcs_kernel(10, 1)
    got = float(
        init_arcs(
            jnp.asarray(pos[None].astype(np.float32)),
            jnp.asarray(exp.astype(np.float32)),
            jnp.float32(1.0),
            jnp.float32(1.0),
            jnp.float32(rep_inv),
            jnp.float32(bg_w),
        )[0]
    )
    assert abs(got - want) / max(abs(want), 1e-9) < 1e-4, (
        f"jax and numba agree on the initial energy: {got:.5f} against {want:.5f}"
    )


def test_jax_batched_driver_runs() -> None:
    """The batched driver runs with the background on."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from gnome3d.mc import jax as mc_jax

    rng = np.random.default_rng(5)
    probs = []
    for _ in range(2):
        n = 24
        pos = rng.normal(0.0, 1.5, (n, 3)).astype(np.float32)
        exp = np.full((n, n), -0.5)
        for i in range(n):
            for j in range(n):
                if 0 < abs(i - j) <= 4:
                    exp[i, j] = -float(1.0 + 0.3 * abs(i - j))
        exp[0, 10] = exp[10, 0] = 1.2
        np.fill_diagonal(exp, 0.0)
        probs.append({"pos": pos, "exp_dist": exp, "step_size": 0.05})
    s = Settings()
    s.background_weight = 0.3
    s.mc_stop_steps = 200
    s.mc_stop_successes = 1
    out = mc_jax.mc_arcs_jax_batch(probs, s, max_iters=3)
    ok = len(out) == 2 and all(np.isfinite(sc) and p.shape == (24, 3) for sc, p in out)
    assert ok, (
        f"two problems come back with finite scores and the right shapes: {[round(float(sc), 3) for sc, _ in out]}"
    )


def test_repulsion_range() -> None:
    """Every repulsion marker cleared, nothing else moves."""
    s = settings()
    s.background_weight = 0.1
    mids = [0, 50_000, 150_000, 600_000, 2_000_000]
    arcs = [(0, 4, 5, 0)]
    mat = arc_expected_matrix(s, mids, arcs)
    same = repulsion_range(mat, mids, s)
    assert same is mat, "on returns the matrix"
    s.arcs_repulsion = False
    out = repulsion_range(mat, mids, s)
    marker = mat == -0.5
    assert bool((out[marker] == 0.0).all()) and bool(marker.any()), (
        f"markers cleared: {int(marker.sum())}"
    )
    assert bool(np.array_equal(out[~marker], mat[~marker])), "every other entry unchanged"
    assert out[0, 4] == mat[0, 4] and out[0, 4] > 0.0, "the loop kept"
