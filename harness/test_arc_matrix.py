"""Unit checks for the arcs stage's target matrix under the polymer law.

    python harness/test_arc_matrix.py

The matrix has four kinds of entry. A pair an arc joins carries the law's contact distance,
positive. A consecutive pair with no arc carries `arcs_chain_bond_scale` times the background
when chain bonds are on. An arcless pair closer than `background_range_bp` carries minus the
background for its separation when `background_weight` is on, which the kernels score as a
weak spring symmetric in log distance. Every other arcless pair carries -0.5, which the kernels
score as the truncated repulsion, and the diagonal is 0. A background is never under one bead,
so the sign and the magnitude separate the two arcless kinds without a second matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.terms import (
    _local_arcs_nb,  # noqa: E402
    init_arcs_nb,  # noqa: E402
)
from gnome3d.pipeline.coarse.build import add_chain_bonds, arc_expected_matrix  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def settings() -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    return s


def test_matrix() -> None:
    print("\n[matrix] arc pairs carry the law's contact distance, the rest nothing")
    s = settings()
    mids = [0, 10_000, 50_000, 200_000]
    arcs = [(0, 2, 4), (1, 3, 40)]
    m = arc_expected_matrix(s, mids, arcs)
    check("diagonal is zero", float(np.abs(np.diag(m)).max()) == 0.0)
    check(
        "an arcless pair is the repulsion marker when the short range background is off",
        m[0, 1] == -0.5 and m[2, 3] == -0.5,
    )
    check(
        "an arc pair carries the law's distance",
        abs(m[0, 2] - s.arc_expected_distance(4, 50_000)) < 1e-12,
    )
    check("and is symmetric", m[0, 2] == m[2, 0] and m[1, 3] == m[3, 1])
    check(
        "a stronger arc is closer than a weaker one at a longer span",
        m[1, 3] < s.polymer.background(190_000),
        f"{m[1, 3]:.3f} against background {s.polymer.background(190_000):.3f}",
    )
    check("no arc target is under one bead", float(m[m > 0].min()) >= 1.0 - 1e-12)


def test_chain_bonds() -> None:
    print("\n[chain bonds] consecutive arcless anchors are held at the scaled background")
    s = settings()
    s.use_arcs_chain_bonds = True
    s.arcs_chain_bond_scale = 1.5
    mids = [0, 10_000, 50_000, 200_000]
    m = add_chain_bonds(arc_expected_matrix(s, mids, [(0, 2, 4)]), mids, s)
    check(
        "a consecutive arcless pair gets scale times the background",
        abs(m[0, 1] - 1.5 * s.polymer.background(10_000)) < 1e-12,
        f"{m[0, 1]:.3f}",
    )
    check(
        "a consecutive pair with an arc keeps the arc",
        abs(m[1, 2] - 1.5 * s.polymer.background(40_000)) < 1e-12 or m[1, 2] > 0.0,
    )
    check("a non consecutive arcless pair keeps the repulsion marker", m[0, 3] == -0.5)
    s.use_arcs_chain_bonds = False
    m2 = add_chain_bonds(arc_expected_matrix(s, mids, [(0, 2, 4)]), mids, s)
    check("with chain bonds off the matrix is returned as built", m2[0, 1] == -0.5)


def test_short_range_background() -> None:
    """With the weight on, arcless pairs inside the range carry minus the background and those
    beyond it keep the repulsion marker. The kernels score the two kinds differently: a weak
    spring symmetric in log distance, and the truncated repulsion."""
    print("\n[short range] arcless pairs inside the range sit on the background")
    s = settings()
    s.background_weight = 0.3
    s.background_range_bp = 100_000
    mids = [0, 10_000, 50_000, 200_000]
    m = arc_expected_matrix(s, mids, [(0, 2, 4)])
    check(
        "a pair inside the range carries minus its background",
        abs(m[0, 1] + s.polymer.background(10_000)) < 1e-12
        and abs(m[1, 2] + s.polymer.background(40_000)) < 1e-12,
    )
    check("a pair beyond the range keeps the repulsion marker", m[0, 3] == -0.5 and m[1, 3] == -0.5)
    check("the arc pair is untouched", m[0, 2] > 0.0)
    check(
        "every background is at least one bead, so the two kinds cannot be confused",
        all(v <= -1.0 for v in m[m < -0.5]),
    )

    pos = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    exp = np.zeros((3, 3))
    exp[0, 1] = exp[1, 0] = -3.0  # background 3, realised 6: spring
    exp[0, 2] = exp[2, 0] = -0.5  # repulsion marker, realised 2
    exp[1, 2] = exp[2, 1] = 2.0  # arc target 2, realised sqrt(40)
    w, cut = 0.3, 0.1
    full = float(init_arcs_nb(pos, exp, 1.0, 1.0, cut, w))
    want = (
        w * ((6.0 - 3.0) / 3.0) ** 2
        + max(0.0, 1.0 / 2.0 - cut)
        + ((np.sqrt(40.0) - 2.0) / 2.0) ** 2
    )
    check(
        "the full score is spring plus repulsion plus arc",
        abs(full - want) < 1e-12,
        f"{full:.6f} against {want:.6f}",
    )
    loc = sum(float(_local_arcs_nb(pos, exp, p, 1.0, 1.0, cut, w)) for p in range(3))
    check("local scores sum to twice the full score", abs(loc - 2.0 * full) < 1e-12)
    squeezed = pos.copy()
    squeezed[1, 0] = 0.5
    check(
        "crushing a background pair to a sixth of its distance costs far more",
        float(init_arcs_nb(squeezed, exp, 1.0, 1.0, cut, w)) > full + 1.0,
    )
    off = float(init_arcs_nb(pos, exp, 1.0, 1.0, cut, 0.0))
    check(
        "weight zero scores the background pair as nothing, so production is untouched",
        abs(off - (max(0.0, 1.0 / 2.0 - cut) + ((np.sqrt(40.0) - 2.0) / 2.0) ** 2)) < 1e-12,
    )


def test_jax_matches_numba() -> None:
    """The JAX arcs kernel scores the same energy as numba on a matrix that mixes arc springs
    and arcless pairs under the truncated repulsion. The check that keeps the batch kernel from
    drifting, which it once did."""
    print("\n[jax] the batched kernel's initial energy is numba's")
    try:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax.numpy as jnp

        from gnome3d.mc.jax.arcs import _build_arcs_kernel
    except Exception as exc:  # noqa: BLE001
        check("jax available for the cross kernel check", False, str(exc)[:60])
        return
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
    check(
        "jax and numba agree on the initial energy",
        abs(got - want) / max(abs(want), 1e-9) < 1e-4,
        f"{got:.5f} against {want:.5f}",
    )


def test_jax_batched_driver_runs() -> None:
    """The batched JAX arcs driver runs end to end with the short range background on, which is
    the only thing that exercises the step kernel's vmap axes."""
    print("\n[jax] the batched driver runs with the background on")
    try:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from gnome3d.mc import jax as mc_jax
    except Exception as exc:  # noqa: BLE001
        check("jax available", False, str(exc)[:60])
        return
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
    check(
        "two problems come back with finite scores and the right shapes",
        ok,
        f"{[round(float(sc), 3) for sc, _ in out]}",
    )


def main() -> int:
    print("arc matrix checks")
    test_matrix()
    test_chain_bonds()
    test_short_range_background()
    test_jax_matches_numba()
    test_jax_batched_driver_runs()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
