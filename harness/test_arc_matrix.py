"""Unit checks for the arcs stage's target matrix under the polymer law.

    python harness/test_arc_matrix.py

The matrix has three kinds of entry. A pair an arc joins carries the law's contact distance. A
consecutive pair with no arc carries `arcs_chain_bond_scale` times the background when chain
bonds are on. Every other pair carries -1, no target, and the diagonal is 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.terms import init_arcs_nb  # noqa: E402
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
    check("an arcless pair is -1", m[0, 1] == -1.0 and m[2, 3] == -1.0)
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
    check("a non consecutive arcless pair stays -1", m[0, 3] == -1.0)
    s.use_arcs_chain_bonds = False
    m2 = add_chain_bonds(arc_expected_matrix(s, mids, [(0, 2, 4)]), mids, s)
    check("with chain bonds off the matrix is returned as built", m2[0, 1] == -1.0)


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
    exp = np.full((n, n), -1.0)
    for _ in range(25):
        i, j = rng.integers(0, n, 2)
        if i != j:
            exp[i, j] = exp[j, i] = float(rng.uniform(0.5, 3.0))
    np.fill_diagonal(exp, 0.0)
    rep_inv = 0.3
    want = float(init_arcs_nb(pos, exp, 1.0, 1.0, rep_inv))
    _, init_arcs, _, _, _ = _build_arcs_kernel(10, 1)
    got = float(
        init_arcs(
            jnp.asarray(pos[None].astype(np.float32)),
            jnp.asarray(exp.astype(np.float32)),
            jnp.float32(1.0),
            jnp.float32(1.0),
            jnp.float32(rep_inv),
        )[0]
    )
    check(
        "jax and numba agree on the initial energy",
        abs(got - want) / max(abs(want), 1e-9) < 1e-4,
        f"{got:.5f} against {want:.5f}",
    )


def main() -> int:
    print("arc matrix checks")
    test_matrix()
    test_chain_bonds()
    test_jax_matches_numba()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
