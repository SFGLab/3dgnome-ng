"""Unit checks for force biased proposals in the arcs MC.

    python harness/test_force_bias.py

A proposal is currently drawn isotropically, so most of a step's work goes into a direction the
energy does not want. Biasing it along the local descent direction makes a proposal more likely
to be accepted and to move further when it is.

The gradient has to be exact or the bias points somewhere useless, so most of this file is a
finite difference check of it against the scorer it is derived from, on each branch of that
scorer separately: a repulsion pair inside the cutoff, one outside it where the term is flat, a
stretched spring and a squeezed one, which have different constants.

Doing this on the arcs energy rather than a gradient solver is deliberate. The repulsion is
singular, so a solver following the gradient sees unbounded forces whenever two anchors approach.
Here the gradient only proposes and the Metropolis rule still rejects, which keeps that safe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.mc.numba.terms import _local_arcs_grad_nb, _local_arcs_nb  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
K_STRETCH, K_SQUEEZE = 1.0, 0.4  # deliberately unequal: the spring is asymmetric


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def fd_grad(pos: np.ndarray, exp: np.ndarray, p: int, cutoff_inv: float, h: float = 1e-6):
    """Central finite difference of the local score with respect to anchor p."""
    g = np.zeros(3)
    for k in range(3):
        a = pos.copy()
        a[p, k] += h
        b = pos.copy()
        b[p, k] -= h
        g[k] = (
            _local_arcs_nb(a, exp, p, K_STRETCH, K_SQUEEZE, cutoff_inv)
            - _local_arcs_nb(b, exp, p, K_STRETCH, K_SQUEEZE, cutoff_inv)
        ) / (2 * h)
    return g


def one_pair(sep: float, e: float, cutoff: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Two anchors `sep` apart on the x axis, with `e` as their expected distance."""
    pos = np.ascontiguousarray(np.array([[0.0, 0.0, 0.0], [sep, 0.3, -0.2]], dtype=np.float64))
    exp = np.array([[0.0, e], [e, 0.0]], dtype=np.float64)
    return pos, exp, 1.0 / cutoff


def agrees(pos, exp, p, cinv, tol=2e-5) -> tuple[bool, float, float]:
    s, gx, gy, gz = _local_arcs_grad_nb(pos, exp, p, K_STRETCH, K_SQUEEZE, cinv)
    ref = _local_arcs_nb(pos, exp, p, K_STRETCH, K_SQUEEZE, cinv)
    fd = fd_grad(pos, exp, p, cinv)
    g = np.array([gx, gy, gz])
    scale = max(np.abs(fd).max(), 1e-9)
    return (
        (s == ref and np.abs(g - fd).max() / scale < tol),
        float(s - ref),
        float(np.abs(g - fd).max() / scale),
    )


def test_score_and_each_branch() -> None:
    for label, sep, e, cutoff in (
        ("a repulsion pair inside the cutoff", 0.8, -1.0, 2.0),
        ("a repulsion pair outside it, where the term is flat", 5.0, -1.0, 2.0),
        ("a stretched spring", 1.4, 0.5, 2.0),
        ("a squeezed spring", 0.3, 0.9, 2.0),
    ):
        pos, exp, cinv = one_pair(sep, e, cutoff)
        ok, ds, dg = agrees(pos, exp, 0, cinv)
        check(
            f"the gradient matches finite differences for {label}",
            ok,
            f"score delta {ds:.2e}, gradient rel {dg:.2e}",
        )


def test_a_real_sized_block() -> None:
    """Every anchor of a block carrying both branches at once."""
    rng = np.random.default_rng(4)
    n = 120
    pos = np.ascontiguousarray(rng.normal(0.0, 1.5, size=(n, 3)))
    exp = np.full((n, n), -1.0)
    r2 = np.random.default_rng(5)
    for _ in range(60):
        i, j = r2.integers(0, n, 2)
        if i != j:
            v = float(r2.uniform(0.3, 0.9))
            exp[i, j] = exp[j, i] = v
    np.fill_diagonal(exp, 0.0)
    cinv = 1.0 / 2.0
    worst_s = worst_g = 0.0
    for p in range(0, n, 7):
        _, ds, dg = agrees(pos, exp, p, cinv)
        worst_s, worst_g = max(worst_s, abs(ds)), max(worst_g, dg)
    check(
        "the score is identical and the gradient matches across a whole block",
        worst_s == 0.0 and worst_g < 2e-5,
        f"worst score delta {worst_s:.2e}, worst gradient rel {worst_g:.2e}",
    )


def run(bias: float) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(6)
    n = 200
    exp = np.full((n, n), -1.0)
    r2 = np.random.default_rng(7)
    for _ in range(int(0.004 * n * n / 2)):
        i, j = r2.integers(0, n, 2)
        if i != j:
            v = float(r2.uniform(0.2, 0.6))
            exp[i, j] = exp[j, i] = v
    np.fill_diagonal(exp, 0.0)
    pos = np.ascontiguousarray(rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32))
    s = Settings()
    s.arcs_repulsion_cutoff_factor = 3.0
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.mc_stop_steps = 5_000
    s.mc_stop_improvement = 0.999
    s.mc_stop_successes = 100
    s.max_temp = 5.0
    s.arcs_force_bias = bias
    seed_numba(9)
    np.random.seed(9)
    return mc_arcs_numba(pos, exp, 0.05, s), pos


def test_off_by_default_and_inert_at_zero() -> None:
    check("the default is zero, which is off", Settings().arcs_force_bias == 0.0)
    a_s, a_p = run(0.0)
    b_s, b_p = run(0.0)
    check("two runs with it off agree bit for bit", a_s == b_s and np.array_equal(a_p, b_p))


def test_a_bias_reaches_the_kernel() -> None:
    """A bias above zero has to change the trajectory, or it is not wired up.

    Whether it helps is measured on real blocks in `playground/force_bias_sweep.py`, not here.
    On a two hundred anchor fixture the two arms land within noise of each other, and the sign of
    that difference flipped when an unrelated 1e-14 change altered the arithmetic, so a unit test
    asserting an improvement would be asserting noise.
    """
    off_s, off_p = run(0.0)
    on_s, on_p = run(0.5)
    check(
        "a bias above zero changes the run",
        not np.array_equal(off_p, on_p),
        f"energy {off_s:.2f} off, {on_s:.2f} on",
    )


def main() -> int:
    print("force biased proposal checks\n")
    test_score_and_each_branch()
    test_a_real_sized_block()
    test_off_by_default_and_inert_at_zero()
    test_a_bias_reaches_the_kernel()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
