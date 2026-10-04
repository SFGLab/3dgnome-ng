"""Unit checks for solving the arcs stage instead of annealing it.

    python harness/test_arcs_solver.py

The arcs landscape is a funnel: ten Monte Carlo starts land within one percent of each other and
temperature is inert. A quasi Newton solver reaches the same minimum on real blocks about thirty
six times faster, with matching energy, ensemble spread and geometry, so the stage is offered as
a solver as well as an annealer.

The energy it minimises has to be the same one `mc_arcs_numba` scores, or the two are not
comparable and nothing else here means anything. That is most of this file: the solver's energy
against the two initialisers the MC builds its score from, and its gradient against finite
differences of itself.

The solver covers the terms production uses, springs, a truncated repulsion and confinement. It
A misspelled solver name must not fall through to the annealer, checked here. Under the batch
executor the solver's energy is evaluated on the JAX device, and that has to agree with the CPU
kernel and reach the same minimum, also checked here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.arcs_solver import arcs_energy_grad, solve_arcs  # noqa: E402
from gnome3d.mc.numba.terms import (  # noqa: E402
    init_arcs_nb,
    init_confine_nb,
    init_excl_nb,
)
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def block(n: int = 90, seed: int = 3) -> tuple[np.ndarray, np.ndarray, Settings]:
    rng = np.random.default_rng(seed)
    exp = np.full((n, n), -1.0)
    r2 = np.random.default_rng(seed + 1)
    for _ in range(max(1, n // 3)):
        i, j = r2.integers(0, n, 2)
        if i != j:
            v = float(r2.uniform(0.3, 0.9))
            exp[i, j] = exp[j, i] = v
    np.fill_diagonal(exp, 0.0)
    pos = np.ascontiguousarray(rng.normal(0.0, 1.5, size=(n, 3)).astype(np.float32))
    s = Settings()
    s.arcs_repulsion_cutoff_factor = 3.0
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    return pos, exp, s


def terms(pos: np.ndarray, exp: np.ndarray, s: Settings):
    """The same derivations `mc_arcs_numba` makes, so both sides score one thing."""
    m = exp > 1e-6
    avg = float(exp[m].mean())
    rep_inv = 1.0 / (float(s.arcs_repulsion_cutoff_factor) * avg)
    n = pos.shape[0]
    cr = float(s.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))
    c = pos.astype(np.float64).mean(0)
    r0 = float(s.exclusion_auto_factor_arcs) * avg if s.exclusion_apply_to_arcs else 0.0
    w = float(s.exclusion_weight) if s.exclusion_apply_to_arcs else 0.0
    return rep_inv, float(c[0]), float(c[1]), float(c[2]), cr, r0, w


def test_energy_is_the_one_the_mc_scores() -> None:
    """Both counting conventions at once: the arc term is unordered, the excluded volume is not."""
    pos, exp, s = block()
    for label, ev in (("without an excluded volume", False), ("with one", True)):
        s.use_excluded_volume = ev
        s.exclusion_apply_to_arcs = ev
        rep_inv, cx, cy, cz, cr, r0, w = terms(pos, exp, s)
        pw = pos.astype(np.float64)
        skip = int(s.exclusion_skip_neighbors)
        want = (
            float(
                init_arcs_nb(
                    pw, exp, float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs), rep_inv
                )
            )
            + float(init_confine_nb(pw, cx, cy, cz, cr, float(s.confinement_weight)))
            + (float(init_excl_nb(pw, r0, w, skip)) if ev else 0.0)
        )
        got, _ = arcs_energy_grad(
            pw.reshape(-1),
            exp,
            float(s.spring_stretch_arcs),
            float(s.spring_squeeze_arcs),
            rep_inv,
            float(s.background_weight),
            cx,
            cy,
            cz,
            cr,
            float(s.confinement_weight),
            r0,
            w,
            skip,
        )
        check(
            f"the solver's energy is the MC's, {label}",
            abs(got - want) / max(abs(want), 1e-9) < 1e-12,
            f"{got:.6f} against {want:.6f}",
        )
    s.use_excluded_volume = False
    s.exclusion_apply_to_arcs = False


def test_gradient_matches_finite_differences() -> None:
    """With the excluded volume on, so its doubled gradient is covered too."""
    pos, exp, s = block(60, 11)
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    rep_inv, cx, cy, cz, cr, r0, w = terms(pos, exp, s)
    args = (
        exp,
        float(s.spring_stretch_arcs),
        float(s.spring_squeeze_arcs),
        rep_inv,
        float(s.background_weight),
        cx,
        cy,
        cz,
        cr,
        float(s.confinement_weight),
        r0,
        w,
        int(s.exclusion_skip_neighbors),
    )
    x = pos.astype(np.float64).reshape(-1)
    _, g = arcs_energy_grad(x, *args)
    h = 1e-6
    worst = 0.0
    for k in range(0, x.size, 17):
        a, b = x.copy(), x.copy()
        a[k] += h
        b[k] -= h
        fd = (arcs_energy_grad(a, *args)[0] - arcs_energy_grad(b, *args)[0]) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(abs(fd), 1e-6))
    check("the gradient matches finite differences", worst < 2e-5, f"worst relative {worst:.2e}")


def test_it_descends() -> None:
    pos, exp, s = block(120, 21)
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    e1, out = solve_arcs(pos, exp, s, iters=200)
    e0, _ = solve_arcs(pos, exp, s, iters=1)
    check("the solver lowers the energy", e1 < e0, f"{e0:,.1f} to {e1:,.1f}")
    check(
        "and returns the shape it was given",
        out.shape == pos.shape and out.dtype == np.float32,
    )


def test_the_tolerance_stops_the_solve_and_zero_leaves_it_alone() -> None:
    """A loose tolerance ends the solve before the cap; at zero the cap is the only stop and
    the result is the one the cap alone gives."""
    pos, exp, s = block(n=120, seed=11)
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.background_weight = 0.1
    s.arcs_solver_tol = 0.0
    e_cap, x_cap = solve_arcs(pos, exp, s, iters=400)
    e_again, x_again = solve_arcs(pos, exp, s, iters=400)
    check("at zero the solve repeats itself", np.array_equal(x_cap, x_again) and e_cap == e_again)
    s.arcs_solver_tol = 1e-2
    e_tol, _ = solve_arcs(pos, exp, s, iters=400)
    check(
        "a loose tolerance stops earlier, at a higher energy",
        e_tol > e_cap * (1.0 + 1e-6),
        f"{e_tol:,.3f} against {e_cap:,.3f}",
    )


def test_loop_weights() -> None:
    """Weights of one are the unweighted energy bit for bit; with weights the solver's energy
    is still the annealer's, term for term, and the device carries the same weighted energy;
    and where two loops compete for one anchor the weighted one wins: an anchor held by a
    strong loop to one partner and a weak loop to another, the partners held apart, sits
    midway without weights and nearer the strong partner with them."""
    pos, exp, s = block(n=120, seed=11)
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.background_weight = 0.1
    rep_inv, cx, cy, cz, cr, r0, wv = terms(pos, exp, s)
    skip = int(s.exclusion_skip_neighbors)
    args = (
        float(s.spring_stretch_arcs),
        float(s.spring_squeeze_arcs),
        rep_inv,
        float(s.background_weight),
        cx,
        cy,
        cz,
        cr,
        float(s.confinement_weight),
        r0,
        wv,
        skip,
    )
    pw = pos.astype(np.float64)
    ones = np.ones(exp.shape, dtype=np.float32)
    e0, g0 = arcs_energy_grad(pw.reshape(-1), exp, *args)
    e1, g1 = arcs_energy_grad(pw.reshape(-1), exp, *args, True, ones)
    check("weights of one are the unweighted energy", e0 == e1 and np.array_equal(g0, g1))
    rng = np.random.default_rng(3)
    w = np.ones(exp.shape, dtype=np.float32)
    arc = exp > 1e-6
    w[arc] = rng.uniform(0.2, 5.0, int(arc.sum())).astype(np.float32)
    w = np.maximum(w, w.T)
    ew, gw = arcs_energy_grad(pw.reshape(-1), exp, *args, True, w)
    want = (
        float(init_arcs_nb(pw, exp, args[0], args[1], rep_inv, args[3], True, w))
        + float(init_confine_nb(pw, cx, cy, cz, cr, float(s.confinement_weight)))
        + float(init_excl_nb(pw, r0, wv, skip))
    )
    check(
        "with weights the solver's energy is still the annealer's",
        abs(ew - want) / max(abs(want), 1e-9) < 1e-12 and ew != e0,
        f"{ew:.6f} against {want:.6f}",
    )
    try:
        import jax  # noqa: F401, PLC0415

        from gnome3d.mc.jax.arcs_energy import DeviceArcsEnergy  # noqa: PLC0415

        ed, gd = DeviceArcsEnergy(exp, args, w)(pw.reshape(-1))
        check(
            "the device carries the weighted energy",
            abs(ed - ew) < 1e-4 * max(1.0, abs(ew))
            and float(np.max(np.abs(gd - gw))) < 1e-3 * (1.0 + float(np.abs(gw).max())),
            f"{ed:,.4f} against {ew:,.4f}",
        )
    except ImportError:
        pass
    n = 3
    exp3 = np.full((n, n), -0.5)
    np.fill_diagonal(exp3, 0.0)
    exp3[0, 1] = exp3[1, 0] = 1.0  # the strong loop
    exp3[0, 2] = exp3[2, 0] = 1.0  # the weak loop
    exp3[1, 2] = exp3[2, 1] = 3.0  # the partners held apart
    start = np.array([[0.0, 0.5, 0.0], [-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    s3 = Settings()
    s3.use_excluded_volume = False
    s3.use_confinement = False
    s3.background_weight = 0.0
    s3.arcs_repulsion_cutoff_factor = 0.0
    w3 = np.ones((n, n), dtype=np.float32)
    w3[0, 1] = w3[1, 0] = 10.0
    _, x_plain = solve_arcs(start.copy(), exp3, s3, iters=500)
    _, x_w = solve_arcs(start.copy(), exp3, s3, iters=500, arc_w=w3)

    def dist(x: np.ndarray, i: int, j: int) -> float:
        return float(np.linalg.norm(x[i] - x[j]))

    check(
        "without weights the anchor sits midway, with them nearer the strong partner",
        abs(dist(x_plain, 0, 1) - dist(x_plain, 0, 2)) < 1e-3
        and dist(x_w, 0, 1) < dist(x_w, 0, 2) - 0.05,
        f"plain {dist(x_plain, 0, 1):.3f} / {dist(x_plain, 0, 2):.3f}, "
        f"weighted {dist(x_w, 0, 1):.3f} / {dist(x_w, 0, 2):.3f}",
    )


def test_the_weight_matrix_follows_the_strength() -> None:
    """At exponent zero there is no matrix; above it an arc pair carries its strength to the
    exponent and every other pair one."""
    from gnome3d.pipeline.coarse.build import arc_weight_matrix  # noqa: PLC0415

    s = Settings()
    mids = [0, 50_000, 120_000, 400_000]
    arcs = [(0, 1, 30), (1, 2, 3), (0, 3, 6)]
    check("exponent zero gives no matrix", arc_weight_matrix(s, mids, arcs) is None)
    s.arc_weight_exponent = 0.5
    law = s.polymer_law()
    w = arc_weight_matrix(s, mids, arcs)
    assert w is not None
    ok = w.shape == (4, 4) and w[2, 3] == 1.0 and w[0, 0] == 1.0
    for i, j, sc in arcs:
        q = law.arc_strength(sc, abs(mids[i] - mids[j]))
        ok = ok and abs(w[i, j] - q**0.5) < 1e-6 and w[i, j] == w[j, i]
    check("an arc pair carries its strength to the exponent, symmetric, others one", ok)


def test_device_energy_matches_the_cpu_kernel() -> None:
    """The device evaluation is the CPU kernel's energy term for term, on a block carrying every
    kind of pair: springs, backgrounds, repulsion, excluded volume and confinement."""
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  skip  JAX not available, the device energy is not checked")
        return
    from gnome3d.mc.jax.arcs_energy import DeviceArcsEnergy

    pos, exp, s = block(n=120, seed=11)
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.background_weight = 0.1
    rng = np.random.default_rng(12)
    for _ in range(60):
        i, j = rng.integers(0, exp.shape[0], 2)
        if i != j and exp[i, j] < 0.0:
            exp[i, j] = exp[j, i] = -float(rng.uniform(0.8, 2.5))
    rep_inv, cx, cy, cz, cr, r0, w = terms(pos, exp, s)
    args = (
        exp,
        float(s.spring_stretch_arcs),
        float(s.spring_squeeze_arcs),
        rep_inv,
        float(s.background_weight),
        cx,
        cy,
        cz,
        cr,
        float(s.confinement_weight),
        r0,
        w,
        int(s.exclusion_skip_neighbors),
    )
    fun = DeviceArcsEnergy(exp, args[1:])
    worst_e = 0.0
    worst_g = 0.0
    for k in range(3):
        x = pos.astype(np.float64).reshape(-1) + rng.normal(0.0, 0.2 * k, size=pos.size)
        e_cpu, g_cpu = arcs_energy_grad(x, *args)
        e_dev, g_dev = fun(x)
        worst_e = max(worst_e, abs(e_dev - e_cpu) / max(abs(e_cpu), 1e-12))
        worst_g = max(
            worst_g, float(np.max(np.abs(g_dev - g_cpu)) / max(np.max(np.abs(g_cpu)), 1e-12))
        )
    check("the device energy is the CPU kernel's", worst_e < 1e-5, f"worst relative {worst_e:.1e}")
    check(
        "the device gradient is the CPU kernel's", worst_g < 1e-4, f"worst relative {worst_g:.1e}"
    )
    e_dev, _ = solve_arcs(pos, exp, s, iters=50, backend="jax")
    e_cpu, _ = solve_arcs(pos, exp, s, iters=50, backend="numba")
    check(
        "the solver on the device reaches the CPU solve's energy",
        abs(e_dev - e_cpu) / max(abs(e_cpu), 1e-12) < 1e-2,
        f"{e_cpu:,.4f} against {e_dev:,.4f}",
    )


def test_off_by_default() -> None:
    check("the stage anneals unless asked otherwise", Settings().arcs_solver == "mc")


def test_an_unknown_name_is_refused() -> None:
    """Falling through to the annealer would run the wrong stage and say nothing."""
    from gnome3d.pipeline.ib.arcs import _run

    pos, exp, s = block(40, 5)
    s.arcs_solver = "lbgfs"  # a plausible typo
    s.steps_arcs = 1
    try:
        _run(
            {
                "anchor_pos": pos,
                "exp_dist": exp,
                "step_size": 0.01,
                "settings": s,
                "seed": 1,
                "anchor_genomic": np.arange(len(pos), dtype=np.int64) * 50_000,
            }  # type: ignore[arg-type]
        )
        ok = False
    except ValueError:
        ok = True
    check("a misspelled solver name is refused, not ignored", ok)


def test_the_batched_runner_solves_on_the_device() -> None:
    """Under the batch executor the solver evaluates its energy on the JAX device, block by
    block on each block's own settings, and lands where the CPU solve lands."""
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  skip  JAX not available, the batched solver is not checked")
        return
    from gnome3d.pipeline.ib.arcs import _batch_run, _run

    pos, exp, s = block(60, 6)
    s.arcs_solver = "lbfgs"
    s.arcs_solver_iters = 60
    # Excluded volume and a short repulsion reach keep the minimum above zero, so the two
    # energies are compared as numbers rather than as noise around zero.
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.arcs_repulsion_cutoff_factor = 1.0
    r2 = np.random.default_rng(7)
    for _ in range(400):
        i, j = r2.integers(0, exp.shape[0], 2)
        if i != j:
            exp[i, j] = exp[j, i] = float(r2.uniform(0.3, 0.9))
    problem = {
        "anchor_pos": pos,
        "exp_dist": exp,
        "step_size": 0.01,
        "settings": s,
        "seed": 1,
        "anchor_genomic": np.arange(len(pos), dtype=np.int64) * 50_000,
    }
    ((e_dev, _),) = _batch_run([problem])  # type: ignore[list-item]
    e_cpu, _ = _run(problem)  # type: ignore[arg-type]
    check(
        "the batch executor solves on the device and reaches the CPU solve's energy",
        abs(e_dev - e_cpu) < 1e-2 * max(abs(e_cpu), 1.0),
        f"{e_cpu:,.4f} against {e_dev:,.4f}",
    )


def main() -> int:
    print("arcs solver checks\n")
    test_energy_is_the_one_the_mc_scores()
    test_gradient_matches_finite_differences()
    test_it_descends()
    test_the_tolerance_stops_the_solve_and_zero_leaves_it_alone()
    test_loop_weights()
    test_the_weight_matrix_follows_the_strength()
    test_device_energy_matches_the_cpu_kernel()
    test_off_by_default()
    test_an_unknown_name_is_refused()
    test_the_batched_runner_solves_on_the_device()
    test_loop_dropout()
    test_factories()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


def test_loop_dropout() -> None:
    """Idea 33: with the setting off every arc is kept and no draw is made; on, a conformation
    keeps each arc with probability q over q plus the scale, reproducibly by its seed."""
    from gnome3d.pipeline.coarse.build import dropout_keep  # noqa: PLC0415

    s = Settings()
    s.polymer_exponent = 0.3
    mids = [i * 20_000 for i in range(400)]
    rng = np.random.default_rng(5)
    # scores 1, 3 and 9 PETs at a law with no arc fit, where the strength is the count itself
    arcs = [
        (int(i), int(i + 1 + rng.integers(1, 5)), int(sc))
        for i, sc in zip(rng.integers(0, 390, 3000), rng.choice([1, 3, 9], 3000))
    ]
    check("off keeps every arc", not s.loop_dropout)
    s.loop_dropout = True
    s.loop_dropout_scale = 1.0
    k1 = dropout_keep(s, mids, arcs, seed=11)
    k2 = dropout_keep(s, mids, arcs, seed=11)
    k3 = dropout_keep(s, mids, arcs, seed=12)
    check("the same seed keeps the same arcs", k1 == k2)
    check("another seed keeps other arcs", k1 != k3)
    for sc, want in ((1, 0.5), (3, 0.75), (9, 0.9)):
        got = np.mean([k for a, k in zip(arcs, k1) if a[2] == sc])
        check(
            f"a {sc} PET loop is kept in {want:.2f} of conformations",
            abs(got - want) < 0.05,
            f"{got:.3f}",
        )
    s.loop_dropout_scale = 3.0
    k4 = dropout_keep(s, mids, arcs, seed=11)
    got = np.mean([k for a, k in zip(arcs, k4) if a[2] == 3])
    check("the scale moves the halfway point", abs(got - 0.5) < 0.05, f"{got:.3f}")


def test_factories() -> None:
    """Idea 35: the factory term on three anchors against its closed form, its gradient against
    finite differences, never below zero, the device against the CPU, and a solve with it pulls
    the active anchors together."""
    from gnome3d.mc.numba.arcs_solver import factory_energy_grad  # noqa: PLC0415

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [5.0, 5.0, 5.0]])
    act = np.array([2.0, 1.0, 0.5, 0.0])
    w, r = 0.7, 2.0
    d = lambda i, j: float(np.linalg.norm(pos[i] - pos[j]))  # noqa: E731
    S = [
        sum(act[j] * np.exp(-d(i, j) / r) for j in range(4) if j != i and act[j] > 0)
        for i in range(4)
    ]
    want = sum(w * act[i] * (np.log1p(act.sum()) - np.log1p(S[i])) for i in range(4) if act[i] > 0)
    e, g = factory_energy_grad(pos.reshape(-1), act, w, r)
    check(
        "the factory energy is its closed form",
        abs(e - want) < 1e-10,
        f"{e:.8f} against {want:.8f}",
    )
    check("the inactive anchor feels nothing", np.all(g[9:12] == 0.0))
    x = pos.reshape(-1).copy()
    num = np.zeros_like(x)
    h = 1e-6
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += h
        xm = x.copy()
        xm[k] -= h
        num[k] = (factory_energy_grad(xp, act, w, r)[0] - factory_energy_grad(xm, act, w, r)[0]) / (
            2 * h
        )
    check(
        "the factory gradient matches finite differences",
        np.max(np.abs(num - g)) < 1e-6,
        f"{np.max(np.abs(num - g)):.2e}",
    )
    rng = np.random.default_rng(2)
    ok = True
    for _ in range(50):
        pr = rng.normal(0, 3, size=(30, 3))
        ar = rng.uniform(0, 2, 30) * (rng.random(30) < 0.6)
        ok &= factory_energy_grad(pr.reshape(-1), ar, w, r)[0] >= 0.0
    check("the term is never below zero", ok)
    try:
        import jax  # noqa: F401, PLC0415

        from gnome3d.mc.jax.arcs_energy import DeviceFactoryEnergy  # noqa: PLC0415

        pr = rng.normal(0, 3, size=(300, 3))
        ar = rng.uniform(0, 2, 300) * (rng.random(300) < 0.6)
        ec, gc = factory_energy_grad(pr.reshape(-1), ar, w, r)
        ed, gd = DeviceFactoryEnergy(ar, w, r)(pr.reshape(-1))
        check(
            "the device carries the factory term",
            abs(ed - ec) / max(abs(ec), 1e-9) < 1e-4
            and np.max(np.abs(gd - gc)) < 1e-3 * (1 + np.max(np.abs(gc))),
            f"{ed:.5f} against {ec:.5f}",
        )
    except ImportError:
        pass
    # a solve: active anchors with no arcs among them end closer with the term than without
    n = 60
    exp = np.full((n, n), -0.5)
    np.fill_diagonal(exp, 0.0)
    for i in range(n - 1):
        exp[i, i + 1] = exp[i + 1, i] = 1.0
    s = Settings()
    s.arcs_repulsion_cutoff_factor = 1.5
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    start = np.ascontiguousarray(rng.normal(0, 4, size=(n, 3)).astype(np.float32))
    active = np.zeros(n, dtype=np.float32)
    active[::6] = 1.0
    _, x0 = solve_arcs(start.copy(), exp, s, iters=300)
    s.factory_weight = 2.0
    s.factory_radius = 2.0
    _, x1 = solve_arcs(start.copy(), exp, s, iters=300, activity=active)
    s.factory_weight = 0.0
    _, x2 = solve_arcs(start.copy(), exp, s, iters=300, activity=active)
    idx = np.where(active > 0)[0]

    def spread(x):
        return float(np.mean([np.linalg.norm(x[i] - x[j]) for i in idx for j in idx if i < j]))

    check("weight zero with an activity leaves the solve alone", np.array_equal(x0, x2))
    check(
        "with the term the active anchors sit closer",
        spread(x1) < spread(x0),
        f"{spread(x1):.2f} against {spread(x0):.2f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
