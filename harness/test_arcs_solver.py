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
Two ways it could be asked for and silently not run, both checked here. A misspelled solver name
must not fall through to the annealer, and the batched runner cannot honour a solver at all, so
it has to say so rather than anneal.
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
            np.zeros(pw.shape[0], dtype=np.int8),
            1.0,
            0.0,
            1.0,
            2.0,
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
        np.zeros(pos.shape[0], dtype=np.int8),
        1.0,
        0.0,
        1.0,
        2.0,
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


def test_the_batched_runner_cannot_honour_it() -> None:
    """The batched runner is a JAX annealer with no solver in it, so asking for one there has
    to fail rather than quietly anneal."""
    from gnome3d.pipeline.ib.arcs import _batch_run

    pos, exp, s = block(40, 6)
    s.arcs_solver = "lbfgs"
    try:
        _batch_run(
            [
                {
                    "anchor_pos": pos,
                    "exp_dist": exp,
                    "step_size": 0.01,
                    "settings": s,
                    "seed": 1,
                    "anchor_genomic": np.arange(len(pos), dtype=np.int64) * 50_000,
                }  # type: ignore[arg-type]
            ]
        )
        ok = False
    except NotImplementedError:
        ok = True
    check("the batch executor refuses a solver rather than annealing", ok)


def test_compartment_term() -> None:
    """The compartment affinity in the solver's energy: a well between like anchors, A against
    A at energy_a and B against B at energy_b, no division by count since the solver is a
    descent and a plateau does not move it. Closed form on one pair, then the gradient."""
    print("\n[compartments] the affinity in the arcs energy")
    pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [0, 5.0, 0]])
    exp = np.zeros((3, 3))
    cls = np.array([1, 1, -1], dtype=np.int8)
    common = (exp, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1)
    e_off, _ = arcs_energy_grad(
        pos.reshape(-1), *common, np.zeros(3, dtype=np.int8), 1.0, 0.0, 1.0, 2.0
    )
    e_on, _ = arcs_energy_grad(pos.reshape(-1), *common, cls, 2.0, 0.7, 1.5, 2.5)
    want = 0.7 * 1.5 * (1.0 - np.exp(-(3.0**2) / (2 * 2.0**2)))
    check(
        "only the like pair carries the well",
        abs((e_on - e_off) - want) < 1e-12,
        f"{e_on - e_off:.6f} vs {want:.6f}",
    )
    rng = np.random.default_rng(5)
    pos, exp, s = block(40, 9)
    cls = rng.choice(np.array([1, -1, 0], dtype=np.int8), size=40)
    rep_inv, cx, cy, cz, cr, r0, w = terms(pos, exp, s)
    args = (exp, 1.0, 1.0, rep_inv, 0.1, cx, cy, cz, cr, 0.5, r0, w, 1, cls, 1.2, 0.3, 1.0, 2.0)
    x = pos.astype(np.float64).reshape(-1)
    _, g = arcs_energy_grad(x, *args)
    h = 1e-6
    worst = 0.0
    for k in range(0, x.size, 13):
        a, b = x.copy(), x.copy()
        a[k] += h
        b[k] -= h
        fd = (arcs_energy_grad(a, *args)[0] - arcs_energy_grad(b, *args)[0]) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(abs(fd), 1e-6))
    check("its gradient matches central differences", worst < 2e-5, f"worst relative {worst:.2e}")
    check("off by default", Settings().compartment_apply_to_arcs is False)


def test_solver_draws_like_anchors_together() -> None:
    """Two A anchors far apart and a B anchor between, no arcs. With the term the A pair ends
    nearer than without it."""
    print("\n[compartments] the solver draws like anchors together")
    pos = np.array([[0.0, 0, 0], [6.0, 0, 0], [12.0, 0, 0]], dtype=np.float32)
    exp = np.full((3, 3), -1.0)
    np.fill_diagonal(exp, 0.0)
    s = Settings()
    s.use_confinement = False
    s.arcs_repulsion_cutoff_factor = 0.0
    _, off = solve_arcs(pos, exp, s, iters=200)
    s.use_compartments = True
    s.compartment_apply_to_arcs = True
    s.compartment_radius_arcs = 4.0
    s.compartment_weight = 1.0
    _, on = solve_arcs(pos, exp, s, iters=200, compartment=np.array([1, -1, 1], dtype=np.int8))
    d_off = float(np.linalg.norm(off[0] - off[2]))
    d_on = float(np.linalg.norm(on[0] - on[2]))
    check("the A pair is nearer with the term", d_on < d_off - 0.5, f"{d_on:.2f} vs {d_off:.2f}")


def main() -> int:
    print("arcs solver checks\n")
    test_energy_is_the_one_the_mc_scores()
    test_gradient_matches_finite_differences()
    test_it_descends()
    test_off_by_default()
    test_an_unknown_name_is_refused()
    test_the_batched_runner_cannot_honour_it()
    test_compartment_term()
    test_solver_draws_like_anchors_together()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
