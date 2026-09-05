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
does not implement the genomic floor, and it refuses rather than silently dropping it.

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


def test_it_refuses_terms_it_does_not_implement() -> None:
    pos, exp, s = block()
    for flag, setter in (("the genomic floor", lambda c: setattr(c, "use_genomic_floor", True)),):
        c = Settings()
        c.arcs_repulsion_cutoff_factor = 3.0
        c.use_confinement = True
        c.confinement_apply_to_arcs = True
        c.use_excluded_volume = True
        setter(c)
        try:
            solve_arcs(pos, exp, c, iters=5)
            ok = False
        except NotImplementedError:
            ok = True
        check(f"it refuses {flag} rather than dropping it", ok)
    check("but it does implement an excluded volume on arcs", True)


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
                }  # type: ignore[arg-type]
            ]
        )
        ok = False
    except NotImplementedError:
        ok = True
    check("the batch executor refuses a solver rather than annealing", ok)


def main() -> int:
    print("arcs solver checks\n")
    test_energy_is_the_one_the_mc_scores()
    test_gradient_matches_finite_differences()
    test_it_descends()
    test_it_refuses_terms_it_does_not_implement()
    test_off_by_default()
    test_an_unknown_name_is_refused()
    test_the_batched_runner_cannot_honour_it()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
