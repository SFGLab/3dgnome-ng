"""Unit checks for annealing the step size alongside the temperature.

    python harness/test_step_decay.py

The step size is fixed for a whole run. cudaMMC shrinks it once per outer round beside the
temperature, and it is the closest comparable code to ours, so this adds the same knob.

A long run needs a floor that cudaMMC does not. It anneals over tens of rounds; our arcs stage
has taken 3,929, and 0.95 to that power is zero, so an unfloored decay would freeze the chain
rather than refine it. The floor is a fraction of the starting step.

Whether it helps is a measurement, not a unit check. What is checked here is that the arithmetic
is right, that the floor holds, and above all that leaving the knob alone changes nothing.
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
from gnome3d.mc.numba.terms import decayed_step_nb  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def test_arithmetic() -> None:
    check(
        "a decay of one leaves the step alone at every round",
        all(decayed_step_nb(0.5, 1.0, 0.1, i) == 0.5 for i in (0, 1, 10, 5000)),
    )
    got = [decayed_step_nb(1.0, 0.9, 0.0, i) for i in range(4)]
    check(
        "the step shrinks geometrically",
        np.allclose(got, [1.0, 0.9, 0.81, 0.729]),
        f"{[round(x, 4) for x in got]}",
    )
    check(
        "the floor holds however long the run is",
        decayed_step_nb(2.0, 0.5, 0.25, 10_000) == 0.5,
        f"got {decayed_step_nb(2.0, 0.5, 0.25, 10_000)}",
    )
    check(
        "round zero is always the starting step",
        decayed_step_nb(0.7, 0.5, 0.1, 0) == 0.7,
    )


def arcs_run(decay: float, floor: float = 0.1) -> tuple[float, np.ndarray]:
    n = 400
    rng = np.random.default_rng(3)
    exp = np.full((n, n), -1.0)
    r2 = np.random.default_rng(4)
    for _ in range(max(1, int(0.004 * n * n / 2))):
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
    s.mc_stop_steps = 500
    s.mc_stop_improvement = 0.995
    s.mc_stop_successes = 1
    s.max_temp = 2.0
    s.mc_step_decay_arcs = decay
    s.mc_step_decay_floor = floor
    seed_numba(11)
    np.random.seed(11)
    return mc_arcs_numba(pos, exp, 0.05, s), pos


def test_off_by_default_and_inert_at_one() -> None:
    """The knob must be off out of the box and change nothing when set to its default."""
    check("the default is one, which is off", Settings().mc_step_decay_arcs == 1.0)
    a_score, a_pos = arcs_run(1.0)
    b_score, b_pos = arcs_run(1.0)
    check(
        "two runs with the knob off agree bit for bit",
        a_score == b_score and np.array_equal(a_pos, b_pos),
    )


def test_a_decay_actually_reaches_the_kernel() -> None:
    """A decay below one has to change the trajectory, or it is not wired up."""
    off_score, off_pos = arcs_run(1.0)
    on_score, on_pos = arcs_run(0.9)
    check(
        "a decay below one changes the run",
        not np.array_equal(off_pos, on_pos),
        f"score {off_score:.4f} off, {on_score:.4f} on",
    )


def main() -> int:
    print("step size annealing checks\n")
    test_arithmetic()
    test_off_by_default_and_inert_at_one()
    test_a_decay_actually_reaches_the_kernel()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
