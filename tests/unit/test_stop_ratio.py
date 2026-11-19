"""Unit checks for the settable convergence ratio of the arcs MC.

    python tests/unit/test_stop_ratio.py

The arcs MC stops when a round improves the score by less than a relative amount. Smooth and
the IB stage pass 2.0, which is not reached.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def block(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exp = np.full((n, n), -1.0)
    r2 = np.random.default_rng(seed + 1)
    for _ in range(max(1, int(0.004 * n * n / 2))):
        i, j = r2.integers(0, n, 2)
        if i != j:
            v = float(r2.uniform(0.2, 0.6))
            exp[i, j] = exp[j, i] = v
    np.fill_diagonal(exp, 0.0)
    pos = np.ascontiguousarray(rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32))
    return pos, exp


def run(ratio: float | None) -> tuple[float, np.ndarray]:
    pos, exp = block(300, 31)
    s = Settings()
    s.arcs_repulsion_cutoff_factor = 3.0
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.max_temp = 5.0
    s.dt_temp = 0.9999
    s.mc_stop_improvement = 0.999
    # With zero successes the plateau branch cannot fire and the ratio alone ends the run.
    s.mc_stop_successes = 0
    s.mc_stop_steps = 20_000
    if ratio is not None:
        s.mc_stop_ratio_arcs = ratio
    seed_numba(3)
    np.random.seed(3)
    return mc_arcs_numba(pos, exp, 0.05, s), pos


def test_default_matches_the_hardcoded_value() -> None:
    check("the default is the value that was hardcoded", Settings().mc_stop_ratio_arcs == 0.9999)
    a_score, a_pos = run(None)
    b_score, b_pos = run(0.9999)
    check(
        "setting it to the default changes nothing",
        a_score == b_score and np.array_equal(a_pos, b_pos),
        f"{a_score:.6f} vs {b_score:.6f}",
    )


def test_the_knob_reaches_the_kernel() -> None:
    """A ratio that the first round exceeds stops the run there and a tight one does not."""
    import io
    import logging

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    lg = logging.getLogger("gnome3d.mc.numba")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)
    rounds = {}
    try:
        for ratio in (0.5, 0.9999):
            buf.truncate(0)
            buf.seek(0)
            run(ratio)
            rounds[ratio] = len(buf.getvalue().splitlines())
    finally:
        lg.removeHandler(h)
    check(
        "a ratio the first round already exceeds stops there",
        rounds[0.5] == 1,
        f"{rounds[0.5]} rounds",
    )
    check(
        "the production ratio runs on",
        rounds[0.9999] > 10,
        f"{rounds[0.9999]} rounds",
    )


def main() -> int:
    print("arcs convergence ratio checks\n")
    test_default_matches_the_hardcoded_value()
    test_the_knob_reaches_the_kernel()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
