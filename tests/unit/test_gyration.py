"""Unit checks for the windowed gyration term of the arcs solver."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.mc.numba.arcs_solver import gyration_energy_grad, gyration_windows  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
LAW = PolymerLaw(nu=0.34, s0_bp=1000)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def mids_for(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.sort(rng.choice(np.arange(0, 2_000_000, 500), n, replace=False)).astype(np.int64)


def test_windows() -> None:
    print("\n[windows] every scale from the smallest, twice")
    m = mids_for(200)
    member, count, target = gyration_windows(m, LAW, 100_000)
    check("every window holds four anchors or more", bool((count >= 4).all()))
    check("targets positive", bool((target > 0).all()))
    check("rows match counts", bool(np.allclose(np.asarray(member.sum(axis=1)).ravel(), count)))
    check("many windows", member.shape[0] > 40, f"{member.shape[0]}")


def test_gradient() -> None:
    print("\n[gradient] against finite differences")
    m = mids_for(80)
    member, count, target = gyration_windows(m, LAW, 100_000)
    x = np.random.default_rng(2).standard_normal(80 * 3) * 3.0
    e, g = gyration_energy_grad(x, member, count, target, 0.7)
    h = 1e-6
    num = np.zeros_like(x)
    for k in range(len(x)):
        d = np.zeros_like(x)
        d[k] = h
        num[k] = (
            gyration_energy_grad(x + d, member, count, target, 0.7)[0]
            - gyration_energy_grad(x - d, member, count, target, 0.7)[0]
        ) / (2 * h)
    err = float(np.abs(num - g).max() / max(np.abs(num).max(), 1e-12))
    check("gradient within 1e-5 relative", err < 1e-5, f"{err:.2e}")
    check("energy non negative", e >= 0.0)


def test_swell() -> None:
    print("\n[swell] a compacted start swells to the law")
    from scipy.optimize import minimize

    m = mids_for(120)
    member, count, target = gyration_windows(m, LAW, 100_000)
    x0 = np.random.default_rng(3).standard_normal(120 * 3) * 0.2

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        return gyration_energy_grad(x, member, count, target, 1.0)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    pos = res.x.reshape(-1, 3)
    cen = (member @ pos) / count[:, None]
    rg = np.sqrt((member @ (pos * pos).sum(axis=1)) / count - (cen * cen).sum(axis=1))
    ratio = rg / target
    check(
        "windows at the law within five percent",
        bool(np.abs(np.median(ratio) - 1) < 0.05),
        f"median {np.median(ratio):.3f}",
    )
    check(
        "spread of window ratios small",
        bool(np.quantile(np.abs(ratio - 1), 0.9) < 0.15),
        f"90% {np.quantile(np.abs(ratio - 1), 0.9):.3f}",
    )


def main() -> int:
    test_windows()
    test_gradient()
    test_swell()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
