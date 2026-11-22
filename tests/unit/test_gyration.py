"""Unit checks for the windowed gyration term of the arcs solver."""

from __future__ import annotations

import numpy as np
import pytest

from gnome3d.mc.numba.arcs_solver import gyration_energy_grad, gyration_windows
from gnome3d.polymer import PolymerLaw

LAW = PolymerLaw(nu=0.34, s0_bp=1000)


def mids_for(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.sort(rng.choice(np.arange(0, 2_000_000, 500), n, replace=False)).astype(np.int64)


def test_windows() -> None:
    """Every scale from the smallest, twice."""
    m = mids_for(200)
    member, count, target = gyration_windows(m, LAW, 100_000)
    assert bool((count >= 4).all()), "every window holds four anchors or more"
    assert bool((target > 0).all()), "targets positive"
    assert bool(np.allclose(np.asarray(member.sum(axis=1)).ravel(), count)), "rows match counts"
    assert member.shape[0] > 40, f"many windows: {member.shape[0]}"


def test_gradient() -> None:
    """Against finite differences."""
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
    assert err < 1e-5, f"gradient within 1e-5 relative: {err:.2e}"
    assert e >= 0.0, "energy non negative"


def test_swell() -> None:
    """A compacted start swells to the law."""
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
    assert bool(np.abs(np.median(ratio) - 1) < 0.05), (
        f"windows at the law within five percent: median {np.median(ratio):.3f}"
    )
    assert bool(np.quantile(np.abs(ratio - 1), 0.9) < 0.15), (
        f"spread of window ratios small: 90% {np.quantile(np.abs(ratio - 1), 0.9):.3f}"
    )


def test_a_smaller_smallest_window_adds_short_windows() -> None:
    m = mids_for(400)
    wide, _count, _target = gyration_windows(m, LAW, 100_000)
    fine, count, target = gyration_windows(m, LAW, 25_000)
    assert fine.shape[0] > wide.shape[0], "the scales under 100 kb add windows"
    spans = np.array([m[row.indices].max() - m[row.indices].min() for row in fine])
    assert spans.min() < 25_000, "a window of the smallest scale spans less than that scale"
    assert bool((count >= 4).all()) and bool((target > 0).all())


def test_solver_reads_the_smallest_window_setting() -> None:
    """Twelve nodes inside 44 kb start at one place. Only windows under 100 kb hold them."""
    from gnome3d.mc.numba.arcs_solver import solve_arcs
    from gnome3d.settings import Settings

    def run(min_bp: int) -> float:
        s = Settings()
        s.polymer = LAW
        s.use_excluded_volume = False
        s.use_confinement = False
        s.background_weight = 0.0
        s.gyration_weight = 1.0
        s.gyration_min_bp = min_bp
        s.arcs_solver_iters = 500
        s.arcs_solver_tol = 1e-12
        rng = np.random.default_rng(0)
        pos = (rng.normal(size=(12, 3)) * 0.01).astype(np.float32)
        mids = (np.arange(12) * 4_000).astype(np.int64)
        _e, out = solve_arcs(pos, np.zeros((12, 12)), s, mids=mids)
        centred = out - out.mean(axis=0)
        return float(np.sqrt((centred**2).sum(axis=1).mean()))

    assert run(0) < 0.05, "with windows from 100 kb the twelve nodes are in no window"
    assert run(25_000) > 0.5, "windows of 25 kb hold them at the size of the law"


def test_the_device_term_is_the_cpu_term() -> None:
    """On positions far from the origin, as the anchors of a chromosome are."""
    pytest.importorskip("jax")
    from gnome3d.mc.jax.arcs_energy import DeviceGyrationEnergy

    m = mids_for(300)
    member, count, target = gyration_windows(m, LAW, 100_000)
    rng = np.random.default_rng(4)
    x = (np.cumsum(rng.standard_normal((300, 3)), axis=0) + 400.0).reshape(-1)
    want, want_grad = gyration_energy_grad(x, member, count, target, 0.7)
    got, got_grad = DeviceGyrationEnergy(member, count, target, 0.7)(x)
    assert got == pytest.approx(want, rel=1e-4), "energy"
    scale = float(np.abs(want_grad).max())
    assert float(np.abs(got_grad - want_grad).max()) < 1e-4 * scale, "gradient"
