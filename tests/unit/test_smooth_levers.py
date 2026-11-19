"""Unit checks for the hard wall, the anchor cap and the coil start of the smooth stage."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.mc import numba as mc_numba
from gnome3d.pipeline.ib.start import coil_start
from gnome3d.settings import Settings


def under(pos: np.ndarray, r: float) -> int:
    """Returns the number of non neighbour pairs closer than r."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    i, j = np.triu_indices(len(pos), k=2)
    return int((d[i, j] < r).sum())


def chain(n: int = 120, gap: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anchors every `gap` beads on a loose random walk and subanchors on the line between them."""
    rng = np.random.default_rng(seed)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[::gap] = True
    fixed[-1] = True
    ai = np.flatnonzero(fixed)
    pos = np.zeros((n, 3), dtype=np.float32)
    walk = np.cumsum(rng.normal(size=(len(ai), 3)) * 2.0, axis=0)
    pos[ai] = walk
    for a, b in zip(ai[:-1], ai[1:], strict=True):
        for j in range(a + 1, b):
            pos[j] = pos[a] + ((j - a) / (b - a)) * (pos[b] - pos[a])
    dtn = np.ones(n - 1, dtype=np.float32)
    return pos, fixed, dtn


def settings(**kw: object) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 0.1
    s.exclusion_radius_smooth = 0.7
    s.use_confinement = False
    s.use_motif_orientation = False
    s.mc_smooth_chains = 1
    s.max_temp_smooth = 5.0
    s.dt_temp_smooth = 0.9999
    s.mc_stop_steps_smooth = 20000
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_off_by_default() -> None:
    s = Settings()
    assert not s.smooth_hard_wall and s.smooth_anchor_cap == 0.0 and s.smooth_start == "line", (
        "all three levers are off by default"
    )


def test_wall_never_raises_the_count() -> None:
    pos, fixed, dtn = chain()
    before = under(pos, 0.7)
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_hard_wall=True))
    after = under(pos, 0.7)
    assert after <= before, (
        f"the wall never lets the count under the radius rise: {before} -> {after}"
    )
    pos2, fixed2, dtn2 = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos2, dtn2, fixed2, 0.5, settings())
    assert after < under(pos2, 0.7), (
        f"and ends lower than the soft term alone: wall {after}, soft {under(pos2, 0.7)}"
    )


def test_wall_holds_once_clear() -> None:
    pos, fixed, dtn = chain()
    pos = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(3))
    start = under(pos, 0.7)
    mc_numba.seed_numba(2)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_hard_wall=True))
    assert under(pos, 0.7) == 0, (
        f"from a coil start the wall drives the count to zero: start {start}, end {under(pos, 0.7)}"
    )


def test_anchor_cap() -> None:
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_numba.seed_numba(4)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_anchor_cap=0.5))
    drift = np.linalg.norm(pos[fixed] - home, axis=1)
    assert drift.max() > 0.0, f"anchors move under the cap: max drift {drift.max():.3f}"
    assert drift.max() <= 0.5 * 1.0 + 1e-6, "but never further than the cap times the bond"
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_numba.seed_numba(4)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings())
    assert np.array_equal(pos[fixed], home), "with the cap at zero anchors do not move"


def test_coil_start() -> None:
    pos, fixed, dtn = chain()
    out = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(5))
    assert np.array_equal(out[fixed], pos[fixed]), "anchors are untouched"
    bonds = np.linalg.norm(np.diff(out.astype(np.float64), axis=0), axis=1)
    inner = ~fixed[:-1] & ~fixed[1:]
    assert np.allclose(bonds[inner], dtn[inner], rtol=0.1), (
        f"subanchor bonds sit within ten percent of their targets: max relative error {np.max(np.abs(bonds[inner] - dtn[inner]) / dtn[inner]):.2e}"
    )
    assert under(out, 0.7) * 4 <= under(pos, 0.7), (
        f"the coil starts with a quarter or less of the line's pairs under the radius: coil {under(out, 0.7)}, line {under(pos, 0.7)}"
    )
    again = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(5))
    assert np.array_equal(out, again), "the start is a function of its seed"
    assert np.linalg.norm(out - pos) > 1.0, "and not the line"


def _jax_available() -> bool:
    try:
        from gnome3d.mc.jax.util import jax_is_available

        return bool(jax_is_available())
    except Exception:
        return False


def test_jax_wall_and_cap() -> None:
    if not _jax_available():
        print("  skip  JAX not available")
        return
    from gnome3d.mc import jax as mc_jax

    s = settings(smooth_hard_wall=True)
    s.mc_smooth_chains = 1
    pos, fixed, dtn = chain()
    before = under(pos, 0.7)
    mc_jax.mc_smooth_jax(pos, dtn, fixed, 0.5, s)
    after = under(pos, 0.7)
    pos2, fixed2, dtn2 = chain()
    mc_jax.mc_smooth_jax(pos2, dtn2, fixed2, 0.5, settings())
    soft = under(pos2, 0.7)
    assert after <= before, f"JAX: the wall never lets the count rise: {before} -> {after}"
    assert after < soft, f"JAX: and ends lower than the soft term alone: wall {after}, soft {soft}"
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_jax.mc_smooth_jax(pos, dtn, fixed, 0.5, settings(smooth_anchor_cap=0.5))
    drift = np.linalg.norm(pos[fixed] - home, axis=1)
    assert 0.0 < drift.max() <= 0.5 + 1e-5, (
        f"JAX: anchors move under the cap but never past it: max drift {drift.max():.3f}"
    )
    # The batched path, two problems in one launch.
    problems = []
    starts = []
    for seed in (1, 2):
        p_, f_, d_ = chain(seed=seed)
        starts.append(under(p_, 0.7))
        problems.append(
            {
                "pos": p_,
                "dtn": d_,
                "fixed": f_,
                "step_size": 0.5,
                "settings": s,
                "seed": seed,
                "char_orientations": None,
                "anchor_neighbors": None,
                "anchor_neighbor_weights": None,
                "heat_dist": None,
                "compartment": None,
            }
        )
    out = mc_jax.mc_smooth_jax_batch(problems, s)
    ends = [under(np.asarray(o[1]), 0.7) for o in out]
    assert all(e <= b for e, b in zip(ends, starts, strict=True)), (
        f"JAX batch: the wall never lets the count rise in either chain: {starts} -> {ends}"
    )


def test_jax_prefetch() -> None:
    if not _jax_available():
        print("  skip  JAX not available")
        return
    from gnome3d.mc import jax as mc_jax

    pos8, fixed8, dtn8 = chain()
    before = under(pos8, 0.7)
    s8 = settings(smooth_hard_wall=True, smooth_prefetch=8)
    s8.mc_smooth_chains = 1
    mc_jax.mc_smooth_jax(pos8, dtn8, fixed8, 0.5, s8)
    after = under(pos8, 0.7)
    assert after <= before, (
        f"JAX prefetch: the wall still never lets the count rise: {before} -> {after}"
    )
    # The chain has the serial law. Over seeds the final energies agree in the mean.
    means = {}
    for k in (1, 8):
        sk = settings(smooth_hard_wall=True, smooth_prefetch=k)
        probs = []
        for seed in range(1, 7):
            p_, f_, d_ = chain(seed=seed)
            probs.append(
                {
                    "pos": p_,
                    "dtn": d_,
                    "fixed": f_,
                    "step_size": 0.5,
                    "settings": sk,
                    "seed": seed,
                    "char_orientations": None,
                    "anchor_neighbors": None,
                    "anchor_neighbor_weights": None,
                    "heat_dist": None,
                    "compartment": None,
                }  # fmt: skip
            )
        means[k] = float(np.mean([float(o[0]) for o in mc_jax.mc_smooth_jax_batch(probs, sk)]))
    assert abs(means[8] - means[1]) <= 0.5 * means[1], (
        f"JAX prefetch: the mean final energy over six seeds matches the serial chain's: serial {means[1]:.4f}, prefetch 8 {means[8]:.4f}"
    )
    assert np.allclose(pos8[fixed8], chain()[0][fixed8]), "JAX prefetch: anchors stay put"
    problems = []
    for seed in (1, 2):
        p, f, d = chain(seed=seed)
        problems.append(
            {
                "pos": p,
                "dtn": d,
                "fixed": f,
                "step_size": 0.5,
                "settings": s8,
                "char_orientations": None,
                "anchor_neighbors": None,
                "anchor_neighbor_weights": None,
                "heat_dist": None,
                "compartment": None,
                "seed": seed,
            }  # fmt: skip
        )
    out = mc_jax.mc_smooth_jax_batch(problems, s8)
    assert len(out) == 2 and all(np.isfinite(float(o[0])) for o in out), (
        "JAX prefetch: the batched path runs"
    )
