"""Unit checks for the three smooth stage levers against within block overlaps.

    python harness/test_smooth_levers.py

The hard wall rejects any move that adds a non neighbour pair under the excluded volume radius
or deepens one that is there, so the count never rises and, once zero, stays zero. The anchor
cap lets an anchor move but rejects a move taking it further than the cap from where the arcs
put it. The coil start places each gap's subanchors on a random bridge between its anchors at
the bond targets instead of on a straight line. Each is checked on a small chain where the
property can be read off the output, and all three off leave the kernel's draws untouched,
which the parity gate confirms on a real region.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.pipeline.ib.start import coil_start  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def under(pos: np.ndarray, r: float) -> int:
    """Non neighbour pairs closer than r."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    i, j = np.triu_indices(len(pos), k=2)
    return int((d[i, j] < r).sum())


def chain(n: int = 120, gap: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anchors every `gap` beads on a loose random walk, subanchors on the line between them."""
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
    check(
        "all three levers are off by default",
        not s.smooth_hard_wall and s.smooth_anchor_cap == 0.0 and s.smooth_start == "line",
    )


def test_wall_never_raises_the_count() -> None:
    pos, fixed, dtn = chain()
    before = under(pos, 0.7)
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_hard_wall=True))
    after = under(pos, 0.7)
    check(
        "the wall never lets the count under the radius rise",
        after <= before,
        f"{before} -> {after}",
    )
    pos2, fixed2, dtn2 = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos2, dtn2, fixed2, 0.5, settings())
    check(
        "and ends lower than the soft term alone",
        after < under(pos2, 0.7),
        f"wall {after}, soft {under(pos2, 0.7)}",
    )


def test_wall_holds_once_clear() -> None:
    pos, fixed, dtn = chain()
    pos = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(3))
    start = under(pos, 0.7)
    mc_numba.seed_numba(2)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_hard_wall=True))
    check(
        "from a coil start the wall drives the count to zero",
        under(pos, 0.7) == 0,
        f"start {start}, end {under(pos, 0.7)}",
    )


def test_anchor_cap() -> None:
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_numba.seed_numba(4)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings(smooth_anchor_cap=0.5))
    drift = np.linalg.norm(pos[fixed] - home, axis=1)
    check("anchors move under the cap", drift.max() > 0.0, f"max drift {drift.max():.3f}")
    check("but never further than the cap times the bond", drift.max() <= 0.5 * 1.0 + 1e-6)
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_numba.seed_numba(4)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, 0.5, settings())
    check("with the cap at zero anchors do not move", np.array_equal(pos[fixed], home))


def test_coil_start() -> None:
    pos, fixed, dtn = chain()
    out = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(5))
    check("anchors are untouched", np.array_equal(out[fixed], pos[fixed]))
    bonds = np.linalg.norm(np.diff(out.astype(np.float64), axis=0), axis=1)
    inner = ~fixed[:-1] & ~fixed[1:]
    check(
        "subanchor bonds sit within ten percent of their targets",
        np.allclose(bonds[inner], dtn[inner], rtol=0.1),
        f"max relative error {np.max(np.abs(bonds[inner] - dtn[inner]) / dtn[inner]):.2e}",
    )
    check(
        "the coil starts with a quarter or less of the line's pairs under the radius",
        under(out, 0.7) * 4 <= under(pos, 0.7),
        f"coil {under(out, 0.7)}, line {under(pos, 0.7)}",
    )
    again = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(5))
    check("the start is a function of its seed", np.array_equal(out, again))
    check("and not the line", np.linalg.norm(out - pos) > 1.0)


def _jax_available() -> bool:
    try:
        from gnome3d.mc.jax.util import jax_is_available

        return bool(jax_is_available())
    except Exception:
        return False


def test_jax_wall_and_cap() -> None:
    """The JAX kernel carries the same two rules, on the single and the batched path."""
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
    check("JAX: the wall never lets the count rise", after <= before, f"{before} -> {after}")
    check("JAX: and ends lower than the soft term alone", after < soft, f"wall {after}, soft {soft}")
    pos, fixed, dtn = chain()
    home = pos[fixed].copy()
    mc_jax.mc_smooth_jax(pos, dtn, fixed, 0.5, settings(smooth_anchor_cap=0.5))
    drift = np.linalg.norm(pos[fixed] - home, axis=1)
    check("JAX: anchors move under the cap but never past it", 0.0 < drift.max() <= 0.5 + 1e-5, f"max drift {drift.max():.3f}")
    # the batched path, two problems in one launch
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
    check(
        "JAX batch: the wall never lets the count rise in either chain",
        all(e <= b for e, b in zip(ends, starts, strict=True)),
        f"{starts} -> {ends}",
    )


def main() -> int:
    print("smooth lever checks\n")
    test_off_by_default()
    test_wall_never_raises_the_count()
    test_wall_holds_once_clear()
    test_anchor_cap()
    test_coil_start()
    test_jax_wall_and_cap()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
