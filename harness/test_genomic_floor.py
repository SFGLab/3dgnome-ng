"""Unit checks for the genomic excluded volume floor.

    python harness/test_genomic_floor.py

The floor gives each arcless anchor pair its own excluded volume radius, growing with genomic
separation as `scale * (s / 1000)^nu`. Arc pairs and the diagonal get zero, which the term treats
as skip. Three properties of the term, the same three every term in this codebase carries.

  * a hand built configuration whose energy is computable in closed form
  * the per bead local scores sum to the full score, the contract the incremental MC update
    depends on
  * the term is non negative over random configurations, which the Metropolis rule requires

Plus the matrix builder's own contract, the bond scale the two pass stage calibrates on, and the
stripped arc matrix that retires the `1/d` repulsion for arcless pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.mc.numba.terms import init_excl_mat_nb, local_excl_mat_nb, seed_numba  # noqa: E402
from gnome3d.pipeline.ib.floor import (  # noqa: E402
    arcs_without_repulsion,
    bond_scale,
    genomic_floor_matrix,
)
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


def pair(d: float, r0: float, w: float) -> float:
    return w * ((r0 - d) / r0) ** 2 if 0.0 < r0 and d < r0 else 0.0


def test_matrix_builder() -> None:
    print("\n[matrix] the floor matrix from genomic separation and the arc matrix")
    gen = [
        (0, 1000, 500),
        (9_500, 10_500, 10_000),
        (99_500, 100_500, 100_000),
        (999_500, 1_000_500, 1_000_000),
    ]
    exp = np.full((4, 4), -1.0)
    exp[0, 2] = exp[2, 0] = 0.25  # one arc
    np.fill_diagonal(exp, 0.0)
    r0 = genomic_floor_matrix(gen, exp, scale=2.0, nu=0.5)
    check(
        "arcless pair follows scale * (s/1000)^nu",
        abs(r0[0, 1] - 2.0 * (9.5**0.5)) < 1e-9,
        f"{r0[0, 1]:.4f}",
    )
    check("arc pair gets zero", r0[0, 2] == 0.0 and r0[2, 0] == 0.0)
    check("diagonal is zero", float(np.abs(np.diag(r0)).max()) == 0.0)
    check("symmetric", np.array_equal(r0, r0.T))
    check("longer separation, larger floor", r0[0, 3] > r0[0, 1] > 0.0)
    stripped = arcs_without_repulsion(exp)
    check(
        "stripped matrix keeps arcs and zeroes repulsion",
        stripped[0, 2] == 0.25 and stripped[0, 1] == 0.0 and stripped[1, 3] == 0.0,
    )
    check("stripped matrix is a copy", exp[0, 1] == -1.0)


def test_closed_form() -> None:
    print("\n[closed form] three anchors on a line")
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    r0 = np.array([[0.0, 5.0, 2.0], [5.0, 0.0, 4.0], [2.0, 4.0, 0.0]])
    w, skip = 0.7, 0
    want = pair(1.0, 5.0, w) + pair(3.0, 2.0, w) + pair(2.0, 4.0, w)
    got = float(init_excl_mat_nb(pos, r0, w, skip))
    check(
        "full energy matches the hand sum (double counted)",
        abs(got - 2.0 * want) < 1e-12,
        f"{got:.6f} vs {2 * want:.6f}",
    )
    got_skip = float(init_excl_mat_nb(pos, r0, w, 1))
    check("skip_neighbors drops |i-j| <= 1", abs(got_skip - 2.0 * pair(3.0, 2.0, w)) < 1e-12)
    r0z = r0.copy()
    r0z[0, 1] = r0z[1, 0] = 0.0
    check(
        "zero radius means skip, not a division",
        np.isfinite(init_excl_mat_nb(pos, r0z, w, 0))
        and abs(
            float(init_excl_mat_nb(pos, r0z, w, 0)) - 2.0 * (pair(3.0, 2.0, w) + pair(2.0, 4.0, w))
        )
        < 1e-12,
    )


def test_local_sums_to_full() -> None:
    print("\n[local] per bead local scores sum to the full score")
    rng = np.random.default_rng(1)
    for n in (5, 40):
        pos = rng.normal(0, 1.5, (n, 3))
        r0 = rng.uniform(0.5, 3.0, (n, n))
        r0 = (r0 + r0.T) / 2
        np.fill_diagonal(r0, 0.0)
        r0[rng.random((n, n)) < 0.3] = 0.0
        r0 = np.minimum(r0, r0.T)
        full = float(init_excl_mat_nb(pos, r0, 0.5, 1))
        loc = sum(float(local_excl_mat_nb(pos, p, r0, 0.5, 1)) for p in range(n))
        check(
            f"n={n}: sum of locals equals full",
            abs(loc - full) < 1e-9 * max(1.0, full),
            f"{loc:.6f} vs {full:.6f}",
        )


def test_non_negative() -> None:
    print("\n[sign] non negative over random configurations")
    rng = np.random.default_rng(2)
    ok = True
    for _ in range(50):
        n = int(rng.integers(3, 30))
        pos = rng.normal(0, rng.uniform(0.1, 5.0), (n, 3))
        r0 = rng.uniform(0.0, 4.0, (n, n))
        r0 = (r0 + r0.T) / 2
        np.fill_diagonal(r0, 0.0)
        ok &= float(init_excl_mat_nb(pos, r0, 0.3, 1)) >= 0.0
        ok &= all(float(local_excl_mat_nb(pos, p, r0, 0.3, 1)) >= 0.0 for p in range(n))
    check("never negative", ok)


def test_bond_scale() -> None:
    print("\n[bond] the scale the two pass stage calibrates on")
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 3.0, 0], [1.0, 3.0, 2.0]])
    check(
        "median consecutive anchor distance",
        abs(bond_scale(pos) - 2.0) < 1e-12,
        f"{bond_scale(pos):.3f}",
    )
    check("single anchor gives zero", bond_scale(pos[:1]) == 0.0)


def test_arcs_driver() -> None:
    print("\n[arcs] with the floor on, arcless distances grow with separation")
    # eight anchors, no arcs, separations spanning 1 kb to 10 Mb; start collapsed
    mids = [0, 1_000, 10_000, 100_000, 1_000_000, 3_000_000, 6_000_000, 10_000_000]
    gen = [(m, m + 500, m + 250) for m in mids]
    n = len(gen)
    exp = np.full((n, n), -1.0)
    np.fill_diagonal(exp, 0.0)
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.exclusion_weight = 1.0
    s.exclusion_skip_neighbors = 0
    s.arcs_repulsion_cutoff_factor = 3.0
    rng = np.random.default_rng(0)
    start = rng.normal(0, 0.2, (n, 3)).astype(np.float32)

    def realised(pos: np.ndarray) -> tuple[float, float]:
        d = lambda i, j: float(np.linalg.norm(pos[i] - pos[j]))  # noqa: E731
        return d(0, 1), d(0, n - 1)

    floor = genomic_floor_matrix(gen, exp, scale=1.0, nu=0.285)
    seed_numba(3)
    pos_off = start.copy()
    mc_arcs_numba(pos_off, exp, 0.1, s)
    near_off, far_off = realised(pos_off)
    seed_numba(3)
    pos_on = start.copy()
    mc_arcs_numba(pos_on, exp, 0.1, s, floor_mat=floor)
    near_on, far_on = realised(pos_on)
    check(
        "floor on pushes the 10 Mb pair beyond where 1/d left it",
        far_on > 1.5 * far_off,
        f"on {far_on:.2f}, off {far_off:.2f}",
    )
    check(
        "floor on: pairs reach their floors",
        far_on >= 0.9 * floor[0, n - 1] and near_on >= 0.9 * floor[0, 1],
        f"floors {floor[0, 1]:.2f}, {floor[0, n - 1]:.2f}",
    )
    check(
        "floor off: no separation dependence",
        far_off < 1.5 * near_off,
        f"1kb {near_off:.2f}, 10Mb {far_off:.2f}",
    )
    seed_numba(3)
    pos_again = start.copy()
    mc_arcs_numba(pos_again, exp, 0.1, s, floor_mat=None)
    check("floor_mat=None is the plain driver", np.array_equal(pos_again, pos_off))


def _chain_toy() -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Six anchors joined by consecutive arcs at 0.3, every longer pair arcless, started from
    noise. Five arcs in series span 1.5 at most, so an explicit floor scale of 0.06 keeps every
    floor reachable while binding on a converged chain."""
    mids = [0, 2_000, 20_000, 200_000, 2_000_000, 8_000_000]
    gen = [(m, m + 500, m + 250) for m in mids]
    n = len(gen)
    exp = np.full((n, n), -1.0)
    for i in range(n - 1):
        exp[i, i + 1] = exp[i + 1, i] = 0.3
    np.fill_diagonal(exp, 0.0)
    rng = np.random.default_rng(5)
    return gen, exp, rng.normal(0, 0.2, (n, 3)).astype(np.float32)


def _toy_settings(flag: bool) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_arcs = True
    s.exclusion_weight = 0.1
    s.exclusion_skip_neighbors = 1
    s.arcs_repulsion_cutoff_factor = 3.0
    s.steps_arcs = 1
    s.use_genomic_floor = flag
    s.genomic_floor_scale = 0.06
    return s


def _floor_checks(tag: str, on: np.ndarray, floor: np.ndarray, n: int) -> None:
    d = lambda i, j: float(np.linalg.norm(on[i] - on[j]))  # noqa: E731
    arcs = [d(i, i + 1) for i in range(n - 1)]
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]
    ratio = [d(i, j) / floor[i, j] for i, j in pairs]
    check(
        f"{tag}: every arc held near its target",
        max(arcs) < 0.6,
        " ".join(f"{a:.2f}" for a in arcs),
    )
    check(
        f"{tag}: arcless pairs sit at their floors",
        sum(r >= 0.9 for r in ratio) >= 8 and float(np.median(ratio)) >= 0.9 and min(ratio) >= 0.6,
        f"{sum(r >= 0.9 for r in ratio)}/10 at floor, median {np.median(ratio):.2f}, min {min(ratio):.2f}",
    )


def test_stage_two_pass() -> None:
    print("\n[stage] numba: the floor pass keeps the arcs and lifts the arcless pairs")
    from gnome3d.pipeline.ib.arcs import _run, floor_for

    gen, exp, start = _chain_toy()
    n = len(gen)

    def run(flag: bool) -> tuple[np.ndarray, dict[str, object], Settings]:
        s = _toy_settings(flag)
        problem = {
            "anchor_pos": start,
            "exp_dist": exp,
            "step_size": 0.1,
            "settings": s,
            "seed": 11,
            "anchor_genomic": gen,
        }
        return np.asarray(_run(problem)[1]), problem, s

    off, _, _ = run(False)
    on, problem, s = run(True)
    floor = floor_for(problem, off, s)
    assert floor is not None
    check("flag off reproduces itself", np.array_equal(off, run(False)[0]))
    check("flag on changes the structure", not np.array_equal(on, off))
    _floor_checks("numba", on, floor, n)


def test_stage_two_pass_jax() -> None:
    from gnome3d.mc.jax.util import jax_is_available

    if not jax_is_available():
        print("\n[stage jax] skipped, JAX not installed")
        return
    print("\n[stage jax] the batched runner does the same")
    from gnome3d.pipeline.ib.arcs import _batch_run, floor_for

    gen, exp, start = _chain_toy()
    n = len(gen)

    def run(flag: bool) -> tuple[np.ndarray, dict[str, object], Settings]:
        s = _toy_settings(flag)
        s.mc_executor_jax_bucket_shapes = False
        problem = {
            "anchor_pos": start,
            "exp_dist": exp,
            "step_size": 0.1,
            "settings": s,
            "seed": 11,
            "anchor_genomic": gen,
        }
        return np.asarray(_batch_run([problem])[0][1]), problem, s

    off, _, _ = run(False)
    on, problem, s = run(True)
    floor = floor_for(problem, off, s)
    assert floor is not None
    check("flag on changes the structure", not np.array_equal(on, off))
    _floor_checks("jax", on, floor, n)


def test_floor_weight() -> None:
    print("\n[weight] the floor carries its own weight, not the excluded volume term's")
    gen, exp, start = _chain_toy()
    n = len(gen)
    d = lambda p, i, j: float(np.linalg.norm(p[i] - p[j]))  # noqa: E731

    def run(weight: float) -> np.ndarray:
        s = _toy_settings(True)
        s.exclusion_weight = 0.01  # the EV term's weight, which must not be the floor's
        s.genomic_floor_weight = weight
        # floors the arc chain cannot fully reach, so the weight decides how far they win
        s.genomic_floor_scale = 0.2
        problem = {
            "anchor_pos": start,
            "exp_dist": exp,
            "step_size": 0.1,
            "settings": s,
            "seed": 11,
            "anchor_genomic": gen,
        }
        from gnome3d.pipeline.ib.arcs import _run

        return np.asarray(_run(problem)[1])

    weak = run(0.01)
    strong = run(5.0)
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]
    check(
        "a heavier floor lifts arcless pairs further than the EV weight alone would",
        np.median([d(strong, i, j) for i, j in pairs])
        > 1.2 * np.median([d(weak, i, j) for i, j in pairs]),
        f"median arcless d: weight 0.01 {np.median([d(weak, i, j) for i, j in pairs]):.3f}, weight 5 {np.median([d(strong, i, j) for i, j in pairs]):.3f}",
    )


def test_jax_agrees() -> None:
    from gnome3d.mc.jax.util import jax_is_available

    if not jax_is_available():
        print("\n[jax] skipped, JAX not installed")
        return
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp

    from gnome3d.mc.jax.arcs import _build_arcs_kernel

    print("\n[jax] matrix radius energy agrees with numba")
    rng = np.random.default_rng(7)
    n = 24
    pos = rng.normal(0, 1.0, (n, 3))
    r0 = rng.uniform(0.0, 2.5, (n, n))
    r0 = (r0 + r0.T) / 2
    np.fill_diagonal(r0, 0.0)
    r0[rng.random((n, n)) < 0.4] = 0.0
    r0 = np.minimum(r0, r0.T)
    w, skip = 0.5, 1
    want = float(init_excl_mat_nb(pos, r0, w, skip))
    bundle = _build_arcs_kernel(50, skip, True)
    init_excl = bundle[2]
    got = float(
        np.asarray(
            init_excl(
                jnp.asarray(pos[None].astype(np.float32)),
                jnp.asarray(r0.astype(np.float32)),
                jnp.float32(w),
                jnp.int32(n),
            )
        )[0]
    )
    rel = abs(got - want) / max(want, 1e-12)
    check("initial floor energy matches numba", rel < 2e-5, f"rel diff {rel:.1e}")


def main() -> int:
    print("genomic floor checks")
    test_matrix_builder()
    test_closed_form()
    test_local_sums_to_full()
    test_non_negative()
    test_bond_scale()
    test_arcs_driver()
    test_stage_two_pass()
    test_stage_two_pass_jax()
    test_floor_weight()
    test_jax_agrees()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
