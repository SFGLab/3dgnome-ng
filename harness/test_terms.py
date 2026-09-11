"""Unit checks for the compartment energy term.

    python harness/test_terms.py

Three properties, which together are what the old scorer harness gave us for the
parity-era terms:

  * a hand-built configuration whose energy is computable in closed form
  * the per-bead local scores sum to the full score, the contract the incremental
    MC update depends on
  * the term is non-negative over random configurations, which the Metropolis rule
    requires because it divides by the running score

Plus the two behavioural checks that a closed form cannot express: the compartment
term actually segregates A from B, and it is inert when its flag is off.

"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.common import affinity_params, init_affinity_scores  # noqa: E402
from gnome3d.mc.numba.heatmap import build_coarse_terms, init_coarse_scores  # noqa: E402
from gnome3d.mc.numba.smooth import mc_smooth_numba  # noqa: E402
from gnome3d.mc.numba.terms import (  # noqa: E402
    init_affinity_nb,
    local_affinity_nb,
    seed_numba,
)
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


def well(d2: float, r0: float) -> float:
    return 1.0 - math.exp(-d2 / (2.0 * r0 * r0))


def norm(n: int) -> float:
    """The kernel divides each pair by the partner count. See the doc."""
    return 1.0 / (n - 1) if n > 1 else 1.0


rng = np.random.default_rng(7)
R0C, EA, EB = 1.5, 1.0, 2.0


def test_affinity() -> None:
    print("\n[affinity] compartment blocks")
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]], dtype=np.float64)
    cls = np.array([1, 1, -1, -1], dtype=np.int8)

    exp_c = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            d2 = float(((pos[i] - pos[j]) ** 2).sum())
            if cls[i] > 0 and cls[j] > 0:
                exp_c += EA * well(d2, R0C) * norm(4)
            elif cls[i] < 0 and cls[j] < 0:
                exp_c += EB * well(d2, R0C) * norm(4)

    got_c = init_affinity_nb(pos, True, cls, R0C, 1.0, EA, EB)
    check("closed form", abs(got_c - exp_c) < 1e-12, f"comp={got_c:.6f}")

    p2 = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float64)
    ab = init_affinity_nb(p2, True, np.array([1, -1], dtype=np.int8), R0C, 1.0, EA, EB)
    none = init_affinity_nb(p2, True, np.array([0, 0], dtype=np.int8), R0C, 1.0, EA, EB)
    check("A-B and unassigned pairs contribute zero", ab == 0.0 and none == 0.0)

    near = np.array([[0.0, 0, 0], [1e-9, 0, 0]], dtype=np.float64)
    far = np.array([[0.0, 0, 0], [1000.0, 0, 0]], dtype=np.float64)
    c2 = np.array([1, 1], dtype=np.int8)
    cn = init_affinity_nb(near, True, c2, R0C, 1.0, EA, EB)
    cf = init_affinity_nb(far, True, c2, R0C, 1.0, EA, EB)
    check(
        "well: zero at contact, saturates at weight",
        cn < 1e-12 and abs(cf - 2 * EA * norm(2)) < 1e-9,
        f"{cn:.3g} -> {cf:.6f}",
    )

    ok = True
    for _ in range(200):
        q = rng.normal(size=(6, 3)) * rng.uniform(0.01, 50)
        c = init_affinity_nb(q, True, rng.integers(-2, 3, 6).astype(np.int8), R0C, 1.0, EA, EB)
        ok &= c >= 0.0
    check("non-negative over 200 random configurations", ok)

    q = rng.normal(size=(30, 3)) * 3.0
    qc = rng.integers(-1, 2, 30).astype(np.int8)
    fc = init_affinity_nb(q, True, qc, R0C, 1.0, EA, EB)
    sc = 0.0
    for p in range(30):
        sc += local_affinity_nb(q, p, True, qc, R0C, 1.0, EA, EB)
    check("local scores sum to the full score", abs(sc - fc) < 1e-9)

    # N-independence: the 1/(N-1) normalisation is what keeps a weight portable.
    per_bead = []
    for n in (20, 200, 2000):
        p = rng.normal(size=(n, 3)) * (n ** (1 / 3))
        c = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
        e = init_affinity_nb(p, True, c, R0C, 1.0, EA, EA)
        per_bead.append(e / n)
    spread = max(per_bead) / min(per_bead)
    check("per-bead energy is N-independent", spread < 1.3, "  ".join(f"{v:.3f}" for v in per_bead))


def test_behaviour() -> None:
    print("\n[behaviour] segregation, and inertness when the flag is off")
    n = 60
    dtn = np.full(n - 1, 2.0, dtype=np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[0] = fixed[-1] = True
    comp = np.where((np.arange(n) // 5) % 2 == 0, 1, -1).astype(np.int8)

    def segregation(p: np.ndarray, c: np.ndarray) -> float:
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        same = (c[:, None] == c[None, :]) & ~np.eye(len(c), dtype=bool)
        return float(d[c[:, None] != c[None, :]].mean() / d[same].mean())

    def run(use: bool) -> float:
        st = Settings()
        st.use_compartments = use
        st.compartment_apply_to_smooth = True
        st.compartment_weight = 50.0
        st.compartment_energy_a = st.compartment_energy_b = 1.0
        st.mc_stop_steps_smooth = 5000
        r = np.random.default_rng(3)
        p = (r.normal(size=(n, 3)) * 5.0).astype(np.float32)
        seed_numba(99)
        mc_smooth_numba(p, dtn, fixed, 0.3, st, None, None, None, None, comp)
        return segregation(p.astype(np.float64), comp)

    off, on = run(False), run(True)
    check("compartment term segregates A from B", on > off, f"cross/within {off:.3f} -> {on:.3f}")

    s_off = Settings()
    aff = affinity_params(s_off, "smooth", 2.0, comp)
    check(
        "flag off: affinity resolves to inert",
        not aff.any_on and init_affinity_scores(np.zeros((3, 3)), aff) == 0.0,
    )

    r = np.random.default_rng(5)
    pa = (r.normal(size=(n, 3)) * 5.0).astype(np.float32)
    pb = pa.copy()
    seed_numba(555)
    sa = mc_smooth_numba(pa, dtn, fixed, 0.3, s_off)
    seed_numba(555)
    sb = mc_smooth_numba(pb, dtn, fixed, 0.3, s_off, None, None, None, None, comp)
    check("flag off: passing the track changes nothing", sa == sb and np.array_equal(pa, pb))

    s_ap = Settings()
    s_ap.use_compartments = True
    s_ap.compartment_apply_to_smooth = False
    s_nt = Settings()
    s_nt.use_compartments = True
    check(
        "apply-flag off and missing-track both stay inert",
        not affinity_params(s_ap, "smooth", 2.0, comp).any_on
        and not affinity_params(s_nt, "smooth", 2.0, None).any_on,
    )

    q = rng.normal(size=(25, 3)) * 8.0
    ct = build_coarse_terms(Settings(), q, 2.0, comp[:25])
    check(
        "flag off: coarse terms resolve to inert",
        not ct.any_on and init_coarse_scores(q, ct) == 0.0,
    )


def test_jax_agrees() -> None:
    """The JAX kernel must carry the same affinity energy as numba.

    Skipped when JAX is absent. Tolerance is float32: the JAX path is f32
    throughout while numba is f64, so exact equality is not the bar.
    """
    print("\n[jax] affinity agreement with numba")
    try:
        import jax.numpy as jnp  # noqa: PLC0415

        from gnome3d.mc.jax.smooth import _build_smooth_kernel, mc_smooth_jax  # noqa: PLC0415
    except ImportError:
        print("  SKIP  jax not installed")
        return

    n = 64
    r = np.random.default_rng(4)
    pos = (r.normal(size=(n, 3)) * 4).astype(np.float32)
    dtn = np.full(n - 1, 2.0, dtype=np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[0] = fixed[-1] = True
    comp = np.where((np.arange(n) // 8) % 2 == 0, 1, -1).astype(np.int8)

    s = Settings()
    s.use_compartments = True
    s.compartment_weight = 1.5
    aff = affinity_params(s, "smooth", float(dtn.mean()), comp)
    nb = init_affinity_scores(np.ascontiguousarray(pos, dtype=np.float64), aff)

    init_affinity = _build_smooth_kernel(
        500, int(s.exclusion_skip_neighbors), False, False, 1, True
    )[9]
    jx = float(
        init_affinity(
            jnp.asarray(pos[None]),
            jnp.asarray(comp),
            jnp.float32(aff.comp_r0),
            jnp.float32(aff.comp_weight),
            jnp.float32(aff.comp_ea),
            jnp.float32(aff.comp_eb),
            jnp.int32(n),
        )[0]
    )
    rel = abs(jx - nb) / max(abs(nb), 1e-9)
    check("initial affinity energy matches numba", rel < 2e-5, f"rel diff {rel:.1e}")

    s_off = Settings()
    pa = np.asarray(mc_smooth_jax(pos.copy(), dtn, fixed, 0.3, s_off) or pos)
    p1, p2 = pos.copy(), pos.copy()
    mc_smooth_jax(p1, dtn, fixed, 0.3, s_off)
    mc_smooth_jax(p2, dtn, fixed, 0.3, s_off, None, None, None, None, comp)
    check("flag off: JAX ignores the track", np.array_equal(np.asarray(p1), np.asarray(p2)))
    del pa


def main() -> int:
    print("compartment energy-term checks")
    test_affinity()
    test_behaviour()
    test_jax_agrees()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
