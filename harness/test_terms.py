"""Unit checks for the epigenome energy terms.

    python harness/test_terms.py

Three properties per term, which together are what the old scorer harness gave us
for the parity-era terms:

  * a hand-built configuration whose energy is computable in closed form
  * the per-bead local scores sum to the full score, the contract the incremental
    MC update depends on
  * the term is non-negative over random configurations, which the Metropolis rule
    requires because it divides by the running score

Plus the two behavioural checks that a closed form cannot express: the compartment
term actually segregates A from B, and every term is inert when its flag is off.

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
    init_chrom_block_nb,
    init_nuclear_nb,
    local_affinity_nb,
    local_chrom_block_nb,
    local_nuclear_nb,
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
R0C, R0B, EA, EB = 1.5, 2.5, 1.0, 2.0


def test_affinity() -> None:
    print("\n[affinity] compartment blocks + accessibility bridging")
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]], dtype=np.float64)
    cls = np.array([1, 1, -1, -1], dtype=np.int8)
    acc = np.array([0.9, 0.1, 0.5, 0.2], dtype=np.float64)

    exp_c = exp_b = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            d2 = float(((pos[i] - pos[j]) ** 2).sum())
            if cls[i] > 0 and cls[j] > 0:
                exp_c += EA * well(d2, R0C) * norm(4)
            elif cls[i] < 0 and cls[j] < 0:
                exp_c += EB * well(d2, R0C) * norm(4)
            exp_b += acc[i] * acc[j] * well(d2, R0B) * norm(4)

    got_c, got_b = init_affinity_nb(pos, True, cls, R0C, 1.0, EA, EB, True, acc, R0B, 1.0)
    check(
        "closed form",
        abs(got_c - exp_c) < 1e-12 and abs(got_b - exp_b) < 1e-12,
        f"comp={got_c:.6f} brdg={got_b:.6f}",
    )

    p2 = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float64)
    ab, _ = init_affinity_nb(
        p2, True, np.array([1, -1], dtype=np.int8), R0C, 1.0, EA, EB, False, acc[:2], R0B, 1.0
    )
    none, _ = init_affinity_nb(
        p2, True, np.array([0, 0], dtype=np.int8), R0C, 1.0, EA, EB, False, acc[:2], R0B, 1.0
    )
    check("A-B and unassigned pairs contribute zero", ab == 0.0 and none == 0.0)

    near = np.array([[0.0, 0, 0], [1e-9, 0, 0]], dtype=np.float64)
    far = np.array([[0.0, 0, 0], [1000.0, 0, 0]], dtype=np.float64)
    c2 = np.array([1, 1], dtype=np.int8)
    a2 = np.array([1.0, 1.0])
    cn, _ = init_affinity_nb(near, True, c2, R0C, 1.0, EA, EB, False, a2, R0B, 1.0)
    cf, _ = init_affinity_nb(far, True, c2, R0C, 1.0, EA, EB, False, a2, R0B, 1.0)
    check(
        "well: zero at contact, saturates at weight",
        cn < 1e-12 and abs(cf - 2 * EA * norm(2)) < 1e-9,
        f"{cn:.3g} -> {cf:.6f}",
    )

    ok = True
    for _ in range(200):
        q = rng.normal(size=(6, 3)) * rng.uniform(0.01, 50)
        c, b = init_affinity_nb(
            q,
            True,
            rng.integers(-2, 3, 6).astype(np.int8),
            R0C,
            1.0,
            EA,
            EB,
            True,
            rng.random(6),
            R0B,
            1.0,
        )
        ok &= c >= 0.0 and b >= 0.0
    check("non-negative over 200 random configurations", ok)

    q = rng.normal(size=(30, 3)) * 3.0
    qc = rng.integers(-1, 2, 30).astype(np.int8)
    qa = rng.random(30)
    fc, fb = init_affinity_nb(q, True, qc, R0C, 1.0, EA, EB, True, qa, R0B, 1.0)
    sc = sb = 0.0
    for p in range(30):
        a, b = local_affinity_nb(q, p, True, qc, R0C, 1.0, EA, EB, True, qa, R0B, 1.0)
        sc += a
        sb += b
    check("local scores sum to the full score", abs(sc - fc) < 1e-9 and abs(sb - fb) < 1e-9)

    # N-independence: the 1/(N-1) normalisation is what keeps a weight portable.
    per_bead = []
    for n in (20, 200, 2000):
        p = rng.normal(size=(n, 3)) * (n ** (1 / 3))
        c = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
        e, _ = init_affinity_nb(p, True, c, R0C, 1.0, EA, EA, False, np.zeros(n), 1.0, 0.0)
        per_bead.append(e / n)
    spread = max(per_bead) / min(per_bead)
    check("per-bead energy is N-independent", spread < 1.3, "  ".join(f"{v:.3f}" for v in per_bead))


def test_nuclear() -> None:
    print("\n[nuclear] lamina shell + nucleolar pull + chromosome territories")
    R1, R2, W = 10.0, 20.0, 400.0
    mid = 0.5 * (R1 + R2)
    pos = np.array([[mid, 0, 0], [R1, 0, 0], [mid, 0, 0]], dtype=np.float64)
    cls = np.array([-1, -1, 1], dtype=np.int8)
    lam, cen = init_nuclear_nb(pos, cls, np.zeros(3), True, W, False, 0.0, 0, 0, 0, R1, R2)
    check(
        "lamina: zero mid-shell, weight at boundary, A exempt",
        abs(lam - W) < 1e-9 and cen == 0.0,
        f"{lam}",
    )

    out = np.array([[0.0, 0, 0], [1e6, 0, 0]], dtype=np.float64)
    lo, _ = init_nuclear_nb(
        out, np.array([-1, -1], dtype=np.int8), np.zeros(2), True, W, False, 0.0, 0, 0, 0, R1, R2
    )
    check("lamina saturates outside the shell", abs(lo - 2 * W) < 1e-6)

    p2 = np.array([[15.0, 0, 0]], dtype=np.float64)
    _l, c2 = init_nuclear_nb(
        p2, np.array([0], dtype=np.int8), np.array([2.0]), False, 0.0, True, 3.0, 0, 0, 0, R1, R2
    )
    check("central force closed form", abs(c2 - 3.0 * 2.0 * (15.0 - R1) ** 2) < 1e-9, f"{c2}")

    p3 = np.array([[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=np.float64)
    cid = np.array([0, 0, 1], dtype=np.int32)
    kc, wt = 0.3, 1e-4
    exp = sum(
        wt
        * (
            kc * abs(p3[i, 0] - p3[j, 0]) ** 4
            - abs(p3[i, 0] - p3[j, 0]) ** 3
            + abs(p3[i, 0] - p3[j, 0]) ** 2
        )
        for i in range(3)
        for j in range(3)
        if i != j and cid[i] == cid[j]
    )
    got = init_chrom_block_nb(p3, cid, kc, wt)
    check(
        "chromosomal blocks closed form, cross-chr excluded", abs(got - exp) < 1e-15, f"{got:.3e}"
    )

    d = np.linspace(0, 50, 20000)
    check("chromosomal-block polynomial is non-negative", (kc * d**4 - d**3 + d**2).min() >= 0.0)

    q = rng.normal(size=(25, 3)) * 8.0
    qc = rng.integers(-1, 2, 25).astype(np.int8)
    qw = rng.random(25)
    qid = rng.integers(0, 3, 25).astype(np.int32)
    fl, fc = init_nuclear_nb(q, qc, qw, True, W, True, 20.0, 1, 2, 3, R1, R2)
    sl = sc = 0.0
    for p in range(25):
        a, b = local_nuclear_nb(q, p, qc, qw, True, W, True, 20.0, 1, 2, 3, R1, R2)
        sl += a
        sc += b
    fb = init_chrom_block_nb(q, qid, kc, wt)
    sb = sum(local_chrom_block_nb(q, p, qid, kc, wt) for p in range(25))
    check(
        "local scores sum to the full score",
        abs(sl - fl) < 1e-9 and abs(sc - fc) < 1e-9 and abs(sb - fb) < 1e-9,
    )


def test_behaviour() -> None:
    print("\n[behaviour] segregation, and inertness when flags are off")
    n = 60
    dtn = np.full(n - 1, 2.0, dtype=np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[0] = fixed[-1] = True
    comp = np.where((np.arange(n) // 5) % 2 == 0, 1, -1).astype(np.int8)
    access = rng.random(n).astype(np.float32)

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
        mc_smooth_numba(p, dtn, fixed, 0.3, st, None, None, None, None, comp, None)
        return segregation(p.astype(np.float64), comp)

    off, on = run(False), run(True)
    check("compartment term segregates A from B", on > off, f"cross/within {off:.3f} -> {on:.3f}")

    s_off = Settings()
    aff = affinity_params(s_off, "smooth", 2.0, comp, access)
    check(
        "flags off: affinity resolves to inert",
        not aff.any_on and init_affinity_scores(np.zeros((3, 3)), aff) == (0.0, 0.0),
    )

    r = np.random.default_rng(5)
    pa = (r.normal(size=(n, 3)) * 5.0).astype(np.float32)
    pb = pa.copy()
    seed_numba(555)
    sa = mc_smooth_numba(pa, dtn, fixed, 0.3, s_off)
    seed_numba(555)
    sb = mc_smooth_numba(pb, dtn, fixed, 0.3, s_off, None, None, None, None, comp, access)
    check("flags off: passing tracks changes nothing", sa == sb and np.array_equal(pa, pb))

    s_ap = Settings()
    s_ap.use_compartments = True
    s_ap.compartment_apply_to_smooth = False
    s_nt = Settings()
    s_nt.use_compartments = True
    check(
        "apply-flag off and missing-track both stay inert",
        not affinity_params(s_ap, "smooth", 2.0, comp, None).any_on
        and not affinity_params(s_nt, "smooth", 2.0, None, None).any_on,
    )

    q = rng.normal(size=(25, 3)) * 8.0
    ct = build_coarse_terms(Settings(), q, 2.0, comp[:25], None, None, None)
    check(
        "flags off: coarse terms resolve to inert",
        not ct.any_on and init_coarse_scores(q, ct) == (0.0, 0.0, 0.0, 0.0, 0.0),
    )

    s2 = Settings()
    s2.use_compartments = s2.use_lamina = True
    s2.use_central_force = s2.use_chromosomal_blocks = True
    ct2 = build_coarse_terms(
        s2, q, 2.0, comp[:25], None, rng.integers(0, 2, 25).astype(np.int32), rng.random(25)
    )
    expected_r2 = s2.nucleus_packing_factor * 2.0 * (25 ** (1 / 3))
    check(
        "nuclear frame follows the constant-density rule",
        abs(ct2.nuc_R2 - expected_r2) < 1e-9
        and abs(ct2.nuc_R1 - expected_r2 * s2.nucleus_inner_fraction ** (1 / 3)) < 1e-9,
        f"R1={ct2.nuc_R1:.3f} R2={ct2.nuc_R2:.3f}",
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
    acc = r.random(n).astype(np.float32)

    s = Settings()
    s.use_compartments = s.use_bridging = True
    s.compartment_weight, s.bridging_weight = 1.5, 0.8
    aff = affinity_params(s, "smooth", float(dtn.mean()), comp, acc)
    nb = sum(init_affinity_scores(np.ascontiguousarray(pos, dtype=np.float64), aff))

    init_affinity = _build_smooth_kernel(
        500, int(s.exclusion_skip_neighbors), False, False, 1, True
    )[9]
    jx = float(
        init_affinity(
            jnp.asarray(pos[None]),
            jnp.asarray(comp),
            jnp.asarray(acc),
            jnp.float32(aff.comp_r0),
            jnp.float32(aff.comp_weight),
            jnp.float32(aff.comp_ea),
            jnp.float32(aff.comp_eb),
            jnp.float32(aff.brdg_r0),
            jnp.float32(aff.brdg_weight),
            jnp.int32(n),
        )[0]
    )
    rel = abs(jx - nb) / max(abs(nb), 1e-9)
    check("initial affinity energy matches numba", rel < 2e-5, f"rel diff {rel:.1e}")

    s_off = Settings()
    pa = np.asarray(mc_smooth_jax(pos.copy(), dtn, fixed, 0.3, s_off) or pos)
    p1, p2 = pos.copy(), pos.copy()
    mc_smooth_jax(p1, dtn, fixed, 0.3, s_off)
    mc_smooth_jax(p2, dtn, fixed, 0.3, s_off, None, None, None, None, comp, acc)
    check("flags off: JAX ignores the tracks", np.array_equal(np.asarray(p1), np.asarray(p2)))
    del pa


def main() -> int:
    print("epigenome energy-term checks")
    test_affinity()
    test_nuclear()
    test_behaviour()
    test_jax_agrees()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
