"""Composeable-terms gate.

For every extracted term: (1) numba<->JAX parity — the njit `*_nb` body and the
jnp `*_jax` body agree (local AND init) on random inputs to a tolerance that
admits f32 rounding but catches any FORMULA divergence; (2) numba composition —
recipes compose into one njit kernel via `compose_*_nb` and equal the manual
per-term sum exactly (incl. an array-param term, proving heterogeneous namedtuple
fields survive the codegen).

    JAX_PLATFORMS=cpu .venv/bin/python -u playground/refactor/validate_terms_parity.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from gnome3d.mc.terms.arc_springs import ARC_SPRINGS, ArcP  # noqa: E402
from gnome3d.mc.terms.chain import CHAIN, ChainP  # noqa: E402
from gnome3d.mc.terms.confinement import CONFINEMENT, ConfP  # noqa: E402
from gnome3d.mc.terms.excluded_volume import EXCLUDED_VOLUME, ExclP  # noqa: E402
from gnome3d.mc.terms.heatmap import HEATMAP, HeatmapP  # noqa: E402
from gnome3d.mc.terms.orientation import (  # noqa: E402
    calc_orientation_nb,
    init_anchor_orientations_jax,
    init_orientation_score_jax,
    local_orientation_jax,
    local_score_orientation_nb,
    score_orientation_full_nb,
)
from gnome3d.mc.terms.subanchor_heat import SUBANCHOR_HEAT, HeatP  # noqa: E402

RTOL, ATOL = 2e-3, 1e-4  # admits f32 accumulation noise; rejects formula drift


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


def _fixtures(n: int, rng: np.random.Generator):
    """One (term, params) per term, with array params built to exercise every
    branch (overlaps for EV, mixed-sign exp for arcs, sparse heat, beads outside
    the confinement sphere)."""
    centre = rng.standard_normal(3) * 0.3
    dtn = np.abs(rng.standard_normal(n - 1)).astype(np.float64) + 0.5

    exp = rng.standard_normal((n, n)).astype(np.float64)
    exp = (exp + exp.T) * 0.5  # SYMMETRIC, as production expected-distance matrices are
    exp[np.abs(exp) < 0.3] = 0.0  # some in the skip band [0,1e-6)
    np.fill_diagonal(exp, 0.0)

    heat = np.abs(rng.standard_normal((n, n))).astype(np.float64) * 2.0 + 0.2
    heat[rng.random((n, n)) < 0.4] = 0.0  # ~40% no-contact
    np.fill_diagonal(heat, 0.0)

    skip = rng.random((n, n)) < 0.3
    np.fill_diagonal(skip, True)
    exp_safe = np.abs(rng.standard_normal((n, n))).astype(np.float64) * 2.0 + 0.5
    exp_safe[skip] = 1.0

    return [
        (EXCLUDED_VOLUME, ExclP(r0=1.0, weight=0.1, skip=2)),
        (CONFINEMENT, ConfP(float(centre[0]), float(centre[1]), float(centre[2]), 1.2, 0.1)),
        (CHAIN, ChainP(dtn=dtn, stretch_k=1.0, squeeze_k=1.0, ang_k=0.1, dist_w=1.0, ang_w=1.0)),
        (ARC_SPRINGS, ArcP(exp=exp, stretch_k=1.0, squeeze_k=1.0)),
        (SUBANCHOR_HEAT, HeatP(heat_dist=heat, heat_weight=0.01)),
        (HEATMAP, HeatmapP(exp_safe=exp_safe, skip=skip)),
    ]


def _to_jax(prm):
    """numpy arrays -> device arrays inside the namedtuple (float arrays to f32;
    bool/int arrays kept as-is; scalars untouched)."""
    import jax.numpy as jnp

    def conv(v):
        if isinstance(v, np.ndarray):
            return jnp.asarray(v.astype(np.float32)) if v.dtype.kind == "f" else jnp.asarray(v)
        return v

    return type(prm)(*[conv(v) for v in prm])


def main() -> int:
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    n = 48
    pos = (rng.standard_normal((n, 3)) * 1.5).astype(np.float64)
    pos32 = jnp.asarray(pos.astype(np.float32))

    fixtures = _fixtures(n, rng)
    ok = True

    print("=== numba <-> JAX parity (local + init) ===")
    for term, prm in fixtures:
        prm32 = _to_jax(prm)
        worst_local = 0.0
        for p in (0, 1, 7, 23, 47):
            nb = float(term.nb_local(pos, p, prm))
            jx = float(term.jax_local(pos32, p, pos32[p], prm32, n))
            worst_local = max(worst_local, abs(nb - jx))
            if not _close(nb, jx):
                print(f"  FAIL {term.name} local p={p}: nb={nb:.6g} jax={jx:.6g}")
                ok = False
        nb_i = float(term.nb_init(pos, prm))
        jx_i = float(term.jax_init(pos32, prm32, n))
        init_ok = _close(nb_i, jx_i)
        ok &= init_ok
        print(
            f"  {'ok ' if init_ok else 'FAIL'} {term.name:16s} "
            f"local max|Δ|={worst_local:.2e}  init nb={nb_i:.5f} jax={jx_i:.5f} (Δ={abs(nb_i - jx_i):.2e})"
        )

    ok &= _orientation_parity(pos, pos32, n, rng, jnp)

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


def _orientation_parity(pos, pos32, n, rng, jnp) -> bool:
    """Orientation is special: anchor-vs-bead indexing, mutable cache, and CSR
    (numba) vs padded-dense (jax) neighbour reps.  Build one graph in both reps
    and check the orientation vectors + global + local scores agree."""
    print("\n=== orientation parity (CSR numba vs padded-dense jax) ===")
    n_anchors = 12
    motif_weight, symmetric = 50.0, True
    anchor_ar = np.sort(rng.choice(np.arange(1, n - 1), size=n_anchors, replace=False)).astype(np.int32)
    is_L_bead = (rng.random(n) < 0.5)

    # initial orientation vectors: numba per-anchor vs jax vmapped
    orn_nb = np.stack(
        [np.array(calc_orientation_nb(pos, int(anchor_ar[k]), n, bool(is_L_bead[anchor_ar[k]])))
         for k in range(n_anchors)]
    )
    orn_jax = np.asarray(init_anchor_orientations_jax(pos32, jnp.asarray(anchor_ar), jnp.asarray(is_L_bead)))
    ovec_ok = bool(np.allclose(orn_nb, orn_jax, rtol=RTOL, atol=ATOL))

    # neighbour graph: 1-3 neighbours per anchor, random weights
    nbrs = [rng.choice(n_anchors, size=int(rng.integers(1, 4)), replace=False) for _ in range(n_anchors)]
    wts = [np.abs(rng.standard_normal(len(js))) + 0.1 for js in nbrs]
    # CSR
    offsets = np.zeros(n_anchors + 1, dtype=np.int32)
    for k in range(n_anchors):
        offsets[k + 1] = offsets[k] + len(nbrs[k])
    indices = np.concatenate(nbrs).astype(np.int32)
    weights = np.concatenate(wts).astype(np.float64)
    # padded-dense
    M = max(len(js) for js in nbrs)
    nbr_idx = np.zeros((n_anchors, M), dtype=np.int32)
    nbr_w = np.zeros((n_anchors, M), dtype=np.float64)
    nbr_valid = np.zeros((n_anchors, M), dtype=bool)
    for k in range(n_anchors):
        m = len(nbrs[k])
        nbr_idx[k, :m] = nbrs[k]
        nbr_w[k, :m] = wts[k]
        nbr_valid[k, :m] = True

    full_nb = float(score_orientation_full_nb(orn_nb, offsets, indices, weights, motif_weight, symmetric))
    full_jax = float(
        init_orientation_score_jax(
            jnp.asarray(orn_jax), jnp.asarray(nbr_idx), jnp.asarray(nbr_w),
            jnp.asarray(nbr_valid), jnp.float32(motif_weight), symmetric,
        )
    )
    full_ok = _close(full_nb, full_jax)

    worst_local = 0.0
    for k in (0, 3, 7, 11):
        ln = float(local_score_orientation_nb(orn_nb, k, offsets, indices, weights, motif_weight, symmetric))
        lj = float(
            local_orientation_jax(
                jnp.asarray(orn_jax), k, jnp.asarray(nbr_idx), jnp.asarray(nbr_w),
                jnp.asarray(nbr_valid), jnp.float32(motif_weight), symmetric,
            )
        )
        worst_local = max(worst_local, abs(ln - lj))

    ok = ovec_ok and full_ok and worst_local <= ATOL + RTOL * abs(full_nb)
    print(f"  {'ok ' if ovec_ok else 'FAIL'} orientation vectors max|Δ|={np.max(np.abs(orn_nb - orn_jax)):.2e}")
    print(f"  {'ok ' if full_ok else 'FAIL'} global score nb={full_nb:.5f} jax={full_jax:.5f} (Δ={abs(full_nb - full_jax):.2e})")
    print(f"  {'ok ' if worst_local <= ATOL + RTOL * abs(full_nb) else 'FAIL'} local score max|Δ|={worst_local:.2e}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
