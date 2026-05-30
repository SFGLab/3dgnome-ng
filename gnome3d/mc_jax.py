"""JAX backend for the smooth-MC + EV + heat + orientation hot path.

Production dryrun profiles (chr22, chr4) show mc_smooth eats 89-96% of total
MC wall time, with the heaviest calls living at N=2000-10000.  The bench in
[playground/bench_jax_smooth_mc.py] shows JAX at f32 is 5-70x faster than
numba on the chain+EV path — JAX wins because xla.vmap + lax.fori_loop fuses
the entire 5000-step annealing AND the O(N) per-step reductions into a
single GPU kernel.

This module implements the smooth-MC "production" energy combo:
  - chain bonds + angles (always)
  - excluded volume (when settings.use_excluded_volume & apply_to_smooth)
  - heat term / subanchor heatmap (when heat_dist is provided)
  - CTCF orientation (when char_orientations is provided)

It does NOT support confinement yet — that's a future port.  The dispatch
gate in [mc.py::mc_smooth] rejects confinement-enabled calls back to numba.

JAX is an optional extras dep; `_ensure_jax()` lazy-imports.  The persistent
compile cache at `~/.cache/gnome3d/jax` makes per-shape compiles a one-time
cost across all runs on a machine.
"""

# NB: no `from __future__ import annotations` — JAX kernels reflect on live
# type objects via decorators.  String-form annotations are fine elsewhere in
# this file but the kernel definitions below are not annotation-sensitive.

import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import F32Array, I32Array, I64Array

if TYPE_CHECKING:
    from .settings import Settings


# ---------------------------------------------------------------------------
# Lazy import + compile-cache setup (thread-safe)
# ---------------------------------------------------------------------------

_JAX_AVAILABLE: bool | None = None  # None = not yet probed
_jax: Any = None
_jnp: Any = None
# Cache key: (n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs)
_kernel_cache: dict[tuple[int, int, bool, bool, int], Any] = {}
# Module-level lock — `ib_workers>1` may have multiple threads racing into
# `_ensure_jax`/`_build_*` simultaneously, causing duplicate banner prints and
# duplicate kernel-build work.
_init_lock = threading.Lock()

# Shape-bucket ladder.  When settings.jax_bucket_shapes is on, every kernel's
# bead count N is padded up to the next bucket so XLA compiles ~one program per
# bucket (8 total) instead of one per distinct region size.  Geometric x2 so
# worst-case padding waste is <2x compute.  N above the top bucket compiles at
# its exact size (rare).
_SHAPE_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
# Separate (finer/smaller) ladders for smooth orientation's anchor count and
# neighbor width — these scale below N, so reusing _SHAPE_BUCKETS would waste a
# lot at small sizes.
_ANCHOR_BUCKETS: tuple[int, ...] = (16, 64, 256, 1024, 4096, 16384)
_NBR_BUCKETS: tuple[int, ...] = (4, 8, 16, 32, 64)
# Tracks which (kind, bucket, signature) kernels have been eagerly precompiled,
# so precompile passes are idempotent across regions / threads.
_precompiled: set[Any] = set()


def _bucket_for(n: int, ladder: tuple[int, ...] = _SHAPE_BUCKETS) -> int:
    """Smallest ladder bucket >= n, or n itself if it exceeds the top bucket."""
    for b in ladder:
        if n <= b:
            return b
    return n


def _ensure_jax() -> bool:
    """Lazy-import JAX.  Returns True on success, False if not installed.
    Idempotent + thread-safe — first caller does the work, others wait."""
    global _JAX_AVAILABLE, _jax, _jnp
    if _JAX_AVAILABLE is not None:
        return _JAX_AVAILABLE
    with _init_lock:
        if _JAX_AVAILABLE is not None:
            return _JAX_AVAILABLE  # another thread won the race
        try:
            import jax  # type: ignore[import-not-found]
            import jax.numpy as jnp  # type: ignore[import-not-found]
        except ImportError:
            _JAX_AVAILABLE = False
            return False
        # f32 is the production dtype — bench showed f64 is 2x slower on
        # consumer GPUs (1/32 throughput) with no quality benefit at these
        # run lengths.  We do NOT enable_x64.
        cache_dir = os.environ.get("GNOME3D_JAX_CACHE", os.path.expanduser("~/.cache/gnome3d/jax"))
        cache_active = False
        try:
            from jax.experimental import compilation_cache  # type: ignore[import-not-found]

            compilation_cache.compilation_cache.set_cache_dir(cache_dir)  # pyright: ignore[reportUnknownMemberType]
            cache_active = True
        except (ImportError, AttributeError):
            pass
        _jax = jax
        _jnp = jnp
        _JAX_AVAILABLE = True
        # Banner — once per process, on stderr so it doesn't mix with CIF stdout.
        try:
            backend: str = str(jax.default_backend())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            _dev = jax.devices()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            devices_str: str = ", ".join(str(d) for d in _dev)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        except Exception:  # noqa: BLE001
            backend = "unknown"
            devices_str = "unknown"
        cache_str = cache_dir if cache_active else "disabled"
        print(
            f"[mc_jax] JAX backend ready: backend={backend} devices=[{devices_str}] "
            f"cache={cache_str}",
            file=__import__("sys").stderr,
            flush=True,
        )
        return True


def is_available() -> bool:
    """Public: True if JAX is importable in the current environment."""
    return _ensure_jax()


# ---------------------------------------------------------------------------
# Kernel construction (cached per kernel signature)
# ---------------------------------------------------------------------------


def _build_smooth_kernel(
    n_steps_per_batch: int,
    excl_skip: int,
    use_heat: bool,
    use_orn: bool,
    max_nbrs: int,
) -> Any:
    """Build (or look up cached) compiled smooth-MC kernel.

    Returns (kernel, init_smooth, init_excl, init_heat, init_orn) — the four
    init functions compute initial scores on-device, vmapped across K chains.

    Static-by-cache-key: n_steps_per_batch, excl_skip, use_heat, use_orn,
    max_nbrs (padding width for the orientation neighbor lists).  JAX further
    shape-specialises on (N, K, n_anchors, n_movable) at runtime — those
    incur per-shape compile cost (cached persistently via
    jax.experimental.compilation_cache).
    """
    cache_key = (n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    # ---- chain energy helpers ----

    def _smooth_len(pa: Any, pb: Any, e: Any, stretch_k: Any, squeeze_k: Any, dist_w: Any) -> Any:
        diff = pa - pb
        d = jnp.sqrt(jnp.sum(diff * diff))
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0, stretch_k, squeeze_k)
        return rel * rel * k * dist_w

    def _smooth_ang(pa: Any, pb: Any, pc: Any, ang_k: Any, ang_w: Any) -> Any:
        v1 = pa - pb
        v2 = pb - pc
        n1 = jnp.sqrt(jnp.sum(v1 * v1))
        n2 = jnp.sqrt(jnp.sum(v2 * v2))
        scale = jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, 1.0)
        cos_a = jnp.sum(v1 * v2) / jnp.maximum(n1 * n2, 1e-30)
        cos_a = jnp.clip(cos_a, -1.0, 1.0)
        ang = 1.0 - (cos_a + 1.0) * 0.5
        return scale * ang * ang * ang * ang_k * ang_w

    def _local_smooth_at(
        pos: Any,
        p_pos: Any,
        p: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        # `n` clips indices into the (possibly bucket-padded) array; `n_active`
        # is the real chain length, so bonds/angles spanning a pad bead (index
        # >= n_active) are masked out.  When unbucketed n_active == n, so this is
        # a no-op.  Pad beads form a contiguous tail, hence a scalar boundary.
        n = pos.shape[0]
        a_pm1 = pos[jnp.maximum(p - 1, 0)]
        bond_L_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n_active - 1)
        bond_L = jnp.where(
            bond_L_ok,
            _smooth_len(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)], stretch_k, squeeze_k, dist_w),
            0.0,
        )
        a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
        bond_R_ok = jnp.logical_and(p >= 0, p < n_active - 1)
        bond_R = jnp.where(
            bond_R_ok,
            _smooth_len(p_pos, a_pp1, dtn[jnp.minimum(p, n - 2)], stretch_k, squeeze_k, dist_w),
            0.0,
        )

        def angle_at(off: int) -> Any:
            i = p + off
            i0 = jnp.clip(i, 0, n - 1)
            i1 = jnp.clip(i + 1, 0, n - 1)
            i2 = jnp.clip(i + 2, 0, n - 1)
            a0 = pos[i0]
            a1 = pos[i1]
            a2 = pos[i2]
            a0 = jnp.where(i == p, p_pos, a0)
            a1 = jnp.where(i + 1 == p, p_pos, a1)
            a2 = jnp.where(i + 2 == p, p_pos, a2)
            valid = jnp.logical_and(i >= 0, i < n_active - 2)
            return jnp.where(valid, _smooth_ang(a0, a1, a2, ang_k, ang_w), 0.0)

        return bond_L + bond_R + angle_at(-2) + angle_at(-1) + angle_at(0)

    # ---- excluded volume helpers ----

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        # Exclude pad beads (idx >= n_active) from the pairwise sum.  Unbucketed
        # n_active == n, so this is a no-op.
        in_range = jnp.logical_and(jnp.abs(idx - p) > excl_skip, idx < n_active)
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    # ---- confinement helper ----
    #
    # Per-bead soft envelope.  Mirrors gnome3d.mc._local_confine_nb:
    #   E(p) = weight * ((|r_p - c| - R) / R)²   if |r_p - c| > R
    #        = 0                                  otherwise
    # Delta factor 1 (single-counted globally).  Always wired into the kernel;
    # weight=0 disables it via XLA constant-folding.

    def _local_confine_at(p_pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any) -> Any:
        dx = p_pos[0] - cx
        dy = p_pos[1] - cy
        dz = p_pos[2] - cz
        r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (r - R) / jnp.maximum(R, 1e-30)
        contrib = weight * rel * rel
        return jnp.where(r > R, contrib, 0.0)

    # ---- heat (subanchor heatmap) helpers ----

    def _local_heat_at(pos: Any, p_pos: Any, p: Any, heat_dist: Any, heat_weight: Any) -> Any:
        """Local heat score for bead p vs all others, evaluated as if pos[p] = p_pos.
        Mirrors gnome3d.mc._local_heat_nb: sum_{i != p, heat_dist[i,p] > 0}
        ((d - heat_dist[i,p]) / heat_dist[i,p])^2 * heat_weight."""
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        exp_d = heat_dist[:, p]  # (N,)
        idx = jnp.arange(n)
        # Skip i == p (the diagonal of heat_dist is zero anyway, but mask
        # explicitly to match numba semantics) and pairs with no contact data
        # (heat_dist < 1e-6).
        active = jnp.logical_and(idx != p, exp_d >= 1e-6)
        exp_d_safe = jnp.maximum(exp_d, 1e-6)
        rel = (d - exp_d_safe) / exp_d_safe
        contrib = rel * rel
        return heat_weight * jnp.sum(jnp.where(active, contrib, 0.0))

    # ---- orientation helpers ----

    def _calc_orientation_at(pos: Any, p: Any, p_pos: Any, ar: Any, is_L: Any) -> Any:
        """Compute the orientation vector for anchor at bead-index `ar`,
        assuming pos[p] is replaced by p_pos.  Returns a normalised (3,) vec.

        Mirrors gnome3d.mc._calc_orientation_nb edge cases:
          - ar == 0:     orn = pos[1]  - pos[0]
          - ar == N-1:   orn = pos[ar] - pos[ar-1]
          - middle:      orn = pos[ar+1] - pos[ar-1]
        Sign-flipped if is_L is True; then L2-normalised."""
        n = pos.shape[0]
        pp1_idx = jnp.minimum(ar + 1, n - 1)
        pm1_idx = jnp.maximum(ar - 1, 0)
        # Substitute p_pos at the right slot if it happens to be one of these
        a_ar = jnp.where(ar == p, p_pos, pos[ar])
        a_pp1 = jnp.where(pp1_idx == p, p_pos, pos[pp1_idx])
        a_pm1 = jnp.where(pm1_idx == p, p_pos, pos[pm1_idx])

        is_first = ar == 0
        is_last = ar == n - 1
        o_first = a_pp1 - a_ar
        o_last = a_ar - a_pm1
        o_mid = a_pp1 - a_pm1
        o = jnp.where(is_first, o_first, jnp.where(is_last, o_last, o_mid))
        o = jnp.where(is_L, -o, o)
        nm = jnp.sqrt(jnp.sum(o * o))
        return jnp.where(nm > 1e-12, o / jnp.maximum(nm, 1e-30), jnp.zeros_like(o))

    def _local_orientation_at(
        anchor_orn: Any,
        k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Local orientation score for anchor k, summed over its (padded)
        neighbors.  Mirrors gnome3d.mc._local_score_orientation_nb."""
        # nbr_idx[k, :] are the neighbor anchor indices (max_nbrs wide, padded
        # with 0 + nbr_valid=False).  nbr_w[k, :] are the per-edge weights.
        neighbors_k = nbr_idx[k]  # (max_nbrs,)
        weights_k = nbr_w[k]  # (max_nbrs,)
        valid_k = nbr_valid[k]  # (max_nbrs,)
        a = anchor_orn[k]  # (3,)
        b = anchor_orn[neighbors_k]  # (max_nbrs, 3)
        b_signed = jnp.where(symmetric, b, -b)
        dot = jnp.sum(a[None, :] * b_signed, axis=1)  # (max_nbrs,)
        ang = 1.0 - (dot + 1.0) * 0.5
        contrib = jnp.where(valid_k, ang * ang * weights_k, 0.0)
        return motif_weight * jnp.sum(contrib)

    # ---- chain (per-batch) body ----

    def chain_batch(
        # state
        pos0: Any,
        ss0: Any,
        se0: Any,
        sh0: Any,
        so0: Any,
        sc0: Any,
        anchor_orn0: Any,
        T0_: Any,
        # static problem data
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        # schedule
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        # RNG
        key: Any,
        # real bead count + real movable count (< padded lengths when bucketed)
        n_active: Any,
        n_movable_active: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        """One batch of `n_steps_per_batch` MC steps for ONE chain.  Returns
        (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, T_f, n_ok)."""
        # `movable` is padded to the bucket; n_movable_active is the real count so
        # the sampler only draws real movable beads (no-op when unbucketed).
        k_p, k_d, k_a = jax.random.split(key, 3)
        idx_picks = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_movable_active)
        ps = movable[idx_picks]
        disps = jax.random.uniform(
            k_d,
            (n_steps_per_batch, 3),
            minval=-step_size,
            maxval=step_size,
            dtype=pos0.dtype,
        )
        accs = jax.random.uniform(k_a, (n_steps_per_batch,), dtype=pos0.dtype)

        def body(i: Any, carry: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = carry
            p = ps[i]
            delta = disps[i]
            u = accs[i]

            score = ss + se + sh + so + sc
            old_p = pos[p]
            new_p = old_p + delta

            # ---- struct (chain bonds + angles) ----
            loc_s_prev = _local_smooth_at(
                pos, old_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            loc_s_curr = _local_smooth_at(
                pos, new_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            ss_new = ss + (loc_s_curr - loc_s_prev)

            # ---- excluded volume ----
            loc_e_prev = _local_excl_at(pos, old_p, p, r0, excl_w, n_active)
            loc_e_curr = _local_excl_at(pos, new_p, p, r0, excl_w, n_active)
            se_new = se + 2.0 * (loc_e_curr - loc_e_prev)

            # ---- heat ----
            if use_heat:
                loc_h_prev = _local_heat_at(pos, old_p, p, heat_dist, heat_weight)
                loc_h_curr = _local_heat_at(pos, new_p, p, heat_dist, heat_weight)
                sh_new = sh + 2.0 * (loc_h_curr - loc_h_prev)
            else:
                sh_new = sh

            # ---- orientation ----
            if use_orn:
                # orn_k = bead_to_anchor_k[p]; if >= 0 this bead is adjacent to
                # an anchor whose orientation depends on p's position.
                orn_k = bead_to_anchor_k[p]  # int, may be -1
                has_orn = orn_k >= 0
                safe_k = jnp.maximum(orn_k, 0)
                # PREV orientation already lives in anchor_orn[safe_k]
                loc_o_prev_raw = _local_orientation_at(
                    anchor_orn,
                    safe_k,
                    nbr_idx,
                    nbr_w,
                    nbr_valid,
                    motif_weight,
                    symmetric,
                )
                loc_o_prev = jnp.where(has_orn, loc_o_prev_raw, 0.0)

                # CURR: recompute anchor's orientation with p moved to new_p
                ar_p = anchor_ar[safe_k]
                is_L_ar = is_L[ar_p]
                new_orn_vec = _calc_orientation_at(pos, p, new_p, ar_p, is_L_ar)
                # Update only that slot in anchor_orn (functional, single scatter)
                anchor_orn_trial = anchor_orn.at[safe_k].set(new_orn_vec)
                loc_o_curr_raw = _local_orientation_at(
                    anchor_orn_trial,
                    safe_k,
                    nbr_idx,
                    nbr_w,
                    nbr_valid,
                    motif_weight,
                    symmetric,
                )
                loc_o_curr = jnp.where(has_orn, loc_o_curr_raw, 0.0)
                so_new = so + 2.0 * (loc_o_curr - loc_o_prev)
            else:
                anchor_orn_trial = anchor_orn
                so_new = so
                has_orn = False
                safe_k = jnp.int32(0)

            # ---- confinement (per-bead, single-counted, delta factor 1) ----
            # When conf_w == 0 the entire contribution folds to zero; XLA
            # eliminates the branch.  Always wired so no new cache key needed.
            loc_c_prev = _local_confine_at(old_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            loc_c_curr = _local_confine_at(new_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            sc_new = sc + (loc_c_curr - loc_c_prev)

            score_new = ss_new + se_new + sh_new + so_new + sc_new

            ok_unc = score_new < score  # smooth uses STRICT less-than
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            sh_next = jnp.where(ok, sh_new, sh)
            so_next = jnp.where(ok, so_new, so)
            sc_next = jnp.where(ok, sc_new, sc)
            if use_orn:
                # Accept = keep anchor_orn_trial; reject = keep anchor_orn.
                # We only modified anchor_orn[safe_k], so equivalently:
                #   anchor_orn_next = anchor_orn_trial if ok else anchor_orn
                anchor_orn_next = jnp.where(ok, anchor_orn_trial, anchor_orn)
            else:
                anchor_orn_next = anchor_orn
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (
                pos_next,
                ss_next,
                se_next,
                sh_next,
                so_next,
                sc_next,
                anchor_orn_next,
                T * dt,
                n_ok_next,
            )

        init = (pos0, ss0, se0, sh0, so0, sc0, anchor_orn0, T0_, jnp.int32(0))
        return jax.lax.fori_loop(0, n_steps_per_batch, body, init)

    # vmap over K chains; problem data and schedule are shared (None).
    # Per-chain: pos, all 5 scores, anchor_orn, key.  T is shared (deterministic).
    in_axes = (
        0,
        0,
        0,
        0,
        0,
        0,  # pos, ss, se, sh, so, sc
        0,  # anchor_orn
        None,  # T0
        None,
        None,  # dtn, movable
        None,  # heat_dist
        None,
        None,  # anchor_ar, bead_to_anchor_k
        None,
        None,
        None,  # nbr_idx, nbr_w, nbr_valid
        None,  # is_L
        None,
        None,
        None,
        None,  # step_size, dt, js, jc
        None,
        None,
        None,
        None,
        None,  # stretch..ang_w
        None,
        None,  # r0, excl_w
        None,  # heat_weight
        None,
        None,  # motif_weight, symmetric
        None,
        None,
        None,
        None,
        None,  # conf_cx, conf_cy, conf_cz, conf_R, conf_w
        0,  # key
        None,  # n_active (shared)
        None,  # n_movable_active (shared)
    )
    out_axes = (0, 0, 0, 0, 0, 0, 0, None, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    @jax.jit
    def kernel(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        keys: Any,
        n_active: Any,
        n_movable_active: Any,
    ) -> Any:
        return batched(
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T,
            dtn,
            movable,
            heat_dist,
            anchor_ar,
            bead_to_anchor_k,
            nbr_idx,
            nbr_w,
            nbr_valid,
            is_L,
            step_size,
            dt,
            js,
            jc,
            stretch_k,
            squeeze_k,
            ang_k,
            dist_w,
            ang_w,
            r0,
            excl_w,
            heat_weight,
            motif_weight,
            symmetric,
            conf_cx,
            conf_cy,
            conf_cz,
            conf_R,
            conf_w,
            keys,
            n_active,
            n_movable_active,
        )

    # ---- full convergence loop, on device ----
    #
    # Wraps the per-batch `batched` kernel with `lax.while_loop`.  Each
    # iteration of the while_loop = one MC batch across all K chains.  The
    # entire annealing runs inside ONE JAX call — no Python sync between
    # batches.  Replaces a Python loop that did one device->host copy per
    # batch (5-10ms × hundreds of batches per smooth call).
    #
    # max_iters is baked in as a static safety cap to prevent runaway loops
    # if convergence never triggers.  At n_steps_per_batch=2000 with
    # max_iters=10000 we cap at 20M MC steps — comfortably above any
    # realistic convergence count.
    _MAX_ITERS: int = 10000

    @jax.jit
    def kernel_full(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T_init: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        n_active: Any,
        n_movable_active: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            _, _, _, _, _, _, _, _, _, iter_i, _, converged = state
            return jnp.logical_and(jnp.logical_not(converged), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, ms_score, iter_i, _, _ = state
            # Derive K per-chain keys deterministically from iter_i
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = batched(
                pos,
                ss,
                se,
                sh,
                so,
                sc,
                anchor_orn,
                T,
                dtn,
                movable,
                heat_dist,
                anchor_ar,
                bead_to_anchor_k,
                nbr_idx,
                nbr_w,
                nbr_valid,
                is_L,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                r0,
                excl_w,
                heat_weight,
                motif_weight,
                symmetric,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                keys,
                n_active,
                n_movable_active,
            )
            score_per_chain = ss + se + sh + so + sc
            best_idx = jnp.argmin(score_per_chain)
            score = score_per_chain[best_idx]
            n_ok_best = n_ok[best_idx]
            plateaued = jnp.logical_and(
                score > stop_improvement * ms_score, n_ok_best < stop_successes
            )
            eps_done = score < score_eps
            converged = jnp.logical_or(plateaued, eps_done)
            return (pos, ss, se, sh, so, sc, anchor_orn, T, score, iter_i + 1, n_ok_best, converged)

        # ms_score init: very large so the first batch never trips the
        # "improvement < threshold" check.  Matches the Python loop's
        # `ms_score = float("inf")` initialiser.
        init_state = (
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T_init,
            jnp.float32(1e30),  # ms_score
            jnp.int32(0),  # iter_i
            jnp.int32(0),  # n_ok_best (filler)
            jnp.bool_(False),  # converged
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        (
            pos_f,
            ss_f,
            se_f,
            sh_f,
            so_f,
            sc_f,
            anchor_orn_f,
            _T_f,
            final_score,
            iter_f,
            _n_ok_best_f,
            converged_f,
        ) = final
        return (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, final_score, iter_f, converged_f)

    # ---- init helpers (one-shot per chain on entry) ----

    def _init_smooth_single(
        pos: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        n = pos.shape[0]

        # SEQUENTIAL (lax.scan) accumulation, matching _init_excl/_init_heat.
        # A tree reduction (jnp.sum) groups differently for a bucket-padded
        # length vs the real length, and that ULP difference gets chaos-amplified
        # by the MC — so the chain init MUST be padding-insensitive.  Scan is:
        # appending the masked-to-zero pad terms never changes the running sum,
        # so bucketed == unbucketed bit-for-bit.  Mask spans a pad bead via
        # n_active (no-op when unbucketed, n_active == n).
        def _bond_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_len(pos[i], pos[i + 1], dtn[i], stretch_k, squeeze_k, dist_w)
            return carry + jnp.where(i + 1 < n_active, val, 0.0), None

        def _angle_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_ang(pos[i], pos[i + 1], pos[i + 2], ang_k, ang_w)
            return carry + jnp.where(i + 2 < n_active, val, 0.0), None

        bonds_total, _ = jax.lax.scan(_bond_body, jnp.float32(0.0), jnp.arange(n - 1))
        angles_total, _ = jax.lax.scan(_angle_body, jnp.float32(0.0), jnp.arange(n - 2))
        return bonds_total + angles_total

    def _init_excl_single(pos: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rel = jnp.maximum(0.0, (r0 - d) / r0)
            contrib = weight * rel * rel
            # Mask pad columns (idx >= n_active); zero the whole row if i is pad.
            in_range = jnp.logical_and(jnp.abs(idx - i) > excl_skip, idx < n_active)
            row = jnp.where(i < n_active, jnp.sum(jnp.where(in_range, contrib, 0.0)), 0.0)
            return carry + row, None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def _init_heat_single(pos: Any, heat_dist: Any, heat_weight: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            exp_d = heat_dist[:, i]
            active = jnp.logical_and(idx != i, exp_d >= 1e-6)
            exp_d_safe = jnp.maximum(exp_d, 1e-6)
            rel = (d - exp_d_safe) / exp_d_safe
            contrib = rel * rel
            return carry + jnp.sum(jnp.where(active, contrib, 0.0)), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return heat_weight * total

    def _init_confine_single(
        pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any, n_active: Any
    ) -> Any:
        """Sum of per-bead confinement contributions.
        Mirrors gnome3d.mc._init_confine_nb.  Sequential (lax.scan) so trailing
        pad beads (masked by n_active) don't perturb the f32 reduction order."""

        def _body(carry: Any, i: Any) -> tuple[Any, None]:
            c = _local_confine_at(pos[i], cx, cy, cz, R, weight)
            return carry + jnp.where(i < n_active, c, 0.0), None

        total, _ = jax.lax.scan(_body, jnp.float32(0.0), jnp.arange(pos.shape[0]))
        return total

    def _init_anchor_orientations_single(
        pos: Any,
        anchor_ar: Any,
        is_L: Any,
    ) -> Any:
        """Compute (n_anchors, 3) initial orientation vectors from anchor
        positions.  is_L is indexed by bead-index (full N).
        """

        def per_anchor(k_idx: Any) -> Any:
            ar = anchor_ar[k_idx]
            is_L_v = is_L[ar]
            # p == -1 sentinel: never matches any index, so substitution branches
            # all fall through to "use pos[...]".  (jnp.int32 cast for safety.)
            return _calc_orientation_at(
                pos, jnp.int32(-1), jnp.zeros((3,), dtype=pos.dtype), ar, is_L_v
            )

        return jax.vmap(per_anchor)(jnp.arange(anchor_ar.shape[0]))

    def _init_orientation_score_single(
        anchor_orn: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Global orientation score (matches _score_orientation_full_nb).

        SEQUENTIAL (lax.scan) accumulation over anchors so that anchor-bucket
        padding (pad anchors have nbr_valid=False -> contribute exactly 0) does
        not perturb the f32 reduction order.  A tree jnp.sum would group an
        A-padded vs n_anchors-real array differently and chaos-amplify."""

        def per_anchor(k_idx: Any) -> Any:
            return _local_orientation_at(
                anchor_orn,
                k_idx,
                nbr_idx,
                nbr_w,
                nbr_valid,
                motif_weight,
                symmetric,
            )

        # Sum of per-anchor local scores gives the global (each local iterates
        # its own neighbor list; symmetric arcs counted from both endpoints).
        # Matches numba _score_orientation_full_nb: per_anchor returns
        # motif_weight * sum_j w_kj*ang_kj^2, so sum_k = motif_weight * global.
        def _scan_body(carry: Any, k_idx: Any) -> tuple[Any, None]:
            return carry + per_anchor(k_idx), None

        total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), jnp.arange(anchor_orn.shape[0]))
        return total

    init_smooth = jax.jit(
        jax.vmap(
            _init_smooth_single,
            in_axes=(0, None, None, None, None, None, None, None),
        )
    )
    init_excl = jax.jit(jax.vmap(_init_excl_single, in_axes=(0, None, None, None)))
    init_heat = jax.jit(jax.vmap(_init_heat_single, in_axes=(0, None, None)))
    init_confine = jax.jit(
        jax.vmap(
            _init_confine_single,
            in_axes=(0, None, None, None, None, None, None),
        )
    )
    init_anchor_orn = jax.jit(jax.vmap(_init_anchor_orientations_single, in_axes=(0, None, None)))
    init_orn_score = jax.jit(
        jax.vmap(
            _init_orientation_score_single,
            in_axes=(0, None, None, None, None, None),
        )
    )

    bundle = (
        kernel,  # per-batch (kept for diagnostics; unused in prod)
        kernel_full,  # full convergence on device — the production path
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
    )
    _kernel_cache[cache_key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Arcs kernel construction (separate from smooth; different energy + schedule)
# ---------------------------------------------------------------------------


def _build_arcs_kernel(n_steps_per_batch: int, excl_skip: int) -> Any:
    """Build (or look up cached) compiled arcs-MC kernel.

    Arcs MC differs from smooth in three ways:
      1. **Energy**: pairwise springs from `exp_dist_mat` with a repulsion
         branch for negative `exp` entries.  No chain bonds, no angles, no
         heat, no orientation.
      2. **Acceptance**: non-strict (`score_new <= score`) vs smooth's strict.
      3. **Convergence**: an additional `stop_when_ratio_above` clause
         (0.9999 in production) that exits early when improvement stalls.

    Cache key: (n_steps_per_batch, excl_skip).  EV support is always wired
    (excl_w=0 disables it at runtime, constant-folded by XLA).
    """
    cache_key = ("arcs", n_steps_per_batch, excl_skip)
    # _kernel_cache is typed for the smooth case; arcs uses string-prefixed
    # tuple keys to share the same dict without collision.
    if cache_key in _kernel_cache:  # pyright: ignore[reportArgumentType]
        return _kernel_cache[cache_key]  # pyright: ignore[reportArgumentType]

    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    def _local_arcs_at(
        pos: Any,
        p_pos: Any,
        p: Any,
        exp_mat: Any,
        stretch_k: Any,
        squeeze_k: Any,
    ) -> Any:
        """Mirror of gnome3d.mc._local_arcs_nb, with bead p virtually at p_pos.
        Three branches per i:
          - i == p:            contribute 0
          - exp[i,p] < 0:      repulsion 1/d (with d clamped to 1e-10 min)
          - exp[i,p] >= 1e-6:  asymmetric spring (d-e)/e
          - else (in [0, 1e-6)): contribute 0 (no arc, no repulsion)
        """
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        e = exp_mat[:, p]
        idx = jnp.arange(n)
        not_self = idx != p
        is_repulse = jnp.logical_and(not_self, e < 0.0)
        is_spring = jnp.logical_and(not_self, e >= 1e-6)

        d_safe = jnp.maximum(d, 1e-10)
        rep = 1.0 / d_safe

        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0, stretch_k, squeeze_k)
        spring = rel * rel * k

        contrib = jnp.where(is_repulse, rep, jnp.where(is_spring, spring, 0.0))
        return jnp.sum(contrib)

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        # Exclude pad beads (idx >= n_active); no-op when unbucketed (n_active==n).
        in_range = jnp.logical_and(jnp.abs(idx - p) > excl_skip, idx < n_active)
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    def _local_confine_at(p_pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any) -> Any:
        """Per-bead soft envelope; see [mc.py::_local_confine_nb]."""
        dx = p_pos[0] - cx
        dy = p_pos[1] - cy
        dz = p_pos[2] - cz
        r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (r - R) / jnp.maximum(R, 1e-30)
        contrib = weight * rel * rel
        return jnp.where(r > R, contrib, 0.0)

    def _init_arcs(pos: Any, exp_mat: Any, stretch_k: Any, squeeze_k: Any) -> Any:
        """O(N^2) init via row-at-a-time scan, summing only upper triangle
        (i < j) to match gnome3d.mc._init_arcs_nb."""
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            e = exp_mat[:, i]
            above = idx > i
            # Match numba: skip e in (-1e-10, 1e-6).
            is_repulse = jnp.logical_and(above, e <= -1e-10)
            is_spring = jnp.logical_and(above, e >= 1e-6)

            d_safe = jnp.maximum(d, 1e-10)
            rep = 1.0 / d_safe
            e_safe = jnp.maximum(e, 1e-6)
            rel = (d - e_safe) / e_safe
            k = jnp.where(rel >= 0, stretch_k, squeeze_k)
            spring = rel * rel * k

            row = jnp.where(is_repulse, rep, jnp.where(is_spring, spring, 0.0))
            return carry + jnp.sum(row), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def _init_excl(pos: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rel = jnp.maximum(0.0, (r0 - d) / r0)
            contrib = weight * rel * rel
            # Mask pad columns (idx >= n_active); zero the whole row if i is pad.
            in_range = jnp.logical_and(jnp.abs(idx - i) > excl_skip, idx < n_active)
            row = jnp.where(i < n_active, jnp.sum(jnp.where(in_range, contrib, 0.0)), 0.0)
            return carry + row, None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def _init_confine(pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any, n_active: Any) -> Any:
        # Sequential (lax.scan) so trailing pad beads don't perturb f32 order.
        def _body(carry: Any, i: Any) -> tuple[Any, None]:
            c = _local_confine_at(pos[i], cx, cy, cz, R, weight)
            return carry + jnp.where(i < n_active, c, 0.0), None

        total, _ = jax.lax.scan(_body, jnp.float32(0.0), jnp.arange(pos.shape[0]))
        return total

    def chain_batch(
        pos0: Any,
        ss0: Any,
        se0: Any,
        sc0: Any,
        T0_: Any,
        exp_mat: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        r0: Any,
        excl_w: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        key: Any,
        n_active: Any,
    ) -> Any:
        k_p, k_d, k_a = jax.random.split(key, 3)
        # Arcs: all real beads movable (mc.py uses np.arange(n)).  Under bucketing
        # pos0 is padded; n_active (dynamic) restricts moves to real beads so pad
        # beads never move (arc term zeroed via exp_mat=0, EV via idx<n_active).
        ps = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_active)
        disps = jax.random.uniform(
            k_d,
            (n_steps_per_batch, 3),
            minval=-step_size,
            maxval=step_size,
            dtype=pos0.dtype,
        )
        accs = jax.random.uniform(k_a, (n_steps_per_batch,), dtype=pos0.dtype)

        def body(i: Any, carry: Any) -> Any:
            pos, ss, se, sc, T, n_ok = carry
            p = ps[i]
            delta = disps[i]
            u = accs[i]

            score = ss + se + sc
            old_p = pos[p]
            new_p = old_p + delta

            loc_s_prev = _local_arcs_at(pos, old_p, p, exp_mat, stretch_k, squeeze_k)
            loc_s_curr = _local_arcs_at(pos, new_p, p, exp_mat, stretch_k, squeeze_k)
            # struct_delta_factor = 1 for arcs (single-counted)
            ss_new = ss + (loc_s_curr - loc_s_prev)

            loc_e_prev = _local_excl_at(pos, old_p, p, r0, excl_w, n_active)
            loc_e_curr = _local_excl_at(pos, new_p, p, r0, excl_w, n_active)
            se_new = se + 2.0 * (loc_e_curr - loc_e_prev)

            # Confinement: per-bead, delta factor 1.  When conf_w=0 the whole
            # contribution folds to zero and XLA elides the branch.
            loc_c_prev = _local_confine_at(old_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            loc_c_curr = _local_confine_at(new_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            sc_new = sc + (loc_c_curr - loc_c_prev)

            score_new = ss_new + se_new + sc_new

            # Arcs uses NON-strict acceptance: score_new <= score.
            ok_unc = score_new <= score
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            sc_next = jnp.where(ok, sc_new, sc)
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (pos_next, ss_next, se_next, sc_next, T * dt, n_ok_next)

        init = (pos0, ss0, se0, sc0, T0_, jnp.int32(0))
        return jax.lax.fori_loop(0, n_steps_per_batch, body, init)

    in_axes = (
        0,
        0,
        0,
        0,
        None,  # pos, ss, se, sc, T0
        None,  # exp_mat
        None,
        None,
        None,
        None,  # step_size, dt, js, jc
        None,
        None,  # stretch_k, squeeze_k
        None,
        None,  # r0, excl_w
        None,
        None,
        None,
        None,
        None,  # conf_cx..conf_w
        0,  # key
        None,  # n_active (shared)
    )
    out_axes = (0, 0, 0, 0, None, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    _MAX_ITERS: int = 10000

    @jax.jit
    def kernel_full(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sc_k: Any,
        T_init: Any,
        exp_mat: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        r0: Any,
        excl_w: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        stop_when_ratio_above: Any,
        n_active: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            _, _, _, _, _, _, iter_i, _, converged = state
            return jnp.logical_and(jnp.logical_not(converged), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sc, T, ms_score, iter_i, _, _ = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, sc, T, n_ok = batched(
                pos,
                ss,
                se,
                sc,
                T,
                exp_mat,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                r0,
                excl_w,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                keys,
                n_active,
            )
            score_per_chain = ss + se + sc
            best_idx = jnp.argmin(score_per_chain)
            score = score_per_chain[best_idx]
            n_ok_best = n_ok[best_idx]

            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(
                score > stop_improvement * ms_score, n_ok_best < stop_successes
            )
            eps_done = score < score_eps
            ratio_done = ratio > stop_when_ratio_above
            converged = jnp.logical_or(jnp.logical_or(plateaued, eps_done), ratio_done)
            return (pos, ss, se, sc, T, score, iter_i + 1, n_ok_best, converged)

        init_state = (
            pos_k,
            ss_k,
            se_k,
            sc_k,
            T_init,
            jnp.float32(1e30),
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(False),
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        pos_f, ss_f, se_f, sc_f, _T_f, final_score, iter_f, _, converged_f = final
        return pos_f, ss_f, se_f, sc_f, final_score, iter_f, converged_f

    init_arcs = jax.jit(jax.vmap(_init_arcs, in_axes=(0, None, None, None)))
    init_excl_arcs = jax.jit(jax.vmap(_init_excl, in_axes=(0, None, None, None)))
    init_confine_arcs = jax.jit(
        jax.vmap(_init_confine, in_axes=(0, None, None, None, None, None, None))
    )

    bundle = (kernel_full, init_arcs, init_excl_arcs, init_confine_arcs)
    _kernel_cache[cache_key] = bundle  # pyright: ignore[reportArgumentType]
    return bundle


def _precompile_arcs(settings: "Settings") -> None:
    """Eagerly compile the arcs kernel for every shape bucket.  Arcs is fully
    N-bucketed (all kernel_full inputs are (K,B,3)/(B,B)/scalar), so one program
    per bucket covers every region.  Same .lower(ShapeDtypeStruct).compile()
    trick as _precompile_heatmap.  Idempotent."""
    if not _ensure_jax():
        return
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp
    sys = __import__("sys")

    K = 1  # arcs has no multichain in production
    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps = int(settings.mc_stop_steps)
    sig = ("arcs", n_steps, excl_skip, K)
    with _init_lock:
        if sig in _precompiled:
            return
        kernel_full, init_arcs, init_excl, init_confine = _build_arcs_kernel(n_steps, excl_skip)
        sds = jax.ShapeDtypeStruct
        f32 = jnp.float32
        key = jax.random.PRNGKey(0)
        T_a = f32(settings.max_temp)
        dt_a = f32(settings.dt_temp)
        js_a = f32(settings.jump_scale)
        jc_a = f32(settings.jump_coef)
        impr_a = f32(settings.mc_stop_improvement)
        succ_a = jnp.int32(settings.mc_stop_successes)
        t0 = __import__("time").perf_counter()
        for b in _SHAPE_BUCKETS:
            pos_a = sds((K, b, 3), np.float32)
            kvec_a = sds((K,), np.float32)
            exp_a = sds((b, b), np.float32)
            try:
                init_arcs.lower(pos_a, exp_a, f32(1.0), f32(1.0)).compile()
                init_excl.lower(pos_a, f32(1.0), f32(0.0), jnp.int32(b)).compile()
                init_confine.lower(
                    pos_a, f32(0.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0), jnp.int32(b)
                ).compile()
                kernel_full.lower(
                    pos_a, kvec_a, kvec_a, kvec_a,  # pos, ss, se, sc
                    T_a, exp_a,
                    f32(0.1),  # step_size (per-call: value irrelevant)
                    dt_a, js_a, jc_a,
                    f32(1.0), f32(1.0),  # stretch_k, squeeze_k
                    f32(1.0), f32(0.0),  # r0, excl_w
                    f32(0.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0),  # conf_cx..conf_w
                    key, impr_a, succ_a,
                    f32(1e-5),  # score_eps (matches mc_arcs_jax hardcode)
                    f32(0.9999),  # stop_when_ratio_above
                    jnp.int32(b),  # n_active
                ).compile()
            except Exception as e:  # noqa: BLE001 - precompile is best-effort
                print(f"[mc_jax] precompile arcs bucket {b} skipped: {e}", file=sys.stderr)
        _precompiled.add(sig)
        dt = __import__("time").perf_counter() - t0
        print(
            f"[mc_jax] precompiled arcs kernel for {len(_SHAPE_BUCKETS)} buckets in {dt:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def mc_arcs_jax(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
    label: str = "",
    verbose: bool = False,
) -> float:
    """JAX backend for mc_arcs.  Supports arc springs + EV + (optional)
    confinement.  Same contract as [mc.mc_arcs].

    Mutates `pos` in place (writes the best-chain final positions back) and
    returns the best chain's final score.
    """
    if not _ensure_jax():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d-ng[jax]` or set mc_backend='numba'."
        )
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    n: int = pos.shape[0]
    if n <= 1:
        return 0.0

    K: int = 1  # arcs has no multichain in production today
    n_steps_per_batch: int = int(settings.mc_stop_steps)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    excl_r0: float
    if use_excl:
        excl_r0 = float(settings.exclusion_radius_arcs)
        if excl_r0 <= 0.0:
            pos_mask = np.asarray(exp_dist_mat) > 1e-6
            factor = float(settings.exclusion_auto_factor_arcs)
            excl_r0 = (
                factor * float(np.asarray(exp_dist_mat)[pos_mask].mean()) if pos_mask.any() else 1.0
            )
    else:
        excl_r0 = 1.0

    # ---- confinement setup (always wired into the kernel; weight=0 disables) ----
    use_conf: bool = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    if use_conf:
        conf_cx_v: float = float(pos[:, 0].mean())
        conf_cy_v: float = float(pos[:, 1].mean())
        conf_cz_v: float = float(pos[:, 2].mean())
        conf_R_v: float = float(settings.confinement_radius_arcs)
        if conf_R_v <= 0.0:
            pos_mask = np.asarray(exp_dist_mat) > 1e-6
            avg_bond = float(np.asarray(exp_dist_mat)[pos_mask].mean()) if pos_mask.any() else 1.0
            pf = float(settings.confinement_packing_factor_arcs)
            conf_R_v = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w_v: float = float(settings.confinement_weight)
    else:
        conf_cx_v = conf_cy_v = conf_cz_v = 0.0
        conf_R_v = 1.0
        conf_w_v = 0.0

    bundle = _build_arcs_kernel(n_steps_per_batch, excl_skip)
    kernel_full, init_arcs, init_excl, init_confine = bundle

    # ---- shape bucketing: pad N up to a bucket.  Pad beads are inert: the arc
    # term is zeroed by exp_mat=0 pad rows/cols (neither spring nor repulsion),
    # EV/confine are masked by n_active, and the move sampler draws from
    # [0, n_active) so pad beads never move.  Result == unbucketed at init
    # (bit-identical); per-step f32 chaos only (arcs uses non-strict acceptance).
    n_active_v: int = n
    if bool(settings.jax_bucket_shapes):
        if settings.jax_precompile_buckets:
            _precompile_arcs(settings)
        B: int = _bucket_for(n)
    else:
        B = n
    pos_f32: F32Array = pos.astype(np.float32)
    exp_mat_np: F32Array = exp_dist_mat.astype(np.float32)
    if B > n:
        pos_f32 = np.concatenate([pos_f32, np.zeros((B - n, 3), dtype=np.float32)], axis=0)
        exp_pad = np.zeros((B, B), dtype=np.float32)
        exp_pad[:n, :n] = exp_mat_np
        exp_mat_np = exp_pad
    pos_k_np: F32Array = np.broadcast_to(pos_f32, (K, B, 3)).copy()

    pos_k = jnp.asarray(pos_k_np)
    exp_mat_j = jnp.asarray(exp_mat_np)
    n_active_j = jnp.int32(n_active_v)

    stretch_k_v: float = float(settings.spring_stretch_arcs)
    squeeze_k_v: float = float(settings.spring_squeeze_arcs)
    ss_k = init_arcs(
        pos_k,
        exp_mat_j,
        jnp.float32(stretch_k_v),
        jnp.float32(squeeze_k_v),
    )
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v), n_active_j)
        if use_excl
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sc_k = (
        init_confine(
            pos_k,
            jnp.float32(conf_cx_v),
            jnp.float32(conf_cy_v),
            jnp.float32(conf_cz_v),
            jnp.float32(conf_R_v),
            jnp.float32(conf_w_v),
            n_active_j,
        )
        if use_conf
        else jnp.zeros((K,), dtype=jnp.float32)
    )

    T = jnp.float32(settings.max_temp)
    dt = jnp.float32(settings.dt_temp)
    js = jnp.float32(settings.jump_scale)
    jc = jnp.float32(settings.jump_coef)
    stretch_k_j = jnp.float32(stretch_k_v)
    squeeze_k_j = jnp.float32(squeeze_k_v)
    r0_j = jnp.float32(excl_r0)
    excl_w_j = jnp.float32(excl_w_v)
    conf_cx_j = jnp.float32(conf_cx_v)
    conf_cy_j = jnp.float32(conf_cy_v)
    conf_cz_j = jnp.float32(conf_cz_v)
    conf_R_j = jnp.float32(conf_R_v)
    conf_w_j = jnp.float32(conf_w_v)
    step_size_j = jnp.float32(step_size)
    stop_improvement = jnp.float32(settings.mc_stop_improvement)
    stop_successes = jnp.int32(settings.mc_stop_successes)
    score_eps = jnp.float32(1e-5)
    stop_when_ratio_above = jnp.float32(0.9999)
    seed_offset: int = abs(hash(label)) % (2**31) if label else 0
    base_key = jax.random.PRNGKey(seed_offset)

    pos_k, ss_k, se_k, sc_k, final_score_best, iter_count, converged_flag = kernel_full(
        pos_k,
        ss_k,
        se_k,
        sc_k,
        T,
        exp_mat_j,
        step_size_j,
        dt,
        js,
        jc,
        stretch_k_j,
        squeeze_k_j,
        r0_j,
        excl_w_j,
        conf_cx_j,
        conf_cy_j,
        conf_cz_j,
        conf_R_j,
        conf_w_j,
        base_key,
        stop_improvement,
        stop_successes,
        score_eps,
        stop_when_ratio_above,
        n_active_j,
    )

    score_per_chain = np.asarray(ss_k + se_k + sc_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)
    if verbose:
        prefix = f"    [{label}] " if label else "    "
        tail = "[done]" if converged_v else "[max-iters reached]"
        print(
            f"{prefix}step {iter_n * n_steps_per_batch:>7,}  "
            f"score={float(final_score_best):.4f}  batches={iter_n}  {tail}",
            flush=True,
        )

    best_k: int = int(np.argmin(score_per_chain))
    # Slice off any bucket padding (pos is (n, 3); pos_k is (K, B, 3), B >= n).
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])


# ---------------------------------------------------------------------------
# Heatmap kernel construction (separate from smooth/arcs; simplest energy)
# ---------------------------------------------------------------------------


def _build_heatmap_kernel(n_steps_per_batch: int, excl_skip: int) -> Any:
    """Build (or look up cached) compiled heatmap-MC kernel.

    Heatmap MC is the simplest of the three JAX kernels:
      - Energy: pairwise distance error vs `exp_dist`, masked by `skip` (the
        diagonal band + zero-frequency cells).  Double-counted (delta factor 2).
      - Optional excluded volume.
      - No chain bonds, angles, heat, orientation, or confinement.
      - Acceptance: non-strict (`score_new <= score`).
      - Convergence uses score_eps=1e-6, the standard plateau check, AND the
        `stop_when_ratio_above`=0.9999 guard (ported from the reference distance MC)
        so sparse/disconnected inter-chr heatmaps can't loop forever.

    Cache key: ("heatmap", n_steps_per_batch, excl_skip).
    """
    cache_key = ("heatmap", n_steps_per_batch, excl_skip)
    if cache_key in _kernel_cache:  # pyright: ignore[reportArgumentType]
        return _kernel_cache[cache_key]  # pyright: ignore[reportArgumentType]

    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    def _local_heatmap_at(pos: Any, p_pos: Any, p: Any, exp_safe: Any, skip: Any) -> Any:
        """Mirror of gnome3d.mc._local_heatmap_nb, with bead p virtually at
        p_pos.  Returns scalar.  `exp_safe[:, p]` is the expected distance
        column (1.0 wherever `skip[:, p]` is True, so the err formula is safe)."""
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        e = exp_safe[:, p]
        skip_col = skip[:, p]
        err = (d - e) / e
        contrib = err * err
        return jnp.sum(jnp.where(skip_col, 0.0, contrib))

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        in_range = jnp.abs(idx - p) > excl_skip
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    def _init_heatmap(pos: Any, exp_safe: Any, skip: Any) -> Any:
        """O(N²) init via row-at-a-time scan."""
        n = pos.shape[0]

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            e = exp_safe[:, i]
            skip_col = skip[:, i]
            err = (d - e) / e
            contrib = err * err
            return carry + jnp.sum(jnp.where(skip_col, 0.0, contrib)), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), jnp.arange(n))
        return total

    def _init_excl(pos: Any, r0: Any, weight: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rel = jnp.maximum(0.0, (r0 - d) / r0)
            contrib = weight * rel * rel
            in_range = jnp.abs(idx - i) > excl_skip
            return carry + jnp.sum(jnp.where(in_range, contrib, 0.0)), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def chain_batch(
        pos0: Any,
        ss0: Any,
        se0: Any,
        T0_: Any,
        exp_safe: Any,
        skip: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        r0: Any,
        excl_w: Any,
        key: Any,
        n_active: Any,
    ) -> Any:
        # Heatmap: all beads movable (mc.py uses np.arange(n)).  Under shape
        # bucketing pos0 is padded to a bucket size, but `n_active` (dynamic, so
        # it does NOT add a compile axis) restricts moves to the real beads
        # [0, n_active); pad beads never move, so they stay far away (EV=0) and
        # their heat rows are masked (skip=True) — fully inert.
        k_p, k_d, k_a = jax.random.split(key, 3)
        ps = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_active)
        disps = jax.random.uniform(
            k_d,
            (n_steps_per_batch, 3),
            minval=-step_size,
            maxval=step_size,
            dtype=pos0.dtype,
        )
        accs = jax.random.uniform(k_a, (n_steps_per_batch,), dtype=pos0.dtype)

        def body(i: Any, carry: Any) -> Any:
            pos, ss, se, T, n_ok = carry
            p = ps[i]
            delta = disps[i]
            u = accs[i]

            score = ss + se
            old_p = pos[p]
            new_p = old_p + delta

            loc_s_prev = _local_heatmap_at(pos, old_p, p, exp_safe, skip)
            loc_s_curr = _local_heatmap_at(pos, new_p, p, exp_safe, skip)
            # struct_delta_factor = 2 for heatmap (double-counted)
            ss_new = ss + 2.0 * (loc_s_curr - loc_s_prev)

            loc_e_prev = _local_excl_at(pos, old_p, p, r0, excl_w)
            loc_e_curr = _local_excl_at(pos, new_p, p, r0, excl_w)
            se_new = se + 2.0 * (loc_e_curr - loc_e_prev)

            score_new = ss_new + se_new

            # Heatmap uses NON-strict acceptance: score_new <= score.
            ok_unc = score_new <= score
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (pos_next, ss_next, se_next, T * dt, n_ok_next)

        init = (pos0, ss0, se0, T0_, jnp.int32(0))
        return jax.lax.fori_loop(0, n_steps_per_batch, body, init)

    in_axes = (
        0,
        0,
        0,
        None,  # pos, ss, se, T0
        None,
        None,  # exp_safe, skip
        None,
        None,
        None,
        None,  # step_size, dt, js, jc
        None,
        None,  # r0, excl_w
        0,  # key
        None,  # n_active (shared)
    )
    out_axes = (0, 0, 0, None, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    _MAX_ITERS: int = 10000

    @jax.jit
    def kernel_full(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        T_init: Any,
        exp_safe: Any,
        skip: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        r0: Any,
        excl_w: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        stop_when_ratio_above: Any,
        n_active: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            _, _, _, _, _, iter_i, _, converged = state
            return jnp.logical_and(jnp.logical_not(converged), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, T, ms_score, iter_i, _, _ = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, T, n_ok = batched(
                pos,
                ss,
                se,
                T,
                exp_safe,
                skip,
                step_size,
                dt,
                js,
                jc,
                r0,
                excl_w,
                keys,
                n_active,
            )
            score_per_chain = ss + se
            best_idx = jnp.argmin(score_per_chain)
            score = score_per_chain[best_idx]
            n_ok_best = n_ok[best_idx]
            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(
                score > stop_improvement * ms_score, n_ok_best < stop_successes
            )
            eps_done = score < score_eps
            # Plateau guard (ports the reference distance-MC guard to heatmap MC):
            # exit when the batch-to-batch score ratio stalls above 0.9999, so
            # sparse/disconnected inter-chr heatmaps don't loop forever (their
            # frustrated components can't be mutually satisfied, so score never
            # reaches score_eps and milestone_success never drops).  Intentional
            # divergence from reference MonteCarloHeatmap.
            ratio_done = ratio > stop_when_ratio_above
            converged = jnp.logical_or(jnp.logical_or(plateaued, eps_done), ratio_done)
            return (pos, ss, se, T, score, iter_i + 1, n_ok_best, converged)

        init_state = (
            pos_k,
            ss_k,
            se_k,
            T_init,
            jnp.float32(1e30),
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(False),
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        pos_f, ss_f, se_f, _T_f, final_score, iter_f, _, converged_f = final
        return pos_f, ss_f, se_f, final_score, iter_f, converged_f

    init_heatmap = jax.jit(jax.vmap(_init_heatmap, in_axes=(0, None, None)))
    init_excl_heatmap = jax.jit(jax.vmap(_init_excl, in_axes=(0, None, None)))

    bundle = (kernel_full, init_heatmap, init_excl_heatmap)
    _kernel_cache[cache_key] = bundle  # pyright: ignore[reportArgumentType]
    return bundle


def _precompile_heatmap(settings: "Settings") -> None:
    """Eagerly compile the heatmap kernel (and its init fns) for every shape
    bucket, so no XLA compile happens mid-run.  Uses .lower(...).compile() with
    ShapeDtypeStruct for the B*B arrays -> compiles without allocating them (a
    32768x32768 f32 would be 4 GB).  Idempotent across regions/threads."""
    if not _ensure_jax():
        return
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp
    sys = __import__("sys")

    K = max(1, int(settings.mc_heatmap_chains))
    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps = int(settings.mc_stop_steps_heatmap)
    sig = ("heatmap", n_steps, excl_skip, K)
    with _init_lock:
        if sig in _precompiled:
            return
        kernel_full, init_heatmap, init_excl = _build_heatmap_kernel(n_steps, excl_skip)
        sds = jax.ShapeDtypeStruct
        sample_key = jax.random.PRNGKey(0)  # concrete -> exact key dtype match
        # NB: only the *avals* (shape+dtype) of these scalars affect the compiled
        # program / cache key — the values are runtime inputs, so the real call
        # hits this cache regardless of value.  We pass settings-derived scalars
        # anyway (matching mc_heatmap_jax) for clarity and to stay correct if a
        # scalar ever becomes trace-relevant.  step_size/r0/excl_w have no settings
        # source (per-call / auto-derived) so they keep dtype-correct placeholders.
        f32 = jnp.float32
        T_a = f32(settings.max_temp_heatmap)
        dt_a = f32(settings.dt_temp_heatmap)
        js_a = f32(settings.jump_scale_heatmap)
        jc_a = f32(settings.jump_coef_heatmap)
        impr_a = f32(settings.mc_stop_improvement_heatmap)
        succ_a = jnp.int32(settings.mc_stop_successes_heatmap)
        t0 = __import__("time").perf_counter()
        for b in _SHAPE_BUCKETS:
            pos_a = sds((K, b, 3), np.float32)
            kvec_a = sds((K,), np.float32)
            exp_a = sds((b, b), np.float32)
            skip_a = sds((b, b), np.bool_)
            try:
                init_heatmap.lower(pos_a, exp_a, skip_a).compile()
                init_excl.lower(pos_a, f32(1.0), f32(0.0)).compile()
                kernel_full.lower(
                    pos_a,
                    kvec_a,  # ss
                    kvec_a,  # se
                    T_a,
                    exp_a,
                    skip_a,
                    f32(0.1),  # step_size (per-call: value irrelevant to compile)
                    dt_a,
                    js_a,
                    jc_a,
                    f32(1.0),  # r0 (auto-derived: value irrelevant)
                    f32(0.0),  # excl_w (per-region: value irrelevant)
                    sample_key,  # base_key
                    impr_a,  # stop_improvement
                    succ_a,  # stop_successes
                    f32(1e-6),  # score_eps (matches mc_heatmap_jax hardcode)
                    f32(0.9999),  # stop_when_ratio_above (matches hardcode)
                    jnp.int32(b),  # n_active
                ).compile()
            except Exception as e:  # noqa: BLE001 - precompile is best-effort
                print(f"[mc_jax] precompile heatmap bucket {b} skipped: {e}", file=sys.stderr)
        _precompiled.add(sig)
        dt = __import__("time").perf_counter() - t0
        print(
            f"[mc_jax] precompiled heatmap kernel for {len(_SHAPE_BUCKETS)} buckets "
            f"(K={K}) in {dt:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def mc_heatmap_jax(
    pos: np.ndarray[Any, Any],
    exp_dist: np.ndarray[Any, Any],
    diag_size: int,
    step_size: float,
    settings: "Settings",
    label: str = "",
    verbose: bool = False,
) -> float:
    """JAX backend for mc_heatmap.  Supports heatmap energy + (optional)
    excluded volume.  Same contract as [mc.mc_heatmap].

    Mutates `pos` in place and returns the best chain's final score.
    """
    if not _ensure_jax():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d-ng[jax]` or set mc_backend='numba'."
        )
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    n: int = pos.shape[0]
    if n <= 1:
        return 0.0

    K: int = max(1, int(settings.mc_heatmap_chains))
    n_steps_per_batch: int = int(settings.mc_stop_steps_heatmap)

    # Build the skip mask: diagonal band of width `diag_size` + zero entries.
    idx = np.arange(n, dtype=np.int64)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip_np = diag_mask | (np.asarray(exp_dist) < 1e-6)
    exp_safe_np = np.where(skip_np, 1.0, exp_dist).astype(np.float32)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(
        settings.exclusion_apply_to_heatmap
    )
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    if use_excl:
        active = np.asarray(exp_dist)[~skip_np]
        excl_r0: float = float(settings.exclusion_radius_heatmap)
        if excl_r0 <= 0.0:
            factor = float(settings.exclusion_auto_factor_heatmap)
            excl_r0 = factor * float(active.mean()) if active.size > 0 else 1.0
    else:
        excl_r0 = 1.0

    bundle = _build_heatmap_kernel(n_steps_per_batch, excl_skip)
    kernel_full, init_heatmap, init_excl = bundle

    # --- shape bucketing: pad N up to a fixed bucket so XLA reuses one compiled
    # kernel across all similarly-sized regions.  Pad beads are placed far apart
    # (EV auto-zero: d >> r0 -> rel=0) with skip=True heat rows (heat auto-zero),
    # and the kernel restricts moves to [0, n_active=n) so pad beads never move.
    # Net contribution is exactly zero -> result identical to the unpadded run.
    if bool(settings.jax_bucket_shapes):
        if settings.jax_precompile_buckets:
            _precompile_heatmap(settings)
        B: int = _bucket_for(n)
    else:
        B = n
    pos_f32: F32Array = pos.astype(np.float32)
    if B > n:
        exp_safe_pad = np.ones((B, B), dtype=np.float32)
        exp_safe_pad[:n, :n] = exp_safe_np
        exp_safe_np = exp_safe_pad
        skip_pad = np.ones((B, B), dtype=np.bool_)  # pad rows/cols skipped -> heat 0
        skip_pad[:n, :n] = skip_np
        skip_np = skip_pad
        # Inert pad beads: base 1e6, spacing 1e4 -> all pad-pad and pad-real
        # distances dwarf any r0 (real coords are O(1e2), r0 is O(1)).
        pad_xyz = np.zeros((B - n, 3), dtype=np.float32)
        pad_xyz[:, 0] = 1.0e6 + np.arange(B - n, dtype=np.float32) * 1.0e4
        pos_f32 = np.concatenate([pos_f32, pad_xyz], axis=0)

    pos_k_np: F32Array = np.broadcast_to(pos_f32, (K, B, 3)).copy()

    pos_k = jnp.asarray(pos_k_np)
    exp_safe_j = jnp.asarray(exp_safe_np)
    skip_j = jnp.asarray(skip_np.astype(np.bool_))
    n_active_j = jnp.int32(n)

    ss_k = init_heatmap(pos_k, exp_safe_j, skip_j)
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v))
        if use_excl
        else jnp.zeros((K,), dtype=jnp.float32)
    )

    T = jnp.float32(settings.max_temp_heatmap)
    dt = jnp.float32(settings.dt_temp_heatmap)
    js = jnp.float32(settings.jump_scale_heatmap)
    jc = jnp.float32(settings.jump_coef_heatmap)
    r0_j = jnp.float32(excl_r0)
    excl_w_j = jnp.float32(excl_w_v)
    step_size_j = jnp.float32(step_size)
    stop_improvement = jnp.float32(settings.mc_stop_improvement_heatmap)
    stop_successes = jnp.int32(settings.mc_stop_successes_heatmap)
    score_eps = jnp.float32(1e-6)
    seed_offset: int = abs(hash(label)) % (2**31) if label else 0
    base_key = jax.random.PRNGKey(seed_offset)

    pos_k, ss_k, se_k, final_score_best, iter_count, converged_flag = kernel_full(
        pos_k,
        ss_k,
        se_k,
        T,
        exp_safe_j,
        skip_j,
        step_size_j,
        dt,
        js,
        jc,
        r0_j,
        excl_w_j,
        base_key,
        stop_improvement,
        stop_successes,
        score_eps,
        jnp.float32(0.9999),  # stop_when_ratio_above: plateau guard (see kernel docstring)
        n_active_j,
    )

    score_per_chain = np.asarray(ss_k + se_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)
    if verbose:
        prefix = f"    [{label}] " if label else "    "
        tail = "[done]" if converged_v else "[max-iters reached]"
        print(
            f"{prefix}step {iter_n * n_steps_per_batch:>7,}  "
            f"score={float(final_score_best):.4f}  batches={iter_n}  {tail}",
            flush=True,
        )

    best_k: int = int(np.argmin(score_per_chain))
    # Slice off any bucket padding (pos is (n, 3); pos_k is (K, B, 3), B >= n).
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])


def _precompile_smooth(
    settings: "Settings", use_heat: bool, use_orn: bool, max_nbrs: int, anchor_frac: float, K: int
) -> None:
    """Eagerly compile the smooth kernel across N buckets for ONE
    (use_heat, use_orn, max_nbrs->M, K) combo.  Smooth specializes on
    (B, A, M, K, use_heat, use_orn); B and A both scale with region size, so we
    compile the realistic (B, A) DIAGONAL: A = bucket(anchor_frac * B) per B
    (use_orn=False has no anchor axis -> A=M=1).  Idempotent per combo via
    _precompiled.  Uses .lower(ShapeDtypeStruct).compile() (no array alloc)."""
    if not _ensure_jax():
        return
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp
    sys = __import__("sys")

    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps = int(settings.mc_stop_steps_smooth)
    M = int(max_nbrs) if use_orn else 1
    sig = ("smooth", n_steps, excl_skip, bool(use_heat), bool(use_orn), M, int(K))
    with _init_lock:
        if sig in _precompiled:
            return
        bundle = _build_smooth_kernel(n_steps, excl_skip, use_heat, use_orn, M)
        kernel_full = bundle[1]
        sds = jax.ShapeDtypeStruct
        f32 = jnp.float32
        key = jax.random.PRNGKey(0)
        T_a = f32(settings.max_temp_smooth)
        dt_a = f32(settings.dt_temp_smooth)
        js_a = f32(settings.jump_scale_smooth)
        jc_a = f32(settings.jump_coef_smooth)
        impr_a = f32(settings.mc_stop_improvement_smooth)
        succ_a = jnp.int32(settings.mc_stop_successes_smooth)
        t0 = __import__("time").perf_counter()
        for b in _SHAPE_BUCKETS:
            a = _bucket_for(max(1, int(anchor_frac * b)), _ANCHOR_BUCKETS) if use_orn else 1
            kvec = sds((K,), np.float32)
            heat_a = sds((b, b), np.float32) if use_heat else sds((1, 1), np.float32)
            try:
                kernel_full.lower(
                    sds((K, b, 3), np.float32),  # pos_k
                    kvec, kvec, kvec, kvec, kvec,  # ss, se, sh, so, sc
                    sds((K, a, 3), np.float32),  # anchor_orn_k
                    T_a,
                    sds((b,), np.float32),  # dtn
                    sds((b,), np.int32),  # movable (int64 -> int32 under x64-off)
                    heat_a,
                    sds((a,), np.int32),  # anchor_ar
                    sds((b,), np.int32),  # bead_to_anchor_k
                    sds((a, M), np.int32),  # nbr_idx
                    sds((a, M), np.float32),  # nbr_w
                    sds((a, M), np.bool_),  # nbr_valid
                    sds((b,), np.bool_),  # is_L
                    f32(0.1),  # step_size (value irrelevant)
                    dt_a, js_a, jc_a,
                    f32(1.0), f32(1.0), f32(0.1), f32(1.0), f32(1.0),  # stretch..ang_w
                    f32(1.0), f32(0.0),  # r0, excl_w
                    f32(1.0),  # heat_weight
                    f32(1.0),  # motif_weight
                    jnp.bool_(True),  # symmetric
                    f32(0.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0),  # conf_cx..conf_w
                    key, impr_a, succ_a,
                    f32(1e-6),  # score_eps (matches mc_smooth_jax hardcode)
                    jnp.int32(b),  # n_active
                    jnp.int32(b),  # n_movable_active
                ).compile()
            except Exception as e:  # noqa: BLE001 - precompile is best-effort
                print(f"[mc_jax] precompile smooth B={b} A={a} skipped: {e}", file=sys.stderr)
        _precompiled.add(sig)
        dt = __import__("time").perf_counter() - t0
        print(
            f"[mc_jax] precompiled smooth kernel: {len(_SHAPE_BUCKETS)} B-buckets "
            f"(heat={use_heat} orn={use_orn} M={M} K={K}) in {dt:.1f}s",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Public entry: mirrors gnome3d.mc.mc_smooth signature
# ---------------------------------------------------------------------------


def mc_smooth_jax(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    label: str = "",
    verbose: bool = False,
    pos_batch: np.ndarray[Any, Any] | None = None,
    return_all: bool = False,
) -> Any:
    """JAX backend for smooth-MC, supporting chain + EV + (optional) heat
    + (optional) orientation + (optional) confinement.

    Mutates `pos` in place (writes the best-chain final positions back) and
    returns the best chain's final score.

    Batched mode (`pos_batch` given, shape (B, N, 3)): run B independent anneals
    from distinct starts in ONE vmapped kernel (K = B), sharing `dtn`/`fixed`/
    `heat`/schedule and using `pos` only as the reference for n/movable/centroid.
    The shared while-loop stops when the BEST of the B chains converges (mc_jax
    convergence is best-of-K).  With `return_all=True` this returns
    `(scores: (B,), finals: (B, N, 3))` as numpy arrays and does NOT mutate
    `pos` — the caller does its own per-trial selection (see solver.py IB phase).
    """
    if not _ensure_jax():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d-ng[jax]` or set mc_backend='numba'."
        )
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    n: int = pos.shape[0]
    if n <= 2:
        return 0.0

    movable_np: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable_np) == 0:
        return 0.0

    if pos_batch is not None:
        if pos_batch.ndim != 3 or pos_batch.shape[1:] != (n, 3):
            raise ValueError(f"pos_batch must have shape (B, {n}, 3); got {pos_batch.shape}")
        K = int(pos_batch.shape[0])
    else:
        K = max(1, int(settings.mc_smooth_chains))
    n_steps_per_batch: int = int(settings.mc_stop_steps_smooth)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    if use_excl:
        excl_r0: float = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            factor = float(settings.exclusion_auto_factor_smooth)
            excl_r0 = factor * float(np.asarray(dtn).mean())
    else:
        excl_r0 = 1.0  # unused but must be valid

    use_heat: bool = heat_dist is not None
    heat_weight_v: float = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    use_orn: bool = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    motif_weight_v: float = float(settings.motif_weight) if use_orn else 0.0
    motifs_symmetric_v: bool = bool(getattr(settings, "motifs_symmetric", True))

    # ---- confinement setup ----
    # Per-bead soft envelope; center = centroid of starting pos, radius from
    # settings (or auto-derived).  Always wired into the kernel; when
    # disabled, conf_w=0 so XLA folds the contribution away.
    use_conf: bool = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    if use_conf:
        conf_cx_v: float = float(pos[:, 0].mean())
        conf_cy_v: float = float(pos[:, 1].mean())
        conf_cz_v: float = float(pos[:, 2].mean())
        conf_R_v: float = float(settings.confinement_radius_smooth)
        if conf_R_v <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R_v = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w_v: float = float(settings.confinement_weight)
    else:
        conf_cx_v = conf_cy_v = conf_cz_v = 0.0
        conf_R_v = 1.0
        conf_w_v = 0.0

    # ---- prepare orientation arrays (padded CSR) ----
    anchor_frac: float = 0.0  # real n_anchors/n; for the precompile (B,A) diagonal
    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar_np: I32Array = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar_np))
        # pad neighbor lists to max width; uniform shape needed for vmap.
        nbr_lists = [list(anchor_neighbors.get(k, [])) for k in range(n_anchors)]
        nbr_w_lists = [list(anchor_neighbor_weights.get(k, [])) for k in range(n_anchors)]
        max_nbrs = max((len(lst) for lst in nbr_lists), default=1)
        max_nbrs = max(max_nbrs, 1)  # at least 1 slot
        nbr_idx_np: I32Array = np.zeros((n_anchors, max_nbrs), dtype=np.int32)
        nbr_w_np: F32Array = np.zeros((n_anchors, max_nbrs), dtype=np.float32)
        nbr_valid_np = np.zeros((n_anchors, max_nbrs), dtype=np.bool_)
        for k_idx in range(n_anchors):
            for m, (jn, wn) in enumerate(zip(nbr_lists[k_idx], nbr_w_lists[k_idx], strict=True)):
                nbr_idx_np[k_idx, m] = int(jn)
                nbr_w_np[k_idx, m] = float(wn)
                nbr_valid_np[k_idx, m] = True
        # bead_to_anchor_k: -1 if bead not adjacent to an anchor; else k.
        bead_to_anchor_k_np: I32Array = np.full(n, -1, dtype=np.int32)
        for k_idx in range(n_anchors):
            ar = int(anchor_ar_np[k_idx])
            if ar > 0:
                bead_to_anchor_k_np[ar - 1] = k_idx
            if ar + 1 < n:
                bead_to_anchor_k_np[ar + 1] = k_idx
        is_L_np = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # Phase-2 bucketing of the ANCHOR-indexed arrays: round n_anchors -> A and
        # max_nbrs -> M so the kernel's orientation shapes come only from the
        # (A, M) ladders, not per-region.  Pad anchors/edges get nbr_valid=False
        # -> contribute exactly 0 to the (scan-summed) orientation score, so this
        # is bit-identical at init.  Pad anchor_ar=0 (its orn is computed but
        # never referenced since no valid edge points to it).
        if bool(settings.jax_bucket_shapes):
            A = _bucket_for(n_anchors, _ANCHOR_BUCKETS)
            M = _bucket_for(max_nbrs, _NBR_BUCKETS)
            anchor_frac = n_anchors / n  # real fraction, before reassignment below
            ap, mp = A - n_anchors, M - max_nbrs
            if ap > 0 or mp > 0:
                anchor_ar_np = np.concatenate([anchor_ar_np, np.zeros(ap, dtype=np.int32)])
                nbr_idx_np = np.pad(nbr_idx_np, ((0, ap), (0, mp)))
                nbr_w_np = np.pad(nbr_w_np, ((0, ap), (0, mp)))
                nbr_valid_np = np.pad(nbr_valid_np, ((0, ap), (0, mp)))  # False pads
                n_anchors, max_nbrs = A, M
    else:
        n_anchors = 1  # placeholder shape
        max_nbrs = 1
        anchor_ar_np = np.zeros(1, dtype=np.int32)
        nbr_idx_np = np.zeros((1, 1), dtype=np.int32)
        nbr_w_np = np.zeros((1, 1), dtype=np.float32)
        nbr_valid_np = np.zeros((1, 1), dtype=np.bool_)
        bead_to_anchor_k_np = np.full(n, -1, dtype=np.int32)
        is_L_np = np.zeros(n, dtype=np.bool_)

    # ---- move state to device (f32) ----
    pos_f32: F32Array = pos.astype(np.float32)
    if pos_batch is not None:
        # Batched mode: K distinct starts (one per trial), not a broadcast.
        pos_k_np = np.ascontiguousarray(pos_batch.astype(np.float32))
    else:
        pos_k_np = np.broadcast_to(pos_f32, (K, n, 3)).copy()
    dtn_np: F32Array = dtn.astype(np.float32)
    heat_np: F32Array
    if use_heat:
        assert heat_dist is not None
        heat_np = heat_dist.astype(np.float32)
    else:
        heat_np = np.zeros((1, 1), dtype=np.float32)  # unused placeholder

    # ---- shape bucketing: pad N up to a bucket so XLA reuses one compiled
    # kernel across all similarly-sized regions.  Pad beads are fully inert:
    # chain/EV/confinement masked by `n_active`, heat rows zeroed, movement
    # restricted to the real movable set via `n_movable_active`.  ALL bead-indexed
    # kernel inputs are padded to B so the kernel's input shapes depend only on B
    # (+ K, max_nbrs) — not on the per-region n/n_movable.  (Orientation's
    # anchor-indexed arrays, shape n_anchors, are NOT yet bucketed -> with
    # use_orn=True the kernel still recompiles per region; that's phase 2.)
    # n_active == n and n_movable_active == len(movable) when unbucketed.
    n_active_v: int = n
    n_movable_v: int = int(movable_np.shape[0])
    if bool(settings.jax_bucket_shapes):
        if settings.jax_precompile_buckets:
            _precompile_smooth(settings, use_heat, use_orn, max_nbrs, anchor_frac, K)
        B: int = _bucket_for(n)
    else:
        B = n
    if B > n:
        n_pad = B - n
        pos_k_np = np.concatenate(
            [pos_k_np, np.zeros((pos_k_np.shape[0], n_pad, 3), dtype=np.float32)], axis=1
        )
        dtn_np = np.concatenate([dtn_np, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat:
            heat_pad = np.zeros((B, B), dtype=np.float32)
            heat_pad[:n, :n] = heat_np
            heat_np = heat_pad
        # bead-indexed arrays -> pad to B (pad beads map to no anchor / never move)
        bead_to_anchor_k_np = np.concatenate(
            [bead_to_anchor_k_np, np.full(n_pad, -1, dtype=np.int32)]
        )
        if is_L_np.shape[0] == n:  # bead-indexed (use_orn=False); anchor-indexed -> phase 2
            is_L_np = np.concatenate([is_L_np, np.zeros(n_pad, dtype=np.bool_)])
        # movable -> pad to B; n_movable_v bounds the sampler so pads never picked
        movable_np = np.concatenate(
            [movable_np, np.zeros(B - movable_np.shape[0], dtype=movable_np.dtype)]
        )

    bundle = _build_smooth_kernel(n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs)
    (
        _kernel_one_batch,
        kernel_full,
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
    ) = bundle

    pos_k = jnp.asarray(pos_k_np)
    dtn_j = jnp.asarray(dtn_np)
    movable_j = jnp.asarray(movable_np)
    heat_j = jnp.asarray(heat_np)
    anchor_ar_j = jnp.asarray(anchor_ar_np)
    bead_to_anchor_k_j = jnp.asarray(bead_to_anchor_k_np)
    nbr_idx_j = jnp.asarray(nbr_idx_np)
    nbr_w_j = jnp.asarray(nbr_w_np)
    nbr_valid_j = jnp.asarray(nbr_valid_np)
    is_L_j = jnp.asarray(is_L_np)
    n_active_j = jnp.int32(n_active_v)
    n_movable_active_j = jnp.int32(n_movable_v)
    seed_offset: int = abs(hash(label)) % (2**31) if label else 0

    # ---- initial scores ----
    ss_k = init_smooth(
        pos_k,
        dtn_j,
        jnp.float32(settings.spring_stretch),
        jnp.float32(settings.spring_squeeze),
        jnp.float32(settings.spring_angular),
        jnp.float32(settings.smooth_dist_weight),
        jnp.float32(settings.smooth_angle_weight),
        n_active_j,
    )
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v), n_active_j)
        if use_excl
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sh_k = (
        init_heat(pos_k, heat_j, jnp.float32(heat_weight_v))
        if use_heat
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sc_k = (
        init_confine(
            pos_k,
            jnp.float32(conf_cx_v),
            jnp.float32(conf_cy_v),
            jnp.float32(conf_cz_v),
            jnp.float32(conf_R_v),
            jnp.float32(conf_w_v),
            n_active_j,
        )
        if use_conf
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    if use_orn:
        anchor_orn_k = init_anchor_orn(pos_k, anchor_ar_j, is_L_j)
        so_k = init_orn_score(
            anchor_orn_k,
            nbr_idx_j,
            nbr_w_j,
            nbr_valid_j,
            jnp.float32(motif_weight_v),
            jnp.bool_(motifs_symmetric_v),
        )
    else:
        anchor_orn_k = jnp.zeros((K, n_anchors, 3), dtype=jnp.float32)
        so_k = jnp.zeros((K,), dtype=jnp.float32)

    T = jnp.float32(settings.max_temp_smooth)
    dt = jnp.float32(settings.dt_temp_smooth)
    js = jnp.float32(settings.jump_scale_smooth)
    jc = jnp.float32(settings.jump_coef_smooth)
    stretch_k_j = jnp.float32(settings.spring_stretch)
    squeeze_k_j = jnp.float32(settings.spring_squeeze)
    ang_k_j = jnp.float32(settings.spring_angular)
    dist_w_j = jnp.float32(settings.smooth_dist_weight)
    ang_w_j = jnp.float32(settings.smooth_angle_weight)
    r0_j = jnp.float32(excl_r0)
    excl_w_j = jnp.float32(excl_w_v)
    heat_w_j = jnp.float32(heat_weight_v)
    motif_w_j = jnp.float32(motif_weight_v)
    symmetric_j = jnp.bool_(motifs_symmetric_v)
    conf_cx_j = jnp.float32(conf_cx_v)
    conf_cy_j = jnp.float32(conf_cy_v)
    conf_cz_j = jnp.float32(conf_cz_v)
    conf_R_j = jnp.float32(conf_R_v)
    conf_w_j = jnp.float32(conf_w_v)
    step_size_j = jnp.float32(step_size)

    stop_improvement = jnp.float32(settings.mc_stop_improvement_smooth)
    stop_successes = jnp.int32(settings.mc_stop_successes_smooth)
    score_eps = jnp.float32(1e-6)
    base_key = jax.random.PRNGKey(seed_offset)

    # ONE JAX call drives the full convergence loop on device.  No per-batch
    # Python sync.  Returns final state + (iter_count, converged) so we can
    # log how the run terminated.
    (
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        _anchor_orn_k_final,
        final_score_best,
        iter_count,
        converged_flag,
    ) = kernel_full(
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        anchor_orn_k,
        T,
        dtn_j,
        movable_j,
        heat_j,
        anchor_ar_j,
        bead_to_anchor_k_j,
        nbr_idx_j,
        nbr_w_j,
        nbr_valid_j,
        is_L_j,
        step_size_j,
        dt,
        js,
        jc,
        stretch_k_j,
        squeeze_k_j,
        ang_k_j,
        dist_w_j,
        ang_w_j,
        r0_j,
        excl_w_j,
        heat_w_j,
        motif_w_j,
        symmetric_j,
        conf_cx_j,
        conf_cy_j,
        conf_cz_j,
        conf_R_j,
        conf_w_j,
        base_key,
        stop_improvement,
        stop_successes,
        score_eps,
        n_active_j,
        n_movable_active_j,
    )

    score_per_chain = np.asarray(ss_k + se_k + sh_k + so_k + sc_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)

    if verbose:
        prefix = f"    [{label}] " if label else "    "
        total_steps = iter_n * n_steps_per_batch
        tail = "[done]" if converged_v else "[max-iters reached]"
        print(
            f"{prefix}step {total_steps:>7,}  score={float(final_score_best):.4f}"
            f"  batches={iter_n}  {tail}",
            flush=True,
        )

    if return_all:
        # Batched mode: hand back every chain's score + final positions; the
        # caller selects per-trial.  Slice off bucket padding (B -> n).  Do NOT
        # mutate `pos`.
        return (
            score_per_chain.astype(np.float64),
            np.asarray(pos_k[:, :n]).astype(np.float32),
        )

    best_k: int = int(np.argmin(score_per_chain))
    # Slice off any bucket padding (pos is (n, 3); pos_k is (K, B, 3), B >= n).
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])
