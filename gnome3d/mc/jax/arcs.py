"""JAX arc-MC kernel: anchor springs to expected distances (+ optional EV / confinement).

`mc_arcs_jax` is the single-problem entry; `mc_arcs_jax_batch` anneals K different
IBs' anchors in one vmapped kernel (region batching).  Both build/compile through
`_build_arcs_kernel` (memoised in `_kernel_cache`).  Arcs is non-strict, so the
batched kernel freezes converged chains once they stop improving.
"""

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.mc.jax.memory import max_k_for_bytes
from gnome3d.mc.jax.util import (
    jax_bucket_for,
    jax_device_budget_bytes,
    jax_is_available,
    log_kernel_done,
    log_kernel_start,
)
from gnome3d.types import F32Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

# Compiled-kernel cache (keyed by kernel signature)
_kernel_cache: dict[Any, Any] = {}

# --- arcs GPU profiling (opt-in via env GNOME3D_ARCS_PROFILE=1) ----------------
# The per-chain conv_iter spread and the extended per-batch log line are ALWAYS on
# (byte-exact, cheap host-side stats).  The compile-vs-execute split, cost_analysis,
# init timing and the per-(K,B) aggregate are gated behind the env flag so a normal
# production run issues the single kernel call byte-identically to before.
_ARCS_PROFILE: bool = os.environ.get("GNOME3D_ARCS_PROFILE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_arcs_profile: dict[tuple[int, int], dict[str, float]] = {}
_arcs_seen_shapes: set[tuple[int, int]] = set()
_arcs_profile_lock = threading.Lock()
_last_batch_diag: dict[str, Any] = {}  # last region-batch diagnostics (read by bench scripts)
# Profiling hook (bench only): when set, overrides the arcs score_eps so the kernel
# "converges" after ~1 outer iter.  Lets the width K-scan measure PER-ITER wall
# without running each launch to full convergence (thousands of iters for a big IB).
# Per-iter compute is constant across iters, so 1 iter is a faithful sample.
_ARCS_FORCE_SCORE_EPS: float | None = None


def _profile_kernel(
    kernel_full_mp: Any, args: tuple[Any, ...], K: int, B: int
) -> tuple[Any, float, float, bool, Any]:
    """Compile/execute split for ONE region-batch (env-gated by GNOME3D_ARCS_PROFILE).

    AOT-compiles via ``.lower(*args).compile()`` to time XLA compilation, then runs
    the kernel exactly ONCE (the MC body can be very long, so it must execute a
    single time).  Returns ``(out, compile_ms, run_ms, cold, cost)``: ``cold`` is
    True the first time this process sees shape ``(K, B)``; ``cost`` is the XLA
    cost_analysis dict (best-effort, ``None`` on failure).  Pure timing - it never
    alters the kernel result."""
    import jax

    shape = (K, B)
    with _arcs_profile_lock:
        cold = shape not in _arcs_seen_shapes
        _arcs_seen_shapes.add(shape)
    compile_ms = -1.0
    cost: Any = None
    compiled: Any = None
    try:
        tc = time.perf_counter()
        compiled = kernel_full_mp.lower(*args).compile()  # the single XLA compile
        compile_ms = (time.perf_counter() - tc) * 1e3
        try:
            cost = compiled.cost_analysis()
        except Exception:  # noqa: BLE001 - introspection is best-effort
            cost = None
    except Exception as e:  # noqa: BLE001 - profiling must never break the run
        LOG.debug("arcs profile: compile timing failed (K=%d B=%d): %s", K, B, e)
    # Run the AOT executable directly when we have it: that skips the Python
    # retrace/dispatch a fresh kernel_full_mp(*args) would add to run_ms (the
    # result is bit-identical).  Fall back to the jitted fn if .compile() failed.
    tr = time.perf_counter()
    out = compiled(*args) if compiled is not None else kernel_full_mp(*args)
    jax.block_until_ready(out)
    run_ms = (time.perf_counter() - tr) * 1e3
    return out, compile_ms, run_ms, cold, cost


def dump_arcs_profile() -> None:
    """Log the per-(K,B) arcs region-batch profile gathered this process.

    Meaningful only when GNOME3D_ARCS_PROFILE=1 (empty otherwise).  Call at the end
    of an arcs stage / run (or from a bench script) to see where arcs GPU wall went:
    per shape - launches, cold compiles, compile vs run vs init time, total iters."""
    with _arcs_profile_lock:
        if not _arcs_profile:
            log.status(LOG, "arcs profile: empty (set GNOME3D_ARCS_PROFILE=1 to collect)")
            return
        items = sorted(_arcs_profile.items(), key=lambda kv: -kv[1]["wall_s"])
        total_wall = sum(r["wall_s"] for _, r in items)
        total_comp = sum(r["compile_ms"] for _, r in items) / 1e3
        log.status(
            LOG,
            "arcs profile: %d (K,B) shapes, wall=%.1fs (compile=%.1fs)",
            len(items),
            total_wall,
            total_comp,
        )
        for (k, b), r in items:
            log.status(
                LOG,
                "  K=%-4d B=%-6d launches=%-3d compiles=%-2d wall=%6.1fs (%2.0f%%) "
                "compile=%5.1fs run=%6.1fs init=%4.1fs iters=%d",
                k,
                b,
                int(r["launches"]),
                int(r["compiles"]),
                r["wall_s"],
                100.0 * r["wall_s"] / max(total_wall, 1e-9),
                r["compile_ms"] / 1e3,
                r["run_ms"] / 1e3,
                r["init_ms"] / 1e3,
                int(r["iters"]),
            )


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

    import jax
    import jax.numpy as jnp

    def _local_arcs_at(
        pos: Any,
        p_pos: Any,
        p: Any,
        exp_mat: Any,
        stretch_k: Any,
        squeeze_k: Any,
        rep_inv_cutoff: Any = 0.0,
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
        rep = jnp.maximum(
            0.0, 1.0 / d_safe - rep_inv_cutoff
        )  # truncate 1/d at cutoff (0 => unbounded)

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

    def _init_arcs(
        pos: Any, exp_mat: Any, stretch_k: Any, squeeze_k: Any, rep_inv_cutoff: Any = 0.0
    ) -> Any:
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
            rep = jnp.maximum(
                0.0, 1.0 / d_safe - rep_inv_cutoff
            )  # truncate 1/d at cutoff (0 => unbounded)
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

    def _init_confine(
        pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any, n_active: Any
    ) -> Any:
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
        rep_inv_cutoff: Any,
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

            loc_s_prev = _local_arcs_at(
                pos, old_p, p, exp_mat, stretch_k, squeeze_k, rep_inv_cutoff
            )
            loc_s_curr = _local_arcs_at(
                pos, new_p, p, exp_mat, stretch_k, squeeze_k, rep_inv_cutoff
            )
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
        None,  # rep_inv_cutoff (shared per-IB)
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
        rep_inv_cutoff: Any,
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
                rep_inv_cutoff,
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

    # --- multi-problem (region-batched) variant: K DIFFERENT IBs in one kernel,
    #     per-chain convergence (cf. the smooth kernel_full_mp).  Per-IB arrays
    #     (exp_mat, step_size, r0, conf_*, n_active) move to axis 0; the schedule,
    #     springs and weights stay shared.
    #
    #     Arcs uses NON-strict acceptance, so a chain that has converged would
    #     DRIFT WORSE if it kept stepping while slower chains finish - so we
    #     FREEZE converged chains (hold their pos/scores).  The strict smooth
    #     path is safe to keep stepping and doesn't need this.
    in_axes_mp = (
        0,
        0,
        0,
        0,  # pos, ss, se, sc (per-chain)
        None,  # T0 (shared)
        0,  # exp_mat (per-IB)
        0,  # step_size (per-IB)
        None,
        None,
        None,  # dt, js, jc (shared)
        None,
        None,  # stretch_k, squeeze_k (shared; boosted IBs aren't batched)
        0,  # r0 (per-IB auto excl radius)
        None,  # excl_w (shared)
        0,
        0,
        0,
        0,  # conf_cx, conf_cy, conf_cz, conf_R (per-IB)
        None,  # conf_w (shared)
        0,  # key (per-chain)
        0,  # n_active (per-IB)
        0,  # rep_inv_cutoff (per-IB)
    )
    batched_mp = jax.vmap(chain_batch, in_axes=in_axes_mp, out_axes=out_axes)

    @jax.jit
    def kernel_full_mp(
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
        rep_inv_cutoff: Any,
        max_iters: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            iter_i = state[6]
            converged = state[8]  # (K,) per-chain
            return jnp.logical_and(jnp.logical_not(jnp.all(converged)), iter_i < max_iters)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sc, T, ms_score, iter_i, _n_ok, conv_prev, conv_iter = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            npos, nss, nse, nsc, nT, n_ok = batched_mp(
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
                rep_inv_cutoff,
            )
            # Freeze chains already converged (non-strict accept could worsen them).
            frozen = conv_prev
            pos = jnp.where(frozen[:, None, None], pos, npos)
            ss = jnp.where(frozen, ss, nss)
            se = jnp.where(frozen, se, nse)
            sc = jnp.where(frozen, sc, nsc)

            score = ss + se + sc  # (K,) per-chain total
            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, n_ok < stop_successes)
            eps_done = score < score_eps
            ratio_done = ratio > stop_when_ratio_above
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, eps_done), ratio_done), conv_prev
            )
            # Diagnostics ONLY: latch the outer-iter at which each chain FIRST
            # converged.  Write-only - never read by cond_fn, never fed to
            # batched_mp / the freeze mask / RNG key derivation - so the MC
            # trajectory and outputs stay bit-identical (sentinel 0 = never).
            newly = jnp.logical_and(converged, jnp.logical_not(conv_prev))
            conv_iter = jnp.where(newly, iter_i + 1, conv_iter)
            return (pos, ss, se, sc, nT, score, iter_i + 1, n_ok, converged, conv_iter)

        init_state = (
            pos_k,
            ss_k,
            se_k,
            sc_k,
            T_init,
            jnp.full((K,), 1e30, dtype=jnp.float32),  # ms_score per-chain
            jnp.int32(0),
            jnp.zeros((K,), dtype=jnp.int32),  # n_ok filler
            jnp.zeros((K,), dtype=jnp.bool_),  # converged per-chain
            jnp.zeros((K,), dtype=jnp.int32),  # conv_iter per-chain (diagnostic; 0 = never)
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        pos_f, ss_f, se_f, sc_f, _T_f, _score_f, iter_f, _nok_f, converged_f, conv_iter_f = final
        return pos_f, ss_f, se_f, sc_f, iter_f, converged_f, conv_iter_f

    init_arcs = jax.jit(jax.vmap(_init_arcs, in_axes=(0, None, None, None, None)))
    init_excl_arcs = jax.jit(jax.vmap(_init_excl, in_axes=(0, None, None, None)))
    init_confine_arcs = jax.jit(
        jax.vmap(_init_confine, in_axes=(0, None, None, None, None, None, None))
    )

    bundle = (kernel_full, init_arcs, init_excl_arcs, init_confine_arcs, kernel_full_mp)
    _kernel_cache[cache_key] = bundle  # pyright: ignore[reportArgumentType]
    return bundle


def mc_arcs_jax(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
) -> float:
    """JAX backend for mc_arcs.  Supports arc springs + EV + (optional)
    confinement.  Same contract as [mc.mc_arcs].

    Mutates `pos` in place (writes the best-chain final positions back) and
    returns the best chain's final score.
    """
    if not jax_is_available():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d-ng[jax]` or set mc_backend='numba'."
        )
    import jax
    import jax.numpy as jnp

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
    kernel_full, init_arcs, init_excl, init_confine, _kernel_full_mp = bundle

    # ---- shape bucketing: pad N up to a bucket.  Pad beads are inert: the arc
    # term is zeroed by exp_mat=0 pad rows/cols (neither spring nor repulsion),
    # EV/confine are masked by n_active, and the move sampler draws from
    # [0, n_active) so pad beads never move.  Result == unbucketed at init
    # (bit-identical); per-step f32 chaos only (arcs uses non-strict acceptance).
    n_active_v: int = n
    if bool(settings.mc_executor_jax_bucket_shapes):
        B: int = jax_bucket_for(n)
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
    rep_factor = float(getattr(settings, "arcs_repulsion_cutoff_factor", 0.0))
    rep_inv_cutoff_v = 0.0
    if rep_factor > 0.0:
        _rm = exp_mat_np > 1e-6
        _rmean = float(exp_mat_np[_rm].mean()) if _rm.any() else 1.0
        if _rmean > 0.0:
            rep_inv_cutoff_v = 1.0 / (rep_factor * _rmean)
    rep_inv_cutoff_j = jnp.float32(rep_inv_cutoff_v)
    ss_k = init_arcs(
        pos_k,
        exp_mat_j,
        jnp.float32(stretch_k_v),
        jnp.float32(squeeze_k_v),
        rep_inv_cutoff_j,
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
    # Per-call RNG diversity keyed on the active scope path (was: the label).
    _seed_src = log.current()
    seed_offset: int = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
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
        rep_inv_cutoff_j,
    )

    score_per_chain = np.asarray(ss_k + se_k + sc_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)
    if LOG.isEnabledFor(logging.DEBUG):
        tail = "[done]" if converged_v else "[max-iters reached]"
        LOG.debug(
            "step %7s  score=%.4f  batches=%d  %s",
            f"{iter_n * n_steps_per_batch:,}",
            float(final_score_best),
            iter_n,
            tail,
        )

    best_k: int = int(np.argmin(score_per_chain))
    # Slice off any bucket padding (pos is (n, 3); pos_k is (K, B, 3), B >= n).
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])


def _prep_arcs_problem_np(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    settings: "Settings",
    B: int,
) -> dict[str, Any]:
    """One IB's arcs kernel inputs as numpy, padded to bucket B.  Pure numpy
    mirror of the per-problem prep in `mc_arcs_jax` (per-IB excl radius +
    confinement envelope); pad beads are inert (exp_mat=0 rows/cols, n_active
    masks EV/confine)."""
    n = int(pos.shape[0])
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    excl_r0 = 1.0
    if use_excl:
        excl_r0 = float(settings.exclusion_radius_arcs)
        if excl_r0 <= 0.0:
            mask = np.asarray(exp_dist_mat) > 1e-6
            factor = float(settings.exclusion_auto_factor_arcs)
            excl_r0 = factor * float(np.asarray(exp_dist_mat)[mask].mean()) if mask.any() else 1.0

    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    if use_conf:
        conf_cx = float(pos[:, 0].mean())
        conf_cy = float(pos[:, 1].mean())
        conf_cz = float(pos[:, 2].mean())
        conf_R = float(settings.confinement_radius_arcs)
        if conf_R <= 0.0:
            mask = np.asarray(exp_dist_mat) > 1e-6
            avg_bond = float(np.asarray(exp_dist_mat)[mask].mean()) if mask.any() else 1.0
            conf_R = float(settings.confinement_packing_factor_arcs) * avg_bond * (n ** (1.0 / 3.0))
    else:
        conf_cx = conf_cy = conf_cz = 0.0
        conf_R = 1.0

    # Non-arc 1/d repulsion cutoff: truncate at factor x mean arc distance so sparse/small IBs
    # don't blow up (the unbounded 1/d has no minimum).  0 => unbounded (faithful to the reference).
    rep_factor = float(getattr(settings, "arcs_repulsion_cutoff_factor", 0.0))
    rep_inv_cutoff = 0.0
    if rep_factor > 0.0:
        _arc_mask = np.asarray(exp_dist_mat) > 1e-6
        _mean_arc = float(np.asarray(exp_dist_mat)[_arc_mask].mean()) if _arc_mask.any() else 1.0
        if _mean_arc > 0.0:
            rep_inv_cutoff = 1.0 / (rep_factor * _mean_arc)

    pos_pad = pos.astype(np.float32)
    exp_pad = exp_dist_mat.astype(np.float32)
    if B > n:
        pos_pad = np.concatenate([pos_pad, np.zeros((B - n, 3), dtype=np.float32)], axis=0)
        ep = np.zeros((B, B), dtype=np.float32)
        ep[:n, :n] = exp_pad
        exp_pad = ep

    return {
        "n": n,
        "pos": pos_pad,  # (B, 3)
        "exp_mat": exp_pad,  # (B, B)
        "excl_r0": excl_r0,
        "conf_cx": conf_cx,
        "conf_cy": conf_cy,
        "conf_cz": conf_cz,
        "conf_R": conf_R,
        "n_active": n,
        "rep_inv_cutoff": rep_inv_cutoff,
    }


def _arcs_tensor_bytes(B: int) -> int:
    """Exact device-tensor bytes for ONE IB of the arcs kernel at bucket B.

    Sums every input/output array of `kernel_full_mp` (each stacked on axis 0, so
    the batch is K times this); dominated by the (B, B) exp-distance tensor.  Used
    by `_resolve_arcs_max_k` with `memory.XLA_PEAK_OVERHEAD` to bound the peak."""
    f4, i4, b1 = 4, 4, 1  # bytes: float32 / int32 / bool
    inp = (
        B * 3 * f4  # pos_k
        + 3 * f4  # ss/se/sc
        + B * B * f4  # exp_k
        + f4  # step_size_k
        + f4  # excl_r0_k
        + 4 * f4  # conf cx,cy,cz,R
        + i4  # n_active_k
    )
    out = B * 3 * f4 + 3 * f4 + b1  # pos_f, scores, converged
    return inp + out


def _resolve_arcs_max_k(big_b: int, settings: "Settings") -> tuple[int, str]:
    """Resolve the arcs region-batch vmap width (IBs per launch).

    `settings.mc_executor_jax_batch_width_arcs` is an integer (flat cap) or
    "auto".  "auto" computes the kernel's exact per-IB device-tensor bytes
    (`_arcs_tensor_bytes`), applies `memory.XLA_PEAK_OVERHEAD`, and solves the
    largest K within the device budget (basis "auto-bytes").  When the budget
    can't be queried (CPU) it falls back to the conservative `32768/B` shape
    heuristic (basis "auto-fallback").  Returns (max_k, basis)."""
    w = str(settings.mc_executor_jax_batch_width_arcs).strip().lower()
    if w != "auto":
        return max(1, int(w)), "explicit"
    budget = jax_device_budget_bytes()
    if budget is None:
        return max(1, 32768 // max(1, big_b)), "auto-fallback"
    return max_k_for_bytes(_arcs_tensor_bytes(big_b), 0, budget), "auto-bytes"


def mc_arcs_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
    max_iters: int | None = None,
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Anneal K *different* IBs' anchors in one vmapped kernel (region batching).

    `max_iters` caps the outer round budget (default `_MAX_ITERS`=10000); the
    hybrid polish passes a low value to bound a slow-to-expand outlier IB.

    Each problem: `pos` (n,3), ``exp_dist`` (n,n), ``step_size`` (float).  All
    share the energy-term flags (caller groups by terms + size bucket).  Returns
    one ``(score, final_pos (n,3))`` per problem, in input order.

    Caps the vmap width (IBs per launch) via
    `settings.mc_executor_jax_batch_width_arcs` - see `_resolve_arcs_max_k`.  The
    stacked ``(B, B)`` exp tensor (4*B^2 bytes/IB) dominates the per-IB footprint.
    Excess IBs run as sequential sub-batches - same discipline as
    `mc_smooth_jax_batch`; the cap is purely an OOM guard."""
    if not problems:
        return []
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    big_b = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    max_k, basis = _resolve_arcs_max_k(big_b, settings)
    if len(problems) <= max_k:
        return _mc_arcs_jax_batch_chunk(problems, settings, max_iters)
    LOG.debug(
        "region-batch[arcs]: %d IBs > max_k=%d (%s) at B=%d; running %d sub-batches",
        len(problems),
        max_k,
        basis,
        big_b,
        -(-len(problems) // max_k),
    )
    out: list[tuple[float, np.ndarray[Any, Any]]] = []
    n_chunks = (len(problems) + max_k - 1) // max_k
    for ci, i in enumerate(range(0, len(problems), max_k), start=1):
        out.extend(
            _mc_arcs_jax_batch_chunk(
                problems[i : i + max_k], settings, max_iters, f", chunk {ci}/{n_chunks}"
            )
        )
    return out


def _mc_arcs_jax_batch_chunk(
    problems: list[dict[str, Any]],
    settings: "Settings",
    max_iters: int | None = None,
    chunk_tag: str = "",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """One vmapped arcs kernel launch for up to max_k IBs.  `max_iters` caps the
    outer round budget (``None`` => 10000); used by the polish to bound an outlier."""
    if not jax_is_available():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    import jax
    import jax.numpy as jnp

    K = len(problems)
    if K == 0:
        return []

    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    B = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    preps = [_prep_arcs_problem_np(p["pos"], p["exp_dist"], settings, B) for p in problems]

    def stack(key: str) -> Any:
        return jnp.asarray(np.stack([pr[key] for pr in preps], axis=0))

    pos_k = stack("pos")  # (K, B, 3)
    exp_k = stack("exp_mat")  # (K, B, B)
    n_active_k = jnp.asarray(np.array([pr["n_active"] for pr in preps], dtype=np.int32))
    excl_r0_k = jnp.asarray(np.array([pr["excl_r0"] for pr in preps], dtype=np.float32))
    conf_cx_k = jnp.asarray(np.array([pr["conf_cx"] for pr in preps], dtype=np.float32))
    conf_cy_k = jnp.asarray(np.array([pr["conf_cy"] for pr in preps], dtype=np.float32))
    conf_cz_k = jnp.asarray(np.array([pr["conf_cz"] for pr in preps], dtype=np.float32))
    conf_R_k = jnp.asarray(np.array([pr["conf_R"] for pr in preps], dtype=np.float32))
    rep_inv_cutoff_k = jnp.asarray(
        np.array([pr["rep_inv_cutoff"] for pr in preps], dtype=np.float32)
    )
    step_size_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    # shared schedule / weights
    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps_per_batch = int(settings.mc_stop_steps)
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    excl_w_v = float(settings.exclusion_weight) if use_excl else 0.0
    conf_w_v = float(settings.confinement_weight) if use_conf else 0.0
    stretch_v = float(settings.spring_stretch_arcs)
    squeeze_v = float(settings.spring_squeeze_arcs)

    bundle = _build_arcs_kernel(n_steps_per_batch, excl_skip)
    _kf, init_arcs, init_excl, init_confine, kernel_full_mp = bundle

    # per-IB initial scores (reuse the validated init helpers)
    def init_one(i: int) -> tuple[Any, Any, Any]:
        p1 = pos_k[i : i + 1]  # (1, B, 3)
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        ss = init_arcs(
            p1, exp_k[i], jnp.float32(stretch_v), jnp.float32(squeeze_v), rep_inv_cutoff_k[i]
        )
        se = (
            init_excl(p1, excl_r0_k[i], jnp.float32(excl_w_v), na)
            if use_excl
            else jnp.zeros((1,), jnp.float32)
        )
        sc = (
            init_confine(
                p1, conf_cx_k[i], conf_cy_k[i], conf_cz_k[i], conf_R_k[i], jnp.float32(conf_w_v), na
            )
            if use_conf
            else jnp.zeros((1,), jnp.float32)
        )
        return ss, se, sc

    t_init = time.perf_counter()
    inits = [init_one(i) for i in range(K)]
    ss_k = jnp.concatenate([x[0] for x in inits])
    se_k = jnp.concatenate([x[1] for x in inits])
    sc_k = jnp.concatenate([x[2] for x in inits])
    init_ms = -1.0
    if _ARCS_PROFILE:
        jax.block_until_ready((ss_k, se_k, sc_k))  # force the O(N^2) init scans
        init_ms = (time.perf_counter() - t_init) * 1e3

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    _kf_args = (
        pos_k,
        ss_k,
        se_k,
        sc_k,
        jnp.float32(settings.max_temp),
        exp_k,
        step_size_k,
        jnp.float32(settings.dt_temp),
        jnp.float32(settings.jump_scale),
        jnp.float32(settings.jump_coef),
        jnp.float32(stretch_v),
        jnp.float32(squeeze_v),
        excl_r0_k,
        jnp.float32(excl_w_v),
        conf_cx_k,
        conf_cy_k,
        conf_cz_k,
        conf_R_k,
        jnp.float32(conf_w_v),
        base_key,
        jnp.float32(settings.mc_stop_improvement),
        jnp.int32(settings.mc_stop_successes),
        jnp.float32(_ARCS_FORCE_SCORE_EPS if _ARCS_FORCE_SCORE_EPS is not None else 1e-5),
        jnp.float32(0.9999),
        n_active_k,
        rep_inv_cutoff_k,
        jnp.int32(max_iters if max_iters is not None else 10000),
    )

    log_kernel_start(LOG, "arcs", "mc", K, B, f"sequential single-bead{chunk_tag}")
    t0 = time.perf_counter()
    if _ARCS_PROFILE:
        out, compile_ms, run_ms, cold, cost = _profile_kernel(kernel_full_mp, _kf_args, K, B)
    else:
        out = kernel_full_mp(*_kf_args)
        compile_ms = run_ms = -1.0
        cold = False
        cost = None
    pos_f, ss_f, se_f, sc_f, iter_f, converged, conv_iter = out
    score_per_chain = np.asarray(ss_f + se_f + sc_f)  # forces device sync
    pos_f_np = np.asarray(pos_f)
    elapsed = time.perf_counter() - t0

    # --- convergence-spread diagnostics (host-only; rides the sync above) ----------
    # conv_iter[k] = outer-iter chain k first converged (0 = never).  When the batch
    # exits via jnp.all(converged), iter_f == max(conv_iter); when it exits via the
    # _MAX_ITERS cap, never-converged chains stay 0 (counted in `never`) and
    # max(conv_iter) over converged chains may be < iter_f.  `wasted` (sum of
    # iter_f - conv_iter over converged chains) is the frozen-but-stepped overshoot.
    it = int(iter_f)
    ci = np.asarray(conv_iter)
    conv_mask = np.asarray(converged)
    done = ci > 0  # chains that converged AND recorded an iter (== conv_mask)
    n_conv = int(conv_mask.sum())
    if done.any():
        cd = ci[done]
        p50 = int(np.median(cd))
        p90 = int(np.percentile(cd, 90))
        mx = int(cd.max())
        wasted = int((it - cd).sum())
    else:
        p50 = p90 = mx = wasted = 0
    never = K - n_conv
    # us/iter from execute-only time when profiling (run_ms excludes the XLA compile
    # that `elapsed` includes on cold launches); else the lumped wall.
    wall_for_iter = run_ms / 1e3 if (_ARCS_PROFILE and run_ms > 0) else elapsed
    us_per_iter = (wall_for_iter / max(it, 1)) * 1e6
    wasted_frac = 100.0 * wasted / max(it * K, 1)

    with _arcs_profile_lock:
        _last_batch_diag.clear()
        _last_batch_diag.update(
            K=K,
            B=B,
            iter_f=it,
            conv_iter=ci,
            converged=conv_mask,
            elapsed_s=elapsed,
            compile_ms=compile_ms,
            run_ms=run_ms,
            init_ms=init_ms,
            cold=cold,
            us_per_iter=us_per_iter,
            p50=p50,
            p90=p90,
            max=mx,
            never=never,
            wasted=wasted,
        )
        if _ARCS_PROFILE:
            rec = _arcs_profile.setdefault(
                (K, B),
                {
                    "launches": 0.0,
                    "compiles": 0.0,
                    "compile_ms": 0.0,
                    "run_ms": 0.0,
                    "init_ms": 0.0,
                    "iters": 0.0,
                    "wall_s": 0.0,
                },
            )
            rec["launches"] += 1
            rec["compiles"] += 1 if cold else 0
            rec["compile_ms"] += max(compile_ms, 0.0)
            rec["run_ms"] += max(run_ms, 0.0)
            rec["init_ms"] += max(init_ms, 0.0)
            rec["iters"] += it
            rec["wall_s"] += elapsed

    prof_tail = ""
    if _ARCS_PROFILE:
        flops = cost.get("flops") if isinstance(cost, dict) else None
        flops_str = f" flops={flops:.2e}" if flops else ""
        warm_str = "COLD" if cold else "warm"
        prof_tail = (
            f" [compile={compile_ms:.0f}ms run={run_ms:.0f}ms "
            f"init={init_ms:.0f}ms {warm_str}{flops_str}]"
        )
    log_kernel_done(
        LOG,
        "arcs",
        "mc",
        K,
        elapsed,
        f"{it} rounds ({it * n_steps_per_batch} steps), {n_conv}/{K} converged "
        f"(median {p50}, slowest {mx}); {wasted_frac:.0f}% wasted{prof_tail}",
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        n_i = pr["n"]
        results.append((float(score_per_chain[i]), pos_f_np[i, :n_i].astype(np.float32)))
    return results
