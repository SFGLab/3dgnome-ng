"""JAX heatmap-energy MC kernel (simplest energy: pairwise distance to expected).

`mc_heatmap_jax` is the single entry; it builds/compiles through
`_build_heatmap_kernel` (memoised in `_kernel_cache`) and can eagerly precompile
every shape bucket via `_precompile_heatmap`.  No orientation / heat / confinement
- double-counted structure plus optional excluded volume only.
"""

# NB: no `from __future__ import annotations` - the JAX kernels reflect on live
# type objects, so the kernel definitions below must see real annotations.

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.types import F32Array
from gnome3d.util import _SHAPE_BUCKETS, jax_bucket_for, jax_is_available

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

# Compiled-kernel cache (keyed by kernel signature), precompile-dedup set, and
# the build lock (ib_workers>1 may race several threads into a kernel build).
_kernel_cache: dict[Any, Any] = {}
_precompiled: set[Any] = set()
_init_lock = threading.Lock()


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

    import jax
    import jax.numpy as jnp

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
        # their heat rows are masked (skip=True) - fully inert.
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
    if not jax_is_available():
        return
    import jax
    import jax.numpy as jnp

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
        # program / cache key - the values are runtime inputs, so the real call
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
                LOG.warning("precompile heatmap bucket %d skipped: %s", b, e)
        _precompiled.add(sig)
        dt = __import__("time").perf_counter() - t0
        log.status(
            LOG,
            "precompiled heatmap kernel for %d buckets (K=%d) in %.1fs",
            len(_SHAPE_BUCKETS),
            K,
            dt,
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
    if bool(settings.mc_executor_jax_bucket_shapes):
        if settings.mc_executor_jax_precompile_buckets:
            _precompile_heatmap(settings)
        B: int = jax_bucket_for(n)
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
    # Per-call RNG diversity keyed on the active scope path (was: the label).
    _seed_src = log.current()
    seed_offset: int = abs(hash(_seed_src)) % (2 ** 31) if _seed_src else 0
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


