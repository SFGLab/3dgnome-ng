"""JAX backend: region-batched MC entry points, shape bucketing, and prep.

JAX wins on the MC hot path because xla.vmap + lax.fori_loop fuses a whole
annealing batch (K IBs in parallel) + the O(N) per-step reductions into one GPU
kernel — profiles show smooth-MC alone is 89-96% of MC wall time at N=2000-10000.

This module is the JAX *plumbing*, not the energy math:
  - `mc_arcs_jax_batch` / `mc_smooth_jax_batch` — region-batched entry points
    (anneal K different IBs at once), used by the pipeline's IB stages;
  - `mc_heatmap_jax` — the chr/segment heatmap kernel (single-problem, used by
    the coarse pipeline);
  - `_prep_*_problem_np` — per-IB numpy prep, padded to shape buckets;
  - `_bucket_for` / `_SHAPE_BUCKETS` — the shape-ladder bucketing.

The actual per-step kernel is the COMPOSEABLE `gnome3d.mc.jax_driver.build_mc_kernel`,
which composes an ordered recipe of `gnome3d.mc.terms` (chain / arc-springs /
excluded-volume / subanchor-heat / orientation / confinement).  The batch entries
here build each IB's term params + init scores and hand a recipe to the driver;
the former hand-written per-stage kernels (`_build_arcs_kernel` /
`_build_smooth_kernel`) were retired once the driver matched them byte-exact.

JAX is an optional extras dep; `_ensure_jax()` lazy-imports.  The persistent
compile cache at `~/.cache/gnome3d/jax` makes per-shape compiles a one-time
cost across all runs on a machine.
"""

# NB: no `from __future__ import annotations` — JAX kernels reflect on live
# type objects via decorators.  String-form annotations are fine elsewhere in
# this file but the kernel definitions below are not annotation-sensitive.

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.types import F32Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

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

def is_available() -> bool:
    """Public: True if JAX is importable in the current environment."""
    return _ensure_jax()


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
    if bool(settings.mc_executor_jax_bucket_shapes):
        if settings.mc_executor_jax_precompile_buckets:
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
    # Per-call RNG diversity keyed on the active scope path (was: the label).
    _seed_src = log.current()
    seed_offset: int = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
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


# ---------------------------------------------------------------------------
# Region-batched smooth-MC: anneal K DIFFERENT IBs in one vmapped kernel.
#
# Profile + kscan (playground/bench/bench_jax_grad_vs_mc.py --kscan) showed a
# K=1 smooth at N~1600 leaves the GPU ~99% idle: total wall is flat from K=1
# to K~16-32.  So instead of annealing hundreds of independent IBs one at a
# time, we pad each to a common bucket and run a whole group as one kernel —
# ~16-50x on the dominant phase, for free.  Every array is per-IB (stacked on
# axis 0) and convergence is per-chain; the kernel itself is the composeable
# `gnome3d.mc.jax_driver.build_mc_kernel` (the per-stage hand-written kernels
# this file used to carry were retired once that driver matched them byte-exact).
# ---------------------------------------------------------------------------


def _prep_smooth_problem_np(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None,
    anchor_neighbors: dict[int, list[int]] | None,
    anchor_neighbor_weights: dict[int, list[float]] | None,
    heat_dist: np.ndarray[Any, Any] | None,
    B: int,
    A: int,
    M: int,
) -> dict[str, Any]:
    """Build one IB's kernel inputs as numpy arrays, padded to a common bucket
    (B beads, A anchors, M neighbours) so a batch of IBs has uniform shapes.

    Pure numpy (no JAX) → unit-testable in isolation.  B/A/M are the batch's
    common bucket (passed in, not derived per-region), and
    arrays are always padded to them (the batched kernel needs uniform shapes
    regardless of the `jax_bucket_shapes` flag).
    """
    n = int(pos.shape[0])
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_heat = heat_dist is not None
    use_orn = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)

    # --- excluded-volume radius (auto-derived per IB if radius<=0) ---
    excl_r0 = 1.0
    if use_excl:
        excl_r0 = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            excl_r0 = float(settings.exclusion_auto_factor_smooth) * float(np.asarray(dtn).mean())

    # --- confinement envelope (centroid + radius, per IB) ---
    if use_conf:
        conf_cx = float(pos[:, 0].mean())
        conf_cy = float(pos[:, 1].mean())
        conf_cz = float(pos[:, 2].mean())
        conf_R = float(settings.confinement_radius_smooth)
        if conf_R <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w = float(settings.confinement_weight)
    else:
        conf_cx = conf_cy = conf_cz = 0.0
        conf_R = 1.0
        conf_w = 0.0

    # --- orientation CSR (anchor-indexed), padded to (A, M) ---
    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar))
        nbr_lists = [list(anchor_neighbors.get(k, [])) for k in range(n_anchors)]
        nbr_w_lists = [list(anchor_neighbor_weights.get(k, [])) for k in range(n_anchors)]
        nbr_idx = np.zeros((n_anchors, M), dtype=np.int32)
        nbr_w = np.zeros((n_anchors, M), dtype=np.float32)
        nbr_valid = np.zeros((n_anchors, M), dtype=np.bool_)
        for k_idx in range(n_anchors):
            for m, (jn, wn) in enumerate(zip(nbr_lists[k_idx], nbr_w_lists[k_idx], strict=True)):
                nbr_idx[k_idx, m] = int(jn)
                nbr_w[k_idx, m] = float(wn)
                nbr_valid[k_idx, m] = True
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        for k_idx in range(n_anchors):
            ar = int(anchor_ar[k_idx])
            if ar > 0:
                bead_to_anchor_k[ar - 1] = k_idx
            if ar + 1 < n:
                bead_to_anchor_k[ar + 1] = k_idx
        is_L = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # pad anchor-indexed arrays to A (pad anchors get nbr_valid=False -> 0 score)
        if A > n_anchors:
            ap = A - n_anchors
            anchor_ar = np.concatenate([anchor_ar, np.zeros(ap, dtype=np.int32)])
            nbr_idx = np.pad(nbr_idx, ((0, ap), (0, 0)))
            nbr_w = np.pad(nbr_w, ((0, ap), (0, 0)))
            nbr_valid = np.pad(nbr_valid, ((0, ap), (0, 0)))
    else:
        anchor_ar = np.zeros(A, dtype=np.int32)
        nbr_idx = np.zeros((A, M), dtype=np.int32)
        nbr_w = np.zeros((A, M), dtype=np.float32)
        nbr_valid = np.zeros((A, M), dtype=np.bool_)
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        is_L = np.zeros(n, dtype=np.bool_)

    # --- bead-indexed arrays, padded to B ---
    movable = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    n_movable = int(movable.shape[0])
    pos_pad = pos.astype(np.float32)
    dtn_pad = dtn.astype(np.float32)
    heat_pad = heat_dist.astype(np.float32) if use_heat else np.zeros((1, 1), dtype=np.float32)
    if B > n:
        n_pad = B - n
        pos_pad = np.concatenate([pos_pad, np.zeros((n_pad, 3), dtype=np.float32)], axis=0)
        dtn_pad = np.concatenate([dtn_pad, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat:
            hp = np.zeros((B, B), dtype=np.float32)
            hp[:n, :n] = heat_pad
            heat_pad = hp
        bead_to_anchor_k = np.concatenate([bead_to_anchor_k, np.full(n_pad, -1, dtype=np.int32)])
        is_L = np.concatenate([is_L, np.zeros(n_pad, dtype=np.bool_)])

    # `movable` lists non-fixed beads, so its length is n_movable (not n) — pad
    # it to B independently of the `B > n` bead-padding above, else an IB whose
    # n lands exactly on the bucket (B == n) keeps a short movable and the
    # batched (K, B) stack goes ragged.  Pad indices are 0 (a valid bead);
    # n_movable bounds the kernel loop, so they're never read.
    if B > n_movable:
        movable = np.concatenate([movable, np.zeros(B - n_movable, dtype=movable.dtype)])

    return {
        "n": n,
        "pos": pos_pad,  # (B, 3)
        "dtn": dtn_pad,  # (B,)
        "movable": movable,  # (B,)
        "heat": heat_pad,  # (B, B) or (1, 1)
        "anchor_ar": anchor_ar,  # (A,)
        "bead_to_anchor_k": bead_to_anchor_k,  # (B,)
        "nbr_idx": nbr_idx,  # (A, M)
        "nbr_w": nbr_w,  # (A, M)
        "nbr_valid": nbr_valid,  # (A, M)
        "is_L": is_L,  # (B,)
        "n_active": n,
        "n_movable": n_movable,
        "excl_r0": excl_r0,
        "conf_cx": conf_cx,
        "conf_cy": conf_cy,
        "conf_cz": conf_cz,
        "conf_R": conf_R,
        "conf_w": conf_w,
    }


def mc_smooth_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Anneal K *different* IBs in one vmapped kernel (region batching).

    `problems` is a list of dicts, each describing one IB's final smooth:
        pos (n,3), dtn (n,), fixed (n,) bool, step_size (float),
        and optionally heat_dist (n,n), char_orientations (n,),
        anchor_neighbors, anchor_neighbor_weights.
    All problems must share the same energy-term flags (use_heat/use_orn/...)
    — the caller groups IBs by (terms, size bucket) before calling.

    Returns one (score, final_pos (n_i, 3)) per problem, in input order.
    Does not mutate the inputs.

    Caps the vmap width at `max(1, 32768 // B)` (the kscan saturation point:
    wider is free, not faster) and runs any excess in sequential sub-batches.
    That cap also bounds the stacked `(K, B, B)` heat tensor to <= ~131072*B
    bytes (~4.3 GB at the largest bucket), so a dense, heat-carrying chromosome
    can't OOM the GPU regardless of how many IBs land in one bucket.
    """
    if not problems:
        return []
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    bsz = [int(p["pos"].shape[0]) for p in problems]
    big_b = max((_bucket_for(n) if bucket else n) for n in bsz)
    max_k = max(1, 32768 // max(1, big_b))
    if len(problems) <= max_k:
        return _mc_smooth_jax_batch_chunk(problems, settings)
    LOG.debug(
        "region-batch: %d IBs > max_k=%d at B=%d; running %d sub-batches",
        len(problems),
        max_k,
        big_b,
        -(-len(problems) // max_k),
    )
    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i in range(0, len(problems), max_k):
        results.extend(_mc_smooth_jax_batch_chunk(problems[i : i + max_k], settings))
    return results


def _mc_smooth_jax_batch_chunk(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """One vmapped kernel launch for up to `max_k` IBs. See `mc_smooth_jax_batch`,
    which chunks to this so the device tensors stay bounded."""
    if not _ensure_jax():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    K = len(problems)
    if K == 0:
        return []

    # Common bucket across the group: pad every IB to the max (B, A, M) so the
    # kernel sees one uniform shape.  Callers should pre-group by bucket so
    # these maxes are tight (wall-clock = slowest IB in the batch).
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    use_orn = (
        problems[0].get("char_orientations") is not None
        and problems[0].get("anchor_neighbors") is not None
        and problems[0].get("anchor_neighbor_weights") is not None
    )
    use_heat = problems[0].get("heat_dist") is not None
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)

    Bs, As, Ms = [], [], []
    for p in problems:
        n_i = int(p["pos"].shape[0])
        Bs.append(_bucket_for(n_i) if bucket else n_i)
        if use_orn:
            anchors_i = int(np.count_nonzero(p["fixed"]))
            nbrs_i = max(
                (len(p["anchor_neighbors"].get(k, [])) for k in range(anchors_i)), default=1
            )
            nbrs_i = max(nbrs_i, 1)
            As.append(_bucket_for(anchors_i, _ANCHOR_BUCKETS) if bucket else anchors_i)
            Ms.append(_bucket_for(nbrs_i, _NBR_BUCKETS) if bucket else nbrs_i)
        else:
            As.append(1)
            Ms.append(1)
    B, A, M = max(Bs), max(As), max(Ms)

    preps = [
        _prep_smooth_problem_np(
            p["pos"],
            p["dtn"],
            p["fixed"],
            settings,
            p.get("char_orientations"),
            p.get("anchor_neighbors"),
            p.get("anchor_neighbor_weights"),
            p.get("heat_dist"),
            B,
            A,
            M,
        )
        for p in problems
    ]

    # --- stack per-IB arrays -> (K, ...) ---
    def stack(key: str) -> Any:
        return jnp.asarray(np.stack([pr[key] for pr in preps], axis=0))

    pos_k = stack("pos")  # (K, B, 3)
    dtn_k = stack("dtn")
    movable_k = stack("movable")
    heat_k = stack("heat")
    anchor_ar_k = stack("anchor_ar")
    b2a_k = stack("bead_to_anchor_k")
    nbr_idx_k = stack("nbr_idx")
    nbr_w_k = stack("nbr_w")
    nbr_valid_k = stack("nbr_valid")
    is_L_k = stack("is_L")
    n_active_k = jnp.asarray(np.array([pr["n_active"] for pr in preps], dtype=np.int32))
    n_movable_k = jnp.asarray(np.array([pr["n_movable"] for pr in preps], dtype=np.int32))
    excl_r0_k = jnp.asarray(np.array([pr["excl_r0"] for pr in preps], dtype=np.float32))
    conf_cx_k = jnp.asarray(np.array([pr["conf_cx"] for pr in preps], dtype=np.float32))
    conf_cy_k = jnp.asarray(np.array([pr["conf_cy"] for pr in preps], dtype=np.float32))
    conf_cz_k = jnp.asarray(np.array([pr["conf_cz"] for pr in preps], dtype=np.float32))
    conf_R_k = jnp.asarray(np.array([pr["conf_R"] for pr in preps], dtype=np.float32))
    conf_w_k = jnp.asarray(np.array([pr["conf_w"] for pr in preps], dtype=np.float32))
    step_size_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    # per-IB springs (uniform from settings for now; small-IB boost groups
    # would carry their own — callers batch boosted IBs separately).
    stretch_k = jnp.full((K,), jnp.float32(settings.spring_stretch))
    squeeze_k = jnp.full((K,), jnp.float32(settings.spring_squeeze))
    ang_k = jnp.full((K,), jnp.float32(settings.spring_angular))

    # --- shared (global) schedule + weights ---
    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps_per_batch = int(settings.mc_stop_steps_smooth)
    heat_weight_v = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    motif_weight_v = float(settings.motif_weight) if use_orn else 0.0
    excl_w_v = float(settings.exclusion_weight) if use_excl else 0.0

    dist_w = jnp.float32(settings.smooth_dist_weight)
    ang_w = jnp.float32(settings.smooth_angle_weight)
    symmetric = jnp.bool_(bool(getattr(settings, "motifs_symmetric", True)))

    # Compose the smooth recipe into one generic kernel (gnome3d.mc.jax_driver) —
    # byte-identical to the former hand-written `_build_smooth_kernel`
    # (validate_driver_smooth).  Each optional term is included ONLY when its
    # setting is on (CHAIN is the always-present structure term); the recipe thus
    # reads as the active settings.  Order = the old ss+se+sh+so+sc sum order, and
    # an omitted term is byte-exact (it contributed exactly 0 in the old kernel).
    from gnome3d.mc.jax_driver import build_mc_kernel
    from gnome3d.mc.terms import CHAIN, CONFINEMENT, EXCLUDED_VOLUME, ORIENTATION, SUBANCHOR_HEAT
    from gnome3d.mc.terms.chain import ChainP
    from gnome3d.mc.terms.confinement import ConfP
    from gnome3d.mc.terms.excluded_volume import ExclP
    from gnome3d.mc.terms.orientation import (
        OrnP,
        init_anchor_orientations_jax,
        init_orientation_score_jax,
    )
    from gnome3d.mc.terms.subanchor_heat import HeatP

    onesK = jnp.ones((K,), jnp.float32)
    # Each spec: (Term, stacked per-IB params, per-IB init fn (na, i) -> score).
    specs: list[tuple[Any, Any, Any]] = [
        (CHAIN, ChainP(dtn_k, stretch_k, squeeze_k, ang_k, onesK * dist_w, onesK * ang_w),
         lambda i, na: CHAIN.jax_init(pos_k[i], ChainP(dtn_k[i], stretch_k[i], squeeze_k[i], ang_k[i], dist_w, ang_w), na)),
    ]
    if use_excl:
        specs.append((EXCLUDED_VOLUME, ExclP(excl_r0_k, onesK * jnp.float32(excl_w_v), (onesK * excl_skip).astype(jnp.int32)),
                      lambda i, na: EXCLUDED_VOLUME.jax_init(pos_k[i], ExclP(excl_r0_k[i], jnp.float32(excl_w_v), jnp.int32(excl_skip)), na)))
    if use_heat:
        specs.append((SUBANCHOR_HEAT, HeatP(heat_k, onesK * jnp.float32(heat_weight_v)),
                      lambda i, na: SUBANCHOR_HEAT.jax_init(pos_k[i], HeatP(heat_k[i], jnp.float32(heat_weight_v)), na)))
    if use_orn:
        specs.append((ORIENTATION, OrnP(b2a_k, anchor_ar_k, is_L_k, nbr_idx_k, nbr_w_k, nbr_valid_k,
                                        onesK * jnp.float32(motif_weight_v), jnp.full((K,), symmetric)),
                      lambda i, na: init_orientation_score_jax(
                          init_anchor_orientations_jax(pos_k[i], anchor_ar_k[i], is_L_k[i]),
                          nbr_idx_k[i], nbr_w_k[i], nbr_valid_k[i], jnp.float32(motif_weight_v), symmetric)))
    if use_conf:
        specs.append((CONFINEMENT, ConfP(conf_cx_k, conf_cy_k, conf_cz_k, conf_R_k, conf_w_k),
                      lambda i, na: CONFINEMENT.jax_init(pos_k[i], ConfP(conf_cx_k[i], conf_cy_k[i], conf_cz_k[i], conf_R_k[i], conf_w_k[i]), na)))

    recipe = [sp[0] for sp in specs]
    term_params = tuple(sp[1] for sp in specs)
    init_fns = [sp[2] for sp in specs]

    # per-IB init: one score component per recipe term (verbatim term inits) + the
    # anchor_orn cache the orientation term mutates (dummy when orientation is off).
    def init_one(i: int) -> tuple[list[Any], Any]:
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        comps = [fn(i, na) for fn in init_fns]
        ao = (
            init_anchor_orientations_jax(pos_k[i], anchor_ar_k[i], is_L_k[i])
            if use_orn
            else jnp.zeros((A, 3), jnp.float32)
        )
        return comps, ao

    inits = [init_one(i) for i in range(K)]
    scores_k = tuple(jnp.asarray([inits[i][0][t] for i in range(K)]) for t in range(len(recipe)))
    anchor_orn_k = jnp.stack([inits[i][1] for i in range(K)])

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    log.status(
        LOG,
        "    smooth kernel: K=%d B=%d A=%d M=%d (heat=%d orn=%d), "
        "compiling/running (first call per shape compiles)...",
        K, B, A, M, int(use_heat), int(use_orn),
    )
    kernel_full_mp = build_mc_kernel(recipe, n_steps_per_batch, strict_accept=True, freeze_converged=False)
    t0 = time.perf_counter()
    out = kernel_full_mp(
        pos_k, scores_k, anchor_orn_k,
        jnp.float32(settings.max_temp_smooth), tuple(term_params), movable_k, step_size_k,
        jnp.float32(settings.dt_temp_smooth), jnp.float32(settings.jump_scale_smooth), jnp.float32(settings.jump_coef_smooth),
        base_key,
        jnp.float32(settings.mc_stop_improvement_smooth), jnp.int32(settings.mc_stop_successes_smooth),
        jnp.float32(1e-6), jnp.float32(np.inf),  # score_eps; stop_ratio=inf disables the (smooth-absent) ratio guard
        n_active_k, n_movable_k,
    )
    pos_f, scores_f, _ao_f, iter_count, converged = out
    score_per_chain = np.asarray(sum(scores_f))  # forces device sync
    pos_f_np = np.asarray(pos_f)

    n_steps_smooth = int(settings.mc_stop_steps_smooth)
    log.status(
        LOG,
        "    smooth region-batch: K=%d B=%d A=%d M=%d (heat=%d orn=%d), "
        "%d batches (%d steps), %d/%d converged, %.1fs",
        K, B, A, M, int(use_heat), int(use_orn),
        int(iter_count), int(iter_count) * n_steps_smooth,
        int(np.asarray(converged).sum()), K, time.perf_counter() - t0,
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        n_i = pr["n"]
        results.append((float(score_per_chain[i]), pos_f_np[i, :n_i].astype(np.float32)))
    return results


# ---------------------------------------------------------------------------
# Region-batched arcs-MC: anneal K DIFFERENT IBs' anchors in one vmapped kernel.
# Mirrors mc_smooth_jax_batch (per-chain convergence, shape bucketing, chunk/OOM
# cap).  The one difference is the kernel freezes converged chains (arcs is
# non-strict) — `build_mc_kernel(..., freeze_converged=True)` in the driver.
# ---------------------------------------------------------------------------


def _prep_arcs_problem_np(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    settings: "Settings",
    B: int,
) -> dict[str, Any]:
    """One IB's arcs kernel inputs as numpy, padded to bucket B (per-IB excl
    radius + confinement envelope); pad beads are inert (exp_mat=0 rows/cols,
    n_active masks EV/confine)."""
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
    }


def mc_arcs_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Anneal K *different* IBs' anchors in one vmapped kernel (region batching).

    Each problem: ``pos`` (n,3), ``exp_dist`` (n,n), ``step_size`` (float).  All
    share the energy-term flags (caller groups by terms + size bucket).  Returns
    one ``(score, final_pos (n,3))`` per problem, in input order.

    Caps the vmap width at ``max(1, 32768 // B)`` (saturation point + bounds the
    stacked (K,B,B) exp tensor), running excess as sequential sub-batches — same
    discipline as `mc_smooth_jax_batch`."""
    if not problems:
        return []
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    big_b = max((_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0])) for p in problems)
    max_k = max(1, 32768 // max(1, big_b))
    if len(problems) <= max_k:
        return _mc_arcs_jax_batch_chunk(problems, settings)
    out: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i in range(0, len(problems), max_k):
        out.extend(_mc_arcs_jax_batch_chunk(problems[i : i + max_k], settings))
    return out


def _mc_arcs_jax_batch_chunk(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """One vmapped arcs kernel launch for up to max_k IBs."""
    if not _ensure_jax():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

    K = len(problems)
    if K == 0:
        return []

    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    B = max((_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0])) for p in problems)
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

    # Compose the arcs recipe [arc_springs, excluded_volume, confinement] into one
    # generic kernel (gnome3d.mc.jax_driver) — byte-identical to the former
    # hand-written `_build_arcs_kernel` (validate_driver_arcs).  Per-IB params ride
    # as each term's namedtuple (shared scalars broadcast to (K,)).
    from gnome3d.mc.jax_driver import build_mc_kernel
    from gnome3d.mc.terms import ARC_SPRINGS, CONFINEMENT, EXCLUDED_VOLUME
    from gnome3d.mc.terms.arc_springs import ArcP
    from gnome3d.mc.terms.confinement import ConfP
    from gnome3d.mc.terms.excluded_volume import ExclP

    onesK = jnp.ones((K,), jnp.float32)
    term_params = (
        ArcP(exp_k, onesK * stretch_v, onesK * squeeze_v),
        ExclP(excl_r0_k, onesK * excl_w_v, (onesK * excl_skip).astype(jnp.int32)),
        ConfP(conf_cx_k, conf_cy_k, conf_cz_k, conf_R_k, onesK * conf_w_v),
    )

    # per-IB initial scores (verbatim term inits — same values as the old bundle).
    def init_one(i: int) -> tuple[Any, Any, Any]:
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        ss = ARC_SPRINGS.jax_init(pos_k[i], ArcP(exp_k[i], jnp.float32(stretch_v), jnp.float32(squeeze_v)), na)
        se = (
            EXCLUDED_VOLUME.jax_init(pos_k[i], ExclP(excl_r0_k[i], jnp.float32(excl_w_v), jnp.int32(excl_skip)), na)
            if use_excl
            else jnp.float32(0.0)
        )
        sc = (
            CONFINEMENT.jax_init(
                pos_k[i], ConfP(conf_cx_k[i], conf_cy_k[i], conf_cz_k[i], conf_R_k[i], jnp.float32(conf_w_v)), na
            )
            if use_conf
            else jnp.float32(0.0)
        )
        return ss, se, sc

    inits = [init_one(i) for i in range(K)]
    ss_k = jnp.asarray([x[0] for x in inits])
    se_k = jnp.asarray([x[1] for x in inits])
    sc_k = jnp.asarray([x[2] for x in inits])

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    kernel_full_mp = build_mc_kernel(
        [ARC_SPRINGS, EXCLUDED_VOLUME, CONFINEMENT],
        n_steps_per_batch,
        strict_accept=False,
        freeze_converged=True,
    )
    # arcs moves any real bead: movable = arange(B), n_movable_active = n_active.
    movable_k = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32), (K, B))
    anchor_dummy = jnp.zeros((K, 1, 3), jnp.float32)  # no orientation term in the arcs recipe
    log.status(
        LOG, "    arcs kernel: K=%d B=%d, compiling/running (first call per shape compiles)...", K, B
    )
    t0 = time.perf_counter()
    out = kernel_full_mp(
        pos_k, (ss_k, se_k, sc_k), anchor_dummy,
        jnp.float32(settings.max_temp), term_params, movable_k, step_size_k,
        jnp.float32(settings.dt_temp), jnp.float32(settings.jump_scale), jnp.float32(settings.jump_coef),
        base_key,
        jnp.float32(settings.mc_stop_improvement), jnp.int32(settings.mc_stop_successes),
        jnp.float32(1e-5), jnp.float32(0.9999), n_active_k, n_active_k,
    )
    pos_f, scores_f, _anchor_f, iter_f, converged = out
    score_per_chain = np.asarray(scores_f[0] + scores_f[1] + scores_f[2])  # forces device sync
    pos_f_np = np.asarray(pos_f)
    log.status(
        LOG, "    arcs region-batch: K=%d B=%d, %d batches (%d steps), %d/%d converged, %.1fs",
        K, B, int(iter_f), int(iter_f) * n_steps_per_batch,
        int(np.asarray(converged).sum()), K, time.perf_counter() - t0,
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        n_i = pr["n"]
        results.append((float(score_per_chain[i]), pos_f_np[i, :n_i].astype(np.float32)))
    return results
