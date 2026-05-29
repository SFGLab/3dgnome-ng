"""JAX backend for the smooth-MC + excluded-volume hot path.

Production dryrun profile (chr22, chr4) shows mc_smooth eats 89-96% of total
MC wall time, with the heavy calls living at N=2000-10000.  Benchmarks
([playground/bench_jax_smooth_mc.py]) show JAX at f32 is 5-70x faster than
numba on that regime — JAX wins because xla.vmap + lax.fori_loop fuses the
entire 5000-step annealing AND the O(N) per-step EV reduction into a single
GPU kernel.

This module implements ONLY the "simple-config" smooth-MC path: chain bonds
+ angles + optional excluded volume.  Orientation / confinement configs are
NOT supported here and fall through to the numba path in [mc.py::mc_smooth].
That matches the existing `_mc_smooth_multichain` gate.

JAX is an OPTIONAL extras dep (pyproject `[jax]`).  Importing this module
without JAX installed must not crash — `_ensure_jax()` lazy-imports.  The
caller dispatch in [mc.py] raises a clear error only when `mc_backend='jax'`
is explicitly set without JAX installed.

Compile cache: first call per (N, n_steps_per_batch, n_movable) shape pays a
~1-50s JAX/XLA compile cost.  We set up `jax.experimental.compilation_cache`
at module init so the cost is paid once per machine, not per process.
"""

# NB: no `from __future__ import annotations` — settings type-checks reference
# Settings at runtime via TYPE_CHECKING; we keep runtime annotations live so
# pyright can resolve them.

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import F32Array, I64Array

if TYPE_CHECKING:
    from .settings import Settings


# ---------------------------------------------------------------------------
# Lazy import + compile-cache setup
# ---------------------------------------------------------------------------

_JAX_AVAILABLE: bool | None = None  # None = not yet probed
_jax: Any = None
_jnp: Any = None
_kernel_cache: dict[tuple[int, int], Any] = {}


def _ensure_jax() -> bool:
    """Lazy-import JAX.  Returns True on success, False if not installed.
    Idempotent — subsequent calls hit the cached `_JAX_AVAILABLE` flag."""
    global _JAX_AVAILABLE, _jax, _jnp
    if _JAX_AVAILABLE is not None:
        return _JAX_AVAILABLE
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except ImportError:
        _JAX_AVAILABLE = False
        return False
    # Float32 is the production dtype for this backend; the benchmark showed
    # f64 is 2x slower on consumer GPUs (1/32 throughput) with no quality
    # benefit at the run lengths we care about.  We do NOT enable_x64.
    cache_dir = os.environ.get(
        "GNOME3D_JAX_CACHE", os.path.expanduser("~/.cache/gnome3d/jax")
    )
    try:
        from jax.experimental import compilation_cache  # type: ignore[import-not-found]

        compilation_cache.compilation_cache.set_cache_dir(cache_dir)  # pyright: ignore[reportUnknownMemberType]
    except (ImportError, AttributeError):
        # older JAX, or API moved — proceed without persistent cache; each
        # process pays compile cost on first call per shape.
        pass
    _jax = jax
    _jnp = jnp
    _JAX_AVAILABLE = True
    return True


def is_available() -> bool:
    """Public: True if JAX is importable in the current environment."""
    return _ensure_jax()


# ---------------------------------------------------------------------------
# Kernel construction (cached per (n_steps_per_batch, excl_skip))
# ---------------------------------------------------------------------------


def _build_smooth_ev_kernel(n_steps_per_batch: int, excl_skip: int) -> Any:
    """Build (or look up cached) compiled smooth-MC + EV kernel.

    The returned callable is jit'd + vmapped over K chains.  It runs ONE batch
    of `n_steps_per_batch` MC steps for K chains in lockstep and returns
    (pos, score_struct, score_excl, T_final, n_ok).

    Static-by-cache-key: `n_steps_per_batch` (shape of pre-gen RNG arrays),
    `excl_skip` (folded into the EV mask).  JAX further shape-specialises on
    (N, K) per call but those are runtime args.
    """
    key = (n_steps_per_batch, excl_skip)
    if key in _kernel_cache:
        return _kernel_cache[key]

    assert _jax is not None and _jnp is not None
    jax = _jax
    jnp = _jnp

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
        pos: Any, p_pos: Any, p: Any, dtn: Any,
        stretch_k: Any, squeeze_k: Any, ang_k: Any, dist_w: Any, ang_w: Any,
    ) -> Any:
        """Local smooth score with bead p's position substituted by p_pos.
        Mirrors gnome3d.mc._local_smooth_nb."""
        n = pos.shape[0]
        a_pm1 = pos[jnp.maximum(p - 1, 0)]
        bond_L_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n - 1)
        bond_L = jnp.where(
            bond_L_ok,
            _smooth_len(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)],
                        stretch_k, squeeze_k, dist_w),
            0.0,
        )
        a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
        bond_R_ok = jnp.logical_and(p >= 0, p < n - 1)
        bond_R = jnp.where(
            bond_R_ok,
            _smooth_len(p_pos, a_pp1, dtn[jnp.minimum(p, n - 2)],
                        stretch_k, squeeze_k, dist_w),
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
            valid = jnp.logical_and(i >= 0, i < n - 2)
            return jnp.where(valid, _smooth_ang(a0, a1, a2, ang_k, ang_w), 0.0)

        return bond_L + bond_R + angle_at(-2) + angle_at(-1) + angle_at(0)

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any) -> Any:
        """Sum over i with |i-p| > excl_skip of weight * ((r0-d)/r0)^2 · [d<r0].
        Returns scalar."""
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        in_range = jnp.abs(idx - p) > excl_skip
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    def _init_smooth(pos: Any, dtn: Any, stretch_k: Any, squeeze_k: Any, ang_k: Any,
                    dist_w: Any, ang_w: Any) -> Any:
        n = pos.shape[0]

        def _bond_at(i: Any) -> Any:
            return _smooth_len(
                pos[i], pos[i + 1], dtn[i], stretch_k, squeeze_k, dist_w
            )

        def _angle_at(i: Any) -> Any:
            return _smooth_ang(pos[i], pos[i + 1], pos[i + 2], ang_k, ang_w)

        bonds = jax.vmap(_bond_at)(jnp.arange(n - 1))
        angles = jax.vmap(_angle_at)(jnp.arange(n - 2))
        return jnp.sum(bonds) + jnp.sum(angles)

    def _init_excl(pos: Any, r0: Any, weight: Any) -> Any:
        """O(N) row-at-a-time scan; avoids the (N, N, 3) materialization that
        would OOM at large N."""
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
        pos0: Any, score_struct0: Any, score_excl0: Any, T0: Any,
        dtn: Any, movable: Any,
        step_size: Any, dt: Any, js: Any, jc: Any,
        stretch_k: Any, squeeze_k: Any, ang_k: Any, dist_w: Any, ang_w: Any,
        r0: Any, excl_w: Any, key: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """One batch of `n_steps_per_batch` MC steps for ONE chain.  Returns
        (pos_f, score_struct_f, score_excl_f, T_f, n_ok)."""
        n_movable = movable.shape[0]

        k_p, k_d, k_a = jax.random.split(key, 3)
        idx_picks = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_movable)
        ps = movable[idx_picks]
        disps = jax.random.uniform(
            k_d, (n_steps_per_batch, 3),
            minval=-step_size, maxval=step_size, dtype=pos0.dtype,
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

            loc_struct_prev = _local_smooth_at(pos, old_p, p, dtn,
                                                stretch_k, squeeze_k, ang_k,
                                                dist_w, ang_w)
            loc_excl_prev = _local_excl_at(pos, old_p, p, r0, excl_w)
            loc_struct_curr = _local_smooth_at(pos, new_p, p, dtn,
                                                stretch_k, squeeze_k, ang_k,
                                                dist_w, ang_w)
            loc_excl_curr = _local_excl_at(pos, new_p, p, r0, excl_w)

            ss_new = ss + (loc_struct_curr - loc_struct_prev)
            se_new = se + 2.0 * (loc_excl_curr - loc_excl_prev)
            score_new = ss_new + se_new

            ok_unc = score_new < score  # smooth: strict
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) \
                       / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (pos_next, ss_next, se_next, T * dt, n_ok_next)

        init = (pos0, score_struct0, score_excl0, T0, jnp.int32(0))
        pos_f, ss_f, se_f, T_f, n_ok_f = jax.lax.fori_loop(
            0, n_steps_per_batch, body, init
        )
        return pos_f, ss_f, se_f, T_f, n_ok_f

    # vmap over K chains; problem arrays (dtn, movable) and schedule scalars
    # are shared.  Per-chain: pos, scores, key.  T is shared (deterministic
    # schedule, identical trajectory for every chain).
    batched = jax.vmap(
        chain_batch,
        in_axes=(0, 0, 0, None,                   # pos, ss, se, T
                 None, None,                       # dtn, movable
                 None, None, None, None,           # step_size, dt, js, jc
                 None, None, None, None, None,     # stretch_k..ang_w
                 None, None,                       # r0, excl_w
                 0),                               # key
        out_axes=(0, 0, 0, None, 0),
    )

    @jax.jit
    def kernel(
        pos_k: Any, ss_k: Any, se_k: Any, T: Any,
        dtn: Any, movable: Any,
        step_size: Any, dt: Any, js: Any, jc: Any,
        stretch_k: Any, squeeze_k: Any, ang_k: Any, dist_w: Any, ang_w: Any,
        r0: Any, excl_w: Any, keys: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        return batched(pos_k, ss_k, se_k, T, dtn, movable,
                       step_size, dt, js, jc,
                       stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                       r0, excl_w, keys)

    # Also expose the init functions vmapped, so the caller can compute initial
    # scores on-device (no host roundtrip).
    init_smooth_vmapped = jax.jit(jax.vmap(
        _init_smooth,
        in_axes=(0, None, None, None, None, None, None),
    ))
    init_excl_vmapped = jax.jit(jax.vmap(_init_excl, in_axes=(0, None, None)))

    bundle = (kernel, init_smooth_vmapped, init_excl_vmapped)
    _kernel_cache[key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Public entry: mirrors gnome3d.mc.mc_smooth signature
# ---------------------------------------------------------------------------


def mc_smooth_jax(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
    label: str = "",
    verbose: bool = False,
) -> float:
    """JAX backend for the simple-config smooth-MC path.

    Same signature contract as [mc.mc_smooth] minus the unsupported optional
    args (orientation, heat_dist).  The dispatcher in [mc.mc_smooth] must gate
    on `settings.mc_backend == "jax"` AND the simple-config check before
    calling this.

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
    if n <= 2:
        return 0.0

    movable_np: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable_np) == 0:
        return 0.0

    K: int = max(1, int(settings.mc_smooth_chains))
    n_steps_per_batch: int = int(settings.mc_stop_steps_smooth)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(
        settings.exclusion_apply_to_smooth
    )
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    if use_excl:
        excl_r0: float = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            factor = float(settings.exclusion_auto_factor_smooth)
            excl_r0 = factor * float(np.asarray(dtn).mean())
    else:
        # r0 unused when excl_w_v == 0, but the kernel still references it —
        # pick something safe & non-zero to avoid div-by-zero.
        excl_r0 = 1.0

    # Move to f32 for JAX (matches the bench finding that f32 is 2x faster
    # with no quality loss on these run lengths).
    pos_f32: F32Array = pos.astype(np.float32)
    pos_k_np: F32Array = np.broadcast_to(pos_f32, (K, n, 3)).copy()
    dtn_np: F32Array = dtn.astype(np.float32)

    kernel, init_smooth, init_excl = _build_smooth_ev_kernel(n_steps_per_batch, excl_skip)

    # Move state to device
    pos_k = jnp.asarray(pos_k_np)
    dtn_j = jnp.asarray(dtn_np)
    movable_j = jnp.asarray(movable_np)
    seed_offset: int = abs(hash(label)) % (2**31) if label else 0
    keys = jax.random.split(jax.random.PRNGKey(seed_offset), K)

    # Initial scores (vmapped across chains, computed on device)
    ss_k = init_smooth(
        pos_k, dtn_j,
        jnp.float32(settings.spring_stretch),
        jnp.float32(settings.spring_squeeze),
        jnp.float32(settings.spring_angular),
        jnp.float32(settings.smooth_dist_weight),
        jnp.float32(settings.smooth_angle_weight),
    )
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v))
        if use_excl else jnp.zeros((K,), dtype=jnp.float32)
    )

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
    step_size_j = jnp.float32(step_size)

    # Outer convergence loop driven from Python.  Each iteration is one
    # compiled JAX kernel call processing `n_steps_per_batch` MC steps × K
    # chains; the cost between iterations is one device->host sync to read
    # the score for the convergence check.
    stop_improvement: float = float(settings.mc_stop_improvement_smooth)
    stop_successes: int = int(settings.mc_stop_successes_smooth)
    score_eps: float = 1e-6
    ms_score: float = float("inf")
    step_i: int = 0
    prefix = f"    [{label}] " if label else "    "

    while True:
        # Fresh per-chain keys per batch, derived from a step counter so the
        # stream is reproducible and never repeats across batches.
        keys = jax.random.split(
            jax.random.PRNGKey(seed_offset + step_i + 1), K
        )

        pos_k, ss_k, se_k, T, n_ok_k = kernel(
            pos_k, ss_k, se_k, T,
            dtn_j, movable_j,
            step_size_j, dt, js, jc,
            stretch_k_j, squeeze_k_j, ang_k_j, dist_w_j, ang_w_j,
            r0_j, excl_w_j, keys,
        )

        # Best score across chains; use it to drive convergence (the unused
        # chains have already lost the race, so their plateauing doesn't
        # matter for the final result).
        score_per_chain = np.asarray(ss_k + se_k)
        n_ok_per_chain = np.asarray(n_ok_k)
        score: float = float(np.min(score_per_chain))
        n_ok_best: int = int(n_ok_per_chain[int(np.argmin(score_per_chain))])

        step_i += n_steps_per_batch
        ratio: float = score / ms_score if ms_score > 0 else 1.0
        converged: bool = (
            (score > stop_improvement * ms_score and n_ok_best < stop_successes)
            or score < score_eps
        )
        if verbose:
            print(
                f"{prefix}step {step_i:>7,}  score={score:.4f}"
                f"  ratio={ratio:.4f}  ok={n_ok_best}/{n_steps_per_batch}"
                + ("  [done]" if converged else ""),
                flush=True,
            )
        if converged:
            break
        ms_score = score

    # Pick best chain and write back to the caller's pos buffer
    best_k: int = int(np.argmin(score_per_chain))
    pos[:] = np.asarray(pos_k[best_k]).astype(pos.dtype)
    return float(score_per_chain[best_k])
