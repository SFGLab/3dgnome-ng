"""Diagnose the ~0.68s JAX overhead floor seen at small N in bench_jax_vs_numba_heatmap.py.

We saw N=64 K=1, N=256 K=1, N=1024 K=1 all ran in ~0.68s — that is independent
of N, so it can't be compute.  Two hypotheses:

  H1: PRNG splits inside fori_loop dominate
      (4 splits/step × 20000 steps × Threefry rounds in a sequential chain)
  H2: Per-step scatters (`pos.at[p].set(...)`) break XLA fusion and force a
      kernel launch per iteration
  H3: float64 on consumer GPUs is 1/32 throughput vs float32 — the PRNG-only
      cost might be dtype-bound

This script tests v1 (baseline) against three v2 variants:

  v2_f64        pre-generated RNG + single scatter per step, float64
  v2_f32        same as v2_f64 but float32
  v2_f64_scan   same as v2_f64 but lax.scan instead of fori_loop
                (sometimes XLA produces a tighter While loop for scan)

Run:
    pip install "jax[cuda12]" numba numpy
    python bench_jax_floor.py

Output: a wide table comparing numba, jax_v1, jax_v2_f64, jax_v2_f32 wall times
plus a separate "compile cost" report so the steady-state cost is unambiguous.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from numba import njit, prange  # type: ignore[import-not-found]

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # we'll cast back to f32 inside v2_f32
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ----------------------------------------------------------------------------
# Numba reference (same kernel as the original bench).
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _local_heatmap_nb(pos, exp_safe, skip_col, p):
    n = pos.shape[0]
    sc = 0.0
    px, py, pz = pos[p, 0], pos[p, 1], pos[p, 2]
    for i in range(n):
        if skip_col[i]:
            continue
        dx = pos[i, 0] - px
        dy = pos[i, 1] - py
        dz = pos[i, 2] - pz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        e = exp_safe[i, p]
        err = (d - e) / e
        sc += err * err
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _init_heatmap_nb(pos, exp_safe, skip):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        for j in range(n):
            if skip[i, j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            e = exp_safe[i, j]
            err = (d - e) / e
            sc += err * err
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _batch_heatmap_chain_nb(pos, exp_safe, skip, step_size, T, dt, js, jc, n_steps, score):
    n = pos.shape[0]
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)
        loc_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        pos[p, 0] += dx; pos[p, 1] += dy; pos[p, 2] += dz
        loc_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_new = score + 2.0 * (loc_curr - loc_prev)
        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < js * math.exp(-jc * (score_new / score) / T)
        if ok:
            score = score_new
        else:
            pos[p, 0] -= dx; pos[p, 1] -= dy; pos[p, 2] -= dz
        T *= dt
    return score


@njit(cache=True, parallel=True, nogil=True)
def _mc_heatmap_kchains_nb(pos_k, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, scores):
    K = pos_k.shape[0]
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]
        s = _init_heatmap_nb(pos, exp_safe, skip)
        scores[k] = _batch_heatmap_chain_nb(pos, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, s)


def run_numba(pos_k, exp_safe, skip, step_size, T0, dt, js, jc, n_steps):
    K = pos_k.shape[0]
    scores = np.zeros(K, dtype=np.float64)
    t0 = time.perf_counter()
    _mc_heatmap_kchains_nb(pos_k, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, scores)
    return time.perf_counter() - t0, scores


# ----------------------------------------------------------------------------
# JAX v1 — same as the previous bench: PRNG split inside the loop, two scatters
# (pos_trial materialized).  Kept as the baseline we're trying to beat.
# ----------------------------------------------------------------------------


def make_jax_v1(dtype):
    def _local_heat(pos, exp_safe, skip_col, p):
        diff = pos - pos[p]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        e = exp_safe[:, p]
        err = (d - e) / e
        return jnp.sum(jnp.where(skip_col, jnp.asarray(0.0, dtype), err * err))

    def _init_heat(pos, exp_safe, skip):
        diff = pos[:, None, :] - pos[None, :, :]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=2))
        err = (d - exp_safe) / exp_safe
        return jnp.sum(jnp.where(skip, jnp.asarray(0.0, dtype), err * err))

    def chain(pos0, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, key):
        n = pos0.shape[0]
        score0 = _init_heat(pos0, exp_safe, skip)

        def body(i, carry):
            pos, score, T, key = carry
            key, k_p, k_d, k_a = jax.random.split(key, 4)
            p = jax.random.randint(k_p, (), 0, n)
            delta = jax.random.uniform(k_d, (3,), minval=-step_size, maxval=step_size, dtype=dtype)
            skip_col = skip[:, p]
            loc_prev = _local_heat(pos, exp_safe, skip_col, p)
            new_p = pos[p] + delta
            pos_trial = pos.at[p].set(new_p)
            loc_curr = _local_heat(pos_trial, exp_safe, skip_col, p)
            score_new = score + 2.0 * (loc_curr - loc_prev)
            ok_unc = score_new <= score
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            u = jax.random.uniform(k_a, (), dtype=dtype)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))
            final_p = jnp.where(ok, new_p, pos[p])
            pos_next = pos.at[p].set(final_p)
            return (pos_next, jnp.where(ok, score_new, score), T * dt, key)

        pos_f, score_f, _, _ = jax.lax.fori_loop(0, n_steps, body, (pos0, score0, T0, key))
        return pos_f, score_f

    vmapped = jax.vmap(chain, in_axes=(0, None, None, None, None, None, None, None, None, 0),
                       out_axes=(0, 0))
    # Keep n_steps static so v1 and v2 are on equal footing (both benefit from
    # constant-folded loop bounds).  Cache key now includes n_steps value.
    return jax.jit(vmapped, static_argnums=(8,))


# ----------------------------------------------------------------------------
# JAX v2 — pre-generated RNG (no splits inside the loop) + single scatter per
# step (no pos_trial materialization).
# ----------------------------------------------------------------------------


def make_jax_v2(dtype, use_scan: bool = False):
    def _local_heat_at(pos, p_pos, exp_col, skip_col):
        # Compute local heat score with bead p's position substituted by p_pos
        # WITHOUT materializing a full N×3 trial array.  For i == p, the diff is
        # zero so contribution is `((0-e)/e)²` — but skip_col[p] is True (the
        # diagonal of the skip mask), so it's filtered.  Same answer as v1.
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        err = (d - exp_col) / exp_col
        return jnp.sum(jnp.where(skip_col, jnp.asarray(0.0, dtype), err * err))

    def _init_heat(pos, exp_safe, skip):
        diff = pos[:, None, :] - pos[None, :, :]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=2))
        err = (d - exp_safe) / exp_safe
        return jnp.sum(jnp.where(skip, jnp.asarray(0.0, dtype), err * err))

    def chain(pos0, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, key):
        n = pos0.shape[0]
        # Pre-generate ALL random numbers for this chain in three big arrays.
        k_p, k_d, k_a = jax.random.split(key, 3)
        ps = jax.random.randint(k_p, (n_steps,), 0, n)
        disps = jax.random.uniform(k_d, (n_steps, 3), minval=-step_size, maxval=step_size, dtype=dtype)
        accs = jax.random.uniform(k_a, (n_steps,), dtype=dtype)

        score0 = _init_heat(pos0, exp_safe, skip)

        def _step_core(carry, p, delta, u):
            pos, score, T = carry
            skip_col = skip[:, p]
            exp_col = exp_safe[:, p]
            old_p = pos[p]
            new_p = old_p + delta
            loc_prev = _local_heat_at(pos, old_p, exp_col, skip_col)
            loc_curr = _local_heat_at(pos, new_p, exp_col, skip_col)
            score_new = score + 2.0 * (loc_curr - loc_prev)
            ok_unc = score_new <= score
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))
            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            return (pos_next, jnp.where(ok, score_new, score), T * dt)

        if use_scan:
            # scan body: (carry, x) -> (new_carry, y)
            def scan_body(carry, x):
                p, delta, u = x
                return _step_core(carry, p, delta, u), None

            (pos_f, score_f, _), _ = jax.lax.scan(
                scan_body, (pos0, score0, T0), (ps, disps, accs)
            )
        else:
            # fori_loop body: (i, val) -> val
            def fori_body(i, carry):
                return _step_core(carry, ps[i], disps[i], accs[i])

            pos_f, score_f, _ = jax.lax.fori_loop(0, n_steps, fori_body, (pos0, score0, T0))
        return pos_f, score_f

    vmapped = jax.vmap(chain, in_axes=(0, None, None, None, None, None, None, None, None, 0),
                       out_axes=(0, 0))
    # n_steps (positional 8) must be static: it's the shape of the pre-gen RNG arrays.
    return jax.jit(vmapped, static_argnums=(8,))


# ----------------------------------------------------------------------------
# Bench driver
# ----------------------------------------------------------------------------


def make_problem(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 3)) * (float(n) ** (1.0 / 3.0))
    idx = np.arange(n)
    gen = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    exp_dist = np.sqrt(gen + 1.0)
    keep = rng.random(exp_dist.shape) < 0.8
    keep = keep | keep.T
    exp_dist = np.where(keep, exp_dist, 0.0)
    np.fill_diagonal(exp_dist, 0.0)
    diag_size = 1
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip = diag_mask | (exp_dist < 1e-6)
    exp_safe = np.where(skip, 1.0, exp_dist)
    return pos.astype(np.float64), exp_safe.astype(np.float64), skip.astype(np.bool_)


def time_jax(kernel, pos_k_np, exp_safe_np, skip_np, step_size, T0, dt, js, jc, n_steps,
             seed, dtype_np):
    """Returns (compile_time, steady_time, scores).

    We call the kernel TWICE.  First call = compile + run.  Second call = run only.
    compile_time = call1 - call2; steady_time = call2.
    """
    pos_k = jnp.asarray(pos_k_np, dtype=dtype_np)
    exp_safe = jnp.asarray(exp_safe_np, dtype=dtype_np)
    skip = jnp.asarray(skip_np, dtype=jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(seed), pos_k.shape[0])

    t0 = time.perf_counter()
    pos1, scores1 = kernel(pos_k, exp_safe, skip,
                           dtype_np.type(step_size), dtype_np.type(T0), dtype_np.type(dt),
                           dtype_np.type(js), dtype_np.type(jc),
                           int(n_steps), keys)
    scores1.block_until_ready()
    t1 = time.perf_counter()

    pos_k2 = jnp.asarray(pos_k_np, dtype=dtype_np)
    keys2 = jax.random.split(jax.random.PRNGKey(seed + 1), pos_k.shape[0])
    t2 = time.perf_counter()
    pos2, scores2 = kernel(pos_k2, exp_safe, skip,
                           dtype_np.type(step_size), dtype_np.type(T0), dtype_np.type(dt),
                           dtype_np.type(js), dtype_np.type(jc),
                           int(n_steps), keys2)
    scores2.block_until_ready()
    t3 = time.perf_counter()

    compile_time = (t1 - t0) - (t3 - t2)
    steady_time = t3 - t2
    return compile_time, steady_time, np.asarray(scores2)


def bench_one(n: int, k: int, n_steps: int, kernels: dict):
    pos, exp_safe, skip = make_problem(n, seed=42)
    pos_k = np.broadcast_to(pos, (k, n, 3)).copy()
    step_size, T0, dt, js, jc = 0.5, 20.0, 0.99995, 50.0, 20.0

    out = {"N": n, "K": k, "steps": n_steps}

    # Numba
    pos_nb = pos_k.copy()
    t_nb, scores_nb = run_numba(pos_nb, exp_safe, skip, step_size, T0, dt, js, jc, n_steps)
    out["numba_s"] = t_nb
    out["numba_best"] = float(np.min(scores_nb))

    for name, (kernel, dtype_np) in kernels.items():
        try:
            compile_t, steady_t, scores = time_jax(
                kernel, pos_k, exp_safe, skip, step_size, T0, dt, js, jc, n_steps, 42, dtype_np
            )
            out[f"{name}_compile_s"] = compile_t
            out[f"{name}_s"] = steady_t
            out[f"{name}_best"] = float(np.min(scores))
        except Exception as ex:
            out[f"{name}_s"] = float("nan")
            out[f"{name}_compile_s"] = float("nan")
            out[f"{name}_best"] = float("nan")
            print(f"  FAIL {name} N={n} K={k}: {type(ex).__name__}: {ex}", flush=True)
    return out


def main():
    if not HAS_JAX:
        print("JAX not installed.  pip install 'jax[cuda12]'", file=sys.stderr)
        sys.exit(2)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    if jax.default_backend() == "cpu":
        print("WARNING: JAX on CPU-XLA, not GPU.", file=sys.stderr)

    # Warmup numba (any shape; cache key is per-function not per-shape).
    p, e, s = make_problem(16, 0); pk = np.broadcast_to(p, (2, 16, 3)).copy()
    sc = np.zeros(2); _mc_heatmap_kchains_nb(pk, e, s, 0.5, 1.0, 0.999, 1.0, 1.0, 32, sc)
    print("Numba warmup done.", flush=True)

    print("Building JAX kernels...", flush=True)
    kernels = {
        "v1_f64":      (make_jax_v1(jnp.float64), np.dtype(np.float64)),
        "v2_f64":      (make_jax_v2(jnp.float64, use_scan=False), np.dtype(np.float64)),
        "v2_f32":      (make_jax_v2(jnp.float32, use_scan=False), np.dtype(np.float32)),
        "v2_scan_f64": (make_jax_v2(jnp.float64, use_scan=True),  np.dtype(np.float64)),
    }
    print("Kernels built. (Each kernel will compile lazily per shape on first call.)\n", flush=True)

    # Grid focused on the regime where the floor matters.  N=4096 included as a
    # sanity check that we didn't regress the large-N JAX win.
    configs = [
        ( 64,   1,  20000),
        ( 64,  32,  20000),
        ( 64, 128,  20000),
        (256,   1,  20000),
        (256,  32,  20000),
        (256, 128,  20000),
        (1024,  1,  20000),
        (1024, 32,  20000),
        (1024,128,  20000),
        (4096,  1,  20000),
        (4096, 32,  20000),
        (4096,128,  20000),
    ]

    # Print steady-state times.
    print("=== STEADY-STATE TIMES (second call, no compile) ===")
    header = f"{'N':>5} {'K':>4} {'steps':>6}  {'numba':>8}  {'v1_f64':>8}  {'v2_f64':>8}  {'v2_f32':>8}  {'v2_scan':>8}"
    print(header)
    print("-" * len(header))
    all_rows = []
    for (n, k, st) in configs:
        r = bench_one(n, k, st, kernels)
        all_rows.append(r)
        print(f"{r['N']:>5} {r['K']:>4} {r['steps']:>6}  "
              f"{r['numba_s']:>8.3f}  "
              f"{r.get('v1_f64_s', float('nan')):>8.3f}  "
              f"{r.get('v2_f64_s', float('nan')):>8.3f}  "
              f"{r.get('v2_f32_s', float('nan')):>8.3f}  "
              f"{r.get('v2_scan_f64_s', float('nan')):>8.3f}",
              flush=True)

    # Then print compile times for context.
    print("\n=== COMPILE TIMES (first call − second call, seconds) ===")
    print(f"{'N':>5} {'K':>4}  {'v1_f64':>8}  {'v2_f64':>8}  {'v2_f32':>8}  {'v2_scan':>8}")
    print("-" * 56)
    for r in all_rows:
        print(f"{r['N']:>5} {r['K']:>4}  "
              f"{r.get('v1_f64_compile_s', float('nan')):>8.2f}  "
              f"{r.get('v2_f64_compile_s', float('nan')):>8.2f}  "
              f"{r.get('v2_f32_compile_s', float('nan')):>8.2f}  "
              f"{r.get('v2_scan_f64_compile_s', float('nan')):>8.2f}",
              flush=True)

    # Quality sanity (best score across K) — should be in the same ballpark.
    print("\n=== BEST SCORES (lower = better; RNGs differ so values won't match exactly) ===")
    print(f"{'N':>5} {'K':>4}  {'numba':>14}  {'v1_f64':>14}  {'v2_f64':>14}  {'v2_f32':>14}")
    for r in all_rows:
        print(f"{r['N']:>5} {r['K']:>4}  "
              f"{r['numba_best']:>14.2f}  "
              f"{r.get('v1_f64_best', float('nan')):>14.2f}  "
              f"{r.get('v2_f64_best', float('nan')):>14.2f}  "
              f"{r.get('v2_f32_best', float('nan')):>14.2f}",
              flush=True)

    print("\nInterpretation:")
    print("  - If v2_f64 << v1_f64: PRNG-in-loop and/or double-scatter were the floor.")
    print("  - If v2_f32 << v2_f64 at small N: float64 throughput was the bottleneck.")
    print("  - If v2_scan ≈ v2_fori: loop construct doesn't matter (scan vs while).")
    print("  - If best scores in f32 column drift wildly from f64: dtype affects MC quality.")


if __name__ == "__main__":
    main()
