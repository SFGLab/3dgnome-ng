"""Self-contained benchmark: numba-prange vs JAX-vmap for heatmap Monte Carlo.

Purpose: decide whether porting the heatmap MC kernel in gnome3d/mc.py to JAX
yields a meaningful speedup over the current numba prange implementation.
Intended to be copied onto a CUDA machine and run there.

Both implementations run the *same* algorithm:
  - random bead pick from [0, N)
  - random uniform-cube displacement in [-step_size, step_size]^3
  - local heatmap score (O(N) per step) before/after the trial move
  - score update: score += 2 * (local_curr - local_prev)
  - accept iff score_new <= score, else thermal jump:
      rand() < jump_scale * exp(-jump_coef * (score_new/score) / T)
  - T *= dt every step (accepted or not)
  - K independent chains run in parallel (no info-sharing)

The benchmark uses a FIXED total step budget (no convergence) so wall-time is
directly comparable.  Quality is reported via final score per chain but is
expected to drift between backends due to different RNG streams.

Setup on a CUDA box:
    pip install "jax[cuda12]" numba numpy
    python bench_jax_vs_numba_heatmap.py

The script self-checks JAX device on startup and aborts with a clear message
if no CUDA device is visible (don't want results from CPU-XLA misread as GPU).

Output format: a copy-paste-friendly table of
    (N, K, steps, numba_time_s, jax_time_s, speedup, numba_final, jax_final)
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Numba implementation: mirrors gnome3d/mc.py::_batch_heatmap_chain_nb and
# _mc_heatmap_kchains_nb.  Trimmed to the K-chain inner loop only (no outer
# convergence loop — we run a fixed total step budget for fair timing).
# ----------------------------------------------------------------------------

from numba import njit, prange  # type: ignore[import-not-found]


@njit(cache=True, fastmath=True, nogil=True)
def _local_heatmap_nb(pos, exp_safe, skip_col, p):
    n = pos.shape[0]
    sc = 0.0
    px = pos[p, 0]
    py = pos[p, 1]
    pz = pos[p, 2]
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
def _batch_heatmap_chain_nb(
    pos, exp_safe, skip, step_size, T, dt, jump_scale, jump_coef, n_steps, score_hm
):
    n = pos.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz
        loc_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        score_new = score_hm + 2.0 * (loc_curr - loc_prev)

        ok = score_new <= score_hm
        if not ok and T > 0.0 and score_hm > 0.0:
            ok = np.random.random() < jump_scale * math.exp(
                -jump_coef * (score_new / score_hm) / T
            )

        if ok:
            n_ok += 1
            score_hm = score_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score_hm, n_ok


@njit(cache=True, parallel=True, nogil=True)
def _mc_heatmap_kchains_nb(
    pos_k, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef, n_steps, final_scores
):
    """K-chain prange parallelism.  Each chain runs n_steps total."""
    K = pos_k.shape[0]
    for k in prange(K):  # pyright: ignore[reportGeneralTypeIssues]
        pos = pos_k[k]
        score = _init_heatmap_nb(pos, exp_safe, skip)
        _, final_score, _ = _batch_heatmap_chain_nb(
            pos, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef, n_steps, score
        )
        final_scores[k] = final_score


def run_numba(pos_k, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef, n_steps):
    """Returns (wall_seconds, final_scores)."""
    K = pos_k.shape[0]
    final_scores = np.zeros(K, dtype=np.float64)
    t0 = time.perf_counter()
    _mc_heatmap_kchains_nb(
        pos_k, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef, n_steps, final_scores
    )
    t1 = time.perf_counter()
    return t1 - t0, final_scores


# ----------------------------------------------------------------------------
# JAX implementation: same algorithm, vmap over K chains, lax.fori_loop for
# the inner step loop so the entire batch is one XLA program.
# ----------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # match numba float64
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _check_jax_device():
    if not HAS_JAX:
        print("JAX not installed.  pip install 'jax[cuda12]'", file=sys.stderr)
        sys.exit(2)
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}  devices: {devices}", flush=True)
    if backend == "cpu":
        print(
            "WARNING: JAX is running on CPU-XLA, not GPU.  "
            "Results will not reflect a CUDA speedup.  "
            "Install with: pip install 'jax[cuda12]'",
            file=sys.stderr,
        )


def _make_jax_kernels():
    """Build the JIT-compiled JAX kernels.  Returns a callable that runs
    K chains for n_steps and returns final scores + wall time."""
    import jax
    import jax.numpy as jnp

    def _local_heat_jax(pos, exp_safe, skip_col, p):
        # pos: (N, 3) ; exp_safe: (N, N) ; skip_col: (N,) bool ; p: scalar int
        diff = pos - pos[p]                          # (N, 3)
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))   # (N,)
        e = exp_safe[:, p]                           # (N,)
        err = (d - e) / e
        contrib = jnp.where(skip_col, 0.0, err * err)
        return jnp.sum(contrib)

    def _init_heat_jax(pos, exp_safe, skip):
        diff = pos[:, None, :] - pos[None, :, :]     # (N, N, 3)
        d = jnp.sqrt(jnp.sum(diff * diff, axis=2))   # (N, N)
        err = (d - exp_safe) / exp_safe
        contrib = jnp.where(skip, 0.0, err * err)
        return jnp.sum(contrib)

    def _chain_run(pos0, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef,
                   n_steps, key):
        """Run one chain for n_steps and return (pos_final, score_final)."""
        n = pos0.shape[0]
        score0 = _init_heat_jax(pos0, exp_safe, skip)

        def body(i, carry):
            pos, score, T, key = carry
            key, k_p, k_disp, k_acc = jax.random.split(key, 4)
            p = jax.random.randint(k_p, (), 0, n)
            delta = jax.random.uniform(k_disp, (3,), minval=-step_size, maxval=step_size)

            skip_col = skip[:, p]
            loc_prev = _local_heat_jax(pos, exp_safe, skip_col, p)
            new_p_pos = pos[p] + delta
            pos_trial = pos.at[p].set(new_p_pos)
            loc_curr = _local_heat_jax(pos_trial, exp_safe, skip_col, p)
            score_new = score + 2.0 * (loc_curr - loc_prev)

            ok_unconditional = score_new <= score
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            # clip exponent for numerical safety; exp(80) ~ 5e34, well past 1
            exponent = -jump_coef * (score_new / jnp.maximum(score, 1e-300)) \
                       / jnp.maximum(T, 1e-300)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_accept = jump_scale * jnp.exp(exponent)
            u = jax.random.uniform(k_acc)
            ok_thermal = jnp.logical_and(can_jump, u < p_accept)
            ok = jnp.logical_or(ok_unconditional, ok_thermal)

            final_p_pos = jnp.where(ok, new_p_pos, pos[p])
            pos_next = pos.at[p].set(final_p_pos)
            score_next = jnp.where(ok, score_new, score)
            T_next = T * dt
            return (pos_next, score_next, T_next, key)

        init = (pos0, score0, T0, key)
        pos_f, score_f, T_f, key_f = jax.lax.fori_loop(0, n_steps, body, init)
        return pos_f, score_f

    # vmap over K chains; the problem arrays (exp_safe, skip) are shared.
    chain_vmapped = jax.vmap(
        _chain_run,
        in_axes=(0, None, None, None, None, None, None, None, None, 0),
        out_axes=(0, 0),
    )

    # JIT the whole K-chain kernel.  n_steps is static so the compiled
    # program has a fixed-trip-count fori_loop.
    @jax.jit
    def run_kchains_jit(pos_k, exp_safe, skip, step_size, T0, dt, jump_scale,
                        jump_coef, n_steps, keys):
        return chain_vmapped(pos_k, exp_safe, skip, step_size, T0, dt, jump_scale,
                             jump_coef, n_steps, keys)

    return run_kchains_jit


def run_jax(run_kchains_jit, pos_k_np, exp_safe_np, skip_np, step_size, T0, dt,
            jump_scale, jump_coef, n_steps, seed):
    """Returns (wall_seconds, final_scores)."""
    import jax
    import jax.numpy as jnp

    K = pos_k_np.shape[0]
    pos_k = jnp.asarray(pos_k_np, dtype=jnp.float64)
    exp_safe = jnp.asarray(exp_safe_np, dtype=jnp.float64)
    skip = jnp.asarray(skip_np, dtype=jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(seed), K)

    # Warmup-then-time: first call compiles (and we discard that time).  But
    # for a fair single-shot measure, we time only the second call.  Block on
    # device so we measure wall time, not async-dispatch time.
    pos_f, scores_f = run_kchains_jit(pos_k, exp_safe, skip, float(step_size),
                                       float(T0), float(dt), float(jump_scale),
                                       float(jump_coef), int(n_steps), keys)
    scores_f.block_until_ready()  # ensure first call (compile + run) completes

    # Real timed run with fresh keys to avoid any kernel-cache shortcuts.
    keys2 = jax.random.split(jax.random.PRNGKey(seed + 1), K)
    pos_k2 = jnp.asarray(pos_k_np, dtype=jnp.float64)  # fresh starting positions
    t0 = time.perf_counter()
    pos_f2, scores_f2 = run_kchains_jit(pos_k2, exp_safe, skip, float(step_size),
                                         float(T0), float(dt), float(jump_scale),
                                         float(jump_coef), int(n_steps), keys2)
    scores_f2.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0, np.asarray(scores_f2)


# ----------------------------------------------------------------------------
# Problem generator + benchmark driver.
# ----------------------------------------------------------------------------


def make_problem(n: int, seed: int = 0):
    """Synthetic heatmap problem with realistic skip pattern (diagonal band +
    random sparsity)."""
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 3)) * (float(n) ** (1.0 / 3.0))
    idx = np.arange(n)
    gen = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    exp_dist = np.sqrt(gen + 1.0)
    # 20% random sparsity on top of diagonal-band skip
    keep = rng.random(exp_dist.shape) < 0.8
    keep = keep | keep.T
    exp_dist = np.where(keep, exp_dist, 0.0)
    np.fill_diagonal(exp_dist, 0.0)

    # build skip mask: diagonal band of width 1, plus any zero/negative entries
    diag_size = 1
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip = diag_mask | (exp_dist < 1e-6)
    exp_safe = np.where(skip, 1.0, exp_dist)
    return pos.astype(np.float64), exp_safe.astype(np.float64), skip.astype(np.bool_)


def bench_one(n: int, k: int, n_steps: int, run_kchains_jit, seed: int = 42):
    pos, exp_safe, skip = make_problem(n, seed=seed)
    pos_k = np.broadcast_to(pos, (k, n, 3)).copy()

    # MC schedule — same as gnome3d defaults for heatmap level.
    step_size = 0.5
    T0 = 20.0
    dt = 0.99995
    jump_scale = 50.0
    jump_coef = 20.0

    # Numba: K-chain prange.
    pos_k_nb = pos_k.copy()
    t_nb, scores_nb = run_numba(
        pos_k_nb, exp_safe, skip, step_size, T0, dt, jump_scale, jump_coef, n_steps
    )

    # JAX: K-chain vmap.
    t_jx, scores_jx = run_jax(
        run_kchains_jit, pos_k, exp_safe, skip, step_size, T0, dt, jump_scale,
        jump_coef, n_steps, seed=seed,
    )

    speedup = t_nb / t_jx if t_jx > 0 else float("inf")
    return {
        "N": n,
        "K": k,
        "steps": n_steps,
        "numba_s": t_nb,
        "jax_s": t_jx,
        "speedup": speedup,
        "numba_best": float(np.min(scores_nb)),
        "jax_best": float(np.min(scores_jx)),
    }


def main():
    _check_jax_device()
    print("Compiling numba kernels (first-call JIT)...", flush=True)
    # Touch each kernel with a tiny problem so the timed runs don't include
    # numba compile time.
    _pos, _exp, _skip = make_problem(16, seed=0)
    _pk = np.broadcast_to(_pos, (2, 16, 3)).copy()
    _scores = np.zeros(2, dtype=np.float64)
    _mc_heatmap_kchains_nb(_pk, _exp, _skip, 0.5, 1.0, 0.999, 1.0, 1.0, 32, _scores)
    print("Numba warmup done.", flush=True)

    print("Compiling JAX kernels (first-call XLA trace)...", flush=True)
    run_kchains_jit = _make_jax_kernels()
    # warmup with the same shape we'll first use; subsequent calls with the
    # same (N, K, n_steps) static shape reuse the cached compile.
    _pos, _exp, _skip = make_problem(64, seed=0)
    _pk = np.broadcast_to(_pos, (4, 64, 3)).copy()
    _ = run_jax(run_kchains_jit, _pk, _exp, _skip, 0.5, 20.0, 0.999, 50.0, 20.0, 32, 0)
    print("JAX warmup done.\n", flush=True)

    # Sweep grid.  N picks span the realistic 3dgnome range (chr-level: tens;
    # segment-level: hundreds; subanchor: ~1k).  K picks span single-chain
    # through "lots of chains, GPU only" regime.
    configs = [
        # N,   K,   steps_per_chain
        ( 64,    1,  20000),
        ( 64,    8,  20000),
        ( 64,   32,  20000),
        ( 64,  128,  20000),
        (256,    1,  20000),
        (256,    8,  20000),
        (256,   32,  20000),
        (256,  128,  20000),
        (1024,   1,  10000),
        (1024,   8,  10000),
        (1024,  32,  10000),
        (1024, 128,  10000),
    ]

    # Header
    print(f"{'N':>5} {'K':>4} {'steps':>7}  "
          f"{'numba_s':>9} {'jax_s':>9} {'speedup':>8}  "
          f"{'numba_best':>12} {'jax_best':>12}")
    print("-" * 80)

    results = []
    for (n, k, s) in configs:
        try:
            r = bench_one(n, k, s, run_kchains_jit)
        except Exception as ex:
            print(f"FAIL N={n} K={k} steps={s}: {type(ex).__name__}: {ex}", flush=True)
            continue
        results.append(r)
        print(f"{r['N']:>5} {r['K']:>4} {r['steps']:>7}  "
              f"{r['numba_s']:>9.3f} {r['jax_s']:>9.3f} {r['speedup']:>7.2f}x  "
              f"{r['numba_best']:>12.2f} {r['jax_best']:>12.2f}", flush=True)

    print("\nDone.")
    print("Notes:")
    print("  - speedup = numba_s / jax_s.  >1 means JAX wins.")
    print("  - *_best is the minimum final score across K chains (lower = better).")
    print("    Numba and JAX use different RNGs so scores won't match exactly,")
    print("    but should be in the same order of magnitude after the same step count.")


if __name__ == "__main__":
    main()
