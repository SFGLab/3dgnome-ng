"""Self-contained benchmark: numba vs JAX for smooth Monte Carlo with excluded volume.

WHY: chr22 and chr4 dryruns showed smooth MC eats 89-96% of total MC time, with
the heavy calls living at N=1024-10000.  The previous bench targeted heatmap
MC, which the profile reveals to be a 0% slice of runtime.  This bench targets
the *actual* hot path.

Smooth MC structure per step:
  - random movable bead p
  - random uniform-cube displacement
  - local chain terms (O(1)): bond lengths to p±1 + angles at p-2..p
  - local excluded-volume term (O(N)): sum over all i with |i-p| > skip of
        weight * ((r0 - d)/r0)^2   if d < r0 else 0
  - struct delta factor = 1 (single-counted)
  - excl  delta factor = 2 (double-counted, matches gnome3d/mc.py convention)
  - Metropolis: ok = (score_new < score) OR rand < jump_scale·exp(-jump_coef·(s_new/s)/T)
  - T *= dt

Important note about the production numba path:
  gnome3d/mc.py::mc_smooth dispatches to a prange-parallel K-chain kernel
  ONLY when "simple_config" is true — i.e., no orientation, no EV, no
  confinement.  With EV on (the dryrun config and any realistic run), smooth
  MC falls back to single-chain.  So in production, smooth-with-EV runs K=1.
  This bench measures K=1 against numba's actual production path, and K>1
  against what numba could theoretically do (single-chain looped K times,
  i.e. NOT prange — there is no prange path for EV today).

Setup on a CUDA box:
    pip install "jax[cuda12]" numba numpy
    python bench_jax_smooth_mc.py
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from numba import njit  # type: ignore[import-not-found]

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # selectively used for v_f64 only
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ----------------------------------------------------------------------------
# Numba reference: replicates _local_smooth_nb + _local_excl_nb + the smooth
# accept/reject loop from gnome3d/mc.py.  Single-chain (matches the production
# path for smooth-with-EV configs).
# ----------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w):
    dx = pos[i, 0] - pos[i + 1, 0]
    dy = pos[i, 1] - pos[i + 1, 1]
    dz = pos[i, 2] - pos[i + 1, 2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    e = dtn[i]
    if e < 1e-6:
        e = 1e-6
    rel = (d - e) / e
    k = stretch_k if rel >= 0.0 else squeeze_k
    return rel * rel * k * dist_w


@njit(cache=True, fastmath=True, nogil=True)
def _smooth_ang_nb(pos, i, ang_k, ang_w):
    v1x = pos[i, 0] - pos[i + 1, 0]
    v1y = pos[i, 1] - pos[i + 1, 1]
    v1z = pos[i, 2] - pos[i + 1, 2]
    v2x = pos[i + 1, 0] - pos[i + 2, 0]
    v2y = pos[i + 1, 1] - pos[i + 2, 1]
    v2z = pos[i + 1, 2] - pos[i + 2, 2]
    n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
    if cos_a > 1.0:
        cos_a = 1.0
    if cos_a < -1.0:
        cos_a = -1.0
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return ang * ang * ang * ang_k * ang_w


@njit(cache=True, fastmath=True, nogil=True)
def _local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    sc = 0.0
    i = p - 1
    if 0 <= i < n - 1:
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    if 0 <= p < n - 1:
        sc += _smooth_len_nb(pos, dtn, p, stretch_k, squeeze_k, dist_w)
    for off in range(-2, 1):
        i = p + off
        if 0 <= i < n - 2:
            sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _local_excl_nb(pos, p, r0, weight, skip):
    n = pos.shape[0]
    err = 0.0
    px, py, pz = pos[p, 0], pos[p, 1], pos[p, 2]
    for i in range(n):
        diff = i - p
        if diff < 0:
            diff = -diff
        if diff <= skip:
            continue
        dx = pos[i, 0] - px
        dy = pos[i, 1] - py
        dz = pos[i, 2] - pz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < r0:
            rel = (r0 - d) / r0
            err += weight * rel * rel
    return err


@njit(cache=True, fastmath=True, nogil=True)
def _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n - 1):
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    for i in range(n - 2):
        sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


@njit(cache=True, fastmath=True, nogil=True)
def _init_excl_nb(pos, r0, weight, skip):
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            if diff <= skip:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < r0:
                rel = (r0 - d) / r0
                err += weight * rel * rel
    return err


@njit(cache=True, fastmath=True, nogil=True)
def _batch_smooth_chain_nb(
    pos, dtn, step_size, T, dt, js, jc, n_steps,
    stretch_k, squeeze_k, ang_k, dist_w, ang_w,
    r0, excl_w, excl_skip,
    score_struct, score_excl,
):
    n = pos.shape[0]
    score = score_struct + score_excl
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = _local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        loc_excl_prev = _local_excl_nb(pos, p, r0, excl_w, excl_skip)

        pos[p, 0] += dx; pos[p, 1] += dy; pos[p, 2] += dz

        loc_struct_curr = _local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        loc_excl_curr = _local_excl_nb(pos, p, r0, excl_w, excl_skip)

        score_struct_new = score_struct + (loc_struct_curr - loc_struct_prev)
        score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)
        score_new = score_struct_new + score_excl_new

        # smooth uses STRICT less-than (matches gnome3d/mc.py::_batch_mc_nb when struct_type==CHAIN)
        ok = score_new < score
        if not ok and T > 0.0 and score > 0.0:
            ok = np.random.random() < js * math.exp(-jc * (score_new / score) / T)

        if ok:
            score = score_new
            score_struct = score_struct_new
            score_excl = score_excl_new
        else:
            pos[p, 0] -= dx; pos[p, 1] -= dy; pos[p, 2] -= dz
        T *= dt
    return score_struct, score_excl


def run_numba_kchains(pos_k, dtn, step_size, T0, dt, js, jc, n_steps,
                     stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                     r0, excl_w, excl_skip):
    """K independent chains, sequential.  Matches production: smooth-with-EV
    has no prange path; the production numba code runs K=1 only."""
    K = pos_k.shape[0]
    final_scores = np.zeros(K, dtype=np.float64)
    t0 = time.perf_counter()
    for k in range(K):
        pos = pos_k[k]
        ss = _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        se = _init_excl_nb(pos, r0, excl_w, excl_skip)
        ss2, se2 = _batch_smooth_chain_nb(
            pos, dtn, step_size, T0, dt, js, jc, n_steps,
            stretch_k, squeeze_k, ang_k, dist_w, ang_w,
            r0, excl_w, excl_skip, ss, se,
        )
        final_scores[k] = ss2 + se2
    return time.perf_counter() - t0, final_scores


# ----------------------------------------------------------------------------
# JAX implementation: vmap over K chains, pre-gen RNG, single-scatter per step,
# float32 by default (per the f32-wins lesson from bench_jax_floor.py).
# ----------------------------------------------------------------------------


def make_jax_smooth_kernel(dtype):
    def _smooth_len(pa, pb, e, stretch_k, squeeze_k, dist_w):
        diff = pa - pb
        d = jnp.sqrt(jnp.sum(diff * diff))
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0, stretch_k, squeeze_k)
        return rel * rel * k * dist_w

    def _smooth_ang(pa, pb, pc, ang_k, ang_w):
        v1 = pa - pb
        v2 = pb - pc
        n1 = jnp.sqrt(jnp.sum(v1 * v1))
        n2 = jnp.sqrt(jnp.sum(v2 * v2))
        # Guard against zero-length vectors (zero contribution)
        scale = jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, 1.0)
        cos_a = jnp.sum(v1 * v2) / jnp.maximum(n1 * n2, 1e-30)
        cos_a = jnp.clip(cos_a, -1.0, 1.0)
        ang = 1.0 - (cos_a + 1.0) * 0.5
        return scale * ang * ang * ang * ang_k * ang_w

    def _local_smooth_at(pos, p_pos, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
        """Local smooth score with bead p's position substituted by p_pos,
        without materializing a trial array.  Mirrors _local_smooth_nb: covers
        bonds (p-1, p), (p, p+1) and angles at i ∈ {p-2, p-1, p}."""
        n = pos.shape[0]
        # Bond at i = p-1: pos[p-1] -- p_pos.  Valid when 0 <= p-1 < n-1, i.e. 1 <= p < n.
        a_pm1 = pos[jnp.maximum(p - 1, 0)]
        bond_left_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n - 1)
        bond_left = jnp.where(
            bond_left_ok,
            _smooth_len(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)], stretch_k, squeeze_k, dist_w),
            0.0,
        )
        # Bond at i = p: p_pos -- pos[p+1].  Valid when 0 <= p < n-1.
        a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
        bond_right_ok = jnp.logical_and(p >= 0, p < n - 1)
        bond_right = jnp.where(
            bond_right_ok,
            _smooth_len(p_pos, a_pp1, dtn[p], stretch_k, squeeze_k, dist_w),
            0.0,
        )

        # Angles touching p: i ∈ {p-2, p-1, p}; each uses beads (i, i+1, i+2).
        # Bead p shows up at position (i+2) for i=p-2, (i+1) for i=p-1, (i) for i=p.
        def angle_at(i):
            a0 = pos[jnp.clip(i, 0, n - 1)]
            a1 = pos[jnp.clip(i + 1, 0, n - 1)]
            a2 = pos[jnp.clip(i + 2, 0, n - 1)]
            # Substitute p_pos where the angle touches bead p
            a0 = jnp.where(i == p, p_pos, a0)
            a1 = jnp.where(i + 1 == p, p_pos, a1)
            a2 = jnp.where(i + 2 == p, p_pos, a2)
            valid = jnp.logical_and(i >= 0, i < n - 2)
            return jnp.where(valid, _smooth_ang(a0, a1, a2, ang_k, ang_w), 0.0)

        ang_sum = angle_at(p - 2) + angle_at(p - 1) + angle_at(p)
        return bond_left + bond_right + ang_sum

    def _local_excl_at(pos, p_pos, p, r0, weight, skip):
        """Sum over all i with |i - p| > skip of weight * ((r0 - d)/r0)² · [d < r0]."""
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        # Mask: pairs with |i - p| > skip (skip self by default via skip >= 0)
        idx = jnp.arange(n)
        in_range = jnp.abs(idx - p) > skip
        return jnp.sum(jnp.where(in_range, contrib, jnp.asarray(0.0, dtype)))

    def _init_smooth(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
        n = pos.shape[0]
        # Bonds: sum over i in [0, n-1) of _smooth_len(pos[i], pos[i+1], dtn[i])
        bonds = jax.vmap(lambda i: _smooth_len(pos[i], pos[i + 1], dtn[i],
                                                stretch_k, squeeze_k, dist_w))(jnp.arange(n - 1))
        # Angles: sum over i in [0, n-2)
        angles = jax.vmap(lambda i: _smooth_ang(pos[i], pos[i + 1], pos[i + 2],
                                                  ang_k, ang_w))(jnp.arange(n - 2))
        return jnp.sum(bonds) + jnp.sum(angles)

    def _init_excl(pos, r0, weight, skip):
        n = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=2))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        in_range = jnp.abs(idx[:, None] - idx[None, :]) > skip
        return jnp.sum(jnp.where(in_range, contrib, jnp.asarray(0.0, dtype)))

    def chain(pos0, dtn, step_size, T0, dt, js, jc, n_steps,
              stretch_k, squeeze_k, ang_k, dist_w, ang_w,
              r0, excl_w, excl_skip, key):
        n = pos0.shape[0]
        # Pre-gen RNG
        k_p, k_d, k_a = jax.random.split(key, 3)
        ps = jax.random.randint(k_p, (n_steps,), 0, n)
        disps = jax.random.uniform(k_d, (n_steps, 3),
                                   minval=-step_size, maxval=step_size, dtype=dtype)
        accs = jax.random.uniform(k_a, (n_steps,), dtype=dtype)

        score_struct0 = _init_smooth(pos0, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        score_excl0 = _init_excl(pos0, r0, excl_w, excl_skip)

        def body(i, carry):
            pos, score_struct, score_excl, T = carry
            p = ps[i]
            delta = disps[i]
            u = accs[i]

            score = score_struct + score_excl

            old_p_pos = pos[p]
            new_p_pos = old_p_pos + delta

            loc_struct_prev = _local_smooth_at(
                pos, old_p_pos, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
            loc_excl_prev = _local_excl_at(pos, old_p_pos, p, r0, excl_w, excl_skip)

            loc_struct_curr = _local_smooth_at(
                pos, new_p_pos, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w
            )
            loc_excl_curr = _local_excl_at(pos, new_p_pos, p, r0, excl_w, excl_skip)

            score_struct_new = score_struct + (loc_struct_curr - loc_struct_prev)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)
            score_new = score_struct_new + score_excl_new

            ok_unc = score_new < score  # smooth uses STRICT less-than
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p_pos, old_p_pos)
            pos_next = pos.at[p].set(final_p)
            score_struct_next = jnp.where(ok, score_struct_new, score_struct)
            score_excl_next = jnp.where(ok, score_excl_new, score_excl)
            return (pos_next, score_struct_next, score_excl_next, T * dt)

        init = (pos0, score_struct0, score_excl0, T0)
        pos_f, ss, se, _ = jax.lax.fori_loop(0, n_steps, body, init)
        return pos_f, ss + se

    # vmap over K chains; problem arrays (dtn) shared
    vmapped = jax.vmap(
        chain,
        in_axes=(0, None,  None, None, None, None, None, None,
                 None, None, None, None, None,
                 None, None, None, 0),
        out_axes=(0, 0),
    )
    # n_steps (positional 7) and excl_skip (positional 15) must be static —
    # n_steps for the RNG array shape, excl_skip is currently an int that XLA
    # would otherwise wrap as a 0-d tracer (fine), but making it static lets
    # the conditional be constant-folded.
    return jax.jit(vmapped, static_argnums=(7, 15))


def time_jax(kernel, pos_k_np, dtn_np, schedule, dtype_np, seed):
    """Returns (compile_time, steady_time, final_scores)."""
    (step_size, T0, dt, js, jc, n_steps,
     stretch_k, squeeze_k, ang_k, dist_w, ang_w,
     r0, excl_w, excl_skip) = schedule

    pos_k = jnp.asarray(pos_k_np, dtype=dtype_np)
    dtn = jnp.asarray(dtn_np, dtype=dtype_np)
    keys = jax.random.split(jax.random.PRNGKey(seed), pos_k.shape[0])

    args = (pos_k, dtn,
            dtype_np.type(step_size), dtype_np.type(T0), dtype_np.type(dt),
            dtype_np.type(js), dtype_np.type(jc),
            int(n_steps),
            dtype_np.type(stretch_k), dtype_np.type(squeeze_k), dtype_np.type(ang_k),
            dtype_np.type(dist_w), dtype_np.type(ang_w),
            dtype_np.type(r0), dtype_np.type(excl_w),
            int(excl_skip),
            keys)

    t0 = time.perf_counter()
    pos1, scores1 = kernel(*args)
    scores1.block_until_ready()
    t1 = time.perf_counter()

    keys2 = jax.random.split(jax.random.PRNGKey(seed + 1), pos_k.shape[0])
    args2 = list(args); args2[-1] = keys2
    args2[0] = jnp.asarray(pos_k_np, dtype=dtype_np)  # fresh starting positions
    t2 = time.perf_counter()
    pos2, scores2 = kernel(*args2)
    scores2.block_until_ready()
    t3 = time.perf_counter()

    return (t1 - t0) - (t3 - t2), t3 - t2, np.asarray(scores2)


# ----------------------------------------------------------------------------
# Bench driver
# ----------------------------------------------------------------------------


def make_problem(n: int, seed: int = 0):
    """Synthetic smooth-MC problem: a noisy chain with realistic-ish bond
    targets (dtn) and starting positions perturbed off a straight line."""
    rng = np.random.default_rng(seed)
    # Mean bond distance: pick something so excluded volume can plausibly fire
    dtn = rng.uniform(0.8, 1.2, size=n - 1).astype(np.float64)
    # Build positions: noisy chain (roughly straight with jitter so EV
    # contributions vary; some pairs will be inside the cutoff)
    pos = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n):
        pos[i] = pos[i - 1] + np.array([dtn[i - 1], 0.0, 0.0])
    pos += rng.normal(0.0, 0.3, size=pos.shape)
    return pos, dtn


def bench_one(n: int, k: int, n_steps: int, kernels: dict):
    pos, dtn = make_problem(n, seed=42)
    pos_k = np.broadcast_to(pos, (k, n, 3)).copy()

    # Smooth MC schedule — matches the dryrun config (which is itself derived
    # from data/GM12878/config.ini's smooth section).
    step_size = 5.0       # noise_smooth from config
    T0 = 5.0              # max_temp
    dt = 0.999            # delta_temp (dryrun-lenient)
    js = 50.0             # jump_temp_scale
    jc = 20.0             # jump_temp_coef
    stretch_k = 0.1       # spring_stretch
    squeeze_k = 0.1       # spring_squeeze
    ang_k = 0.1           # spring_angular
    dist_w = 1.0          # smooth_dist_weight
    ang_w = 1.0           # smooth_angle_weight
    # Excluded volume (auto-radius from mean dtn × default factor 0.5)
    r0 = 0.5 * float(dtn.mean())
    excl_w = 0.1          # exclusion_weight (config)
    excl_skip = 1         # exclusion_skip_neighbors

    schedule = (step_size, T0, dt, js, jc, n_steps,
                stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                r0, excl_w, excl_skip)

    out = {"N": n, "K": k, "steps": n_steps}

    # Numba (matches production: smooth-with-EV runs K=1 sequentially; we
    # loop K times to match work).
    pos_nb = pos_k.copy()
    t_nb, scores_nb = run_numba_kchains(
        pos_nb, dtn, step_size, T0, dt, js, jc, n_steps,
        stretch_k, squeeze_k, ang_k, dist_w, ang_w,
        r0, excl_w, excl_skip,
    )
    out["numba_s"] = t_nb
    out["numba_best"] = float(np.min(scores_nb))

    for name, (kernel, dtype_np) in kernels.items():
        try:
            ct, st, scores = time_jax(kernel, pos_k, dtn, schedule, dtype_np, 42)
            out[f"{name}_compile_s"] = ct
            out[f"{name}_s"] = st
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

    # Warmup numba (one tiny call so the JIT cache is warm)
    p, d = make_problem(16, 0)
    pk = np.broadcast_to(p, (2, 16, 3)).copy()
    _ = run_numba_kchains(pk, d, 0.1, 1.0, 0.999, 1.0, 1.0, 32,
                          0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 0.1, 1)
    print("Numba warmup done.", flush=True)

    print("Building JAX kernels...", flush=True)
    kernels = {
        "jax_f32": (make_jax_smooth_kernel(jnp.float32), np.dtype(np.float32)),
        "jax_f64": (make_jax_smooth_kernel(jnp.float64), np.dtype(np.float64)),
    }
    print("Kernels built.\n", flush=True)

    # Grid: focus on the realistic smooth-MC N range from the dryruns.
    # chr4 mode is N=2000-4000; chr22 has a peak at N=10116.  K=1 matches
    # production (smooth+EV has no prange path today); K=8/32 explores what
    # GPU vmap could add.
    configs = [
        # N,   K,   n_steps
        (1024,   1,  5000),
        (1024,   8,  5000),
        (1024,  32,  5000),
        (2048,   1,  5000),
        (2048,   8,  5000),
        (2048,  32,  5000),
        (4096,   1,  5000),
        (4096,   8,  5000),
        (4096,  32,  5000),
        (8192,   1,  5000),
        (8192,   8,  5000),
        (8192,  32,  5000),
    ]

    print("=== STEADY-STATE TIMES (second call, no compile) ===")
    header = f"{'N':>5} {'K':>4} {'steps':>6}  {'numba':>9}  {'jax_f32':>9}  {'jax_f64':>9}  {'speedup':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for (n, k, st) in configs:
        r = bench_one(n, k, st, kernels)
        rows.append(r)
        nb = r["numba_s"]; jf = r.get("jax_f32_s", float("nan"))
        spd = nb / jf if jf > 0 else float("nan")
        print(f"{r['N']:>5} {r['K']:>4} {r['steps']:>6}  "
              f"{nb:>9.3f}  {jf:>9.3f}  {r.get('jax_f64_s', float('nan')):>9.3f}  "
              f"{spd:>8.2f}x", flush=True)

    print("\n=== COMPILE TIMES (first - second call, seconds) ===")
    print(f"{'N':>5} {'K':>4}  {'jax_f32':>9}  {'jax_f64':>9}")
    print("-" * 36)
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  "
              f"{r.get('jax_f32_compile_s', float('nan')):>9.2f}  "
              f"{r.get('jax_f64_compile_s', float('nan')):>9.2f}", flush=True)

    print("\n=== BEST SCORES (lower = better; RNGs differ) ===")
    print(f"{'N':>5} {'K':>4}  {'numba':>14}  {'jax_f32':>14}  {'jax_f64':>14}")
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  "
              f"{r['numba_best']:>14.2f}  "
              f"{r.get('jax_f32_best', float('nan')):>14.2f}  "
              f"{r.get('jax_f64_best', float('nan')):>14.2f}", flush=True)

    print("\nNotes:")
    print("  - numba column matches production: smooth+EV has no prange path,")
    print("    so 'K chains' = K sequential single-chain runs.")
    print("  - JAX K>1 is genuine vmap parallelism — this is what's NOT available")
    print("    in numba today for the smooth-with-EV path.")
    print("  - If jax_f64 best-scores diverge wildly from numba, the algorithm")
    print("    has drifted between implementations.")


if __name__ == "__main__":
    main()
