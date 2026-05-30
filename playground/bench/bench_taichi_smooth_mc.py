"""Self-contained benchmark: Taichi vs numba for smooth Monte Carlo with EV.

Why Taichi: one of the few frameworks that runs the SAME source code on
CUDA, Metal, Vulkan, and CPU.  If the perf is competitive with JAX, this
solves the entire deployment matrix (Mac dev / NVIDIA prod / AMD / CPU CI)
from a single backend.

Design choice: the whole annealing loop lives inside ONE @ti.kernel.
  - Outer `for k in range(K)` is Taichi's top-level loop → auto-parallelized
    across chains (one thread per chain).
  - Inner sequential MC step loop runs serially within each thread.
  - The O(N) excluded-volume reduction also runs serially within each thread.

Limitation to be honest about: this design only exposes K-way parallelism.
A CUDA box with ~80 SMs × 32 warps = ~80k available threads can use this
efficiently only when K is large.  At K=1, we're using 0.001% of the GPU.
This is fundamentally different from JAX which vmaps K chains AND parallelizes
the O(N) reduction within each chain via XLA fusion.  Treat the K=1 column as
a lower bound; the realistic question is K=8/32.

Setup on a CUDA box:
    pip install taichi numba numpy
    python bench_taichi_smooth_mc.py
"""

# NB: no `from __future__ import annotations` — PEP 563 string forms break
# Taichi's @ti.kernel argument-type extraction (it inspects live type objects,
# not strings).

import math
import sys
import time

import numpy as np
from numba import njit  # type: ignore[import-not-found]

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False


# ----------------------------------------------------------------------------
# Numba reference (copied verbatim from bench_jax_smooth_mc.py)
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
def _batch_smooth_chain_nb(pos, dtn, step_size, T, dt, js, jc, n_steps,
                          stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                          r0, excl_w, excl_skip, score_struct, score_excl):
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
# Taichi implementation: ONE @ti.kernel containing the entire MC loop.
#
# Outer parallel: K chains.  Inner serial: MC steps + O(N) reductions.
# All RNG happens inside the kernel via ti.random (one Threefry-equivalent
# stream per chain, derived from the kernel's per-thread RNG state).
# ----------------------------------------------------------------------------


def make_taichi_mc():
    """Build and return the Taichi MC kernel.  Must be called AFTER ti.init."""

    # Taichi 1.7's annotation syntax: pass ndim only; dtype is inferred from the
    # numpy array at call time.  (`dtype=ti.f32, ndim=3` form raises
    # TaichiSyntaxError on 1.7.4.)
    @ti.kernel
    def mc_smooth_kchains(
        pos: ti.types.ndarray(ndim=3),       # (K, N, 3) f32
        dtn: ti.types.ndarray(ndim=1),       # (N-1,) f32
        final_score: ti.types.ndarray(ndim=1),  # (K,) f32 out
        n_steps: ti.i32,
        step_size: ti.f32,
        T0: ti.f32, dt: ti.f32, js: ti.f32, jc: ti.f32,
        stretch_k: ti.f32, squeeze_k: ti.f32, ang_k: ti.f32,
        dist_w: ti.f32, ang_w: ti.f32,
        r0: ti.f32, excl_w: ti.f32, excl_skip: ti.i32,
    ):
        # `ti.loop_config(serialize=False)` is the default for the outer
        # range(K) loop — Taichi auto-parallelises it.  Each value of k runs
        # in its own thread.
        for k in range(pos.shape[0]):
            N = pos.shape[1]

            # ---- initial structure score (bonds + angles) ----
            score_struct: ti.f32 = 0.0
            for i in range(N - 1):
                dx = pos[k, i, 0] - pos[k, i + 1, 0]
                dy = pos[k, i, 1] - pos[k, i + 1, 1]
                dz = pos[k, i, 2] - pos[k, i + 1, 2]
                d = ti.sqrt(dx * dx + dy * dy + dz * dz)
                e = ti.max(dtn[i], 1e-6)
                rel = (d - e) / e
                kk = stretch_k if rel >= 0.0 else squeeze_k
                score_struct += rel * rel * kk * dist_w
            for i in range(N - 2):
                v1x = pos[k, i, 0] - pos[k, i + 1, 0]
                v1y = pos[k, i, 1] - pos[k, i + 1, 1]
                v1z = pos[k, i, 2] - pos[k, i + 1, 2]
                v2x = pos[k, i + 1, 0] - pos[k, i + 2, 0]
                v2y = pos[k, i + 1, 1] - pos[k, i + 2, 1]
                v2z = pos[k, i + 1, 2] - pos[k, i + 2, 2]
                n1 = ti.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
                n2 = ti.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
                if n1 >= 1e-12 and n2 >= 1e-12:
                    cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
                    cos_a = ti.min(1.0, ti.max(-1.0, cos_a))
                    ang = 1.0 - (cos_a + 1.0) * 0.5
                    score_struct += ang * ang * ang * ang_k * ang_w

            # ---- initial excluded-volume score (O(N^2)) ----
            score_excl: ti.f32 = 0.0
            for i in range(N):
                for j in range(N):
                    diff = i - j
                    if diff < 0:
                        diff = -diff
                    if diff > excl_skip:
                        dx = pos[k, i, 0] - pos[k, j, 0]
                        dy = pos[k, i, 1] - pos[k, j, 1]
                        dz = pos[k, i, 2] - pos[k, j, 2]
                        d = ti.sqrt(dx * dx + dy * dy + dz * dz)
                        if d < r0:
                            rel = (r0 - d) / r0
                            score_excl += excl_w * rel * rel

            T = T0
            score = score_struct + score_excl

            # ---- MC step loop (serial within this chain) ----
            for _ in range(n_steps):
                p = ti.random(ti.i32) % N
                dx = (ti.random(ti.f32) * 2.0 - 1.0) * step_size
                dy = (ti.random(ti.f32) * 2.0 - 1.0) * step_size
                dz = (ti.random(ti.f32) * 2.0 - 1.0) * step_size

                # local smooth score at p (bonds (p-1,p) & (p,p+1), angles at p-2..p)
                # Compute loc_prev (using current pos[k, p, :]) and loc_curr
                # (using pos[k, p, :] + delta).  The two share the same loop
                # over the ±2 chain neighbourhood, so we inline both.
                old_x = pos[k, p, 0]
                old_y = pos[k, p, 1]
                old_z = pos[k, p, 2]
                new_x = old_x + dx
                new_y = old_y + dy
                new_z = old_z + dz

                loc_struct_prev: ti.f32 = 0.0
                loc_struct_curr: ti.f32 = 0.0

                # left bond i = p-1: bead (p-1) -- bead p
                if p - 1 >= 0 and p - 1 < N - 1:
                    ax = pos[k, p - 1, 0]; ay = pos[k, p - 1, 1]; az = pos[k, p - 1, 2]
                    e = ti.max(dtn[p - 1], 1e-6)
                    # prev
                    bx = old_x - ax; by = old_y - ay; bz = old_z - az
                    d = ti.sqrt(bx * bx + by * by + bz * bz)
                    rel = (d - e) / e
                    kk = stretch_k if rel >= 0.0 else squeeze_k
                    loc_struct_prev += rel * rel * kk * dist_w
                    # curr
                    bx = new_x - ax; by = new_y - ay; bz = new_z - az
                    d = ti.sqrt(bx * bx + by * by + bz * bz)
                    rel = (d - e) / e
                    kk = stretch_k if rel >= 0.0 else squeeze_k
                    loc_struct_curr += rel * rel * kk * dist_w

                # right bond i = p: bead p -- bead (p+1)
                if p >= 0 and p < N - 1:
                    ax = pos[k, p + 1, 0]; ay = pos[k, p + 1, 1]; az = pos[k, p + 1, 2]
                    e = ti.max(dtn[p], 1e-6)
                    # prev
                    bx = old_x - ax; by = old_y - ay; bz = old_z - az
                    d = ti.sqrt(bx * bx + by * by + bz * bz)
                    rel = (d - e) / e
                    kk = stretch_k if rel >= 0.0 else squeeze_k
                    loc_struct_prev += rel * rel * kk * dist_w
                    bx = new_x - ax; by = new_y - ay; bz = new_z - az
                    d = ti.sqrt(bx * bx + by * by + bz * bz)
                    rel = (d - e) / e
                    kk = stretch_k if rel >= 0.0 else squeeze_k
                    loc_struct_curr += rel * rel * kk * dist_w

                # angles at i in {p-2, p-1, p}; each uses (i, i+1, i+2).
                for off in range(-2, 1):
                    i = p + off
                    if i >= 0 and i < N - 2:
                        # Beads (i, i+1, i+2).  We substitute (new_x,new_y,new_z)
                        # for whichever bead equals p; the others are read from pos.
                        a0x = pos[k, i, 0];     a0y = pos[k, i, 1];     a0z = pos[k, i, 2]
                        a1x = pos[k, i + 1, 0]; a1y = pos[k, i + 1, 1]; a1z = pos[k, i + 1, 2]
                        a2x = pos[k, i + 2, 0]; a2y = pos[k, i + 2, 1]; a2z = pos[k, i + 2, 2]

                        # PREV: pos[p] is already (old_x,old_y,old_z) — but we
                        # read from pos[] which still holds old_*.  So prev
                        # naturally uses old at index p.
                        v1x = a0x - a1x; v1y = a0y - a1y; v1z = a0z - a1z
                        v2x = a1x - a2x; v2y = a1y - a2y; v2z = a1z - a2z
                        n1 = ti.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
                        n2 = ti.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
                        if n1 >= 1e-12 and n2 >= 1e-12:
                            cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
                            cos_a = ti.min(1.0, ti.max(-1.0, cos_a))
                            ang = 1.0 - (cos_a + 1.0) * 0.5
                            loc_struct_prev += ang * ang * ang * ang_k * ang_w

                        # CURR: substitute new_* where the bead equals p
                        c0x = new_x if i == p else a0x
                        c0y = new_y if i == p else a0y
                        c0z = new_z if i == p else a0z
                        c1x = new_x if (i + 1) == p else a1x
                        c1y = new_y if (i + 1) == p else a1y
                        c1z = new_z if (i + 1) == p else a1z
                        c2x = new_x if (i + 2) == p else a2x
                        c2y = new_y if (i + 2) == p else a2y
                        c2z = new_z if (i + 2) == p else a2z
                        v1x = c0x - c1x; v1y = c0y - c1y; v1z = c0z - c1z
                        v2x = c1x - c2x; v2y = c1y - c2y; v2z = c1z - c2z
                        n1 = ti.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
                        n2 = ti.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
                        if n1 >= 1e-12 and n2 >= 1e-12:
                            cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
                            cos_a = ti.min(1.0, ti.max(-1.0, cos_a))
                            ang = 1.0 - (cos_a + 1.0) * 0.5
                            loc_struct_curr += ang * ang * ang * ang_k * ang_w

                # excluded-volume local score: sum over i with |i-p|>skip
                loc_excl_prev: ti.f32 = 0.0
                loc_excl_curr: ti.f32 = 0.0
                for i in range(N):
                    diff = i - p
                    if diff < 0:
                        diff = -diff
                    if diff > excl_skip:
                        ax = pos[k, i, 0]; ay = pos[k, i, 1]; az = pos[k, i, 2]
                        bx = ax - old_x; by = ay - old_y; bz = az - old_z
                        d_prev = ti.sqrt(bx * bx + by * by + bz * bz)
                        if d_prev < r0:
                            rel = (r0 - d_prev) / r0
                            loc_excl_prev += excl_w * rel * rel
                        bx = ax - new_x; by = ay - new_y; bz = az - new_z
                        d_curr = ti.sqrt(bx * bx + by * by + bz * bz)
                        if d_curr < r0:
                            rel = (r0 - d_curr) / r0
                            loc_excl_curr += excl_w * rel * rel

                score_struct_new = score_struct + (loc_struct_curr - loc_struct_prev)
                score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)
                score_new = score_struct_new + score_excl_new

                ok = score_new < score
                if not ok and T > 0.0 and score > 0.0:
                    ratio = score_new / ti.max(score, 1e-30)
                    expo = -jc * ratio / ti.max(T, 1e-30)
                    expo = ti.min(80.0, ti.max(-80.0, expo))
                    p_acc = js * ti.exp(expo)
                    ok = ti.random(ti.f32) < p_acc

                if ok:
                    pos[k, p, 0] = new_x
                    pos[k, p, 1] = new_y
                    pos[k, p, 2] = new_z
                    score = score_new
                    score_struct = score_struct_new
                    score_excl = score_excl_new
                T *= dt

            final_score[k] = score_struct + score_excl

    return mc_smooth_kchains


def run_taichi_kchains(kernel, pos_k_np, dtn_np, step_size, T0, dt, js, jc, n_steps,
                      stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                      r0, excl_w, excl_skip):
    """Time a full K-chain MC run via Taichi.  Returns (compile_time, steady_time, scores)."""
    K = pos_k_np.shape[0]

    pos_a = pos_k_np.astype(np.float32, copy=True)
    dtn_a = dtn_np.astype(np.float32, copy=True)
    final_a = np.zeros(K, dtype=np.float32)

    # First call: compile + run.  Sync before stopping the timer.
    t0 = time.perf_counter()
    kernel(pos_a, dtn_a, final_a,
           int(n_steps), float(step_size),
           float(T0), float(dt), float(js), float(jc),
           float(stretch_k), float(squeeze_k), float(ang_k),
           float(dist_w), float(ang_w),
           float(r0), float(excl_w), int(excl_skip))
    ti.sync()
    t1 = time.perf_counter()

    # Reset and run a second time for steady-state timing.
    pos_a2 = pos_k_np.astype(np.float32, copy=True)
    final_a2 = np.zeros(K, dtype=np.float32)
    t2 = time.perf_counter()
    kernel(pos_a2, dtn_a, final_a2,
           int(n_steps), float(step_size),
           float(T0), float(dt), float(js), float(jc),
           float(stretch_k), float(squeeze_k), float(ang_k),
           float(dist_w), float(ang_w),
           float(r0), float(excl_w), int(excl_skip))
    ti.sync()
    t3 = time.perf_counter()

    return (t1 - t0) - (t3 - t2), t3 - t2, final_a2


# ----------------------------------------------------------------------------
# Bench driver
# ----------------------------------------------------------------------------


def make_problem(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dtn = rng.uniform(0.8, 1.2, size=n - 1).astype(np.float64)
    pos = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n):
        pos[i] = pos[i - 1] + np.array([dtn[i - 1], 0.0, 0.0])
    pos += rng.normal(0.0, 0.3, size=pos.shape)
    return pos, dtn


def main():
    if not HAS_TAICHI:
        print("Taichi not installed.  pip install taichi", file=sys.stderr)
        sys.exit(2)

    # ti.init must be called BEFORE @ti.kernel definitions are touched.
    # ti.gpu picks CUDA on NVIDIA, Metal on Mac, Vulkan elsewhere.
    ti.init(arch=ti.gpu)
    print(f"Taichi arch: {ti.cfg.arch}  taichi={ti.__version__}", flush=True)

    # Warmup numba
    p, d = make_problem(16, 0)
    pk = np.broadcast_to(p, (2, 16, 3)).copy()
    _ = run_numba_kchains(pk, d, 0.1, 1.0, 0.999, 1.0, 1.0, 32,
                          0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 0.1, 1)
    print("Numba warmup done.\n", flush=True)

    print("Building Taichi kernel...", flush=True)
    mc_kernel = make_taichi_mc()
    print("Kernel defined (lazy-compiles on first call).\n", flush=True)

    sched = dict(
        step_size=5.0, T0=5.0, dt=0.999, js=50.0, jc=20.0,
        stretch_k=0.1, squeeze_k=0.1, ang_k=0.1, dist_w=1.0, ang_w=1.0,
        excl_w=0.1, excl_skip=1,
    )

    configs = [
        (1024,  1, 5000), (1024,  8, 5000), (1024, 32, 5000),
        (2048,  1, 5000), (2048,  8, 5000), (2048, 32, 5000),
        (4096,  1, 5000), (4096,  8, 5000), (4096, 32, 5000),
        (8192,  1, 5000), (8192,  8, 5000), (8192, 32, 5000),
    ]

    print("=== STEADY-STATE TIMES ===")
    header = (f"{'N':>5} {'K':>4} {'steps':>6}  {'numba':>9}  "
              f"{'taichi':>9}  {'speedup':>9}")
    print(header)
    print("-" * len(header))
    rows = []
    for (n, k, st) in configs:
        pos, dtn = make_problem(n, seed=42)
        pos_k = np.broadcast_to(pos, (k, n, 3)).copy()
        r0 = 0.5 * float(dtn.mean())

        # Numba
        pos_nb = pos_k.copy()
        t_nb, scores_nb = run_numba_kchains(
            pos_nb, dtn, sched["step_size"], sched["T0"], sched["dt"],
            sched["js"], sched["jc"], st,
            sched["stretch_k"], sched["squeeze_k"], sched["ang_k"],
            sched["dist_w"], sched["ang_w"],
            r0, sched["excl_w"], sched["excl_skip"],
        )

        # Taichi
        try:
            ct, t_ti, scores_ti = run_taichi_kchains(
                mc_kernel, pos_k, dtn,
                sched["step_size"], sched["T0"], sched["dt"],
                sched["js"], sched["jc"], st,
                sched["stretch_k"], sched["squeeze_k"], sched["ang_k"],
                sched["dist_w"], sched["ang_w"],
                r0, sched["excl_w"], sched["excl_skip"],
            )
        except Exception as ex:
            ct, t_ti, scores_ti = float("nan"), float("nan"), np.array([float("nan")])
            print(f"  FAIL N={n} K={k}: {type(ex).__name__}: {ex}", flush=True)

        spd = t_nb / t_ti if t_ti > 0 else float("nan")
        print(f"{n:>5} {k:>4} {st:>6}  {t_nb:>9.3f}  {t_ti:>9.3f}  {spd:>8.2f}x",
              flush=True)
        rows.append({
            "N": n, "K": k, "steps": st,
            "numba_s": t_nb, "taichi_s": t_ti, "taichi_compile_s": ct,
            "numba_best": float(np.min(scores_nb)),
            "taichi_best": float(np.min(scores_ti)),
        })

    print("\n=== COMPILE TIMES (first - second call, seconds) ===")
    print(f"{'N':>5} {'K':>4}  {'taichi':>9}")
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  {r['taichi_compile_s']:>9.2f}", flush=True)

    print("\n=== BEST SCORES (lower = better) ===")
    print(f"{'N':>5} {'K':>4}  {'numba':>14}  {'taichi':>14}")
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  {r['numba_best']:>14.2f}  "
              f"{r['taichi_best']:>14.2f}", flush=True)

    print("\nNotes:")
    print("  - taichi uses K-way parallelism only (one thread per chain).  At K=1")
    print("    only one CUDA thread runs the whole MC loop — expect terrible numbers.")
    print("  - At K=32 we use ~32 threads of an 80k-thread GPU; <0.05% utilization.")
    print("    Real CUDA wins need within-chain parallelism (block-level reduction),")
    print("    which would be a more involved Taichi program.")
    print("  - JAX wins here because xla.vmap + xla fusion gets BOTH K-parallelism")
    print("    AND O(N) reduction parallelism in one kernel.")


if __name__ == "__main__":
    main()
