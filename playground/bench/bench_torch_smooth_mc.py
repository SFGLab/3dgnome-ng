"""PyTorch port of the smooth-MC + EV kernel, benched against numba.

Companion to bench_jax_smooth_mc.py.  The deployment story requires support
for Apple Silicon (MPS), NVIDIA (CUDA), AMD (ROCm), and CPU-only fallback —
PyTorch is the only single framework that covers all four targets.

What this bench measures:
  - numba             (reference, matches gnome3d/mc.py production path)
  - torch_eager_f32   (PyTorch eager mode, no compile — establishes torch overhead)
  - torch_compile_f32 (PyTorch with torch.compile, mode='reduce-overhead')
                       reduce-overhead enables CUDA graphs on NVIDIA, falls back
                       to default mode on MPS/ROCm/CPU

Device selection is automatic: CUDA > MPS > CPU.  Prints which device was used
so the result table can be interpreted.  Run the same script on a Mac dev box
and a CUDA box to see the deployment-relevant numbers.

Install:
    Mac:     pip install torch numba numpy        # torch ships with MPS
    CUDA:    pip install torch numba numpy        # default whl ships with CUDA
    ROCm:    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from numba import njit  # type: ignore[import-not-found]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ----------------------------------------------------------------------------
# Numba reference (copied verbatim from bench_jax_smooth_mc.py so this file
# is self-contained).  Single-chain matches the production path for
# smooth-with-EV configs.
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
# PyTorch implementation: batched K chains via leading dim, pre-gen RNG,
# Python for-loop calling a (compiled) step function.
#
# Choice: manual batching over leading K dim (not torch.func.vmap) because
# torch.compile + vmap interaction has version-dependent quirks; manual
# batching has identical perf and is rock-solid across torch versions.
# ----------------------------------------------------------------------------


def pick_device():
    """CUDA > MPS > CPU.  Print and return."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_init_smooth(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    """Initial chain-spring + angle score.  pos: (K, N, 3)  dtn: (N-1,)  -> (K,)"""
    diff = pos[:, :-1, :] - pos[:, 1:, :]                   # (K, N-1, 3)
    d = torch.linalg.norm(diff, dim=-1)                     # (K, N-1)
    e = torch.clamp(dtn, min=1e-6)                          # (N-1,)
    rel = (d - e) / e                                       # (K, N-1)
    k = torch.where(rel >= 0, stretch_k, squeeze_k)
    bonds = (rel * rel * k * dist_w).sum(dim=-1)            # (K,)

    v1 = pos[:, :-2, :] - pos[:, 1:-1, :]                   # (K, N-2, 3)
    v2 = pos[:, 1:-1, :] - pos[:, 2:, :]                    # (K, N-2, 3)
    n1 = torch.linalg.norm(v1, dim=-1)
    n2 = torch.linalg.norm(v2, dim=-1)
    cos_a = (v1 * v2).sum(-1) / torch.clamp(n1 * n2, min=1e-30)
    cos_a = torch.clamp(cos_a, -1.0, 1.0)
    valid = (n1 > 1e-12) & (n2 > 1e-12)
    ang = 1.0 - (cos_a + 1.0) * 0.5
    contrib = torch.where(valid, ang * ang * ang * ang_k * ang_w, torch.zeros_like(ang))
    angles = contrib.sum(dim=-1)                            # (K,)
    return bonds + angles


def _torch_init_excl(pos, r0, weight, skip):
    """Initial excluded-volume score (double-counted), per chain.
    pos: (K, N, 3)  -> (K,)"""
    N = pos.shape[1]
    diff = pos[:, :, None, :] - pos[:, None, :, :]          # (K, N, N, 3)
    d = torch.linalg.norm(diff, dim=-1)                     # (K, N, N)
    rel = torch.clamp((r0 - d) / r0, min=0.0)
    contrib = weight * rel * rel
    idx = torch.arange(N, device=pos.device)
    in_range = (idx[:, None] - idx[None, :]).abs() > skip   # (N, N)
    return torch.where(in_range, contrib, torch.zeros_like(contrib)).sum(dim=(-2, -1))


def build_torch_step(N: int, K: int, dtn: torch.Tensor, dtype: torch.dtype, device: torch.device,
                    stretch_k: float, squeeze_k: float, ang_k: float,
                    dist_w: float, ang_w: float,
                    r0: float, excl_w: float, excl_skip: int,
                    dt: float, js: float, jc: float,
                    compile_step: bool = True):
    """Build a (K-batched) MC step function.  Returns a callable that takes
    (pos, score_struct, score_excl, T, p_batch, delta_batch, u_batch) and
    returns new state."""
    # Bind scalar constants as zero-d tensors so torch.compile traces them
    # as constants (no recompile when a config dict changes — they're baked
    # in here at build time, which matches production).
    stretch_k_t = torch.tensor(stretch_k, dtype=dtype, device=device)
    squeeze_k_t = torch.tensor(squeeze_k, dtype=dtype, device=device)
    ang_k_t = torch.tensor(ang_k, dtype=dtype, device=device)
    dist_w_t = torch.tensor(dist_w, dtype=dtype, device=device)
    ang_w_t = torch.tensor(ang_w, dtype=dtype, device=device)
    r0_t = torch.tensor(r0, dtype=dtype, device=device)
    excl_w_t = torch.tensor(excl_w, dtype=dtype, device=device)
    dt_t = torch.tensor(dt, dtype=dtype, device=device)
    js_t = torch.tensor(js, dtype=dtype, device=device)
    jc_t = torch.tensor(jc, dtype=dtype, device=device)

    idx_N = torch.arange(N, device=device)
    ar_K = torch.arange(K, device=device)  # hoisted once at build time

    def _local_smooth_at(pos, p_pos, p_batch):
        """pos: (K, N, 3), p_pos: (K, 3), p_batch: (K,) int -> (K,)
        Mirrors _local_smooth_nb: bonds (p-1, p) and (p, p+1), angles at
        i in {p-2, p-1, p}.  Uses safe indexing + validity masks because
        p can be 0 or N-1."""

        # Bond i=p-1 between pos[p-1] and p_pos.  Valid: 1 <= p < N (so p-1 in [0, N-2])
        pm1 = torch.clamp(p_batch - 1, min=0)
        pp1 = torch.clamp(p_batch + 1, max=N - 1)
        a_pm1 = pos[ar_K, pm1]                              # (K, 3)
        a_pp1 = pos[ar_K, pp1]                              # (K, 3)

        # Left bond: (a_pm1, p_pos), target distance dtn[p-1]
        bond_left_valid = (p_batch - 1 >= 0) & (p_batch - 1 < N - 1)
        diff_L = a_pm1 - p_pos
        d_L = torch.linalg.norm(diff_L, dim=-1)
        e_L = torch.clamp(dtn[pm1], min=1e-6)
        rel_L = (d_L - e_L) / e_L
        kL = torch.where(rel_L >= 0, stretch_k_t, squeeze_k_t)
        bond_L = torch.where(bond_left_valid, rel_L * rel_L * kL * dist_w_t,
                             torch.zeros_like(d_L))

        # Right bond: (p_pos, a_pp1), target distance dtn[p]
        bond_right_valid = (p_batch >= 0) & (p_batch < N - 1)
        diff_R = p_pos - a_pp1
        d_R = torch.linalg.norm(diff_R, dim=-1)
        # dtn index = p (clamped for safety on the masked-out elements)
        e_R = torch.clamp(dtn[torch.clamp(p_batch, max=N - 2)], min=1e-6)
        rel_R = (d_R - e_R) / e_R
        kR = torch.where(rel_R >= 0, stretch_k_t, squeeze_k_t)
        bond_R = torch.where(bond_right_valid, rel_R * rel_R * kR * dist_w_t,
                             torch.zeros_like(d_R))

        # Angles at i in {p-2, p-1, p}; each uses beads (i, i+1, i+2).
        # The angle term touches p_pos when one of i, i+1, i+2 equals p.
        def angle_term(i_offset):
            i = p_batch + i_offset                          # (K,)
            i_clamped = torch.clamp(i, min=0, max=N - 3)
            i1 = torch.clamp(i + 1, min=0, max=N - 1)
            i2 = torch.clamp(i + 2, min=0, max=N - 1)
            a0 = pos[ar_K, i_clamped]
            a1 = pos[ar_K, i1]
            a2 = pos[ar_K, i2]
            # Substitute p_pos wherever this bead is p
            a0 = torch.where((i == p_batch).unsqueeze(-1), p_pos, a0)
            a1 = torch.where((i + 1 == p_batch).unsqueeze(-1), p_pos, a1)
            a2 = torch.where((i + 2 == p_batch).unsqueeze(-1), p_pos, a2)
            v1 = a0 - a1
            v2 = a1 - a2
            n1 = torch.linalg.norm(v1, dim=-1)
            n2 = torch.linalg.norm(v2, dim=-1)
            cos_a = (v1 * v2).sum(-1) / torch.clamp(n1 * n2, min=1e-30)
            cos_a = torch.clamp(cos_a, -1.0, 1.0)
            ang = 1.0 - (cos_a + 1.0) * 0.5
            term = ang * ang * ang * ang_k_t * ang_w_t
            valid = (i >= 0) & (i < N - 2) & (n1 > 1e-12) & (n2 > 1e-12)
            return torch.where(valid, term, torch.zeros_like(term))

        ang_sum = angle_term(-2) + angle_term(-1) + angle_term(0)
        return bond_L + bond_R + ang_sum

    def _local_excl_at(pos, p_pos, p_batch):
        """pos: (K, N, 3), p_pos: (K, 3), p_batch: (K,) int -> (K,)
        Sum over i with |i - p| > excl_skip of weight * ((r0 - d)/r0)² for d < r0."""
        diff = pos - p_pos.unsqueeze(1)                      # (K, N, 3)
        d = torch.linalg.norm(diff, dim=-1)                  # (K, N)
        rel = torch.clamp((r0_t - d) / r0_t, min=0.0)
        contrib = excl_w_t * rel * rel
        diff_idx = (idx_N.unsqueeze(0) - p_batch.unsqueeze(1)).abs()  # (K, N)
        in_range = diff_idx > excl_skip
        return torch.where(in_range, contrib, torch.zeros_like(contrib)).sum(dim=-1)

    def step(pos, score_struct, score_excl, T, p_batch, delta_batch, u_batch):
        """One MC step, batched over K chains.  Mutates pos in place.
        pos:           (K, N, 3)
        score_struct:  (K,)
        score_excl:    (K,)
        T:             scalar tensor
        p_batch:       (K,) int64
        delta_batch:   (K, 3)
        u_batch:       (K,)  uniform [0, 1)
        """
        score = score_struct + score_excl
        old_p_pos = pos[ar_K, p_batch]                      # (K, 3) — view
        new_p_pos = old_p_pos + delta_batch

        loc_struct_prev = _local_smooth_at(pos, old_p_pos, p_batch)
        loc_excl_prev = _local_excl_at(pos, old_p_pos, p_batch)
        loc_struct_curr = _local_smooth_at(pos, new_p_pos, p_batch)
        loc_excl_curr = _local_excl_at(pos, new_p_pos, p_batch)

        score_struct_new = score_struct + (loc_struct_curr - loc_struct_prev)
        score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)
        score_new = score_struct_new + score_excl_new

        ok_unc = score_new < score                          # smooth: strict
        can_jump = (T > 0) & (score > 0)
        exponent = -jc_t * (score_new / torch.clamp(score, min=1e-30)) / torch.clamp(T, min=1e-30)
        exponent = torch.clamp(exponent, -80.0, 80.0)
        p_acc = js_t * torch.exp(exponent)
        ok = ok_unc | (can_jump & (u_batch < p_acc))

        # Functional update via index_put (returns a new tensor).  In-place
        # mutation defeats CUDA graphs ("skipping cudagraphs due to mutated
        # inputs") which is the whole point of mode='reduce-overhead'.
        # On eager this pays an N×3×K clone per step (≫ numba), but eager on
        # GPU is launch-bound regardless — only compile is competitive.
        final_p_pos = torch.where(ok.unsqueeze(-1), new_p_pos, old_p_pos)
        pos_next = pos.index_put((ar_K, p_batch), final_p_pos)

        score_struct_next = torch.where(ok, score_struct_new, score_struct)
        score_excl_next = torch.where(ok, score_excl_new, score_excl)
        T_next = T * dt_t
        return pos_next, score_struct_next, score_excl_next, T_next

    if compile_step:
        # mode='default' = inductor fusion without CUDA graph capture.
        # We tried 'reduce-overhead' (CUDA graphs) and hit:
        #   "accessing tensor output of CUDAGraphs that has been overwritten by
        #    a subsequent run"
        # because each step's output aliases the next step's input in the
        # Python loop, and the graph allocator recycles the buffer before we
        # finish reading it.  cudagraph_mark_step_begin() didn't break the
        # cycle for this access pattern.  'default' gives us the kernel-fusion
        # win (~1 launch per step instead of ~50 in eager) without graph
        # replay; per-step Python+launch overhead remains.
        step = torch.compile(step, mode="default", fullgraph=True, dynamic=False)
    return step


def run_torch_kchains(pos_k_np, dtn_np, step_size, T0, dt, js, jc, n_steps,
                     stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                     r0, excl_w, excl_skip, dtype, device, compile_step,
                     seed):
    """Time a full K-chain run.  Returns (compile_time, steady_time, scores)."""
    K, N, _ = pos_k_np.shape

    pos = torch.from_numpy(pos_k_np).to(dtype=dtype, device=device)
    dtn = torch.from_numpy(dtn_np).to(dtype=dtype, device=device)

    step = build_torch_step(N, K, dtn, dtype, device,
                            stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                            r0, excl_w, excl_skip,
                            dt, js, jc, compile_step=compile_step)

    # Pre-generate RNG (all on device).  Same shape every iteration so
    # the compiled graph specializes once.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ps_cpu = torch.randint(0, N, (n_steps, K), generator=gen, dtype=torch.int64)
    disps_cpu = (torch.rand((n_steps, K, 3), generator=gen, dtype=torch.float32) * 2 - 1) * step_size
    accs_cpu = torch.rand((n_steps, K), generator=gen, dtype=torch.float32)
    ps = ps_cpu.to(device=device)
    disps = disps_cpu.to(device=device, dtype=dtype)
    accs = accs_cpu.to(device=device, dtype=dtype)

    score_struct = _torch_init_smooth(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
    score_excl = _torch_init_excl(pos, r0, excl_w, excl_skip)
    T_t = torch.tensor(T0, dtype=dtype, device=device)

    # When torch.compile uses CUDA graphs (mode='reduce-overhead' on CUDA),
    # each replay reuses the output buffer of the previous replay.  We must
    # call `cudagraph_mark_step_begin()` before each invocation to tell the
    # allocator the previous outputs have been consumed.  No-op on non-CUDA
    # devices and on the eager path.
    cuda_mark = getattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    def _run():
        nonlocal pos, score_struct, score_excl, T_t
        for i in range(n_steps):
            cuda_mark()
            pos, score_struct, score_excl, T_t = step(
                pos, score_struct, score_excl, T_t, ps[i], disps[i], accs[i]
            )

    # First call: compile + run.  Sync before stopping the timer.
    t0 = time.perf_counter()
    _run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    # Reset state and run a second time to measure steady-state.
    pos = torch.from_numpy(pos_k_np).to(dtype=dtype, device=device)
    score_struct = _torch_init_smooth(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
    score_excl = _torch_init_excl(pos, r0, excl_w, excl_skip)
    T_t = torch.tensor(T0, dtype=dtype, device=device)

    # New RNG for the second run (different stream → don't get cached results)
    gen2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    ps2 = torch.randint(0, N, (n_steps, K), generator=gen2, dtype=torch.int64).to(device)
    disps2 = ((torch.rand((n_steps, K, 3), generator=gen2, dtype=torch.float32) * 2 - 1)
              * step_size).to(device=device, dtype=dtype)
    accs2 = torch.rand((n_steps, K), generator=gen2, dtype=torch.float32).to(device=device, dtype=dtype)
    ps, disps, accs = ps2, disps2, accs2

    t2 = time.perf_counter()
    _run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t3 = time.perf_counter()

    final_scores = (score_struct + score_excl).detach().cpu().numpy()
    return (t1 - t0) - (t3 - t2), t3 - t2, final_scores


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
    if not HAS_TORCH:
        print("PyTorch not installed.", file=sys.stderr)
        sys.exit(2)

    device = pick_device()
    print(f"PyTorch device: {device}   torch={torch.__version__}", flush=True)

    # Warmup numba
    p, d = make_problem(16, 0)
    pk = np.broadcast_to(p, (2, 16, 3)).copy()
    _ = run_numba_kchains(pk, d, 0.1, 1.0, 0.999, 1.0, 1.0, 32,
                          0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 0.1, 1)
    print("Numba warmup done.\n", flush=True)

    # Smooth MC schedule — same as the JAX bench
    sched = dict(
        step_size=5.0, T0=5.0, dt=0.999, js=50.0, jc=20.0,
        stretch_k=0.1, squeeze_k=0.1, ang_k=0.1, dist_w=1.0, ang_w=1.0,
        excl_w=0.1, excl_skip=1,
    )

    # Same grid as the JAX bench so numbers are comparable
    configs = [
        (1024,  1, 5000), (1024,  8, 5000), (1024, 32, 5000),
        (2048,  1, 5000), (2048,  8, 5000), (2048, 32, 5000),
        (4096,  1, 5000), (4096,  8, 5000), (4096, 32, 5000),
        (8192,  1, 5000), (8192,  8, 5000), (8192, 32, 5000),
    ]

    print("=== STEADY-STATE TIMES ===")
    header = (f"{'N':>5} {'K':>4} {'steps':>6}  {'numba':>9}  "
              f"{'torch_eager':>11}  {'torch_compile':>13}  "
              f"{'eager_x':>8}  {'compile_x':>9}")
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

        # Torch eager
        try:
            ct_e, t_e, scores_e = run_torch_kchains(
                pos_k, dtn, sched["step_size"], sched["T0"], sched["dt"],
                sched["js"], sched["jc"], st,
                sched["stretch_k"], sched["squeeze_k"], sched["ang_k"],
                sched["dist_w"], sched["ang_w"],
                r0, sched["excl_w"], sched["excl_skip"],
                dtype=torch.float32, device=device, compile_step=False, seed=42,
            )
        except Exception as ex:
            ct_e, t_e, scores_e = float("nan"), float("nan"), np.array([float("nan")])
            print(f"  eager FAIL N={n} K={k}: {type(ex).__name__}: {ex}", flush=True)

        # Torch compile
        try:
            ct_c, t_c, scores_c = run_torch_kchains(
                pos_k, dtn, sched["step_size"], sched["T0"], sched["dt"],
                sched["js"], sched["jc"], st,
                sched["stretch_k"], sched["squeeze_k"], sched["ang_k"],
                sched["dist_w"], sched["ang_w"],
                r0, sched["excl_w"], sched["excl_skip"],
                dtype=torch.float32, device=device, compile_step=True, seed=42,
            )
        except Exception as ex:
            ct_c, t_c, scores_c = float("nan"), float("nan"), np.array([float("nan")])
            print(f"  compile FAIL N={n} K={k}: {type(ex).__name__}: {ex}", flush=True)

        eager_x = t_nb / t_e if t_e > 0 else float("nan")
        compile_x = t_nb / t_c if t_c > 0 else float("nan")
        print(f"{n:>5} {k:>4} {st:>6}  {t_nb:>9.3f}  "
              f"{t_e:>11.3f}  {t_c:>13.3f}  "
              f"{eager_x:>7.2f}x  {compile_x:>8.2f}x", flush=True)
        rows.append({
            "N": n, "K": k, "steps": st,
            "numba_s": t_nb, "torch_eager_s": t_e, "torch_compile_s": t_c,
            "torch_eager_compile_s": ct_e, "torch_compile_compile_s": ct_c,
            "numba_best": float(np.min(scores_nb)),
            "torch_eager_best": float(np.min(scores_e)),
            "torch_compile_best": float(np.min(scores_c)),
        })

    print("\n=== COMPILE TIMES (first - second call, seconds) ===")
    print(f"{'N':>5} {'K':>4}  {'eager':>9}  {'compile':>9}")
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  "
              f"{r['torch_eager_compile_s']:>9.2f}  "
              f"{r['torch_compile_compile_s']:>9.2f}", flush=True)

    print("\n=== BEST SCORES (lower = better) ===")
    print(f"{'N':>5} {'K':>4}  {'numba':>14}  {'torch_eager':>14}  {'torch_compile':>14}")
    for r in rows:
        print(f"{r['N']:>5} {r['K']:>4}  {r['numba_best']:>14.2f}  "
              f"{r['torch_eager_best']:>14.2f}  {r['torch_compile_best']:>14.2f}", flush=True)

    print("\nNotes:")
    print("  - eager_x / compile_x = numba_s / torch_*_s.  >1 means torch wins.")
    print("  - 'compile' time on first call includes torch.compile tracing +")
    print("    inductor codegen + (on CUDA) CUDA graph capture.")
    print("  - On MPS, torch.compile falls back to default mode (no CUDA graphs).")
    print("  - Best-score columns should be within 10% across backends; if not,")
    print("    the algorithm port has drifted.")


if __name__ == "__main__":
    main()
