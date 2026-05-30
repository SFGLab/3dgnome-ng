"""Micro-bench: ONE full-state gradient step vs ONE single-bead MC sweep.

QUESTION THIS ANSWERS
---------------------
At production heatmap bead counts (e.g. `[chr5 IB] heat dist matrix: 12359
beads (11348 movable)`), is the smooth+heat MC bottleneck the *sequential
single-bead* move shape, and would a *full-state gradient* move be faster on a
GPU?

The two shapes, for "advance every bead once":

  single-bead MC : ~N sequential fori_loop iterations, each an O(N) heat-column
                   reduction. Latency-bound — ~N tiny serial kernels, GPU mostly
                   idle inside each. Cannot parallelize across beads: move i+1
                   depends on accept/reject of move i (shared mutable pos/score).

  gradient step  : ONE fused O(N^2) pairwise pass (the N-body force pattern),
                   moving all N beads at once. Throughput-bound — saturates the
                   GPU. Cost grows with N^2 but in a single parallel op.

If the hypothesis holds, grad/sweep speedup should GROW with N, and be large at
N>=8000. On CPU the gap is a conservative LOWER BOUND (few cores); the real
number needs a CUDA/ROCm box.

WHAT IS COMPARED
----------------
Both use the SAME energy (chain bonds + angles + dense heat), formulas copied
verbatim from gnome3d/mc_jax.py so this is apples-to-apples on the same backend
and dtype. This intentionally does NOT involve numba — the existing
playground/bench_jax_smooth_mc.py already covers numba-vs-JAX for MC. The new
variable here is the move *shape* (sequential vs full-state), not the language.

  * MC sweep    : faithful fori_loop, n_movable single-bead moves, ratio-based
                  acceptance (score_new < score) OR rand < js*exp(-jc*(s_new/s)/T),
                  matching gnome3d/mc_jax.py::_build_smooth_kernel body.
  * grad step   : jax.value_and_grad(energy) + one overdamped-Langevin update
                  x <- x - lr*grad + sqrt(2*lr*T)*noise.  We time the grad eval
                  (the dominant cost; the update is a cheap elementwise op).

CAVEATS (read before trusting a number)
----------------------------------------
* This measures THROUGHPUT PER STEP, not convergence. A gradient step and an MC
  sweep are not equal units of optimization progress — gradient descent on a
  frustrated non-convex energy can stall in worse minima than annealing. The
  full prototype must validate final heat-score / Rg / bond-KS, not just speed.
  This bench only answers "is the per-step throughput gap real and large?".
* heat_dist is generated DENSE (every off-diagonal entry active), matching
  production subanchor expected-distance matrices (avg_dist is filled for all
  pairs). If a real case is sparser, both sides get cheaper but the shape
  argument is unchanged.
* f32, no x64 — the production dtype. Reverse-mode grad through the heat
  row-scan is wrapped in jax.checkpoint so peak memory stays O(N), not O(N^2).

USAGE
-----
    pip install "jax[cuda12]"        # on the CUDA box; or "jax[rocm6]"
    python playground/bench_jax_grad_vs_mc.py
    python playground/bench_jax_grad_vs_mc.py 1000 4000 12359   # custom N list
"""

from __future__ import annotations

import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Production dtype: f32, NO enable_x64 (bench in mc_jax docstring: f64 is 2x
# slower on consumer GPUs at these run lengths, no quality benefit).

# ---- schedule / weight constants (representative smooth-phase values) -------
STRETCH_K = np.float32(1.0)
SQUEEZE_K = np.float32(1.0)
ANG_K = np.float32(0.1)
DIST_W = np.float32(1.0)
ANG_W = np.float32(1.0)
HEAT_WEIGHT = np.float32(1.0)
T0 = np.float32(2.0)  # mid-anneal temperature (not the initial 20.0)
DT = np.float32(0.99995)
JUMP_SCALE = np.float32(50.0)
JUMP_COEF = np.float32(20.0)
STEP_SIZE = np.float32(0.1)
LR = np.float32(1e-3)  # Langevin step (untuned — speed test only)


# ===========================================================================
# Energy terms — copied verbatim from gnome3d/mc_jax.py
# ===========================================================================


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
    scale = jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, 1.0)
    cos_a = jnp.sum(v1 * v2) / jnp.maximum(n1 * n2, 1e-30)
    cos_a = jnp.clip(cos_a, -1.0, 1.0)
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return scale * ang * ang * ang * ang_k * ang_w


def _local_heat_at(pos, p_pos, p, heat_dist, heat_weight):
    """Heat score for bead p vs all others (verbatim from mc_jax). O(N)."""
    n = pos.shape[0]
    diff = pos - p_pos
    d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    exp_d = heat_dist[:, p]
    idx = jnp.arange(n)
    active = jnp.logical_and(idx != p, exp_d >= 1e-6)
    exp_d_safe = jnp.maximum(exp_d, 1e-6)
    rel = (d - exp_d_safe) / exp_d_safe
    contrib = rel * rel
    return heat_weight * jnp.sum(jnp.where(active, contrib, 0.0))


# ===========================================================================
# Full-state energy (for the gradient path).  Mirrors _init_smooth_single +
# _init_heat_single.  Heat row-scan is checkpointed -> grad memory O(N).
# ===========================================================================


def make_energy_fn(n, heat_dist):
    idx = jnp.arange(n)

    def energy(pos):
        # chain bonds + angles (cheap, O(N), vmapped)
        bonds = jax.vmap(
            lambda i: _smooth_len(pos[i], pos[i + 1], DTN[i], STRETCH_K, SQUEEZE_K, DIST_W)
        )(jnp.arange(n - 1))
        angles = jax.vmap(lambda i: _smooth_ang(pos[i], pos[i + 1], pos[i + 2], ANG_K, ANG_W))(
            jnp.arange(n - 2)
        )
        chain = jnp.sum(bonds) + jnp.sum(angles)

        # dense heat — O(N^2), row-scan with remat to bound reverse-mode memory
        @jax.checkpoint
        def row(carry, i):
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            exp_d = heat_dist[:, i]
            active = jnp.logical_and(idx != i, exp_d >= 1e-6)
            exp_d_safe = jnp.maximum(exp_d, 1e-6)
            rel = (d - exp_d_safe) / exp_d_safe
            return carry + jnp.sum(jnp.where(active, rel * rel, 0.0)), None

        heat_total, _ = jax.lax.scan(row, jnp.float32(0.0), idx)
        return chain + HEAT_WEIGHT * heat_total

    return energy


# DTN is shared module-level (set per-N in run()).  Kept global so the closures
# above capture the current value without re-threading it everywhere.
DTN = None


# ===========================================================================
# MC sweep kernel — faithful single-bead loop (verbatim accept rule)
# ===========================================================================


def make_mc_sweep(n, n_movable, heat_dist):
    movable_idx = jnp.arange(n_movable, dtype=jnp.int32)  # beads 0..n_movable-1 movable

    def local_chain(pos, p, p_pos):
        """Sum of chain terms touching bead p (<=2 bonds, <=3 angles), masked at ends."""

        # bonds (p-1,p) and (p,p+1)
        def bond(i, j):
            ok = jnp.logical_and(i >= 0, j <= n - 1)
            si = jnp.clip(i, 0, n - 1)
            sj = jnp.clip(j, 0, n - 1)
            pi = jnp.where(i == p, p_pos, pos[si])
            pj = jnp.where(j == p, p_pos, pos[sj])
            return jnp.where(
                ok,
                _smooth_len(pi, pj, DTN[jnp.clip(si, 0, n - 2)], STRETCH_K, SQUEEZE_K, DIST_W),
                0.0,
            )

        b = bond(p - 1, p) + bond(p, p + 1)

        # angles at triples (p-2,p-1,p),(p-1,p,p+1),(p,p+1,p+2)
        def ang(a, m, c):
            ok = jnp.logical_and(a >= 0, c <= n - 1)
            sa, sm, sc = jnp.clip(a, 0, n - 1), jnp.clip(m, 0, n - 1), jnp.clip(c, 0, n - 1)
            pa = jnp.where(a == p, p_pos, pos[sa])
            pm = jnp.where(m == p, p_pos, pos[sm])
            pc = jnp.where(c == p, p_pos, pos[sc])
            return jnp.where(ok, _smooth_ang(pa, pm, pc, ANG_K, ANG_W), 0.0)

        a = ang(p - 2, p - 1, p) + ang(p - 1, p, p + 1) + ang(p, p + 1, p + 2)
        return b + a

    def sweep(pos, score0, key):
        def body(_, carry):
            pos, score, T, key, n_ok = carry
            key, kp, kd, ku = jax.random.split(key, 4)
            # random movable bead
            p = movable_idx[jax.random.randint(kp, (), 0, n_movable)]
            old_p = pos[p]
            disp = jax.random.uniform(kd, (3,), minval=-STEP_SIZE, maxval=STEP_SIZE).astype(
                jnp.float32
            )
            new_p = old_p + disp

            ch_prev = local_chain(pos, p, old_p)
            ch_curr = local_chain(pos, p, new_p)
            h_prev = _local_heat_at(pos, old_p, p, heat_dist, HEAT_WEIGHT)
            h_curr = _local_heat_at(pos, new_p, p, heat_dist, HEAT_WEIGHT)
            score_new = score + (ch_curr - ch_prev) + 2.0 * (h_curr - h_prev)

            ok_unc = score_new < score
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -JUMP_COEF * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = JUMP_SCALE * jnp.exp(exponent)
            u = jax.random.uniform(ku, ())
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            pos = pos.at[p].set(jnp.where(ok, new_p, old_p))
            score = jnp.where(ok, score_new, score)
            return (pos, score, T * DT, key, n_ok + jnp.where(ok, 1, 0))

        init = (pos, score0, T0, key, jnp.int32(0))
        pos, score, _, _, n_ok = jax.lax.fori_loop(0, n_movable, body, init)
        return pos, score, n_ok

    return sweep


# ===========================================================================
# Driver
# ===========================================================================


def make_problem(n, seed=0):
    """Random chain + DENSE positive heat_dist (expected-distance style)."""
    rng = np.random.default_rng(seed)
    # self-avoiding-ish random walk start
    steps = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    pos = np.cumsum(steps, axis=0).astype(np.float32)
    # consecutive rest lengths
    dtn = np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32)
    dtn = np.append(dtn, dtn[-1])
    # dense expected-distance matrix: grows with genomic separation, all active
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    heat = (1.0 + np.sqrt(sep)).astype(np.float32)  # >0 everywhere off-diagonal
    np.fill_diagonal(heat, 0.0)
    return pos, dtn, heat


def timeit(fn, reps=5):
    fn()  # warmup / compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def run(n_list):
    global DTN
    print(f"JAX {jax.__version__} | backend={jax.default_backend()} | devices={jax.devices()}")
    print("dtype=float32  reps=5(median)  heat=DENSE\n")
    hdr = f"{'N':>7} {'movable':>8} {'MC sweep ms':>13} {'grad step ms':>14} {'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))

    for n in n_list:
        n_movable = max(1, int(n * 0.92))  # ~production movable fraction
        pos_np, dtn_np, heat_np = make_problem(n)
        DTN = jnp.asarray(dtn_np)
        heat = jnp.asarray(heat_np)
        pos = jnp.asarray(pos_np)
        key = jax.random.PRNGKey(0)

        # ---- MC sweep ----
        sweep = jax.jit(make_mc_sweep(n, n_movable, heat))
        # initial score for the loop (cheap, off the clock)
        energy = jax.jit(make_energy_fn(n, heat))
        score0 = jax.block_until_ready(energy(pos))
        mc_ms = timeit(lambda: sweep(pos, score0, key)) * 1e3

        # ---- gradient step (value_and_grad + Langevin update) ----
        vg = jax.jit(jax.value_and_grad(make_energy_fn(n, heat)))

        def grad_step():
            val, g = vg(pos)
            noise = jax.random.normal(key, pos.shape, dtype=jnp.float32)
            new = pos - LR * g + jnp.sqrt(2.0 * LR * T0) * noise
            return val, new

        grad_ms = timeit(grad_step) * 1e3

        speedup = mc_ms / grad_ms if grad_ms > 0 else float("inf")
        print(f"{n:>7} {n_movable:>8} {mc_ms:>13.2f} {grad_ms:>14.2f} {speedup:>8.1f}x")


def run_kscan(n, k_list):
    """Batch-width experiment. Each MC step is latency-bound with a near-idle
    GPU (the bench above showed ~20us/step independent of N). So vmapping the
    sweep over K independent chains should cost ~the same wall-clock as K=1 —
    i.e. K structures for the price of ~1. If per-structure time drops roughly
    linearly with K while total wall stays ~flat, batching reps/regions is
    nearly-free parallelism and is the real throughput lever."""
    global DTN
    print(f"\nK-scan at N={n}  (vmap over K independent chains)")
    hdr = f"{'K':>5} {'total ms':>10} {'per-struct ms':>14} {'vs K=1':>8}"
    print(hdr)
    print("-" * len(hdr))

    n_movable = max(1, int(n * 0.92))
    pos_np, dtn_np, heat_np = make_problem(n)
    DTN = jnp.asarray(dtn_np)
    heat = jnp.asarray(heat_np)
    energy = jax.jit(make_energy_fn(n, heat))
    sweep1 = make_mc_sweep(n, n_movable, heat)  # un-jitted; vmap then jit

    base_per = None
    for k in k_list:
        pos_k = jnp.broadcast_to(jnp.asarray(pos_np), (k, n, 3))
        keys = jax.random.split(jax.random.PRNGKey(0), k)
        score0 = jax.block_until_ready(energy(jnp.asarray(pos_np)))
        scores = jnp.broadcast_to(score0, (k,))
        # vmap chains: pos and key per-chain, heat/score shared semantics.
        batched = jax.jit(jax.vmap(sweep1, in_axes=(0, 0, 0)))
        total_ms = timeit(lambda: batched(pos_k, scores, keys)) * 1e3
        per = total_ms / k
        if base_per is None:
            base_per = per
        print(f"{k:>5} {total_ms:>10.2f} {per:>14.3f} {base_per / per:>7.1f}x")


def make_orientation(fixed, half_window=4):
    """Synthesize realistic CTCF orientation inputs for mc_smooth_jax.
    Anchors = the fixed beads; each anchor neighbors a window of nearby anchors
    (anchor-space indices, as the kernel indexes anchor_orn[j]).  max_nbrs ~ 2*W."""
    n_anchors = int(fixed.sum())
    chars = ["L" if i % 2 == 0 else "R" for i in range(n_anchors)]
    neighbors = {}
    weights = {}
    for k in range(n_anchors):
        nb = [
            j for j in range(max(0, k - half_window), min(n_anchors, k + half_window + 1)) if j != k
        ]
        neighbors[k] = nb
        weights[k] = [1.0] * len(nb)
    return chars, neighbors, weights


def run_real(n, b_list, n_steps_per_batch=2000, with_orn=False):
    """Production-faithful batch test: time the REAL gnome3d.mc_jax.mc_smooth_jax
    (chain + excluded volume, the IB heat-estimate energy) at width B.

    The IB phase (solver.py:1032) runs n_reps*n_steps independent anneals as a
    sequential python loop, each a chains=1 call. mc_smooth_jax already vmaps
    internally over settings.mc_smooth_chains, so:
        baseline  = chains=1, called B times   (today's IB loop)
        batched   = chains=B, called once       (the proposed batched entry)
    Same kernel, same energy, no throwaway code. Convergence is judged on the
    BEST of the K chains (mc_jax.py:718) so batching never waits for the slowest.

    NOTE: each call runs to convergence, so absolute time depends on the random
    structure; the per-structure RATIO is the signal. Reduce n_steps_per_batch
    if a single B=max call is too slow on your box.
    """
    import sys as _sys

    _sys.path.insert(0, ".")
    from gnome3d.mc_jax import mc_smooth_jax
    from gnome3d.settings import Settings

    s = Settings()
    s.mc_backend = "jax"
    s.mc_backend_apply_to_smooth = True
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.mc_stop_steps_smooth = int(n_steps_per_batch)
    if not hasattr(s, "motif_weight"):
        s.motif_weight = 1.0

    pos_np, dtn_np, _ = make_problem(n)
    fixed = np.zeros(n, dtype=bool)
    fixed[::12] = True  # ~8% fixed anchors, matching production movable fraction
    step_size = 0.5

    orn_kw = {}
    if with_orn:
        chars, nbrs, wts = make_orientation(fixed)
        orn_kw = {
            "char_orientations": chars,
            "anchor_neighbors": nbrs,
            "anchor_neighbor_weights": wts,
        }

    energy_label = "chain+EV+orientation" if with_orn else "chain+EV"
    print(f"\nREAL mc_smooth_jax  N={n}  {energy_label}  steps/batch={n_steps_per_batch}")
    print("baseline = chains=1 called B times ; batched = chains=B called once")
    hdr = f"{'B':>4} {'seq total ms':>13} {'batch total ms':>15} {'seq/struct':>11} {'batch/struct':>13} {'speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    for b in b_list:
        # baseline: B sequential chains=1 calls (fresh copy — call mutates in place)
        s.mc_smooth_chains = 1

        def seq():
            out = 0.0
            for _ in range(b):
                p = pos_np.copy()
                out += mc_smooth_jax(p, dtn_np, fixed, step_size, s, **orn_kw)
            return out

        seq_ms = timeit(seq, reps=3) * 1e3

        # batched: one chains=B call
        s.mc_smooth_chains = b

        def batched_call():
            p = pos_np.copy()
            return mc_smooth_jax(p, dtn_np, fixed, step_size, s, **orn_kw)

        batch_ms = timeit(batched_call, reps=3) * 1e3

        sp = (seq_ms / b) / (batch_ms / b) if batch_ms > 0 else float("inf")
        print(
            f"{b:>4} {seq_ms:>13.1f} {batch_ms:>15.1f} "
            f"{seq_ms / b:>11.2f} {batch_ms / b:>13.2f} {sp:>7.1f}x"
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--real" in sys.argv:
        n = int(args[0]) if args else 12345
        steps = int(args[1]) if len(args) > 1 else 2000
        run_real(n, [1, 4, 8, 16], n_steps_per_batch=steps, with_orn="--orn" in sys.argv)
    elif "--kscan" in sys.argv:
        n = int(args[0]) if args else 8000
        run_kscan(n, [1, 4, 16, 64, 256])
    else:
        ns = [int(x) for x in args] if args else [1000, 2000, 4000, 8000, 12359]
        run(ns)
