"""JAX color-gather checkerboard arcs-MC kernel (opt-in: mc_executor_jax_arcs_kernel='checker').

Arcs energy = a few springs + ALL-PAIRS 1/d repulsion (+EV +confine) — a force-directed
N-body layout.  Sequential single-bead MC is latency-bound on GPU (it leaves the GPU ~idle,
~12 us/step), so plain region-batching can't beat multi-core CPU.  This kernel breaks the
sequential dependency with an APPROXIMATE SPATIAL CHECKERBOARD:

  - each sweep recolors anchors by a spatial grid (8-color 3D parity, same-color anchors
    >= cell apart), then updates each color's anchors SIMULTANEOUSLY;
  - within a color, far-pair repulsion is treated as stale (small error — the 1/d delta is
    near-dominated by close neighbours, and same-colour anchors are >= cell apart);
  - per color it GATHERS only that colour's <= maxc anchors (jnp.nonzero static size) and
    computes the (maxc, B) delta instead of the full (B, B) — the key speedup.

The cell is the robust median nearest-neighbour distance x4, recomputed once per outer-iter
(O(B^2), ~2% of a batch) so it tracks the structure as it expands from the collapsed seed.

Validated equal-energy to sequential single-bead MC on real chr1 IBs (N=462..1146,
E_chk/E_seq=1.00); ~5x faster than the dense kernel on the large bottleneck IBs and
decisively faster than multi-core CPU.  See playground/arcs_checker_*.py for the bench/derivation.
"""

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.mc.jax.arcs import _prep_arcs_problem_np, _resolve_arcs_max_k
from gnome3d.mc.jax.shrink import run_shrinking
from gnome3d.mc.jax.util import jax_bucket_for, jax_is_available

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

_kernel_cache: dict[Any, Any] = {}
_init_lock = threading.Lock()
_MAX_ITERS: int = 10000


def _build_checker_kernel(n_sweeps: int, excl_skip: int, maxc: int) -> Any:
    """Build (or look up cached) compiled checkerboard kernel.  Static over
    (n_sweeps, excl_skip, maxc); JAX specialises on the (B,B) shapes per call."""
    cache_key = ("checker", n_sweeps, excl_skip, maxc)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    import jax
    import jax.numpy as jnp

    def _arc_E(d: Any, e: Any, stretch: Any, squeeze: Any) -> Any:
        rep = 1.0 / jnp.maximum(d, 1e-10)
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0.0, stretch, squeeze)
        return jnp.where(e < 0.0, rep, jnp.where(e >= 1e-6, rel * rel * k, 0.0))

    def _energy_total(
        pos: Any, exp: Any, stretch: Any, squeeze: Any,
        r0: Any, excl_w: Any, cx: Any, cy: Any, cz: Any, R: Any, conf_w: Any, n_active: Any,
    ) -> Any:
        """Total arcs energy (arcs single-count + excl double-count + confine), pad-masked."""
        b = pos.shape[0]
        idx = jnp.arange(b)
        active = idx < n_active
        eye = idx[:, None] == idx[None, :]
        upper = idx[:, None] < idx[None, :]
        d = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        tot = jnp.sum(jnp.where(upper, _arc_E(d, exp, stretch, squeeze), 0.0))  # pad exp=0 -> 0
        far = jnp.abs(idx[:, None] - idx[None, :]) > excl_skip
        far = far & jnp.logical_not(eye) & active[:, None] & active[None, :]
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        tot = tot + jnp.sum(jnp.where(far, excl_w * rel * rel, 0.0))  # double-counted
        ctr = jnp.array([cx, cy, cz])
        r = jnp.sqrt(jnp.sum((pos - ctr) ** 2, axis=-1))
        conf = jnp.where(jnp.logical_and(active, r > R), conf_w * ((r - R) / R) ** 2, 0.0)
        return tot + jnp.sum(conf)

    def chain_checker(
        pos0: Any, score0: Any, T0: Any, exp: Any, step: Any, dt: Any, js: Any, jc: Any,
        stretch: Any, squeeze: Any, r0: Any, excl_w: Any,
        cx: Any, cy: Any, cz: Any, R: Any, conf_w: Any, n_active: Any, key: Any,
    ) -> Any:
        """One outer-iter for ONE chain: pick a median-nn cell, run n_sweeps colored sweeps,
        return (pos, exact_total_E, T, n_ok, max_color_cnt)."""
        b = pos0.shape[0]
        idx_all = jnp.arange(b)
        active = idx_all < n_active
        ctr = jnp.array([cx, cy, cz])
        expT = exp.T
        eargs = (r0, excl_w, cx, cy, cz, R, conf_w, n_active)

        # --- cell = 2 * mean nearest-neighbour distance over S probe anchors ---
        # Probe a SUBSET (not all (B,B) pairs) and use the MEAN (not a sort-median): the full
        # (B,B) distance + jnp.sort made XLA compile pathologically (20+min) and FAIL ptxas at
        # large batch widths.  The cell is only a heuristic scale, so the estimate is fine.
        # mod-3 (27-colour) needs >=3 cells/dim; the 2x keeps that for small/collapsed IBs.
        big = jnp.float32(1e30)
        S = 64
        stride = jnp.maximum(n_active // S, 1)
        probe = jnp.minimum(jnp.arange(S) * stride, jnp.maximum(n_active - 1, 0))  # (S,) probe idx
        pp = pos0[probe]                                                           # (S, 3)
        dpr = jnp.sqrt(jnp.sum((pp[:, None, :] - pos0[None, :, :]) ** 2, axis=-1))  # (S, B)
        mask = (probe[:, None] == idx_all[None, :]) | jnp.logical_not(active[None, :])
        nn_pr = jnp.min(jnp.where(mask, big, dpr), axis=1)                         # (S,) per-probe nn
        valid = jnp.arange(S) < n_active
        mean_nn = jnp.sum(jnp.where(valid, nn_pr, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
        cell = jnp.maximum(2.0 * mean_nn, 1e-10)

        def sweep_body(_sw: Any, carry: Any) -> Any:
            pos, score, T, n_ok, mx = carry
            cellidx = jnp.floor(pos / cell).astype(jnp.int32)
            # 27-colour mod-3 spatial parity (vs 8-colour mod-2): cuts the same-colour
            # (stale-repulsion) partner fraction ~1/8 -> ~1/27, removing the compaction
            # bias the all-pairs 1/d repulsion staleness causes on dense structures.
            m3 = jnp.mod(cellidx, 3)
            color = m3[:, 0] * 9 + m3[:, 1] * 3 + m3[:, 2]
            k_m, k_u = jax.random.split(jax.random.fold_in(key, _sw + 1))
            move = jax.random.uniform(k_m, (b, 3), minval=-step, maxval=step, dtype=pos.dtype)
            u = jax.random.uniform(k_u, (b,), dtype=pos.dtype)

            def color_body(c: Any, c2: Any) -> Any:
                pos, score, T, n_ok, mx = c2
                mask_c = jnp.logical_and(color == c, active)  # pad anchors never move
                count_c = jnp.sum(mask_c)
                idx_c = jnp.nonzero(mask_c, size=maxc, fill_value=0)[0]
                valid = jnp.arange(maxc) < count_c
                pos_c = pos[idx_c]
                new_c = pos_c + move[idx_c]
                exp_c = expT[idx_c]                            # (maxc, B); pad cols exp=0
                self_m = idx_c[:, None] == idx_all[None, :]
                d_old = jnp.sqrt(jnp.sum((pos_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
                d_mov = jnp.sqrt(jnp.sum((new_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
                a_old = jnp.where(self_m, 0.0, _arc_E(d_old, exp_c, stretch, squeeze))
                a_mov = jnp.where(self_m, 0.0, _arc_E(d_mov, exp_c, stretch, squeeze))
                delta = jnp.sum(a_mov - a_old, axis=1)
                far = jnp.abs(idx_c[:, None] - idx_all[None, :]) > excl_skip
                far = far & jnp.logical_not(self_m) & active[None, :]  # mask pad cols out of EV
                rel_o = jnp.maximum(0.0, (r0 - d_old) / r0)
                rel_m = jnp.maximum(0.0, (r0 - d_mov) / r0)
                delta = delta + 2.0 * jnp.sum(jnp.where(far, excl_w * (rel_m * rel_m - rel_o * rel_o), 0.0), axis=1)
                ro = jnp.sqrt(jnp.sum((pos_c - ctr) ** 2, axis=-1))
                rn = jnp.sqrt(jnp.sum((new_c - ctr) ** 2, axis=-1))
                co = jnp.where(ro > R, conf_w * ((ro - R) / R) ** 2, 0.0)
                cn = jnp.where(rn > R, conf_w * ((rn - R) / R) ** 2, 0.0)
                delta = delta + (cn - co)
                can_jump = jnp.logical_and(T > 0.0, score > 0.0)
                expo = jnp.clip(-jc * ((score + delta) / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30), -80.0, 80.0)
                ok = jnp.logical_or(delta <= 0.0, jnp.logical_and(can_jump, u[idx_c] < js * jnp.exp(expo)))
                ok = jnp.logical_and(ok, valid)
                pos = pos.at[idx_c].add(jnp.where(ok[:, None], move[idx_c], 0.0))
                score = score + jnp.sum(jnp.where(ok, delta, 0.0))
                n_ok = n_ok + jnp.sum(ok)
                T = T * dt ** count_c
                return pos, score, T, n_ok, jnp.maximum(mx, count_c.astype(jnp.int32))

            return jax.lax.fori_loop(0, 27, color_body, (pos, score, T, n_ok, mx))

        init = (pos0, score0, T0, jnp.int32(0), jnp.int32(0))
        pos, _score, T, n_ok, mx = jax.lax.fori_loop(0, n_sweeps, sweep_body, init)
        return pos, _energy_total(pos, exp, stretch, squeeze, *eargs), T, n_ok, mx

    in_axes = (
        0, 0, 0,        # pos, score, T (per-chain)
        0,              # exp (per-IB)
        0,              # step (per-IB)
        None, None, None,  # dt, js, jc (shared)
        None, None,     # stretch, squeeze (shared)
        0, None,        # r0 (per-IB), excl_w (shared)
        0, 0, 0, 0,     # cx, cy, cz, R (per-IB)
        None,           # conf_w (shared)
        0, 0,           # n_active (per-IB), key (per-chain)
    )
    batched = jax.vmap(chain_checker, in_axes=in_axes, out_axes=(0, 0, 0, 0, 0))

    @jax.jit
    def kernel_chunk(
        carry: Any, problem: Any, scalars: Any, base_key: Any, max_iters: Any, iter_base: Any,
    ) -> Any:
        """Run the batched checker on this (possibly-shrunk + padded) set of chains until
        all converge OR ``max_iters`` outer-iters elapse; returns the carry + iters run.
        ``iter_base`` continues the RNG stream + conv-iter numbering across chunks."""
        pos, score, T, ms0, conv0, ci0 = carry
        exp_k, step_k, r0_k, cx_k, cy_k, cz_k, R_k, n_active_k, succ_k, chain_id = problem
        (dt, js, jc, stretch, squeeze, excl_w, conf_w,
         stop_improvement, score_eps, stop_ratio) = scalars

        def cond_fn(state: Any) -> Any:
            conv, li = state[4], state[6]
            return jnp.logical_and(jnp.logical_not(jnp.all(conv)), li < max_iters)

        def body_fn(state: Any) -> Any:
            pos, score, T, ms_score, conv_prev, conv_iter, li = state
            # Per-chain RNG keyed on the chain's GLOBAL id (not its batch lane), so a chain's
            # random stream is invariant to which others share its (shrinking) batch.
            giter = iter_base + li + 1
            keys = jax.vmap(
                lambda cid: jax.random.fold_in(jax.random.fold_in(base_key, cid), giter)
            )(chain_id)
            npos, nscore, nT, nok, _mcnt = batched(
                pos, score, T, exp_k, step_k, dt, js, jc, stretch, squeeze, r0_k, excl_w,
                cx_k, cy_k, cz_k, R_k, conf_w, n_active_k, keys,
            )
            # freeze converged chains (non-strict accept could drift them worse)
            frozen = conv_prev
            pos = jnp.where(frozen[:, None, None], pos, npos)
            score = jnp.where(frozen, score, nscore)
            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, nok < succ_k)
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, score < score_eps), ratio > stop_ratio),
                conv_prev,
            )
            newly = jnp.logical_and(converged, jnp.logical_not(conv_prev))
            conv_iter = jnp.where(newly, iter_base + li + 1, conv_iter)
            return pos, score, nT, score, converged, conv_iter, li + 1

        init = (pos, score, T, ms0, conv0, ci0, jnp.int32(0))
        pos, score, T, ms, conv, ci, li = jax.lax.while_loop(cond_fn, body_fn, init)
        return (pos, score, T, ms, conv, ci), li

    init_energy = jax.jit(jax.vmap(
        _energy_total, in_axes=(0, 0, None, None, 0, None, 0, 0, 0, 0, None, 0)))

    bundle = (kernel_chunk, init_energy)
    _kernel_cache[cache_key] = bundle
    return bundle


def mc_arcs_checker_jax_batch(
    problems: list[dict[str, Any]], settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Checkerboard analogue of `mc_arcs_jax_batch`: anneal K IBs' anchors in one vmapped
    colored-sweep kernel.  Same signature / return ((score, pos(n,3)) per problem)."""
    if not problems:
        return []
    if not jax_is_available():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    big_b = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    max_k, _basis = _resolve_arcs_max_k(big_b, settings)
    if len(problems) <= max_k:
        return _checker_chunk(problems, settings)
    out: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i in range(0, len(problems), max_k):
        out.extend(_checker_chunk(problems[i : i + max_k], settings))
    return out


def _checker_chunk(
    problems: list[dict[str, Any]], settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    import jax
    import jax.numpy as jnp

    K = len(problems)
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    B = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    preps = [_prep_arcs_problem_np(p["pos"], p["exp_dist"], settings, B) for p in problems]

    def stack(key: str, dtype: Any) -> Any:
        return jnp.asarray(np.array([pr[key] for pr in preps], dtype=dtype))

    pos_k = jnp.asarray(np.stack([pr["pos"] for pr in preps], axis=0))          # (K,B,3)
    exp_k = jnp.asarray(np.stack([pr["exp_mat"] for pr in preps], axis=0))      # (K,B,B)
    n_active_k = stack("n_active", np.int32)
    r0_k = stack("excl_r0", np.float32)
    cx_k, cy_k, cz_k = stack("conf_cx", np.float32), stack("conf_cy", np.float32), stack("conf_cz", np.float32)
    R_k = stack("conf_R", np.float32)
    step_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    excl_skip = int(settings.exclusion_skip_neighbors)
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    excl_w = jnp.float32(settings.exclusion_weight if use_excl else 0.0)
    conf_w = jnp.float32(settings.confinement_weight if use_conf else 0.0)
    stretch = jnp.float32(settings.spring_stretch_arcs)
    squeeze = jnp.float32(settings.spring_squeeze_arcs)
    maxc = max(8, B // 9)  # 27-color parity => ~B/27 per color; B/9 = 3x safety (no overflow)
    # >=4 sweeps/outer-iter so each convergence check spans a meaningful batch (else a
    # single sweep's tiny score change trips ratio>stop_ratio prematurely).
    stop_steps = max(int(settings.mc_stop_steps), 1)
    n_sweeps = max(4, stop_steps // B)

    # Scale the plateau accept-threshold per IB to the checker's proposals/outer-iter
    # (n_sweeps * n_active movable beads) so `n_ok < successes` fires at the SAME accept
    # RATE as the numba sequential (mc_stop_successes per mc_stop_steps proposals) -
    # without this the parallel checker never plateaus and over-optimizes the structure.
    n_active_arr = np.array([pr["n_active"] for pr in preps], np.float64)
    succ_k = jnp.asarray(
        np.maximum(1.0, settings.mc_stop_successes * n_sweeps * n_active_arr / stop_steps).astype(np.float32)
    )

    kernel_chunk, init_energy = _build_checker_kernel(n_sweeps, excl_skip, maxc)

    score_k = init_energy(pos_k, exp_k, stretch, squeeze, r0_k, excl_w, cx_k, cy_k, cz_k, R_k, conf_w, n_active_k)

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    carry = (
        pos_k, score_k, jnp.full((K,), jnp.float32(settings.max_temp)),
        jnp.full((K,), jnp.float32(1e30)), jnp.zeros((K,), jnp.bool_), jnp.zeros((K,), jnp.int32),
    )
    problem = (exp_k, step_k, r0_k, cx_k, cy_k, cz_k, R_k, n_active_k, succ_k,
               jnp.arange(K, dtype=jnp.int32))
    scalars = (
        jnp.float32(settings.dt_temp), jnp.float32(settings.jump_scale), jnp.float32(settings.jump_coef),
        stretch, squeeze, excl_w, conf_w,
        jnp.float32(settings.mc_stop_improvement), jnp.float32(1e-5), jnp.float32(0.9999),
    )

    log.status(LOG, "    arcs[checker]: annealing %d IBs (%d beads each; 27-colour spatial gather, "
               "<=%d beads/colour, %d sweeps/round)...", K, B, maxc, n_sweeps)
    t0 = time.perf_counter()
    out_pos, out_score, out_ci, total = run_shrinking(
        kernel_chunk, carry, problem, scalars, base_key, max_total=_MAX_ITERS)
    ci = out_ci[out_ci > 0]
    log.status(
        LOG,
        "    arcs[checker]: %d IBs done in %.1fs - %d rounds; IBs converged at round: "
        "median %d, slowest %d",
        K, time.perf_counter() - t0, total,
        int(np.median(ci)) if ci.size else 0, int(ci.max()) if ci.size else 0,
    )

    return [(float(out_score[i]), out_pos[i][: pr["n"]].astype(np.float32)) for i, pr in enumerate(preps)]
