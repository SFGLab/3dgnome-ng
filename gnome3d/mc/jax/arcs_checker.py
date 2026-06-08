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

        # --- robust cell = 4 * median nearest-neighbour distance over active anchors ---
        d0 = jnp.sqrt(jnp.sum((pos0[:, None, :] - pos0[None, :, :]) ** 2, axis=-1))
        big = jnp.float32(1e30)
        d0 = jnp.where((idx_all[:, None] == idx_all[None, :]) | jnp.logical_not(active[None, :]), big, d0)
        nn = jnp.where(active, jnp.min(d0, axis=1), big)
        med_nn = jnp.sort(nn)[jnp.maximum(n_active // 2, 0)]
        cell = jnp.maximum(4.0 * med_nn, 1e-10)

        def sweep_body(_sw: Any, carry: Any) -> Any:
            pos, score, T, n_ok, mx = carry
            cellidx = jnp.floor(pos / cell).astype(jnp.int32)
            color = (cellidx[:, 0] & 1) * 4 + (cellidx[:, 1] & 1) * 2 + (cellidx[:, 2] & 1)
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

            return jax.lax.fori_loop(0, 8, color_body, (pos, score, T, n_ok, mx))

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
    def kernel_checker_mp(
        pos_k: Any, score_k: Any, T_init: Any, exp_k: Any, step_k: Any,
        dt: Any, js: Any, jc: Any, stretch: Any, squeeze: Any, r0_k: Any, excl_w: Any,
        cx_k: Any, cy_k: Any, cz_k: Any, R_k: Any, conf_w: Any, base_key: Any,
        stop_improvement: Any, stop_successes: Any, score_eps: Any, stop_ratio: Any, n_active_k: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            iter_i, converged = state[4], state[6]
            return jnp.logical_and(jnp.logical_not(jnp.all(converged)), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, score, T, ms_score, iter_i, _n_ok, conv_prev, conv_iter = state
            keys = jax.random.split(jax.random.fold_in(base_key, iter_i + 1), K)
            npos, nscore, nT, nok, _mcnt = batched(
                pos, score, T, exp_k, step_k, dt, js, jc, stretch, squeeze, r0_k, excl_w,
                cx_k, cy_k, cz_k, R_k, conf_w, n_active_k, keys,
            )
            # freeze converged chains (non-strict accept could drift them worse)
            frozen = conv_prev
            pos = jnp.where(frozen[:, None, None], pos, npos)
            score = jnp.where(frozen, score, nscore)
            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, nok < stop_successes)
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, score < score_eps), ratio > stop_ratio),
                conv_prev,
            )
            newly = jnp.logical_and(converged, jnp.logical_not(conv_prev))
            conv_iter = jnp.where(newly, iter_i + 1, conv_iter)
            return pos, score, nT, score, iter_i + 1, nok, converged, conv_iter

        init = (
            pos_k, score_k, T_init,
            jnp.full((K,), 1e30, dtype=jnp.float32),
            jnp.int32(0),
            jnp.zeros((K,), dtype=jnp.int32),
            jnp.zeros((K,), dtype=jnp.bool_),
            jnp.zeros((K,), dtype=jnp.int32),
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init)
        pos_f, score_f, _T, _ms, iter_f, _nok, conv_f, conviter_f = final
        return pos_f, score_f, iter_f, conv_f, conviter_f

    init_energy = jax.jit(jax.vmap(
        _energy_total, in_axes=(0, 0, None, None, 0, None, 0, 0, 0, 0, None, 0)))

    bundle = (kernel_checker_mp, init_energy)
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
    maxc = max(8, B // 4)  # 8-color parity => ~B/8 per color; B/4 = 2x safety (no overflow)
    n_sweeps = max(1, int(settings.mc_stop_steps) // B)  # ~mc_stop_steps bead-moves / outer-iter

    kernel_checker_mp, init_energy = _build_checker_kernel(n_sweeps, excl_skip, maxc)

    score_k = init_energy(pos_k, exp_k, stretch, squeeze, r0_k, excl_w, cx_k, cy_k, cz_k, R_k, conf_w, n_active_k)

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)
    T_init = jnp.full((K,), jnp.float32(settings.max_temp))

    log.status(LOG, "    arcs[checker] kernel: K=%d B=%d maxc=%d n_sweeps=%d, running...", K, B, maxc, n_sweeps)
    t0 = time.perf_counter()
    pos_f, score_f, iter_f, conv_f, conviter_f = kernel_checker_mp(
        pos_k, score_k, T_init, exp_k, step_k,
        jnp.float32(settings.dt_temp), jnp.float32(settings.jump_scale), jnp.float32(settings.jump_coef),
        stretch, squeeze, r0_k, excl_w, cx_k, cy_k, cz_k, R_k, conf_w, base_key,
        jnp.float32(settings.mc_stop_improvement), jnp.int32(settings.mc_stop_successes),
        jnp.float32(1e-5), jnp.float32(0.9999), n_active_k,
    )
    score = np.asarray(score_f)  # device sync
    pos_np = np.asarray(pos_f)
    it = int(iter_f)
    ci = np.asarray(conviter_f)
    done = ci > 0
    p50 = int(np.median(ci[done])) if done.any() else 0
    log.status(
        LOG,
        "    arcs[checker] kernel: K=%d B=%d, %d iters (~%d sweeps), %d/%d converged, "
        "conv p50/max=%d/%d, %.1fs",
        K, B, it, it * n_sweeps, int(np.asarray(conv_f).sum()), K,
        p50, int(ci.max()) if ci.size else 0, time.perf_counter() - t0,
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        results.append((float(score[i]), pos_np[i, : pr["n"]].astype(np.float32)))
    return results
