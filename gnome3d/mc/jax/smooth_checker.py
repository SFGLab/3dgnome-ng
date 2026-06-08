"""JAX color-gather checkerboard smooth-MC kernel (opt-in: mc_executor_jax_smooth_kernel='checker').

Smooth energy = chain bonds + angles (LOCAL, span ±2) + heat (dense all-pairs springs to
heat_dist) + EV (all-pairs spatial) + confine.  Like arcs, the all-pairs heat+EV make a large
smooth IB latency-bound on GPU (B reaches ~15k on chr1).  This kernel breaks the sequential
dependency with a 24-colour checkerboard:

  - `color = (p mod 3) * 8 + spatial-parity(pos[p])` (24 colours).  chain-mod-3 makes the
    bond/angle delta EXACT (a colour-c bead's structural neighbours p±2 are always colours
    c+1/c+2 — never moving); spatial-8 handles the all-pairs heat+EV (far-stale, near-dominated).
  - per colour it gathers that colour's <= maxc movable beads and computes the (maxc, B) heat/EV
    delta + a small (maxc, 5) structural window — instead of the full (B, B).

Fixed anchor beads never move (masked like pad beads).  ORIENTATION (CTCF) is OMITTED: during
smooth it only updates on anchor moves and anchors are fixed, so it's a constant — the produced
structures are correct, but the returned score excludes that constant offset.  Validated
equal-energy to sequential single-bead smooth on real chr1 IBs.  See playground/smooth_checker_*.py.
"""

import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.mc.jax.memory import max_k_for_bytes
from gnome3d.mc.jax.shrink import run_shrinking
from gnome3d.mc.jax.util import (
    jax_bucket_for,
    jax_device_budget_bytes,
    jax_is_available,
    log_kernel_done,
    log_kernel_start,
)

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

_kernel_cache: dict[Any, Any] = {}
_init_lock = threading.Lock()
_MAX_ITERS: int = 10000


def _build_smooth_checker_kernel(n_sweeps: int, excl_skip: int, maxc: int) -> Any:
    cache_key = ("smooth_checker", n_sweeps, excl_skip, maxc)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    import jax
    import jax.numpy as jnp

    def _len_E(d, e, sk, qk, dw):
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        return rel * rel * jnp.where(rel >= 0.0, sk, qk) * dw

    def _ang_E(v1, v2, ak, aw):
        n1 = jnp.sqrt(jnp.sum(v1 * v1, axis=-1))
        n2 = jnp.sqrt(jnp.sum(v2 * v2, axis=-1))
        cos_a = jnp.clip(jnp.sum(v1 * v2, axis=-1) / jnp.maximum(n1 * n2, 1e-30), -1.0, 1.0)
        ang = 1.0 - (cos_a + 1.0) * 0.5
        return jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, ang * ang * ang * ak * aw)

    def _struct_delta(pos, new, idx, dtn, sk, qk, ak, dw, aw, n_active):
        b = pos.shape[0]
        g = lambda off: pos[jnp.clip(idx + off, 0, b - 1)]
        pm2, pm1, p0, pp1, pp2 = g(-2), g(-1), g(0), g(1), g(2)
        newp = new[idx]
        dL = dtn[jnp.clip(idx - 1, 0, b - 1)]
        dR = dtn[jnp.clip(idx, 0, b - 1)]
        nrm = lambda a: jnp.sqrt(jnp.sum(a * a, axis=-1))
        bL = _len_E(nrm(pm1 - newp), dL, sk, qk, dw) - _len_E(nrm(pm1 - p0), dL, sk, qk, dw)
        bR = _len_E(nrm(newp - pp1), dR, sk, qk, dw) - _len_E(nrm(p0 - pp1), dR, sk, qk, dw)
        bond = jnp.where(jnp.logical_and(idx >= 1, idx < n_active), bL, 0.0) + jnp.where(
            idx < n_active - 1, bR, 0.0
        )
        a2 = _ang_E(pm2 - pm1, pm1 - newp, ak, aw) - _ang_E(pm2 - pm1, pm1 - p0, ak, aw)
        a1 = _ang_E(pm1 - newp, newp - pp1, ak, aw) - _ang_E(pm1 - p0, p0 - pp1, ak, aw)
        a0 = _ang_E(newp - pp1, pp1 - pp2, ak, aw) - _ang_E(p0 - pp1, pp1 - pp2, ak, aw)
        ang = (
            jnp.where(jnp.logical_and(idx >= 2, idx < n_active), a2, 0.0)
            + jnp.where(jnp.logical_and(idx >= 1, idx < n_active - 1), a1, 0.0)
            + jnp.where(idx < n_active - 2, a0, 0.0)
        )
        return bond + ang

    def _heat_ev_conf(pos, new, idx, heat, hw, r0, ew, cx, cy, cz, R, cw, n_active):
        b = pos.shape[0]
        idx_all = jnp.arange(b)
        active = idx_all < n_active
        self_m = idx[:, None] == idx_all[None, :]
        pos_c, new_c = pos[idx], new[idx]
        d_old = jnp.sqrt(jnp.sum((pos_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        d_mov = jnp.sqrt(jnp.sum((new_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        h_c = heat.T[idx]
        hmask = jnp.logical_and(
            jnp.logical_and(h_c > 1e-6, jnp.logical_not(self_m)), active[None, :]
        )
        h_safe = jnp.maximum(h_c, 1e-6)
        ro, rm = (d_old - h_safe) / h_safe, (d_mov - h_safe) / h_safe
        heat_d = 2.0 * hw * jnp.sum(jnp.where(hmask, rm * rm - ro * ro, 0.0), axis=1)
        far = jnp.logical_and(
            jnp.abs(idx[:, None] - idx_all[None, :]) > excl_skip, jnp.logical_not(self_m)
        )
        far = jnp.logical_and(far, active[None, :])
        eo, em = jnp.maximum(0.0, (r0 - d_old) / r0), jnp.maximum(0.0, (r0 - d_mov) / r0)
        ev_d = 2.0 * jnp.sum(jnp.where(far, ew * (em * em - eo * eo), 0.0), axis=1)
        ctr = jnp.array([cx, cy, cz])
        rro = jnp.sqrt(jnp.sum((pos_c - ctr) ** 2, axis=-1))
        rrn = jnp.sqrt(jnp.sum((new_c - ctr) ** 2, axis=-1))
        co = jnp.where(rro > R, cw * ((rro - R) / R) ** 2, 0.0)
        cn = jnp.where(rrn > R, cw * ((rrn - R) / R) ** 2, 0.0)
        return heat_d + ev_d + (cn - co)

    def _deltas(
        pos, move, idx, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, cx, cy, cz, R, cw, n_active
    ):
        new = pos + move
        return _struct_delta(pos, new, idx, dtn, sk, qk, ak, dw, aw, n_active) + _heat_ev_conf(
            pos, new, idx, heat, hw, r0, ew, cx, cy, cz, R, cw, n_active
        )

    def _energy(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, cx, cy, cz, R, cw, n_active):
        b = pos.shape[0]
        idx = jnp.arange(b)
        active = idx < n_active
        eye = idx[:, None] == idx[None, :]
        d = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        diff = pos[:-1] - pos[1:]
        bonds = _len_E(jnp.sqrt(jnp.sum(diff * diff, axis=-1)), dtn[: b - 1], sk, qk, dw)
        tot = jnp.sum(jnp.where(jnp.arange(b - 1) < n_active - 1, bonds, 0.0))
        angs = _ang_E(diff[:-1], diff[1:], ak, aw)
        tot = tot + jnp.sum(jnp.where(jnp.arange(b - 2) < n_active - 2, angs, 0.0))
        hmask = jnp.logical_and(
            jnp.logical_and(heat > 1e-6, jnp.logical_not(eye)), active[:, None] & active[None, :]
        )
        rel = (d - jnp.maximum(heat, 1e-6)) / jnp.maximum(heat, 1e-6)
        tot = tot + hw * jnp.sum(jnp.where(hmask, rel * rel, 0.0))
        far = jnp.logical_and(
            jnp.abs(idx[:, None] - idx[None, :]) > excl_skip, jnp.logical_not(eye)
        )
        far = jnp.logical_and(far, active[:, None] & active[None, :])
        rl = jnp.maximum(0.0, (r0 - d) / r0)
        tot = tot + jnp.sum(jnp.where(far, ew * rl * rl, 0.0))
        ctr = jnp.array([cx, cy, cz])
        rr = jnp.sqrt(jnp.sum((pos - ctr) ** 2, axis=-1))
        return tot + jnp.sum(
            jnp.where(jnp.logical_and(active, rr > R), cw * ((rr - R) / R) ** 2, 0.0)
        )

    def chain_checker(
        pos0,
        score0,
        T0,
        dtn,
        heat,
        movable,
        step,
        dt,
        js,
        jc,
        sk,
        qk,
        ak,
        dw,
        aw,
        hw,
        r0,
        ew,
        cx,
        cy,
        cz,
        R,
        cw,
        n_active,
        key,
    ):
        b = pos0.shape[0]
        idx_all = jnp.arange(b)
        chainc = (idx_all % 3) * 8
        eargs = (sk, qk, ak, dw, aw, hw, r0, ew, cx, cy, cz, R, cw, n_active)
        # cell = 4 * mean nn over S PROBE beads (subset).  The full (B,B) distance + jnp.sort
        # made XLA compile pathologically / fail ptxas at large B (up to 32k) or large batch
        # widths; the cell is only a heuristic scale, so a subsampled mean is fine.
        big = jnp.float32(1e30)
        active = idx_all < n_active
        S = 64
        stride = jnp.maximum(n_active // S, 1)
        probe = jnp.minimum(jnp.arange(S) * stride, jnp.maximum(n_active - 1, 0))
        pp = pos0[probe]
        dpr = jnp.sqrt(jnp.sum((pp[:, None, :] - pos0[None, :, :]) ** 2, axis=-1))  # (S, B)
        mask = (probe[:, None] == idx_all[None, :]) | jnp.logical_not(active[None, :])
        nn_pr = jnp.min(jnp.where(mask, big, dpr), axis=1)
        valid = jnp.arange(S) < n_active
        mean_nn = jnp.sum(jnp.where(valid, nn_pr, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
        cell = jnp.maximum(4.0 * mean_nn, 1e-10)

        def sweep_body(sw, carry):
            pos, score, T, n_ok, mx = carry
            ci = jnp.floor(pos / cell).astype(jnp.int32)
            color = chainc + (ci[:, 0] & 1) * 4 + (ci[:, 1] & 1) * 2 + (ci[:, 2] & 1)
            k_m, k_u = jax.random.split(jax.random.fold_in(key, sw + 1))
            move = jax.random.uniform(k_m, (b, 3), minval=-step, maxval=step, dtype=pos.dtype)
            u = jax.random.uniform(k_u, (b,), dtype=pos.dtype)

            def color_body(c, c2):
                pos, score, T, n_ok, mx = c2
                mask_c = jnp.logical_and(color == c, movable)
                count_c = jnp.sum(mask_c)
                idx_c = jnp.nonzero(mask_c, size=maxc, fill_value=0)[0]
                valid = jnp.arange(maxc) < count_c
                delta = _deltas(
                    pos,
                    move,
                    idx_c,
                    dtn,
                    heat,
                    sk,
                    qk,
                    ak,
                    dw,
                    aw,
                    hw,
                    r0,
                    ew,
                    cx,
                    cy,
                    cz,
                    R,
                    cw,
                    n_active,
                )
                can_jump = jnp.logical_and(T > 0.0, score > 0.0)
                expo = jnp.clip(
                    -jc * ((score + delta) / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30),
                    -80.0,
                    80.0,
                )
                ok = jnp.logical_or(
                    delta < 0.0, jnp.logical_and(can_jump, u[idx_c] < js * jnp.exp(expo))
                )  # STRICT
                ok = jnp.logical_and(ok, valid)
                pos = pos.at[idx_c].add(jnp.where(ok[:, None], move[idx_c], 0.0))
                score = score + jnp.sum(jnp.where(ok, delta, 0.0))
                n_ok = n_ok + jnp.sum(ok)
                return pos, score, T * dt**count_c, n_ok, jnp.maximum(mx, count_c.astype(jnp.int32))

            return jax.lax.fori_loop(0, 24, color_body, (pos, score, T, n_ok, mx))

        init = (pos0, score0, T0, jnp.int32(0), jnp.int32(0))
        pos, _s, T, n_ok, mx = jax.lax.fori_loop(0, n_sweeps, sweep_body, init)
        return pos, _energy(pos, dtn, heat, *eargs), T, n_ok, mx

    in_axes = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        None,
        0,
        0,
        0,
        0,
        None,
        0,
        0,
    )
    batched = jax.vmap(chain_checker, in_axes=in_axes, out_axes=(0, 0, 0, 0, 0))

    @jax.jit
    def kernel_chunk(carry, problem, scalars, base_key, max_iters, iter_base):
        """Run the batched smooth checker on this (shrunk + padded) chain set until all
        converge OR ``max_iters`` outer-iters; returns the carry + iters run.  See
        gnome3d/mc/jax/shrink.py for the host-side shrinking driver."""
        pos, score, T, ms0, conv0, ci0 = carry
        (
            dtn_k,
            heat_k,
            movable_k,
            step_k,
            r0_k,
            cx_k,
            cy_k,
            cz_k,
            R_k,
            n_active_k,
            succ_k,
            chain_id,
        ) = problem
        (dt, js, jc, sk, qk, ak, dw, aw, hw, ew, cw, stop_improvement, score_eps, stop_ratio) = (
            scalars
        )

        def cond_fn(state):
            return jnp.logical_and(jnp.logical_not(jnp.all(state[4])), state[6] < max_iters)

        def body_fn(state):
            pos, score, T, ms, conv_prev, conv_iter, li = state
            # per-chain RNG (global id) -> compaction-invariant; see arcs_checker.
            giter = iter_base + li + 1
            keys = jax.vmap(
                lambda cid: jax.random.fold_in(jax.random.fold_in(base_key, cid), giter)
            )(chain_id)
            npos, nscore, nT, nok, _mc = batched(
                pos,
                score,
                T,
                dtn_k,
                heat_k,
                movable_k,
                step_k,
                dt,
                js,
                jc,
                sk,
                qk,
                ak,
                dw,
                aw,
                hw,
                r0_k,
                ew,
                cx_k,
                cy_k,
                cz_k,
                R_k,
                cw,
                n_active_k,
                keys,
            )
            frozen = conv_prev
            pos = jnp.where(frozen[:, None, None], pos, npos)
            score = jnp.where(frozen, score, nscore)
            ratio = score / jnp.maximum(ms, 1e-30)
            plateaued = jnp.logical_and(score > stop_improvement * ms, nok < succ_k)
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, score < score_eps), ratio > stop_ratio),
                conv_prev,
            )
            conv_iter = jnp.where(
                jnp.logical_and(converged, jnp.logical_not(conv_prev)),
                iter_base + li + 1,
                conv_iter,
            )
            return pos, score, nT, score, converged, conv_iter, li + 1

        init = (pos, score, T, ms0, conv0, ci0, jnp.int32(0))
        pos, score, T, ms, conv, ci, li = jax.lax.while_loop(cond_fn, body_fn, init)
        return (pos, score, T, ms, conv, ci), li

    init_energy = jax.jit(
        jax.vmap(
            _energy,
            in_axes=(0, 0, 0, None, None, None, None, None, None, 0, None, 0, 0, 0, 0, None, 0),
        )
    )

    bundle = (kernel_chunk, init_energy)
    _kernel_cache[cache_key] = bundle
    return bundle


def _prep(p: dict[str, Any], settings: "Settings", B: int) -> dict[str, Any]:
    pos = np.asarray(p["pos"], np.float32)
    dtn = np.asarray(p["dtn"], np.float32)
    fixed = np.asarray(p["fixed"], np.bool_)
    n = pos.shape[0]
    heat = p.get("heat_dist")
    heat = np.asarray(heat, np.float32) if heat is not None else np.zeros((n, n), np.float32)
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    ew = float(settings.exclusion_weight) if use_excl else 0.0
    r0 = float(settings.exclusion_radius_smooth)
    if use_excl and r0 <= 0.0:
        m = dtn > 1e-6
        r0 = float(settings.exclusion_auto_factor_smooth) * float(dtn[m].mean()) if m.any() else 1.0
    r0 = r0 if r0 > 0 else 1.0
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    cx = cy = cz = 0.0
    R = 1.0
    cw = 0.0
    if use_conf:
        cx, cy, cz = float(pos[:, 0].mean()), float(pos[:, 1].mean()), float(pos[:, 2].mean())
        R = float(settings.confinement_radius_smooth)
        if R <= 0.0:
            m = dtn > 1e-6
            ab = float(dtn[m].mean()) if m.any() else 1.0
            R = float(settings.confinement_packing_factor_smooth) * ab * (n ** (1.0 / 3.0))
        cw = float(settings.confinement_weight)
    if B > n:
        pos = np.concatenate([pos, np.zeros((B - n, 3), np.float32)], axis=0)
        dtn = np.concatenate([dtn, np.zeros(B - n, np.float32)])
        fixed = np.concatenate([fixed, np.ones(B - n, np.bool_)])  # pad beads never move
        hp = np.zeros((B, B), np.float32)
        hp[:n, :n] = heat
        heat = hp
    return {
        "n": n,
        "pos": pos,
        "dtn": dtn,
        "heat": heat,
        "movable": ~fixed,
        "r0": r0,
        "ew": ew,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "R": R,
        "cw": cw,
        "n_active": n,
    }


def _bytes(B: int) -> int:
    f4 = 4
    return B * B * f4 + B * 3 * f4 + B * f4 + B  # heat (B,B) dominates


def mc_smooth_checker_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Checkerboard analogue of mc_smooth_jax_batch (same (score, pos) return).  Score
    EXCLUDES the constant orientation offset (see module docstring)."""
    if not problems:
        return []
    if not jax_is_available():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    big_b = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    budget = jax_device_budget_bytes()
    max_k = max_k_for_bytes(_bytes(big_b), 0, budget) if budget else max(1, 16384 // max(1, big_b))
    # Bound the SINGLE largest device allocation, not just the total.  run_shrinking pads the
    # active batch up to the next power of two (shrink._ceil_pow2), so one launch materialises a
    # CONTIGUOUS (pow2(max_k), big_b, big_b) energy tensor.  At big_b=16384 that reached 8 GiB and
    # OOM'd a fragmented BFC pool that still had ample *total* free memory (the budget above only
    # bounds the total, not the largest contiguous block).  Cap max_k so that padded tensor stays
    # <= ~1/8 of the pool, floored to a power of two so _ceil_pow2 can't round past the cap.
    # ONLY for the fragmenting BFC pool: vmm (CUDA virtual memory) and platform back a large
    # logical tensor with non-contiguous physical pages, so they place it fine - skip the cap
    # there (XLA_PYTHON_CLIENT_ALLOCATOR selects the allocator) to keep full batch width.
    alloc = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").lower()
    if budget and alloc not in ("vmm", "platform"):
        single = max(1, int(0.125 * budget) // (big_b * big_b * 4))
        max_k = max(1, min(max_k, 1 << (single.bit_length() - 1)))
    if len(problems) <= max_k:
        return _chunk(problems, settings)
    out: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i in range(0, len(problems), max_k):
        out.extend(_chunk(problems[i : i + max_k], settings))
    return out


def _chunk(
    problems: list[dict[str, Any]], settings: "Settings"
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    import jax
    import jax.numpy as jnp

    K = len(problems)
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    B = max(
        (jax_bucket_for(int(p["pos"].shape[0])) if bucket else int(p["pos"].shape[0]))
        for p in problems
    )
    preps = [_prep(p, settings, B) for p in problems]
    st = lambda k, dt: jnp.asarray(np.array([pr[k] for pr in preps], dtype=dt))

    pos_k = jnp.asarray(np.stack([pr["pos"] for pr in preps], 0))
    dtn_k = jnp.asarray(np.stack([pr["dtn"] for pr in preps], 0))
    heat_k = jnp.asarray(np.stack([pr["heat"] for pr in preps], 0))
    movable_k = jnp.asarray(np.stack([pr["movable"] for pr in preps], 0))
    n_active_k = st("n_active", np.int32)
    r0_k, R_k = st("r0", np.float32), st("R", np.float32)
    cx_k, cy_k, cz_k = st("cx", np.float32), st("cy", np.float32), st("cz", np.float32)
    step_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], np.float32))

    excl_skip = int(settings.exclusion_skip_neighbors)
    ew = jnp.float32(preps[0]["ew"])
    cw = jnp.float32(preps[0]["cw"])
    sk, qk = jnp.float32(settings.spring_stretch), jnp.float32(settings.spring_squeeze)
    ak = jnp.float32(settings.spring_angular)
    dw, aw = jnp.float32(settings.smooth_dist_weight), jnp.float32(settings.smooth_angle_weight)
    hw = jnp.float32(settings.subanchor_heatmap_dist_weight)
    maxc = max(8, B // 8)  # 24-colour parity => ~B/24 per colour, B/8 = ~3x safety
    # >=4 sweeps/check so a single sweep's tiny score change doesn't trip ratio prematurely
    # (n_sweeps=mc_stop_steps//B was 1 whenever B>mc_stop_steps -> premature stop, long bonds).
    stop_steps = max(int(settings.mc_stop_steps_smooth), 1)
    n_sweeps = max(4, stop_steps // B)

    # Scale the plateau accept-threshold per IB to the checker's proposals/outer-iter
    # (n_sweeps * #movable beads) so `n_ok < successes` fires at the SAME accept rate as
    # the numba sequential (else the parallel checker over- or under-converges vs numba).
    movable_cnt = np.array([int(pr["movable"].sum()) for pr in preps], np.float64)
    succ_k = jnp.asarray(
        np.maximum(
            1.0, settings.mc_stop_successes_smooth * n_sweeps * movable_cnt / stop_steps
        ).astype(np.float32)
    )

    kernel_chunk, init_energy = _build_smooth_checker_kernel(n_sweeps, excl_skip, maxc)
    score_k = init_energy(
        pos_k,
        dtn_k,
        heat_k,
        sk,
        qk,
        ak,
        dw,
        aw,
        hw,
        r0_k,
        ew,
        cx_k,
        cy_k,
        cz_k,
        R_k,
        cw,
        n_active_k,
    )

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    carry = (
        pos_k,
        score_k,
        jnp.full((K,), jnp.float32(settings.max_temp_smooth)),
        jnp.full((K,), jnp.float32(1e30)),
        jnp.zeros((K,), jnp.bool_),
        jnp.zeros((K,), jnp.int32),
    )
    problem = (
        dtn_k,
        heat_k,
        movable_k,
        step_k,
        r0_k,
        cx_k,
        cy_k,
        cz_k,
        R_k,
        n_active_k,
        succ_k,
        jnp.arange(K, dtype=jnp.int32),
    )
    scalars = (
        jnp.float32(settings.dt_temp_smooth),
        jnp.float32(settings.jump_scale_smooth),
        jnp.float32(settings.jump_coef_smooth),
        sk,
        qk,
        ak,
        dw,
        aw,
        hw,
        ew,
        cw,
        jnp.float32(settings.mc_stop_improvement_smooth),
        jnp.float32(1e-6),
        jnp.float32(0.9999),
    )

    log_kernel_start(
        LOG,
        "smooth",
        "checker",
        K,
        B,
        f"24-colour gather, <={maxc}/colour, {n_sweeps} sweeps/round",
    )
    t0 = time.perf_counter()
    out_pos, out_score, out_ci, total = run_shrinking(
        kernel_chunk, carry, problem, scalars, base_key, max_total=_MAX_ITERS
    )
    ci = out_ci[out_ci > 0]
    med = int(np.median(ci)) if ci.size else 0
    slow = int(ci.max()) if ci.size else 0
    log_kernel_done(
        LOG,
        "smooth",
        "checker",
        K,
        time.perf_counter() - t0,
        f"{total} rounds, {ci.size}/{K} converged (median {med}, slowest {slow})",
    )
    return [
        (float(out_score[i]), out_pos[i][: pr["n"]].astype(np.float32))
        for i, pr in enumerate(preps)
    ]
