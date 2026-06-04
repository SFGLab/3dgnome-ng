"""JAX smooth-MC kernel: chain bonds + angles, optional EV / heat / CTCF orientation.

This is the dominant MC hot path (mc_smooth is 89-96% of MC wall time at
N=2000-10000).  `mc_smooth_jax` is the single-problem entry; `mc_smooth_jax_batch`
anneals K different IBs in one vmapped kernel (region batching).  Both build /
compile through `_build_smooth_kernel` (memoised in `_kernel_cache`) and can
eagerly precompile every shape bucket via `_precompile_smooth`.

Confinement is not supported here - the dispatch gate in the pipeline routes
confinement-enabled smooth calls back to numba.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log
from gnome3d.mc.jax.memory import max_k_for_bytes
from gnome3d.mc.jax.util import (
    ANCHOR_BUCKETS,
    NBR_BUCKETS,
    SHAPE_BUCKETS,
    jax_bucket_for,
    jax_device_budget_bytes,
    jax_is_available,
)
from gnome3d.types import F32Array, I32Array, I64Array

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

# Compiled-kernel cache (keyed by kernel signature)
_kernel_cache: dict[Any, Any] = {}
_precompiled: set[Any] = set()
_init_lock = threading.Lock()


def _build_smooth_kernel(
    n_steps_per_batch: int,
    excl_skip: int,
    use_heat: bool,
    use_orn: bool,
    max_nbrs: int,
) -> Any:
    """Build (or look up cached) compiled smooth-MC kernel.

    Returns (kernel, init_smooth, init_excl, init_heat, init_orn) - the four
    init functions compute initial scores on-device, vmapped across K chains.

    Static-by-cache-key: n_steps_per_batch, excl_skip, use_heat, use_orn,
    max_nbrs (padding width for the orientation neighbor lists).  JAX further
    shape-specialises on (N, K, n_anchors, n_movable) at runtime - those
    incur per-shape compile cost (cached persistently via
    jax.experimental.compilation_cache).
    """
    cache_key = (n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    import jax
    import jax.numpy as jnp

    # ---- chain energy helpers ----

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
        pos: Any,
        p_pos: Any,
        p: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        # `n` clips indices into the (possibly bucket-padded) array; `n_active`
        # is the real chain length, so bonds/angles spanning a pad bead (index
        # >= n_active) are masked out.  When unbucketed n_active == n, so this is
        # a no-op.  Pad beads form a contiguous tail, hence a scalar boundary.
        n = pos.shape[0]
        a_pm1 = pos[jnp.maximum(p - 1, 0)]
        bond_L_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n_active - 1)
        bond_L = jnp.where(
            bond_L_ok,
            _smooth_len(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)], stretch_k, squeeze_k, dist_w),
            0.0,
        )
        a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
        bond_R_ok = jnp.logical_and(p >= 0, p < n_active - 1)
        bond_R = jnp.where(
            bond_R_ok,
            _smooth_len(p_pos, a_pp1, dtn[jnp.minimum(p, n - 2)], stretch_k, squeeze_k, dist_w),
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
            valid = jnp.logical_and(i >= 0, i < n_active - 2)
            return jnp.where(valid, _smooth_ang(a0, a1, a2, ang_k, ang_w), 0.0)

        return bond_L + bond_R + angle_at(-2) + angle_at(-1) + angle_at(0)

    # ---- excluded volume helpers ----

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        # Exclude pad beads (idx >= n_active) from the pairwise sum.  Unbucketed
        # n_active == n, so this is a no-op.
        in_range = jnp.logical_and(jnp.abs(idx - p) > excl_skip, idx < n_active)
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    # ---- confinement helper ----
    #
    # Per-bead soft envelope.  Mirrors gnome3d.mc._local_confine_nb:
    #   E(p) = weight * ((|r_p - c| - R) / R)²   if |r_p - c| > R
    #        = 0                                  otherwise
    # Delta factor 1 (single-counted globally).  Always wired into the kernel;
    # weight=0 disables it via XLA constant-folding.

    def _local_confine_at(p_pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any) -> Any:
        dx = p_pos[0] - cx
        dy = p_pos[1] - cy
        dz = p_pos[2] - cz
        r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (r - R) / jnp.maximum(R, 1e-30)
        contrib = weight * rel * rel
        return jnp.where(r > R, contrib, 0.0)

    # ---- heat (subanchor heatmap) helpers ----

    def _local_heat_at(pos: Any, p_pos: Any, p: Any, heat_dist: Any, heat_weight: Any) -> Any:
        """Local heat score for bead p vs all others, evaluated as if pos[p] = p_pos.
        Mirrors gnome3d.mc._local_heat_nb: sum_{i != p, heat_dist[i,p] > 0}
        ((d - heat_dist[i,p]) / heat_dist[i,p])^2 * heat_weight."""
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        exp_d = heat_dist[:, p]  # (N,)
        idx = jnp.arange(n)
        # Skip i == p (the diagonal of heat_dist is zero anyway, but mask
        # explicitly to match numba semantics) and pairs with no contact data
        # (heat_dist < 1e-6).
        active = jnp.logical_and(idx != p, exp_d >= 1e-6)
        exp_d_safe = jnp.maximum(exp_d, 1e-6)
        rel = (d - exp_d_safe) / exp_d_safe
        contrib = rel * rel
        return heat_weight * jnp.sum(jnp.where(active, contrib, 0.0))

    # ---- orientation helpers ----

    def _calc_orientation_at(pos: Any, p: Any, p_pos: Any, ar: Any, is_L: Any) -> Any:
        """Compute the orientation vector for anchor at bead-index `ar`,
        assuming pos[p] is replaced by p_pos.  Returns a normalised (3,) vec.

        Mirrors gnome3d.mc._calc_orientation_nb edge cases:
          - ar == 0:     orn = pos[1]  - pos[0]
          - ar == N-1:   orn = pos[ar] - pos[ar-1]
          - middle:      orn = pos[ar+1] - pos[ar-1]
        Sign-flipped if is_L is True; then L2-normalised."""
        n = pos.shape[0]
        pp1_idx = jnp.minimum(ar + 1, n - 1)
        pm1_idx = jnp.maximum(ar - 1, 0)
        # Substitute p_pos at the right slot if it happens to be one of these
        a_ar = jnp.where(ar == p, p_pos, pos[ar])
        a_pp1 = jnp.where(pp1_idx == p, p_pos, pos[pp1_idx])
        a_pm1 = jnp.where(pm1_idx == p, p_pos, pos[pm1_idx])

        is_first = ar == 0
        is_last = ar == n - 1
        o_first = a_pp1 - a_ar
        o_last = a_ar - a_pm1
        o_mid = a_pp1 - a_pm1
        o = jnp.where(is_first, o_first, jnp.where(is_last, o_last, o_mid))
        o = jnp.where(is_L, -o, o)
        nm = jnp.sqrt(jnp.sum(o * o))
        return jnp.where(nm > 1e-12, o / jnp.maximum(nm, 1e-30), jnp.zeros_like(o))

    def _local_orientation_at(
        anchor_orn: Any,
        k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Local orientation score for anchor k, summed over its (padded)
        neighbors.  Mirrors gnome3d.mc._local_score_orientation_nb."""
        # nbr_idx[k, :] are the neighbor anchor indices (max_nbrs wide, padded
        # with 0 + nbr_valid=False).  nbr_w[k, :] are the per-edge weights.
        neighbors_k = nbr_idx[k]  # (max_nbrs,)
        weights_k = nbr_w[k]  # (max_nbrs,)
        valid_k = nbr_valid[k]  # (max_nbrs,)
        a = anchor_orn[k]  # (3,)
        b = anchor_orn[neighbors_k]  # (max_nbrs, 3)
        b_signed = jnp.where(symmetric, b, -b)
        dot = jnp.sum(a[None, :] * b_signed, axis=1)  # (max_nbrs,)
        ang = 1.0 - (dot + 1.0) * 0.5
        contrib = jnp.where(valid_k, ang * ang * weights_k, 0.0)
        return motif_weight * jnp.sum(contrib)

    # ---- chain (per-batch) body ----

    def chain_batch(
        # state
        pos0: Any,
        ss0: Any,
        se0: Any,
        sh0: Any,
        so0: Any,
        sc0: Any,
        anchor_orn0: Any,
        T0_: Any,
        # static problem data
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        # schedule
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        # RNG
        key: Any,
        # real bead count + real movable count (< padded lengths when bucketed)
        n_active: Any,
        n_movable_active: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        """One batch of `n_steps_per_batch` MC steps for ONE chain.  Returns
        (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, T_f, n_ok)."""
        # `movable` is padded to the bucket; n_movable_active is the real count so
        # the sampler only draws real movable beads (no-op when unbucketed).
        k_p, k_d, k_a = jax.random.split(key, 3)
        idx_picks = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_movable_active)
        ps = movable[idx_picks]
        disps = jax.random.uniform(
            k_d,
            (n_steps_per_batch, 3),
            minval=-step_size,
            maxval=step_size,
            dtype=pos0.dtype,
        )
        accs = jax.random.uniform(k_a, (n_steps_per_batch,), dtype=pos0.dtype)

        def body(i: Any, carry: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = carry
            p = ps[i]
            delta = disps[i]
            u = accs[i]

            score = ss + se + sh + so + sc
            old_p = pos[p]
            new_p = old_p + delta

            # ---- struct (chain bonds + angles) ----
            loc_s_prev = _local_smooth_at(
                pos, old_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            loc_s_curr = _local_smooth_at(
                pos, new_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            ss_new = ss + (loc_s_curr - loc_s_prev)

            # ---- excluded volume ----
            loc_e_prev = _local_excl_at(pos, old_p, p, r0, excl_w, n_active)
            loc_e_curr = _local_excl_at(pos, new_p, p, r0, excl_w, n_active)
            se_new = se + 2.0 * (loc_e_curr - loc_e_prev)

            # ---- heat ----
            if use_heat:
                loc_h_prev = _local_heat_at(pos, old_p, p, heat_dist, heat_weight)
                loc_h_curr = _local_heat_at(pos, new_p, p, heat_dist, heat_weight)
                sh_new = sh + 2.0 * (loc_h_curr - loc_h_prev)
            else:
                sh_new = sh

            # ---- orientation ----
            if use_orn:
                # orn_k = bead_to_anchor_k[p]; if >= 0 this bead is adjacent to
                # an anchor whose orientation depends on p's position.
                orn_k = bead_to_anchor_k[p]  # int, may be -1
                has_orn = orn_k >= 0
                safe_k = jnp.maximum(orn_k, 0)
                # PREV orientation already lives in anchor_orn[safe_k]
                loc_o_prev_raw = _local_orientation_at(
                    anchor_orn,
                    safe_k,
                    nbr_idx,
                    nbr_w,
                    nbr_valid,
                    motif_weight,
                    symmetric,
                )
                loc_o_prev = jnp.where(has_orn, loc_o_prev_raw, 0.0)

                # CURR: recompute anchor's orientation with p moved to new_p
                ar_p = anchor_ar[safe_k]
                is_L_ar = is_L[ar_p]
                new_orn_vec = _calc_orientation_at(pos, p, new_p, ar_p, is_L_ar)
                # Update only that slot in anchor_orn (functional, single scatter)
                anchor_orn_trial = anchor_orn.at[safe_k].set(new_orn_vec)
                loc_o_curr_raw = _local_orientation_at(
                    anchor_orn_trial,
                    safe_k,
                    nbr_idx,
                    nbr_w,
                    nbr_valid,
                    motif_weight,
                    symmetric,
                )
                loc_o_curr = jnp.where(has_orn, loc_o_curr_raw, 0.0)
                so_new = so + 2.0 * (loc_o_curr - loc_o_prev)
            else:
                anchor_orn_trial = anchor_orn
                so_new = so
                has_orn = False
                safe_k = jnp.int32(0)

            # ---- confinement (per-bead, single-counted, delta factor 1) ----
            # When conf_w == 0 the entire contribution folds to zero; XLA
            # eliminates the branch.  Always wired so no new cache key needed.
            loc_c_prev = _local_confine_at(old_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            loc_c_curr = _local_confine_at(new_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            sc_new = sc + (loc_c_curr - loc_c_prev)

            score_new = ss_new + se_new + sh_new + so_new + sc_new

            ok_unc = score_new < score  # smooth uses STRICT less-than
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))

            final_p = jnp.where(ok, new_p, old_p)
            pos_next = pos.at[p].set(final_p)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            sh_next = jnp.where(ok, sh_new, sh)
            so_next = jnp.where(ok, so_new, so)
            sc_next = jnp.where(ok, sc_new, sc)
            if use_orn:
                # Accept = keep anchor_orn_trial; reject = keep anchor_orn.
                # We only modified anchor_orn[safe_k], so equivalently:
                #   anchor_orn_next = anchor_orn_trial if ok else anchor_orn
                anchor_orn_next = jnp.where(ok, anchor_orn_trial, anchor_orn)
            else:
                anchor_orn_next = anchor_orn
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (
                pos_next,
                ss_next,
                se_next,
                sh_next,
                so_next,
                sc_next,
                anchor_orn_next,
                T * dt,
                n_ok_next,
            )

        init = (pos0, ss0, se0, sh0, so0, sc0, anchor_orn0, T0_, jnp.int32(0))
        return jax.lax.fori_loop(0, n_steps_per_batch, body, init)

    # vmap over K chains; problem data and schedule are shared (None).
    # Per-chain: pos, all 5 scores, anchor_orn, key.  T is shared (deterministic).
    in_axes = (
        0,
        0,
        0,
        0,
        0,
        0,  # pos, ss, se, sh, so, sc
        0,  # anchor_orn
        None,  # T0
        None,
        None,  # dtn, movable
        None,  # heat_dist
        None,
        None,  # anchor_ar, bead_to_anchor_k
        None,
        None,
        None,  # nbr_idx, nbr_w, nbr_valid
        None,  # is_L
        None,
        None,
        None,
        None,  # step_size, dt, js, jc
        None,
        None,
        None,
        None,
        None,  # stretch..ang_w
        None,
        None,  # r0, excl_w
        None,  # heat_weight
        None,
        None,  # motif_weight, symmetric
        None,
        None,
        None,
        None,
        None,  # conf_cx, conf_cy, conf_cz, conf_R, conf_w
        0,  # key
        None,  # n_active (shared)
        None,  # n_movable_active (shared)
    )
    out_axes = (0, 0, 0, 0, 0, 0, 0, None, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    @jax.jit
    def kernel(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        keys: Any,
        n_active: Any,
        n_movable_active: Any,
    ) -> Any:
        return batched(
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T,
            dtn,
            movable,
            heat_dist,
            anchor_ar,
            bead_to_anchor_k,
            nbr_idx,
            nbr_w,
            nbr_valid,
            is_L,
            step_size,
            dt,
            js,
            jc,
            stretch_k,
            squeeze_k,
            ang_k,
            dist_w,
            ang_w,
            r0,
            excl_w,
            heat_weight,
            motif_weight,
            symmetric,
            conf_cx,
            conf_cy,
            conf_cz,
            conf_R,
            conf_w,
            keys,
            n_active,
            n_movable_active,
        )

    # ---- full convergence loop, on device ----
    #
    # Wraps the per-batch `batched` kernel with `lax.while_loop`.  Each
    # iteration of the while_loop = one MC batch across all K chains.  The
    # entire annealing runs inside ONE JAX call - no Python sync between
    # batches.  Replaces a Python loop that did one device->host copy per
    # batch (5-10ms × hundreds of batches per smooth call).
    #
    # max_iters is baked in as a static safety cap to prevent runaway loops
    # if convergence never triggers.  At n_steps_per_batch=2000 with
    # max_iters=10000 we cap at 20M MC steps - comfortably above any
    # realistic convergence count.
    _MAX_ITERS: int = 10000

    @jax.jit
    def kernel_full(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T_init: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        n_active: Any,
        n_movable_active: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            _, _, _, _, _, _, _, _, _, iter_i, _, converged = state
            return jnp.logical_and(jnp.logical_not(converged), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, ms_score, iter_i, _, _ = state
            # Derive K per-chain keys deterministically from iter_i
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = batched(
                pos,
                ss,
                se,
                sh,
                so,
                sc,
                anchor_orn,
                T,
                dtn,
                movable,
                heat_dist,
                anchor_ar,
                bead_to_anchor_k,
                nbr_idx,
                nbr_w,
                nbr_valid,
                is_L,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                r0,
                excl_w,
                heat_weight,
                motif_weight,
                symmetric,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                keys,
                n_active,
                n_movable_active,
            )
            score_per_chain = ss + se + sh + so + sc
            best_idx = jnp.argmin(score_per_chain)
            score = score_per_chain[best_idx]
            n_ok_best = n_ok[best_idx]
            plateaued = jnp.logical_and(
                score > stop_improvement * ms_score, n_ok_best < stop_successes
            )
            eps_done = score < score_eps
            converged = jnp.logical_or(plateaued, eps_done)
            return (pos, ss, se, sh, so, sc, anchor_orn, T, score, iter_i + 1, n_ok_best, converged)

        # ms_score init: very large so the first batch never trips the
        # "improvement < threshold" check.  Matches the Python loop's
        # `ms_score = float("inf")` initialiser.
        init_state = (
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T_init,
            jnp.float32(1e30),  # ms_score
            jnp.int32(0),  # iter_i
            jnp.int32(0),  # n_ok_best (filler)
            jnp.bool_(False),  # converged
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        (
            pos_f,
            ss_f,
            se_f,
            sh_f,
            so_f,
            sc_f,
            anchor_orn_f,
            _T_f,
            final_score,
            iter_f,
            _n_ok_best_f,
            converged_f,
        ) = final
        return (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, final_score, iter_f, converged_f)

    # ---- init helpers (one-shot per chain on entry) ----

    def _init_smooth_single(
        pos: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        n = pos.shape[0]

        # SEQUENTIAL (lax.scan) accumulation, matching _init_excl/_init_heat.
        # A tree reduction (jnp.sum) groups differently for a bucket-padded
        # length vs the real length, and that ULP difference gets chaos-amplified
        # by the MC - so the chain init MUST be padding-insensitive.  Scan is:
        # appending the masked-to-zero pad terms never changes the running sum,
        # so bucketed == unbucketed bit-for-bit.  Mask spans a pad bead via
        # n_active (no-op when unbucketed, n_active == n).
        def _bond_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_len(pos[i], pos[i + 1], dtn[i], stretch_k, squeeze_k, dist_w)
            return carry + jnp.where(i + 1 < n_active, val, 0.0), None

        def _angle_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_ang(pos[i], pos[i + 1], pos[i + 2], ang_k, ang_w)
            return carry + jnp.where(i + 2 < n_active, val, 0.0), None

        bonds_total, _ = jax.lax.scan(_bond_body, jnp.float32(0.0), jnp.arange(n - 1))
        angles_total, _ = jax.lax.scan(_angle_body, jnp.float32(0.0), jnp.arange(n - 2))
        return bonds_total + angles_total

    def _init_excl_single(pos: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rel = jnp.maximum(0.0, (r0 - d) / r0)
            contrib = weight * rel * rel
            # Mask pad columns (idx >= n_active); zero the whole row if i is pad.
            in_range = jnp.logical_and(jnp.abs(idx - i) > excl_skip, idx < n_active)
            row = jnp.where(i < n_active, jnp.sum(jnp.where(in_range, contrib, 0.0)), 0.0)
            return carry + row, None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def _init_heat_single(pos: Any, heat_dist: Any, heat_weight: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            exp_d = heat_dist[:, i]
            active = jnp.logical_and(idx != i, exp_d >= 1e-6)
            exp_d_safe = jnp.maximum(exp_d, 1e-6)
            rel = (d - exp_d_safe) / exp_d_safe
            contrib = rel * rel
            return carry + jnp.sum(jnp.where(active, contrib, 0.0)), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return heat_weight * total

    def _init_confine_single(
        pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any, n_active: Any
    ) -> Any:
        """Sum of per-bead confinement contributions.
        Mirrors gnome3d.mc._init_confine_nb.  Sequential (lax.scan) so trailing
        pad beads (masked by n_active) don't perturb the f32 reduction order."""

        def _body(carry: Any, i: Any) -> tuple[Any, None]:
            c = _local_confine_at(pos[i], cx, cy, cz, R, weight)
            return carry + jnp.where(i < n_active, c, 0.0), None

        total, _ = jax.lax.scan(_body, jnp.float32(0.0), jnp.arange(pos.shape[0]))
        return total

    def _init_anchor_orientations_single(
        pos: Any,
        anchor_ar: Any,
        is_L: Any,
    ) -> Any:
        """Compute (n_anchors, 3) initial orientation vectors from anchor
        positions.  is_L is indexed by bead-index (full N).
        """

        def per_anchor(k_idx: Any) -> Any:
            ar = anchor_ar[k_idx]
            is_L_v = is_L[ar]
            # p == -1 sentinel: never matches any index, so substitution branches
            # all fall through to "use pos[...]".  (jnp.int32 cast for safety.)
            return _calc_orientation_at(
                pos, jnp.int32(-1), jnp.zeros((3,), dtype=pos.dtype), ar, is_L_v
            )

        return jax.vmap(per_anchor)(jnp.arange(anchor_ar.shape[0]))

    def _init_orientation_score_single(
        anchor_orn: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Global orientation score (matches _score_orientation_full_nb).

        SEQUENTIAL (lax.scan) accumulation over anchors so that anchor-bucket
        padding (pad anchors have nbr_valid=False -> contribute exactly 0) does
        not perturb the f32 reduction order.  A tree jnp.sum would group an
        A-padded vs n_anchors-real array differently and chaos-amplify."""

        def per_anchor(k_idx: Any) -> Any:
            return _local_orientation_at(
                anchor_orn,
                k_idx,
                nbr_idx,
                nbr_w,
                nbr_valid,
                motif_weight,
                symmetric,
            )

        # Sum of per-anchor local scores gives the global (each local iterates
        # its own neighbor list; symmetric arcs counted from both endpoints).
        # Matches numba _score_orientation_full_nb: per_anchor returns
        # motif_weight * sum_j w_kj*ang_kj^2, so sum_k = motif_weight * global.
        def _scan_body(carry: Any, k_idx: Any) -> tuple[Any, None]:
            return carry + per_anchor(k_idx), None

        total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), jnp.arange(anchor_orn.shape[0]))
        return total

    init_smooth = jax.jit(
        jax.vmap(
            _init_smooth_single,
            in_axes=(0, None, None, None, None, None, None, None),
        )
    )
    init_excl = jax.jit(jax.vmap(_init_excl_single, in_axes=(0, None, None, None)))
    init_heat = jax.jit(jax.vmap(_init_heat_single, in_axes=(0, None, None)))
    init_confine = jax.jit(
        jax.vmap(
            _init_confine_single,
            in_axes=(0, None, None, None, None, None, None),
        )
    )
    init_anchor_orn = jax.jit(jax.vmap(_init_anchor_orientations_single, in_axes=(0, None, None)))
    init_orn_score = jax.jit(
        jax.vmap(
            _init_orientation_score_single,
            in_axes=(0, None, None, None, None, None),
        )
    )

    # ---- multi-problem variant: K DIFFERENT IBs in one kernel ----
    #
    # The single-problem path above vmaps K restarts of ONE problem (problem
    # arrays shared, None in `in_axes`).  Region-batching instead wants K
    # *different* IBs annealed together to fill the GPU (per profile, a K=1
    # smooth at N=1607 is ~99% GPU-idle; K=8 costs ~the same wall).  Two
    # differences from the single-problem path, nothing else:
    #   (a) every per-IB array is vmapped (axis 0) instead of shared.
    #   (b) convergence is PER-CHAIN - each IB stops on its own criterion and
    #       the device while-loop runs until ALL chains have converged (or the
    #       max-iters cap).  Smooth uses strict acceptance, so a chain that
    #       converges early and keeps stepping can only hold or improve - never
    #       drift worse - which makes run-to-all-converged safe.  Wall-clock of
    #       a batch = its slowest IB, so the caller buckets IBs by size.
    #
    # The per-chain step body (`chain_batch`) and the init scores are reused
    # verbatim - the caller computes per-IB init scores with the same init
    # helpers and stacks them.  `batched_mp`/`kernel_full_mp` only compile when
    # the region-batched entry actually calls them.
    in_axes_mp = (
        0,
        0,
        0,
        0,
        0,
        0,  # pos, ss, se, sh, so, sc  (per-chain)
        0,  # anchor_orn (per-chain)
        None,  # T0 (shared schedule start)
        0,
        0,  # dtn, movable (per-IB)
        0,  # heat_dist (per-IB)
        0,
        0,  # anchor_ar, bead_to_anchor_k (per-IB)
        0,
        0,
        0,  # nbr_idx, nbr_w, nbr_valid (per-IB)
        0,  # is_L (per-IB)
        0,  # step_size (per-IB)
        None,
        None,
        None,  # dt, js, jc (shared schedule)
        0,
        0,
        0,  # stretch_k, squeeze_k, ang_k (per-IB; small-IB boost varies)
        None,
        None,  # dist_w, ang_w (global weights)
        0,  # r0 (per-IB auto excl radius)
        None,  # excl_w (global)
        None,  # heat_weight (global)
        None,
        None,  # motif_weight, symmetric (global)
        0,
        0,
        0,
        0,
        0,  # conf_cx, conf_cy, conf_cz, conf_R, conf_w (per-IB)
        0,  # keys (per-chain)
        0,  # n_active (per-IB)
        0,  # n_movable_active (per-IB)
    )
    batched_mp = jax.vmap(chain_batch, in_axes=in_axes_mp, out_axes=out_axes)

    @jax.jit
    def kernel_full_mp(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T_init: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        anchor_ar: Any,
        bead_to_anchor_k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        n_active: Any,
        n_movable_active: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            iter_i = state[9]
            converged = state[11]  # (K,) per-chain
            return jnp.logical_and(jnp.logical_not(jnp.all(converged)), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, ms_score, iter_i, _, conv_prev = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = batched_mp(
                pos,
                ss,
                se,
                sh,
                so,
                sc,
                anchor_orn,
                T,
                dtn,
                movable,
                heat_dist,
                anchor_ar,
                bead_to_anchor_k,
                nbr_idx,
                nbr_w,
                nbr_valid,
                is_L,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                r0,
                excl_w,
                heat_weight,
                motif_weight,
                symmetric,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                keys,
                n_active,
                n_movable_active,
            )
            score = ss + se + sh + so + sc  # (K,) per-chain total
            # Per-chain plateau: improvement below threshold AND too few accepts
            # this batch.  ms_score is the previous batch's per-chain score.
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, n_ok < stop_successes)
            eps_done = score < score_eps
            # Latch convergence: once a chain converges it stays converged, so
            # `jnp.all` can terminate even if a cold chain's score wobbles.
            converged = jnp.logical_or(jnp.logical_or(plateaued, eps_done), conv_prev)
            return (pos, ss, se, sh, so, sc, anchor_orn, T, score, iter_i + 1, n_ok, converged)

        init_state = (
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T_init,
            jnp.full((K,), 1e30, dtype=jnp.float32),  # ms_score per-chain
            jnp.int32(0),  # iter_i
            jnp.zeros((K,), dtype=jnp.int32),  # n_ok (filler)
            jnp.zeros((K,), dtype=jnp.bool_),  # converged per-chain
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        (
            pos_f,
            ss_f,
            se_f,
            sh_f,
            so_f,
            sc_f,
            anchor_orn_f,
            _T_f,
            final_score,
            iter_f,
            _n_ok_f,
            converged_f,
        ) = final
        return (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, final_score, iter_f, converged_f)

    bundle = (
        kernel,  # per-batch (kept for diagnostics; unused in prod)
        kernel_full,  # full convergence on device - the production path
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
        kernel_full_mp,  # region-batched (K different IBs); per-chain convergence
    )
    _kernel_cache[cache_key] = bundle
    return bundle


def _precompile_smooth(
    settings: "Settings", use_heat: bool, use_orn: bool, max_nbrs: int, anchor_frac: float, K: int
) -> None:
    """Eagerly compile the smooth kernel across N buckets for ONE
    (use_heat, use_orn, max_nbrs->M, K) combo.  Smooth specializes on
    (B, A, M, K, use_heat, use_orn); B and A both scale with region size, so we
    compile the realistic (B, A) DIAGONAL: A = bucket(anchor_frac * B) per B
    (use_orn=False has no anchor axis -> A=M=1).  Idempotent per combo via
    _precompiled.  Uses .lower(ShapeDtypeStruct).compile() (no array alloc)."""
    if not jax_is_available():
        return
    import jax
    import jax.numpy as jnp

    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps = int(settings.mc_stop_steps_smooth)
    M = int(max_nbrs) if use_orn else 1
    sig = ("smooth", n_steps, excl_skip, bool(use_heat), bool(use_orn), M, int(K))
    with _init_lock:
        if sig in _precompiled:
            return
        bundle = _build_smooth_kernel(n_steps, excl_skip, use_heat, use_orn, M)
        kernel_full = bundle[1]
        sds = jax.ShapeDtypeStruct
        f32 = jnp.float32
        key = jax.random.PRNGKey(0)
        T_a = f32(settings.max_temp_smooth)
        dt_a = f32(settings.dt_temp_smooth)
        js_a = f32(settings.jump_scale_smooth)
        jc_a = f32(settings.jump_coef_smooth)
        impr_a = f32(settings.mc_stop_improvement_smooth)
        succ_a = jnp.int32(settings.mc_stop_successes_smooth)
        t0 = __import__("time").perf_counter()
        for b in SHAPE_BUCKETS:
            a = jax_bucket_for(max(1, int(anchor_frac * b)), ANCHOR_BUCKETS) if use_orn else 1
            kvec = sds((K,), np.float32)
            heat_a = sds((b, b), np.float32) if use_heat else sds((1, 1), np.float32)
            try:
                kernel_full.lower(
                    sds((K, b, 3), np.float32),  # pos_k
                    kvec,
                    kvec,
                    kvec,
                    kvec,
                    kvec,  # ss, se, sh, so, sc
                    sds((K, a, 3), np.float32),  # anchor_orn_k
                    T_a,
                    sds((b,), np.float32),  # dtn
                    sds((b,), np.int32),  # movable (int64 -> int32 under x64-off)
                    heat_a,
                    sds((a,), np.int32),  # anchor_ar
                    sds((b,), np.int32),  # bead_to_anchor_k
                    sds((a, M), np.int32),  # nbr_idx
                    sds((a, M), np.float32),  # nbr_w
                    sds((a, M), np.bool_),  # nbr_valid
                    sds((b,), np.bool_),  # is_L
                    f32(0.1),  # step_size (value irrelevant)
                    dt_a,
                    js_a,
                    jc_a,
                    f32(1.0),
                    f32(1.0),
                    f32(0.1),
                    f32(1.0),
                    f32(1.0),  # stretch..ang_w
                    f32(1.0),
                    f32(0.0),  # r0, excl_w
                    f32(1.0),  # heat_weight
                    f32(1.0),  # motif_weight
                    jnp.bool_(True),  # symmetric
                    f32(0.0),
                    f32(0.0),
                    f32(0.0),
                    f32(1.0),
                    f32(0.0),  # conf_cx..conf_w
                    key,
                    impr_a,
                    succ_a,
                    f32(1e-6),  # score_eps (matches mc_smooth_jax hardcode)
                    jnp.int32(b),  # n_active
                    jnp.int32(b),  # n_movable_active
                ).compile()
            except Exception as e:  # noqa: BLE001 - precompile is best-effort
                LOG.warning("precompile smooth B=%d A=%d skipped: %s", b, a, e)
        _precompiled.add(sig)
        dt = __import__("time").perf_counter() - t0
        log.status(
            LOG,
            "precompiled smooth kernel: %d B-buckets (heat=%s orn=%s M=%d K=%d) in %.1fs",
            len(SHAPE_BUCKETS),
            use_heat,
            use_orn,
            M,
            K,
            dt,
        )


def mc_smooth_jax(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    pos_batch: np.ndarray[Any, Any] | None = None,
    return_all: bool = False,
) -> Any:
    """JAX backend for smooth-MC, supporting chain + EV + (optional) heat
    + (optional) orientation + (optional) confinement.

    Mutates `pos` in place (writes the best-chain final positions back) and
    returns the best chain's final score.

    Batched mode (`pos_batch` given, shape (B, N, 3)): run B independent anneals
    from distinct starts in ONE vmapped kernel (K = B), sharing `dtn`/`fixed`/
    `heat`/schedule and using `pos` only as the reference for n/movable/centroid.
    The shared while-loop stops when the BEST of the B chains converges (mc_jax
    convergence is best-of-K).  With `return_all=True` this returns
    `(scores: (B,), finals: (B, N, 3))` as numpy arrays and does NOT mutate
    `pos` - the caller does its own per-trial selection (see solver.py IB phase).
    """
    if not jax_is_available():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d-ng[jax]` or set mc_backend='numba'."
        )
    import jax
    import jax.numpy as jnp

    n: int = pos.shape[0]
    if n <= 2:
        return 0.0

    movable_np: I64Array = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    if len(movable_np) == 0:
        return 0.0

    if pos_batch is not None:
        if pos_batch.ndim != 3 or pos_batch.shape[1:] != (n, 3):
            raise ValueError(f"pos_batch must have shape (B, {n}, 3); got {pos_batch.shape}")
        K = int(pos_batch.shape[0])
    else:
        K = max(1, int(settings.mc_smooth_chains))
    n_steps_per_batch: int = int(settings.mc_stop_steps_smooth)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    if use_excl:
        excl_r0: float = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            factor = float(settings.exclusion_auto_factor_smooth)
            excl_r0 = factor * float(np.asarray(dtn).mean())
    else:
        excl_r0 = 1.0  # unused but must be valid

    use_heat: bool = heat_dist is not None
    heat_weight_v: float = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    use_orn: bool = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    motif_weight_v: float = float(settings.motif_weight) if use_orn else 0.0
    motifs_symmetric_v: bool = bool(getattr(settings, "motifs_symmetric", True))

    # ---- confinement setup ----
    # Per-bead soft envelope; center = centroid of starting pos, radius from
    # settings (or auto-derived).  Always wired into the kernel; when
    # disabled, conf_w=0 so XLA folds the contribution away.
    use_conf: bool = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    if use_conf:
        conf_cx_v: float = float(pos[:, 0].mean())
        conf_cy_v: float = float(pos[:, 1].mean())
        conf_cz_v: float = float(pos[:, 2].mean())
        conf_R_v: float = float(settings.confinement_radius_smooth)
        if conf_R_v <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R_v = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w_v: float = float(settings.confinement_weight)
    else:
        conf_cx_v = conf_cy_v = conf_cz_v = 0.0
        conf_R_v = 1.0
        conf_w_v = 0.0

    # ---- prepare orientation arrays (padded CSR) ----
    anchor_frac: float = 0.0  # real n_anchors/n; for the precompile (B,A) diagonal
    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar_np: I32Array = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar_np))
        # pad neighbor lists to max width; uniform shape needed for vmap.
        nbr_lists = [list(anchor_neighbors.get(k, [])) for k in range(n_anchors)]
        nbr_w_lists = [list(anchor_neighbor_weights.get(k, [])) for k in range(n_anchors)]
        max_nbrs = max((len(lst) for lst in nbr_lists), default=1)
        max_nbrs = max(max_nbrs, 1)  # at least 1 slot
        nbr_idx_np: I32Array = np.zeros((n_anchors, max_nbrs), dtype=np.int32)
        nbr_w_np: F32Array = np.zeros((n_anchors, max_nbrs), dtype=np.float32)
        nbr_valid_np = np.zeros((n_anchors, max_nbrs), dtype=np.bool_)
        for k_idx in range(n_anchors):
            for m, (jn, wn) in enumerate(zip(nbr_lists[k_idx], nbr_w_lists[k_idx], strict=True)):
                nbr_idx_np[k_idx, m] = int(jn)
                nbr_w_np[k_idx, m] = float(wn)
                nbr_valid_np[k_idx, m] = True
        # bead_to_anchor_k: -1 if bead not adjacent to an anchor; else k.
        bead_to_anchor_k_np: I32Array = np.full(n, -1, dtype=np.int32)
        for k_idx in range(n_anchors):
            ar = int(anchor_ar_np[k_idx])
            if ar > 0:
                bead_to_anchor_k_np[ar - 1] = k_idx
            if ar + 1 < n:
                bead_to_anchor_k_np[ar + 1] = k_idx
        is_L_np = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # Phase-2 bucketing of the ANCHOR-indexed arrays: round n_anchors -> A and
        # max_nbrs -> M so the kernel's orientation shapes come only from the
        # (A, M) ladders, not per-region.  Pad anchors/edges get nbr_valid=False
        # -> contribute exactly 0 to the (scan-summed) orientation score, so this
        # is bit-identical at init.  Pad anchor_ar=0 (its orn is computed but
        # never referenced since no valid edge points to it).
        if bool(settings.mc_executor_jax_bucket_shapes):
            A = jax_bucket_for(n_anchors, ANCHOR_BUCKETS)
            M = jax_bucket_for(max_nbrs, NBR_BUCKETS)
            anchor_frac = n_anchors / n  # real fraction, before reassignment below
            ap, mp = A - n_anchors, M - max_nbrs
            if ap > 0 or mp > 0:
                anchor_ar_np = np.concatenate([anchor_ar_np, np.zeros(ap, dtype=np.int32)])
                nbr_idx_np = np.pad(nbr_idx_np, ((0, ap), (0, mp)))
                nbr_w_np = np.pad(nbr_w_np, ((0, ap), (0, mp)))
                nbr_valid_np = np.pad(nbr_valid_np, ((0, ap), (0, mp)))  # False pads
                n_anchors, max_nbrs = A, M
    else:
        n_anchors = 1  # placeholder shape
        max_nbrs = 1
        anchor_ar_np = np.zeros(1, dtype=np.int32)
        nbr_idx_np = np.zeros((1, 1), dtype=np.int32)
        nbr_w_np = np.zeros((1, 1), dtype=np.float32)
        nbr_valid_np = np.zeros((1, 1), dtype=np.bool_)
        bead_to_anchor_k_np = np.full(n, -1, dtype=np.int32)
        is_L_np = np.zeros(n, dtype=np.bool_)

    # ---- move state to device (f32) ----
    pos_f32: F32Array = pos.astype(np.float32)
    if pos_batch is not None:
        # Batched mode: K distinct starts (one per trial), not a broadcast.
        pos_k_np = np.ascontiguousarray(pos_batch.astype(np.float32))
    else:
        pos_k_np = np.broadcast_to(pos_f32, (K, n, 3)).copy()
    dtn_np: F32Array = dtn.astype(np.float32)
    heat_np: F32Array
    if use_heat:
        assert heat_dist is not None
        heat_np = heat_dist.astype(np.float32)
    else:
        heat_np = np.zeros((1, 1), dtype=np.float32)  # unused placeholder

    # ---- shape bucketing: pad N up to a bucket so XLA reuses one compiled
    # kernel across all similarly-sized regions.  Pad beads are fully inert:
    # chain/EV/confinement masked by `n_active`, heat rows zeroed, movement
    # restricted to the real movable set via `n_movable_active`.  ALL bead-indexed
    # kernel inputs are padded to B so the kernel's input shapes depend only on B
    # (+ K, max_nbrs) - not on the per-region n/n_movable.  (Orientation's
    # anchor-indexed arrays, shape n_anchors, are NOT yet bucketed -> with
    # use_orn=True the kernel still recompiles per region; that's phase 2.)
    # n_active == n and n_movable_active == len(movable) when unbucketed.
    n_active_v: int = n
    n_movable_v: int = int(movable_np.shape[0])
    if bool(settings.mc_executor_jax_bucket_shapes):
        if settings.mc_executor_jax_precompile_buckets:
            _precompile_smooth(settings, use_heat, use_orn, max_nbrs, anchor_frac, K)
        B: int = jax_bucket_for(n)
    else:
        B = n
    if B > n:
        n_pad = B - n
        pos_k_np = np.concatenate(
            [pos_k_np, np.zeros((pos_k_np.shape[0], n_pad, 3), dtype=np.float32)], axis=1
        )
        dtn_np = np.concatenate([dtn_np, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat:
            heat_pad = np.zeros((B, B), dtype=np.float32)
            heat_pad[:n, :n] = heat_np
            heat_np = heat_pad
        # bead-indexed arrays -> pad to B (pad beads map to no anchor / never move)
        bead_to_anchor_k_np = np.concatenate(
            [bead_to_anchor_k_np, np.full(n_pad, -1, dtype=np.int32)]
        )
        if is_L_np.shape[0] == n:  # bead-indexed (use_orn=False); anchor-indexed -> phase 2
            is_L_np = np.concatenate([is_L_np, np.zeros(n_pad, dtype=np.bool_)])
        # movable -> pad to B; n_movable_v bounds the sampler so pads never picked
        movable_np = np.concatenate(
            [movable_np, np.zeros(B - movable_np.shape[0], dtype=movable_np.dtype)]
        )

    bundle = _build_smooth_kernel(n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs)
    (
        _kernel_one_batch,
        kernel_full,
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
        _kernel_full_mp,  # region-batched entry uses this; single-problem path ignores it
    ) = bundle

    pos_k = jnp.asarray(pos_k_np)
    dtn_j = jnp.asarray(dtn_np)
    movable_j = jnp.asarray(movable_np)
    heat_j = jnp.asarray(heat_np)
    anchor_ar_j = jnp.asarray(anchor_ar_np)
    bead_to_anchor_k_j = jnp.asarray(bead_to_anchor_k_np)
    nbr_idx_j = jnp.asarray(nbr_idx_np)
    nbr_w_j = jnp.asarray(nbr_w_np)
    nbr_valid_j = jnp.asarray(nbr_valid_np)
    is_L_j = jnp.asarray(is_L_np)
    n_active_j = jnp.int32(n_active_v)
    n_movable_active_j = jnp.int32(n_movable_v)
    # Per-call RNG diversity keyed on the active scope path (was: the label).
    _seed_src = log.current()
    seed_offset: int = abs(hash(_seed_src)) % (2**31) if _seed_src else 0

    # ---- initial scores ----
    ss_k = init_smooth(
        pos_k,
        dtn_j,
        jnp.float32(settings.spring_stretch),
        jnp.float32(settings.spring_squeeze),
        jnp.float32(settings.spring_angular),
        jnp.float32(settings.smooth_dist_weight),
        jnp.float32(settings.smooth_angle_weight),
        n_active_j,
    )
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v), n_active_j)
        if use_excl
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sh_k = (
        init_heat(pos_k, heat_j, jnp.float32(heat_weight_v))
        if use_heat
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sc_k = (
        init_confine(
            pos_k,
            jnp.float32(conf_cx_v),
            jnp.float32(conf_cy_v),
            jnp.float32(conf_cz_v),
            jnp.float32(conf_R_v),
            jnp.float32(conf_w_v),
            n_active_j,
        )
        if use_conf
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    if use_orn:
        anchor_orn_k = init_anchor_orn(pos_k, anchor_ar_j, is_L_j)
        so_k = init_orn_score(
            anchor_orn_k,
            nbr_idx_j,
            nbr_w_j,
            nbr_valid_j,
            jnp.float32(motif_weight_v),
            jnp.bool_(motifs_symmetric_v),
        )
    else:
        anchor_orn_k = jnp.zeros((K, n_anchors, 3), dtype=jnp.float32)
        so_k = jnp.zeros((K,), dtype=jnp.float32)

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
    heat_w_j = jnp.float32(heat_weight_v)
    motif_w_j = jnp.float32(motif_weight_v)
    symmetric_j = jnp.bool_(motifs_symmetric_v)
    conf_cx_j = jnp.float32(conf_cx_v)
    conf_cy_j = jnp.float32(conf_cy_v)
    conf_cz_j = jnp.float32(conf_cz_v)
    conf_R_j = jnp.float32(conf_R_v)
    conf_w_j = jnp.float32(conf_w_v)
    step_size_j = jnp.float32(step_size)

    stop_improvement = jnp.float32(settings.mc_stop_improvement_smooth)
    stop_successes = jnp.int32(settings.mc_stop_successes_smooth)
    score_eps = jnp.float32(1e-6)
    base_key = jax.random.PRNGKey(seed_offset)

    # ONE JAX call drives the full convergence loop on device.  No per-batch
    # Python sync.  Returns final state + (iter_count, converged) so we can
    # log how the run terminated.
    (
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        _anchor_orn_k_final,
        final_score_best,
        iter_count,
        converged_flag,
    ) = kernel_full(
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        anchor_orn_k,
        T,
        dtn_j,
        movable_j,
        heat_j,
        anchor_ar_j,
        bead_to_anchor_k_j,
        nbr_idx_j,
        nbr_w_j,
        nbr_valid_j,
        is_L_j,
        step_size_j,
        dt,
        js,
        jc,
        stretch_k_j,
        squeeze_k_j,
        ang_k_j,
        dist_w_j,
        ang_w_j,
        r0_j,
        excl_w_j,
        heat_w_j,
        motif_w_j,
        symmetric_j,
        conf_cx_j,
        conf_cy_j,
        conf_cz_j,
        conf_R_j,
        conf_w_j,
        base_key,
        stop_improvement,
        stop_successes,
        score_eps,
        n_active_j,
        n_movable_active_j,
    )

    score_per_chain = np.asarray(ss_k + se_k + sh_k + so_k + sc_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)

    if LOG.isEnabledFor(logging.DEBUG):
        tail = "[done]" if converged_v else "[max-iters reached]"
        LOG.debug(
            "step %7s  score=%.4f  batches=%d  %s",
            f"{iter_n * n_steps_per_batch:,}",
            float(final_score_best),
            iter_n,
            tail,
        )

    if return_all:
        # Batched mode: hand back every chain's score + final positions; the
        # caller selects per-trial.  Slice off bucket padding (B -> n).  Do NOT
        # mutate `pos`.
        return (
            score_per_chain.astype(np.float64),
            np.asarray(pos_k[:, :n]).astype(np.float32),
        )

    best_k: int = int(np.argmin(score_per_chain))
    # Slice off any bucket padding (pos is (n, 3); pos_k is (K, B, 3), B >= n).
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])


def _prep_smooth_problem_np(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None,
    anchor_neighbors: dict[int, list[int]] | None,
    anchor_neighbor_weights: dict[int, list[float]] | None,
    heat_dist: np.ndarray[Any, Any] | None,
    B: int,
    A: int,
    M: int,
) -> dict[str, Any]:
    """Build one IB's kernel inputs as numpy arrays, padded to a common bucket
    (B beads, A anchors, M neighbours) so a batch of IBs has uniform shapes.

    Pure numpy (no JAX) → unit-testable in isolation.  Mirrors the per-problem
    prep inside `mc_smooth_jax` exactly; the only difference is that B/A/M are
    passed in (the batch's common bucket) instead of derived per-region, and
    arrays are always padded to them (the batched kernel needs uniform shapes
    regardless of the `mc_executor_jax_bucket_shapes` flag).
    """
    n = int(pos.shape[0])
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_heat = heat_dist is not None
    use_orn = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)

    # --- excluded-volume radius (auto-derived per IB if radius<=0) ---
    excl_r0 = 1.0
    if use_excl:
        excl_r0 = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            excl_r0 = float(settings.exclusion_auto_factor_smooth) * float(np.asarray(dtn).mean())

    # --- confinement envelope (centroid + radius, per IB) ---
    if use_conf:
        conf_cx = float(pos[:, 0].mean())
        conf_cy = float(pos[:, 1].mean())
        conf_cz = float(pos[:, 2].mean())
        conf_R = float(settings.confinement_radius_smooth)
        if conf_R <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w = float(settings.confinement_weight)
    else:
        conf_cx = conf_cy = conf_cz = 0.0
        conf_R = 1.0
        conf_w = 0.0

    # --- orientation CSR (anchor-indexed), padded to (A, M) ---
    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar))
        nbr_lists = [list(anchor_neighbors.get(k, [])) for k in range(n_anchors)]
        nbr_w_lists = [list(anchor_neighbor_weights.get(k, [])) for k in range(n_anchors)]
        nbr_idx = np.zeros((n_anchors, M), dtype=np.int32)
        nbr_w = np.zeros((n_anchors, M), dtype=np.float32)
        nbr_valid = np.zeros((n_anchors, M), dtype=np.bool_)
        for k_idx in range(n_anchors):
            for m, (jn, wn) in enumerate(zip(nbr_lists[k_idx], nbr_w_lists[k_idx], strict=True)):
                nbr_idx[k_idx, m] = int(jn)
                nbr_w[k_idx, m] = float(wn)
                nbr_valid[k_idx, m] = True
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        for k_idx in range(n_anchors):
            ar = int(anchor_ar[k_idx])
            if ar > 0:
                bead_to_anchor_k[ar - 1] = k_idx
            if ar + 1 < n:
                bead_to_anchor_k[ar + 1] = k_idx
        is_L = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # pad anchor-indexed arrays to A (pad anchors get nbr_valid=False -> 0 score)
        if A > n_anchors:
            ap = A - n_anchors
            anchor_ar = np.concatenate([anchor_ar, np.zeros(ap, dtype=np.int32)])
            nbr_idx = np.pad(nbr_idx, ((0, ap), (0, 0)))
            nbr_w = np.pad(nbr_w, ((0, ap), (0, 0)))
            nbr_valid = np.pad(nbr_valid, ((0, ap), (0, 0)))
    else:
        anchor_ar = np.zeros(A, dtype=np.int32)
        nbr_idx = np.zeros((A, M), dtype=np.int32)
        nbr_w = np.zeros((A, M), dtype=np.float32)
        nbr_valid = np.zeros((A, M), dtype=np.bool_)
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        is_L = np.zeros(n, dtype=np.bool_)

    # --- bead-indexed arrays, padded to B ---
    movable = np.ascontiguousarray(np.where(~fixed)[0], dtype=np.int64)
    n_movable = int(movable.shape[0])
    pos_pad = pos.astype(np.float32)
    dtn_pad = dtn.astype(np.float32)
    heat_pad = heat_dist.astype(np.float32) if use_heat else np.zeros((1, 1), dtype=np.float32)
    if B > n:
        n_pad = B - n
        pos_pad = np.concatenate([pos_pad, np.zeros((n_pad, 3), dtype=np.float32)], axis=0)
        dtn_pad = np.concatenate([dtn_pad, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat:
            hp = np.zeros((B, B), dtype=np.float32)
            hp[:n, :n] = heat_pad
            heat_pad = hp
        bead_to_anchor_k = np.concatenate([bead_to_anchor_k, np.full(n_pad, -1, dtype=np.int32)])
        is_L = np.concatenate([is_L, np.zeros(n_pad, dtype=np.bool_)])

    # `movable` lists non-fixed beads, so its length is n_movable (not n) - pad
    # it to B independently of the `B > n` bead-padding above, else an IB whose
    # n lands exactly on the bucket (B == n) keeps a short movable and the
    # batched (K, B) stack goes ragged.  Pad indices are 0 (a valid bead);
    # n_movable bounds the kernel loop, so they're never read.
    if B > n_movable:
        movable = np.concatenate([movable, np.zeros(B - n_movable, dtype=movable.dtype)])

    return {
        "n": n,
        "pos": pos_pad,  # (B, 3)
        "dtn": dtn_pad,  # (B,)
        "movable": movable,  # (B,)
        "heat": heat_pad,  # (B, B) or (1, 1)
        "anchor_ar": anchor_ar,  # (A,)
        "bead_to_anchor_k": bead_to_anchor_k,  # (B,)
        "nbr_idx": nbr_idx,  # (A, M)
        "nbr_w": nbr_w,  # (A, M)
        "nbr_valid": nbr_valid,  # (A, M)
        "is_L": is_L,  # (B,)
        "n_active": n,
        "n_movable": n_movable,
        "excl_r0": excl_r0,
        "conf_cx": conf_cx,
        "conf_cy": conf_cy,
        "conf_cz": conf_cz,
        "conf_R": conf_R,
        "conf_w": conf_w,
    }


def _smooth_tensor_bytes(B: int, A: int, M: int, use_heat: bool, use_orn: bool) -> int:
    """Exact device-tensor bytes for ONE IB of the smooth kernel at (B, A, M).

    Sums every input and output array of `kernel_full_mp` (each is stacked on
    axis 0, so the whole batch is K times this).  Dominated by the (B, B) heat
    tensor when heat is on; orn adds the (A, *) anchor/neighbor arrays.  Used by
    `_resolve_smooth_max_k` with `memory.XLA_PEAK_OVERHEAD` to bound the peak."""
    f4, i4, b1 = 4, 4, 1  # bytes: float32 / int32 / bool
    A_ = A if use_orn else 1
    M_ = M if use_orn else 1
    inp = (
        B * 3 * f4  # pos_k
        + 5 * f4  # ss/se/sh/so/sc
        + A_ * 3 * f4  # anchor_orn_k
        + B * f4  # dtn_k
        + B * i4  # movable_k
        + (B * B * f4 if use_heat else f4)  # heat_k
        + A_ * i4  # anchor_ar_k
        + B * i4  # bead_to_anchor_k
        + A_ * M_ * (i4 + f4 + b1)  # nbr idx / w / valid
        + B * b1  # is_L_k
        + f4  # step_size_k
        + 6 * f4  # excl_r0 + conf cx,cy,cz,R,w
        + 2 * i4  # n_active / n_movable
    )
    out = B * 3 * f4 + 5 * f4 + A_ * 3 * f4 + b1  # pos_f, scores, anchor_orn, converged
    return inp + out


def _resolve_smooth_max_k(
    big_b: int, big_a: int, big_m: int, use_heat: bool, use_orn: bool, settings: "Settings"
) -> tuple[int, str]:
    """Resolve the smooth region-batch vmap width (IBs per launch).

    `settings.mc_executor_jax_batch_width_smooth` is either an integer (a flat
    cap) or "auto".  "auto" computes the kernel's exact per-IB device-tensor bytes
    (`_smooth_tensor_bytes`), applies `memory.XLA_PEAK_OVERHEAD`, and solves the
    largest K within the device budget (basis "auto-bytes").  When the budget
    can't be queried (CPU backend) it falls back to the conservative `32768/B`
    shape heuristic (basis "auto-fallback").  Returns (max_k, basis)."""
    w = str(settings.mc_executor_jax_batch_width_smooth).strip().lower()
    if w != "auto":
        return max(1, int(w)), "explicit"
    budget = jax_device_budget_bytes()
    if budget is None:
        return max(1, 32768 // max(1, big_b)), "auto-fallback"
    per_ib = _smooth_tensor_bytes(big_b, big_a, big_m, use_heat, use_orn)
    return max_k_for_bytes(per_ib, 0, budget), "auto-bytes"


def mc_smooth_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Anneal K *different* IBs in one vmapped kernel (region batching).

    `problems` is a list of dicts, each describing one IB's final smooth:
        pos (n,3), dtn (n,), fixed (n,) bool, step_size (float),
        and optionally heat_dist (n,n), char_orientations (n,),
        anchor_neighbors, anchor_neighbor_weights.
    All problems must share the same energy-term flags (use_heat/use_orn/...)
    - the caller groups IBs by (terms, size bucket) before calling.

    Returns one (score, final_pos (n_i, 3)) per problem, in input order.
    Does not mutate the inputs.

    Caps the vmap width (IBs per launch) via `settings.mc_executor_jax_batch_width_smooth`
    - see `_resolve_smooth_max_k`.  Excess IBs run in sequential sub-batches.  The
    cap is purely an OOM guard (a wider launch is never slower than more serial
    sub-batches); "auto" sizes it to what device memory allows.
    """
    if not problems:
        return []
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    bsz = [int(p["pos"].shape[0]) for p in problems]
    big_b = max((jax_bucket_for(n) if bucket else n) for n in bsz)
    use_heat = problems[0].get("heat_dist") is not None
    use_orn = (
        problems[0].get("char_orientations") is not None
        and problems[0].get("anchor_neighbors") is not None
        and problems[0].get("anchor_neighbor_weights") is not None
    )
    # Group maxes of the anchor (A) and neighbor-width (M) axes, matching the
    # padding the chunk applies, so the footprint model sees the real shapes.
    big_a, big_m = 1, 1
    if use_orn:
        a_list, m_list = [], []
        for p in problems:
            anchors_i = int(np.count_nonzero(p["fixed"]))
            nbrs_i = max(
                (len(p["anchor_neighbors"].get(k, [])) for k in range(anchors_i)), default=1
            )
            a_list.append(jax_bucket_for(anchors_i, ANCHOR_BUCKETS) if bucket else anchors_i)
            m_list.append(jax_bucket_for(max(nbrs_i, 1), NBR_BUCKETS) if bucket else max(nbrs_i, 1))
        big_a, big_m = max(a_list), max(m_list)
    max_k, basis = _resolve_smooth_max_k(big_b, big_a, big_m, use_heat, use_orn, settings)
    if len(problems) <= max_k:
        return _mc_smooth_jax_batch_chunk(problems, settings)
    LOG.debug(
        "region-batch[smooth]: %d IBs > max_k=%d (%s) at B=%d heat=%s; running %d sub-batches",
        len(problems),
        max_k,
        basis,
        big_b,
        use_heat,
        -(-len(problems) // max_k),
    )
    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i in range(0, len(problems), max_k):
        results.extend(_mc_smooth_jax_batch_chunk(problems[i : i + max_k], settings))
    return results


def _mc_smooth_jax_batch_chunk(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """One vmapped kernel launch for up to `max_k` IBs. See `mc_smooth_jax_batch`,
    which chunks to this so the device tensors stay bounded."""
    if not jax_is_available():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    import jax
    import jax.numpy as jnp

    K = len(problems)
    if K == 0:
        return []

    # Common bucket across the group: pad every IB to the max (B, A, M) so the
    # kernel sees one uniform shape.  Callers should pre-group by bucket so
    # these maxes are tight (wall-clock = slowest IB in the batch).
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    use_orn = (
        problems[0].get("char_orientations") is not None
        and problems[0].get("anchor_neighbors") is not None
        and problems[0].get("anchor_neighbor_weights") is not None
    )
    use_heat = problems[0].get("heat_dist") is not None
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)

    Bs, As, Ms = [], [], []
    for p in problems:
        n_i = int(p["pos"].shape[0])
        Bs.append(jax_bucket_for(n_i) if bucket else n_i)
        if use_orn:
            anchors_i = int(np.count_nonzero(p["fixed"]))
            nbrs_i = max(
                (len(p["anchor_neighbors"].get(k, [])) for k in range(anchors_i)), default=1
            )
            nbrs_i = max(nbrs_i, 1)
            As.append(jax_bucket_for(anchors_i, ANCHOR_BUCKETS) if bucket else anchors_i)
            Ms.append(jax_bucket_for(nbrs_i, NBR_BUCKETS) if bucket else nbrs_i)
        else:
            As.append(1)
            Ms.append(1)
    B, A, M = max(Bs), max(As), max(Ms)

    preps = [
        _prep_smooth_problem_np(
            p["pos"],
            p["dtn"],
            p["fixed"],
            settings,
            p.get("char_orientations"),
            p.get("anchor_neighbors"),
            p.get("anchor_neighbor_weights"),
            p.get("heat_dist"),
            B,
            A,
            M,
        )
        for p in problems
    ]

    # --- stack per-IB arrays -> (K, ...) ---
    def stack(key: str) -> Any:
        return jnp.asarray(np.stack([pr[key] for pr in preps], axis=0))

    pos_k = stack("pos")  # (K, B, 3)
    dtn_k = stack("dtn")
    movable_k = stack("movable")
    heat_k = stack("heat")
    anchor_ar_k = stack("anchor_ar")
    b2a_k = stack("bead_to_anchor_k")
    nbr_idx_k = stack("nbr_idx")
    nbr_w_k = stack("nbr_w")
    nbr_valid_k = stack("nbr_valid")
    is_L_k = stack("is_L")
    n_active_k = jnp.asarray(np.array([pr["n_active"] for pr in preps], dtype=np.int32))
    n_movable_k = jnp.asarray(np.array([pr["n_movable"] for pr in preps], dtype=np.int32))
    excl_r0_k = jnp.asarray(np.array([pr["excl_r0"] for pr in preps], dtype=np.float32))
    conf_cx_k = jnp.asarray(np.array([pr["conf_cx"] for pr in preps], dtype=np.float32))
    conf_cy_k = jnp.asarray(np.array([pr["conf_cy"] for pr in preps], dtype=np.float32))
    conf_cz_k = jnp.asarray(np.array([pr["conf_cz"] for pr in preps], dtype=np.float32))
    conf_R_k = jnp.asarray(np.array([pr["conf_R"] for pr in preps], dtype=np.float32))
    conf_w_k = jnp.asarray(np.array([pr["conf_w"] for pr in preps], dtype=np.float32))
    step_size_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    # per-IB springs (uniform from settings for now; small-IB boost groups
    # would carry their own - callers batch boosted IBs separately).
    stretch_k = jnp.full((K,), jnp.float32(settings.spring_stretch))
    squeeze_k = jnp.full((K,), jnp.float32(settings.spring_squeeze))
    ang_k = jnp.full((K,), jnp.float32(settings.spring_angular))

    # --- shared (global) schedule + weights ---
    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps_per_batch = int(settings.mc_stop_steps_smooth)
    heat_weight_v = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    motif_weight_v = float(settings.motif_weight) if use_orn else 0.0
    excl_w_v = float(settings.exclusion_weight) if use_excl else 0.0

    bundle = _build_smooth_kernel(n_steps_per_batch, excl_skip, use_heat, use_orn, M)
    (
        _kb,
        _kf,
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
        kernel_full_mp,
    ) = bundle

    # --- per-IB initial scores (one-shot; reuse the validated init helpers) ---
    dist_w = jnp.float32(settings.smooth_dist_weight)
    ang_w = jnp.float32(settings.smooth_angle_weight)
    symmetric = jnp.bool_(bool(getattr(settings, "motifs_symmetric", True)))

    def init_one(i: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        p1 = pos_k[i : i + 1]  # (1, B, 3)
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        ss = init_smooth(p1, dtn_k[i], stretch_k[i], squeeze_k[i], ang_k[i], dist_w, ang_w, na)
        se = (
            init_excl(p1, excl_r0_k[i], jnp.float32(excl_w_v), na)
            if use_excl
            else jnp.zeros((1,), jnp.float32)
        )
        sh = (
            init_heat(p1, heat_k[i], jnp.float32(heat_weight_v))
            if use_heat
            else jnp.zeros((1,), jnp.float32)
        )
        sc = (
            init_confine(p1, conf_cx_k[i], conf_cy_k[i], conf_cz_k[i], conf_R_k[i], conf_w_k[i], na)
            if use_conf
            else jnp.zeros((1,), jnp.float32)
        )
        if use_orn:
            ao = init_anchor_orn(p1, anchor_ar_k[i], is_L_k[i])
            so = init_orn_score(
                ao, nbr_idx_k[i], nbr_w_k[i], nbr_valid_k[i], jnp.float32(motif_weight_v), symmetric
            )
        else:
            ao = jnp.zeros((1, A, 3), jnp.float32)
            so = jnp.zeros((1,), jnp.float32)
        return ss, se, sh, so, sc, ao

    inits = [init_one(i) for i in range(K)]
    ss_k = jnp.concatenate([x[0] for x in inits])
    se_k = jnp.concatenate([x[1] for x in inits])
    sh_k = jnp.concatenate([x[2] for x in inits])
    so_k = jnp.concatenate([x[3] for x in inits])
    sc_k = jnp.concatenate([x[4] for x in inits])
    anchor_orn_k = jnp.concatenate([x[5] for x in inits])

    _seed_src = log.current()
    seed_offset = abs(hash(_seed_src)) % (2**31) if _seed_src else 0
    base_key = jax.random.PRNGKey(seed_offset)

    log.status(
        LOG,
        "    smooth kernel: K=%d B=%d A=%d M=%d (heat=%d orn=%d), compiling/running...",
        K,
        B,
        A,
        M,
        int(use_heat),
        int(use_orn),
    )
    t0 = time.perf_counter()
    out = kernel_full_mp(
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        anchor_orn_k,
        jnp.float32(settings.max_temp_smooth),
        dtn_k,
        movable_k,
        heat_k,
        anchor_ar_k,
        b2a_k,
        nbr_idx_k,
        nbr_w_k,
        nbr_valid_k,
        is_L_k,
        step_size_k,
        jnp.float32(settings.dt_temp_smooth),
        jnp.float32(settings.jump_scale_smooth),
        jnp.float32(settings.jump_coef_smooth),
        stretch_k,
        squeeze_k,
        ang_k,
        dist_w,
        ang_w,
        excl_r0_k,
        jnp.float32(excl_w_v),
        jnp.float32(heat_weight_v),
        jnp.float32(motif_weight_v),
        symmetric,
        conf_cx_k,
        conf_cy_k,
        conf_cz_k,
        conf_R_k,
        conf_w_k,
        base_key,
        jnp.float32(settings.mc_stop_improvement_smooth),
        jnp.int32(settings.mc_stop_successes_smooth),
        jnp.float32(1e-6),
        n_active_k,
        n_movable_k,
    )
    pos_f, ss_f, se_f, sh_f, so_f, sc_f, _ao_f, _final_score, iter_count, converged = out
    score_per_chain = np.asarray(ss_f + se_f + sh_f + so_f + sc_f)  # forces device sync
    pos_f_np = np.asarray(pos_f)

    n_steps_smooth = int(settings.mc_stop_steps_smooth)
    log.status(
        LOG,
        "    smooth kernel: K=%d B=%d A=%d M=%d (heat=%d orn=%d), "
        "%d batches (%d steps), %d/%d converged, %.1fs",
        K,
        B,
        A,
        M,
        int(use_heat),
        int(use_orn),
        int(iter_count),
        int(iter_count) * n_steps_smooth,
        int(np.asarray(converged).sum()),
        K,
        time.perf_counter() - t0,
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        n_i = pr["n"]
        results.append((float(score_per_chain[i]), pos_f_np[i, :n_i].astype(np.float32)))
    return results
