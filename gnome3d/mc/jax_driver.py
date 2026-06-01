"""Generic composeable JAX MC driver.

`build_mc_kernel(terms, n_steps, ...)` returns a region-batched (vmapped +
``lax.while_loop``) MC kernel that composes an ordered *recipe* of energy terms
(``gnome3d.mc.terms``) — replacing the three hand-written ``_build_*_kernel``
builders in ``gnome3d.mc.jax`` with one.  The term tuple is a STATIC closure, so
the per-step ``for term in terms`` loop unrolls at trace time (no runtime
dispatch); XLA sees a fused program equivalent to the old hand-written kernels.

Each term's per-IB parameters travel as its shared ``namedtuple`` (one stacked
instance per term, every leaf shape ``(K, ...)`` — shared scalars/bools broadcast
to ``(K,)``), vmapped on axis 0.  Per-step math mirrors the old kernels exactly:
sample a movable bead (``movable[idx]``; arcs/heatmap pass ``movable=arange``),
propose a uniform displacement, accumulate ``score += delta_factor*(curr-prev)``
per term IN RECIPE ORDER, accept by strict/non-strict Metropolis + the annealed
jump, anneal ``T *= dt``.  RNG draw order (``split(key,3)`` → idx, disp, acc)
matches the old kernels, so the result is byte-identical to the hand-written
kernel a recipe replaces (proven for arcs + smooth).

A *stateful* term (orientation) carries the ``anchor_orn`` cache through the loop
via its ``jax_step`` (recompute-on-move); at most one stateful term per recipe.
Stateless recipes pass a dummy ``anchor_orn``; the orientation branch is then
traced out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gnome3d.mc.terms.base import Term

_MAX_ITERS = 10000
_driver_cache: dict[Any, Any] = {}


def build_mc_kernel(
    terms: Sequence[Term],
    n_steps: int,
    *,
    strict_accept: bool,
    freeze_converged: bool,
) -> Any:
    """Build (and cache) the region-batched kernel for a recipe.

    ``strict_accept``   — ``score_new < score`` (smooth) vs ``<=`` (arcs/heatmap).
    ``freeze_converged``— hold converged chains' state (arcs/heatmap: non-strict
    accept would drift a converged chain worse if it kept stepping).
    """
    terms = tuple(terms)
    key = (
        tuple((t.name, t.delta_factor) for t in terms),
        int(n_steps), bool(strict_accept), bool(freeze_converged),
    )
    if key in _driver_cache:
        return _driver_cache[key]

    import jax
    import jax.numpy as jnp

    n_terms = len(terms)
    factors = tuple(float(t.delta_factor) for t in terms)
    locals_ = tuple(t.jax_local for t in terms)
    steps_ = tuple(t.jax_step for t in terms)  # None for stateless terms

    def chain_batch(
        pos0: Any, scores0: Any, anchor_orn0: Any, T0: Any, term_params: Any, movable: Any,
        step_size: Any, dt: Any, js: Any, jc: Any, key_: Any, n_active: Any, n_movable_active: Any,
    ) -> Any:
        k_p, k_d, k_a = jax.random.split(key_, 3)
        ps = movable[jax.random.randint(k_p, (n_steps,), 0, n_movable_active)]
        disps = jax.random.uniform(
            k_d, (n_steps, 3), minval=-step_size, maxval=step_size, dtype=pos0.dtype
        )
        accs = jax.random.uniform(k_a, (n_steps,), dtype=pos0.dtype)

        def body(i: Any, carry: Any) -> Any:
            pos, scores, anchor_orn, T, n_ok = carry
            p = ps[i]
            old_p = pos[p]
            new_p = old_p + disps[i]
            total = sum(scores)

            anchor_orn_trial = anchor_orn
            new_scores = list(scores)
            for ti in range(n_terms):
                if steps_[ti] is not None:  # stateful term (orientation)
                    delta, anchor_orn_trial = steps_[ti](
                        pos, p, old_p, new_p, term_params[ti], anchor_orn, n_active
                    )
                else:
                    delta = locals_[ti](pos, p, new_p, term_params[ti], n_active) - locals_[ti](
                        pos, p, old_p, term_params[ti], n_active
                    )
                new_scores[ti] = scores[ti] + factors[ti] * delta
            total_new = sum(new_scores)

            ok_unc = total_new < total if strict_accept else total_new <= total
            can_jump = jnp.logical_and(T > 0, total > 0)
            exponent = jnp.clip(-jc * (total_new / jnp.maximum(total, 1e-30)) / jnp.maximum(T, 1e-30), -80.0, 80.0)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, accs[i] < js * jnp.exp(exponent)))

            pos_next = pos.at[p].set(jnp.where(ok, new_p, old_p))
            scores_next = tuple(jnp.where(ok, ns, s) for ns, s in zip(new_scores, scores, strict=True))
            anchor_orn_next = jnp.where(ok, anchor_orn_trial, anchor_orn)
            return (pos_next, scores_next, anchor_orn_next, T * dt, n_ok + jnp.where(ok, 1, 0))

        return jax.lax.fori_loop(
            0, n_steps, body, (pos0, tuple(scores0), anchor_orn0, T0, jnp.int32(0))
        )

    scores_axes = tuple(0 for _ in terms)
    in_axes = (0, scores_axes, 0, None, 0, 0, 0, None, None, None, 0, 0, 0)
    out_axes = (0, scores_axes, 0, None, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    @jax.jit
    def kernel_full_mp(
        pos_k: Any, scores_k: Any, anchor_orn_k: Any, T_init: Any, term_params_k: Any, movable_k: Any,
        step_size_k: Any, dt: Any, js: Any, jc: Any, base_key: Any,
        stop_improvement: Any, stop_successes: Any, score_eps: Any, stop_ratio: Any,
        n_active_k: Any, n_movable_active_k: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            iter_i, converged = state[4], state[5]
            return jnp.logical_and(jnp.logical_not(jnp.all(converged)), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, scores, anchor_orn, ms_score, iter_i, conv_prev, T = state
            keys = jax.random.split(jax.random.fold_in(base_key, iter_i + 1), K)
            npos, nscores, nanchor, nT, n_ok = batched(
                pos, scores, anchor_orn, T, term_params_k, movable_k, step_size_k,
                dt, js, jc, keys, n_active_k, n_movable_active_k,
            )
            if freeze_converged:
                frozen = conv_prev
                pos = jnp.where(frozen[:, None, None], pos, npos)
                scores = tuple(jnp.where(frozen, s, ns) for s, ns in zip(scores, nscores, strict=True))
                anchor_orn = jnp.where(frozen[:, None, None], anchor_orn, nanchor)
            else:
                pos, scores, anchor_orn = npos, nscores, nanchor
            score = sum(scores)
            ratio = score / jnp.maximum(ms_score, 1e-30)
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, n_ok < stop_successes)
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, score < score_eps), ratio > stop_ratio), conv_prev
            )
            return (pos, scores, anchor_orn, score, iter_i + 1, converged, nT)

        init = (
            pos_k, tuple(scores_k), anchor_orn_k,
            jnp.full((K,), 1e30, dtype=jnp.float32), jnp.int32(0), jnp.zeros((K,), dtype=jnp.bool_), T_init,
        )
        pos_f, scores_f, anchor_f, _ms, iter_f, converged_f, _T = jax.lax.while_loop(cond_fn, body_fn, init)
        return pos_f, scores_f, anchor_f, iter_f, converged_f

    _driver_cache[key] = kernel_full_mp
    return kernel_full_mp
