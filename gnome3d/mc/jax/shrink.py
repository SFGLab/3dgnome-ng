"""Host-side batch-shrinking driver for the checkerboard MC kernels.

A batched checker `while_loop` runs every chain until the *last* one converges — and a
`vmap`'d kernel recomputes already-converged ("frozen") lanes every iteration.  On a bucket
of many small IBs the convergence spread is huge (e.g. p50=105 / max=2448 outer-iters), so
~95% of the GPU work is spent re-grinding chains that finished long ago.

This driver removes that waste: it runs the kernel for a *bounded, growing* chunk of
iterations, then on the host drops the chains that converged and re-launches only the
survivors (compacted, padded up to a power-of-two so XLA only specialises ~log2(K) batch
shapes).  The per-chain MC dynamics and convergence criteria are unchanged — only *which*
chains are in each launch differs — so it is a pure speedup (the RNG stream a chain sees
shifts when it's compacted, so results are statistically- not byte-equivalent to the
single-launch path; the checker is already a stochastic approximation).
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def _ceil_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def run_shrinking(
    chunk_fn: Callable[..., Any],
    carry: tuple[Any, ...],
    problem: tuple[Any, ...],
    scalars: tuple[Any, ...],
    base_key: Any,
    *,
    conv_idx: int = 4,
    pos_idx: int = 0,
    score_idx: int = 1,
    conviter_idx: int = 5,
    init_cap: int = 32,
    max_cap: int = 512,
    max_total: int = 100000,
) -> tuple[list[Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    """Drive ``chunk_fn`` to convergence over a shrinking active set.

    ``carry`` is the evolving per-chain state (leading dim = K); ``problem`` the per-chain
    static inputs; ``scalars`` the shared (non-per-chain) constants.  ``chunk_fn(carry,
    problem, scalars, base_key, max_iters, iter_base) -> (carry', iters_run)`` runs the
    batched checker until all lanes converge OR ``max_iters`` is reached.  ``conv_idx`` /
    ``pos_idx`` / ``score_idx`` / ``conviter_idx`` locate those fields in ``carry``.

    Returns ``(pos_per_chain, score_per_chain, conviter_per_chain, total_iters)`` indexed by
    the ORIGINAL chain order."""
    import jax.numpy as jnp

    K = int(carry[pos_idx].shape[0])
    out_pos: list[Any] = [None] * K
    out_score = np.zeros(K, np.float32)
    out_ci = np.zeros(K, np.int32)
    active = list(range(K))

    def _pad(arrays: tuple[Any, ...], bk: int, is_carry: bool) -> tuple[Any, ...]:
        out = []
        for i, a in enumerate(arrays):
            n = a.shape[0]
            if n >= bk:
                out.append(a)
                continue
            tail = (bk - n,) + a.shape[1:]
            if is_carry and i == conv_idx:
                pad = jnp.ones(tail, a.dtype)  # pad lanes are pre-converged -> inert
            else:
                pad = jnp.broadcast_to(a[:1], tail)  # frozen, value irrelevant
            out.append(jnp.concatenate([a, pad], axis=0))
        return tuple(out)

    iter_base, cap, total = 0, init_cap, 0
    while active and iter_base < max_total:
        n = len(active)
        bk = _ceil_pow2(n)
        ncarry, ran = chunk_fn(
            _pad(carry, bk, True), _pad(problem, bk, False), scalars, base_key,
            jnp.int32(cap), jnp.int32(iter_base),
        )
        ran = int(ran)
        iter_base += ran
        total = iter_base
        conv = np.asarray(ncarry[conv_idx])[:n]
        conv_lanes = np.nonzero(conv)[0]
        if conv_lanes.size:
            cpos = np.asarray(ncarry[pos_idx][conv_lanes])
            csc = np.asarray(ncarry[score_idx][conv_lanes])
            cci = np.asarray(ncarry[conviter_idx][conv_lanes])
            for j, li in enumerate(conv_lanes):
                oi = active[int(li)]
                out_pos[oi] = cpos[j]
                out_score[oi] = csc[j]
                out_ci[oi] = cci[j]
        surv = np.nonzero(~conv)[0]
        if surv.size == 0:
            active = []
            break
        surv_j = jnp.asarray(surv)
        carry = tuple(a[surv_j] for a in ncarry[:6])  # gather survivors from padded output
        problem = tuple(a[surv_j] for a in problem)   # problem is size n (pre-pad); surv < n
        active = [active[int(li)] for li in surv]
        cap = min(cap * 2, max_cap)

    # any chains still active hit max_total — take their current (best-effort) state
    if active:
        cpos = np.asarray(carry[pos_idx])
        csc = np.asarray(carry[score_idx])
        cci = np.asarray(carry[conviter_idx])
        for li, oi in enumerate(active):
            out_pos[oi] = cpos[li]
            out_score[oi] = float(csc[li])
            out_ci[oi] = int(cci[li]) if int(cci[li]) > 0 else total
    return out_pos, out_score, out_ci, total
