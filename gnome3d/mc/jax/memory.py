"""Device-memory sizing for the region-batched JAX kernels (smooth, arcs).

We size the vmap width K so the kernel's peak device memory fits the budget.
The peak cannot be predicted exactly, and we learned why the hard way:

  * XLA's static `compiled.memory_analysis()` UNDERREPORTS the runtime peak - it
    omits the compute scratch and the internal buffer duplication XLA does, so
    sizing off it picked a K that OOM'd (it saw ~1x the tensor bytes; the kernel
    needed ~3x).
  * Summing the tensor shapes by hand also can't match XLA: its own scheduler
    counts the input/output arguments at ~2x the exact tensor bytes (buffer
    duplication), and then allocates compute scratch on top.

What IS exactly computable is the total bytes of every input/output tensor (their
shapes are known).  XLA's overhead on top of that is an empirical constant,
`XLA_PEAK_OVERHEAD`, calibrated from a measured OOM.  Each kernel reports its
exact per-IB / fixed tensor bytes; `max_k_for_bytes` applies the overhead and
solves for K.  No per-tensor fudge, no guessed `B^2` model - one honest factor on
an exact byte count.
"""

from __future__ import annotations

# Real peak device bytes / exact tensor bytes.  Calibrated from an observed OOM:
# a K=16, B=16384 heat+orn smooth launch had 16 GiB of tensors, but XLA's
# scheduler counted 32 GiB of I/O args (2x: buffer duplication) and the runtime
# then tried a further 16 GiB of scratch -> ~48 GiB real peak = 2.99x the tensor
# bytes.  Rounded up to 3.5 for margin (fragmentation, the smaller terms).  This
# is a pure OOM guard: too-low K only adds serial sub-batches; an OOM kills the
# run.  If a heat group still OOMs, this is the one number to raise.
XLA_PEAK_OVERHEAD = 3.5


def max_k_for_bytes(per_ib: int, fixed: int, budget: int) -> int:
    """Largest K with `(fixed + per_ib*K) * XLA_PEAK_OVERHEAD <= budget`.

    `per_ib` / `fixed` are the exact device-tensor bytes that do / don't scale
    with the vmap width K (a kernel computes them from its known shapes)."""
    if per_ib <= 0:
        return 1
    avail = budget / XLA_PEAK_OVERHEAD - fixed
    if avail <= 0:
        return 1
    return max(1, int(avail // per_ib))
