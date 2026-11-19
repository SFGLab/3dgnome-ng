"""Device memory sizing for the batched JAX kernels.

The vmap width K is the largest whose peak device memory fits the budget. The
total bytes of the input and output tensors are known exactly. XLA needs a multiple of that,
`XLA_PEAK_OVERHEAD`, which is measured. `compiled.memory_analysis()` reports less than the
peak at run time and is not used.
"""

from __future__ import annotations

# Peak device bytes over exact tensor bytes. A measured launch needed 2.99 times its tensor bytes. A
# K that is too low only adds sub-batches and an out of memory error ends the run.
XLA_PEAK_OVERHEAD = 3.5


def max_k_for_bytes(per_ib: int, fixed: int, budget: int) -> int:
    """Returns the largest K with `(fixed + per_ib * K) * XLA_PEAK_OVERHEAD <= budget`.

    `per_ib` and `fixed` are the device tensor bytes that scale and do not scale with K.
    """
    if per_ib <= 0:
        return 1
    avail = budget / XLA_PEAK_OVERHEAD - fixed
    if avail <= 0:
        return 1
    return max(1, int(avail // per_ib))


# The share of the available host RAM that one batch may use. The host holds one (K, B, B) staging
# buffer before the transfer.
HOST_BUDGET_FRACTION = 0.5


def host_available_bytes() -> int | None:
    """Returns the available host RAM, `MemAvailable`, or None when it cannot be read. Linux only."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def max_k_for_host(per_ib_host: int) -> int | None:
    """Returns the largest K whose host staging buffer fits `HOST_BUDGET_FRACTION` of the RAM, or
    None when host memory cannot be read.

    The batch stages its inputs on the host and the heat tensor makes that staging larger than
    the device allocation.
    """
    avail = host_available_bytes()
    if avail is None or per_ib_host <= 0:
        return None
    return max(1, int((avail * HOST_BUDGET_FRACTION) // per_ib_host))
