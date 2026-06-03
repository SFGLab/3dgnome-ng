"""Device-memory sizing for the region-batched JAX kernels (smooth, arcs).

Each kernel vmaps K independent IBs on axis 0, so its peak device memory is
~affine in K: `peak(K) = fixed + per_ib*K`.  Rather than hand-model the bytes -
the (K,B,B) heat/exp tensor plus whatever XLA duplicates and scratches, a
multiple that varies by shape and backend - we MEASURE the compiled executable:
lower+compile at K=1 and K=2 (abstract `ShapeDtypeStruct` shapes, so nothing is
allocated), read XLA's `memory_analysis()`, fit the line, and solve for the
largest K within budget.

A kernel plugs in by supplying `peak_at(K) -> int | None` (build its lowering
args at width K, lower, compile, then `compiled_peak_bytes`).  The fit + budget
solve + caching live here, shared across kernels; only the arg-builder, which is
necessarily specific to each kernel's signature, lives in the kernel module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def compiled_peak_bytes(compiled: Any) -> int | None:
    """Peak device bytes of a compiled XLA executable, from its
    `memory_analysis()`: input arguments + outputs + temporary scratch, less the
    aliased buffers that are shared between input and output (donated).  `None`
    when the analysis is unavailable (older jaxlib / backend)."""
    try:
        ma = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 - any backend without memory_analysis
        return None
    if ma is None:
        return None

    def g(name: str) -> int:
        return int(getattr(ma, name, 0) or 0)

    return (
        g("argument_size_in_bytes")
        + g("output_size_in_bytes")
        + g("temp_size_in_bytes")
        - g("alias_size_in_bytes")
    )


def measured_max_k(
    peak_at: Callable[[int], int | None],
    budget: int,
    cache: dict[Any, tuple[int, int]],
    key: Any,
) -> int | None:
    """Largest vmap width K whose measured peak device memory fits `budget`.

    Measures `peak_at` at K=1 and K=2, fits `peak(K) = fixed + per_ib*K`, and
    returns `max(1, (budget - fixed) // per_ib)`.  The two coefficients are
    cached by `key` so each shape/term signature is measured once.  `None` if
    measurement fails (so the caller can fall back to an analytic model)."""
    coeffs = cache.get(key)
    if coeffs is None:
        m1 = peak_at(1)
        m2 = peak_at(2)
        if m1 is None or m2 is None:
            return None
        per_ib = max(1, m2 - m1)  # marginal bytes per added IB
        fixed = max(0, m1 - per_ib)  # K-independent overhead
        coeffs = (per_ib, fixed)
        cache[key] = coeffs
    per_ib, fixed = coeffs
    avail = budget - fixed
    if avail <= 0:
        return 1
    return max(1, int(avail // per_ib))
