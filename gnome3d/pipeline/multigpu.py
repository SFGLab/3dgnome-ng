"""Multi-GPU batch sharding for the JAX batch strategy.

The engine never touches SLURM or job submission - those stay external.  It simply uses whatever
``jax.devices()`` reports (the GPUs the launcher made visible, e.g. via CUDA_VISIBLE_DEVICES) and
shards work across them.  A batch's IBs are independent MC problems, so a group splits cleanly:
each device gets a contiguous slice and runs its own sub-launch concurrently.  ``jax.default_device``
is thread-local (validated), so a per-device worker thread places its sub-launch on its device with
no cross-talk; each worker also runs in a COPY of the caller's context so it inherits the log scope
(the kernels seed ``base_key`` from ``log.current()``, a ContextVar empty in a fresh thread).

Reproducibility note: the batched kernels key each IB's per-step RNG on its POSITION in the launch,
so a different split (i.e. a different GPU count) yields a statistically-equivalent but NOT
byte-identical random realisation vs single-device.  Multi-GPU is purely a production-speed lever;
the fixed-seed reproducibility gate runs single-GPU.
"""

import contextvars
import threading
from collections.abc import Callable, Sequence
from typing import Any


def visible_devices(limit: int = 0) -> list[Any]:
    """JAX devices to shard across: GPUs if any are present, else all devices (covers the CPU
    multi-device simulation used for local tests).  ``limit`` > 0 caps the count."""
    import jax

    devs = list(jax.devices())
    gpus = [d for d in devs if getattr(d, "platform", "") in ("gpu", "cuda", "rocm")]
    devs = gpus or devs
    return devs[:limit] if limit > 0 else devs


def _spans(n_items: int, n_parts: int) -> list[tuple[int, int]]:
    """Balanced contiguous ``[lo, hi)`` ranges over ``n_items`` - remainder spread over the first
    parts so chunk sizes differ by at most one (the IBs in a group are equal-shape => equal work)."""
    base, rem = divmod(n_items, n_parts)
    out: list[tuple[int, int]] = []
    lo = 0
    for i in range(n_parts):
        hi = lo + base + (1 if i < rem else 0)
        if hi > lo:
            out.append((lo, hi))
        lo = hi
    return out


def run_sharded(
    batch_fn: Callable[[list[Any]], list[Any]],
    problems: list[Any],
    devices: Sequence[Any],
) -> list[Any]:
    """Run ``batch_fn(problems)`` split across ``devices``, one concurrent sub-launch per device,
    returning results in the ORIGINAL problem order.  Falls back to a single in-line call for one
    device / one problem.  Pure data-parallel - the IBs are independent, so there is no
    cross-device communication; each device just gets a slice."""
    import jax

    n = min(len(devices), len(problems))
    if n <= 1:
        return batch_fn(problems)

    spans = _spans(len(problems), n)
    results: list[list[Any] | None] = [None] * len(spans)
    errors: list[BaseException | None] = [None] * len(spans)

    def _worker(i: int, lo: int, hi: int) -> None:
        try:
            with jax.default_device(devices[i]):
                results[i] = batch_fn(problems[lo:hi])
        except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread below
            errors[i] = exc

    # Run each worker inside a COPY of the main thread's context so it inherits the log scope
    # (a ContextVar, empty in a fresh thread): the kernels seed their RNG from log.current(), so
    # without this the workers' base_key would diverge from the single-device path.
    threads = [
        threading.Thread(
            target=contextvars.copy_context().run, args=(_worker, i, lo, hi), name=f"mgpu-{i}"
        )
        for i, (lo, hi) in enumerate(spans)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for exc in errors:
        if exc is not None:
            raise exc

    out: list[Any] = []
    for r in results:
        out.extend(r or [])
    return out
