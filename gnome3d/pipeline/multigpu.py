"""Runs batch launches on several JAX devices.

The engine uses the devices `jax.devices()` reports. `jax.default_device` is thread local.
One worker thread per device places its launches there. Each worker runs in a copy of the
caller's context. The kernels seed from `log.current()` and a new thread has an empty
scope.
"""

import contextvars
import threading
from collections.abc import Callable, Sequence
from typing import Any


def visible_devices(limit: int = 0) -> list[Any]:
    """Returns the JAX devices to use, the GPUs when there are any and every device otherwise.
    `limit` above 0 caps the count.
    """
    import jax

    from gnome3d.mc.jax.util import jax_is_available

    jax_is_available()

    devs = list(jax.devices())
    gpus = [d for d in devs if getattr(d, "platform", "") in ("gpu", "cuda", "rocm")]
    devs = gpus or devs
    return devs[:limit] if limit > 0 else devs


def _spans(n_items: int, n_parts: int) -> list[tuple[int, int]]:
    """Returns balanced contiguous `[lo, hi)` ranges over `n_items`, sizes at most one apart."""
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
    """Runs `batch_fn(problems)` split across `devices` and returns the results in the order of
    `problems`.

    The kernels key the RNG of each IB on its position in the launch. The split changes the
    random draw.
    """
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

    # Each worker runs in a copy of the caller's context. The kernels seed from `log.current()`,
    # which is empty in a new thread.
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


def run_group_parallel(
    group_fn: Callable[[int], None],
    n_groups: int,
    devices: Sequence[Any],
) -> None:
    """Runs `group_fn(gi)` for every group index across `devices`.

    One worker thread per device takes the next unclaimed group and runs it with that device as
    `jax.default_device`. A group stays whole on one device. The results are byte identical to
    a run on one device.

    `group_fn` must be safe in a worker thread and must not touch shared scheduler state. The
    caller applies the results in group order.
    """
    import jax

    n_dev = min(len(devices), n_groups)
    if n_dev <= 1:
        for gi in range(n_groups):
            group_fn(gi)
        return

    next_gi = 0
    lock = threading.Lock()
    errors: list[BaseException] = []

    def _worker(dev: Any) -> None:
        nonlocal next_gi
        while True:
            with lock:
                if next_gi >= n_groups or errors:
                    return
                gi = next_gi
                next_gi += 1
            try:
                with jax.default_device(dev):
                    group_fn(gi)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread below
                with lock:
                    errors.append(exc)
                return

    threads = [
        threading.Thread(
            target=contextvars.copy_context().run, args=(_worker, devices[i]), name=f"mgpu-g{i}"
        )
        for i in range(n_dev)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
