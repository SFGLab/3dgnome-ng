import logging
import os
import threading

from gnome3d import log

LOG = log.get("mc.jax.util")

_JAX_AVAILABLE: bool | None = None  # None before the first probe
_init_lock = threading.Lock()

# The shape buckets. With `jax_bucket_shapes` the bead count of a kernel is padded to the next
# bucket. That XLA compiles one program per bucket. The steps are x2 up to 4096 and about x1.25
# above. The heat tensor is (N, N) and padding costs the square of the ratio. An N above the
# top bucket compiles at its exact size.
SHAPE_BUCKETS: tuple[int, ...] = (
    256,
    512,
    1024,
    2048,
    4096,
    5120,
    6400,
    8192,
    10240,
    12800,
    16384,
    20480,
    25600,
    32768,
)

# Finer buckets for the anchor count and the neighbour width of the orientation term.
ANCHOR_BUCKETS: tuple[int, ...] = (16, 64, 256, 1024, 4096, 16384)
NBR_BUCKETS: tuple[int, ...] = (4, 8, 16, 32, 64)


def log_kernel_start(
    logger: logging.Logger, stage: str, kernel: str, k: int, b: int, detail: str
) -> None:
    """Logs the start of a JAX kernel, `arcs[mc]: 719 IBs x 256 beads - <detail>, running...`."""
    log.status(
        logger, "    %s[%s]: %d IBs x %d beads - %s, running...", stage, kernel, k, b, detail
    )


def log_kernel_done(
    logger: logging.Logger, stage: str, kernel: str, k: int, secs: float, summary: str
) -> None:
    """Logs the end of a JAX kernel, `arcs[mc]: 719 IBs in 268.1s - <summary>`."""
    log.status(logger, "    %s[%s]: %d IBs in %.1fs - %s", stage, kernel, k, secs, summary)


def jax_is_available() -> bool:
    """Imports JAX on first use. Returns False when it is not installed."""
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is not None:
        return _JAX_AVAILABLE

    with _init_lock:
        if _JAX_AVAILABLE is not None:
            return _JAX_AVAILABLE
        from gnome3d.mc.jax.share import configure_gpu_pool

        configure_gpu_pool()
        try:
            import jax  # type: ignore[import-not-found]
            import jax.numpy as jnp  # type: ignore[import-not-found]
        except ImportError:
            _JAX_AVAILABLE = False
            return False

        cache_dir = os.environ.get("GNOME3D_JAX_CACHE", os.path.expanduser("~/.cache/gnome3d/jax"))
        cache_active = False
        try:
            from jax.experimental import compilation_cache  # type: ignore[import-not-found]

            compilation_cache.compilation_cache.set_cache_dir(cache_dir)  # pyright: ignore[reportUnknownMemberType]
            cache_active = True
        except (ImportError, AttributeError):
            pass

        _JAX_AVAILABLE = True

        try:
            backend: str = str(jax.default_backend())
            _dev = jax.devices()
            devices_str: str = ", ".join(str(d) for d in _dev)
        except Exception:  # noqa: BLE001
            backend = "unknown"
            devices_str = "unknown"

        cache_str = cache_dir if cache_active else "disabled"

        log.status(
            LOG,
            "JAX backend ready: backend=%s devices=[%s] cache=%s",
            backend,
            devices_str,
            cache_str,
        )

        return True


_DEVICE_TOTAL_BYTES: int | None = None
_DEVICE_PROBED: bool = False
_BUDGET_LOGGED: bool = False


def _physical_device_total_bytes() -> int | None:
    """Returns the total bytes of GPU 0 by NVML or nvidia-smi, cached, or None. The vmm and
    platform allocators report an empty `memory_stats()`.
    """
    global _DEVICE_TOTAL_BYTES, _DEVICE_PROBED
    if _DEVICE_PROBED:
        return _DEVICE_TOTAL_BYTES
    _DEVICE_PROBED = True
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        _DEVICE_TOTAL_BYTES = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
        return _DEVICE_TOTAL_BYTES
    except Exception:  # noqa: BLE001 - NVML missing / no GPU
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split()
        if out:
            _DEVICE_TOTAL_BYTES = int(float(out[0])) * 1024 * 1024  # MiB to bytes
    except Exception:  # noqa: BLE001 - nvidia-smi missing / unparseable
        pass
    return _DEVICE_TOTAL_BYTES


def jax_device_budget_bytes(fraction: float = 0.95) -> int | None:
    """Returns the usable device memory in bytes for sizing batched kernels, or None on the CPU
    backend.

    The BFC allocator preallocates a pool and `memory_stats()['bytes_limit']` is the budget. The
    vmm and platform allocators grow on demand and the physical total of the device is read.
    `fraction` leaves room for kernel scratch.
    """
    if not jax_is_available():
        return None
    try:
        import jax  # type: ignore[import-not-found]

        stats = jax.devices()[0].memory_stats() or {}  # pyright: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001 - any backend without memory_stats
        stats = {}
    alloc = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").lower()
    if alloc in ("vmm", "platform"):
        total = stats.get("bytes_reservable_limit") or _physical_device_total_bytes()
        share = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
        if total and share:
            # These allocators do not cap a process. Its share of the card is the budget.
            total = int(int(total) * float(share))
    else:
        total = stats.get("bytes_limit")
    if not total:
        return None
    budget = int(int(total) * fraction)
    global _BUDGET_LOGGED
    if not _BUDGET_LOGGED:
        _BUDGET_LOGGED = True
        log.status(
            LOG,
            "JAX kernel-batch budget: %.1f GiB (allocator=%s)",
            budget / 2**30,
            alloc or "default",
        )
    return budget


def stable_seed_offset(src: str, base: int | None = None) -> int:
    """Returns the PRNG offset of a JAX kernel, stable across processes.

    The offset comes from a blake2b digest. Python salts `hash()` per process. `src`
    separates concurrent kernels and is normally the active scope path. `base` mixes in the seed
    of the DAG node.
    """
    import hashlib

    v = int.from_bytes(hashlib.blake2b(src.encode("utf-8"), digest_size=8).digest(), "big")
    if base is not None:
        v ^= (int(base) & 0xFFFFFFFFFFFFFFFF) * 0x9E3779B97F4A7C15
    return int(v % (2**31))


def jax_bucket_for(n: int, ladder: tuple[int, ...] = SHAPE_BUCKETS) -> int:
    """Returns the smallest bucket >= n, or n above the top bucket."""
    for b in ladder:
        if n <= b:
            return b
    return n
