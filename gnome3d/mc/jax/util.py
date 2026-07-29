import logging
import os
import threading

from gnome3d import log

LOG = log.get("mc.jax.util")

_JAX_AVAILABLE: bool | None = None  # None = not yet probed
_init_lock = threading.Lock()

# Shape-bucket ladder.  When settings.jax_bucket_shapes is on, every kernel's
# bead count N is padded up to the next bucket so XLA compiles ~one program per
# bucket (8 total) instead of one per distinct region size.  Geometric x2 so
# worst-case padding waste is <2x compute.  N above the top bucket compiles at
# its exact size (rare).
SHAPE_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Separate (finer/smaller) ladders for smooth orientation's anchor count and
# neighbor width - these scale below N, so reusing _SHAPE_BUCKETS would waste a
# lot at small sizes.
ANCHOR_BUCKETS: tuple[int, ...] = (16, 64, 256, 1024, 4096, 16384)
NBR_BUCKETS: tuple[int, ...] = (4, 8, 16, 32, 64)


def log_kernel_start(
    logger: logging.Logger, stage: str, kernel: str, k: int, b: int, detail: str
) -> None:
    """Standard JAX-kernel START line - one format for every mc/checker/hybrid kernel (arcs +
    smooth): ``arcs[checker]: 719 IBs x 256 beads - <detail>, running...``."""
    log.status(
        logger, "    %s[%s]: %d IBs x %d beads - %s, running...", stage, kernel, k, b, detail
    )


def log_kernel_done(
    logger: logging.Logger, stage: str, kernel: str, k: int, secs: float, summary: str
) -> None:
    """Standard JAX-kernel DONE line: ``arcs[checker]: 719 IBs in 268.1s - <summary>``."""
    log.status(logger, "    %s[%s]: %d IBs in %.1fs - %s", stage, kernel, k, secs, summary)


def jax_is_available() -> bool:
    """Lazy-import JAX. Returns True on success, False if not installed."""
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is not None:
        return _JAX_AVAILABLE

    with _init_lock:
        if _JAX_AVAILABLE is not None:
            return _JAX_AVAILABLE
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
    """Total physical bytes of GPU 0, independent of the XLA allocator.  vmm / platform expose an
    EMPTY ``memory_stats()``, so the only reliable budget source under them is the device itself -
    read it via NVML, then nvidia-smi.  Cached; ``None`` if neither is available."""
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
            _DEVICE_TOTAL_BYTES = int(float(out[0])) * 1024 * 1024  # MiB -> bytes
    except Exception:  # noqa: BLE001 - nvidia-smi missing / unparseable
        pass
    return _DEVICE_TOTAL_BYTES


def jax_device_budget_bytes(fraction: float = 0.95) -> int | None:
    """Best-effort usable device memory (bytes) for sizing batched kernels.

    BFC preallocates a fixed pool, so ``memory_stats()['bytes_limit']`` IS the usable budget.
    vmm / platform grow on demand and expose an EMPTY ``memory_stats()``, so we read the physical
    device total (NVML / nvidia-smi) instead - allocator-independent.  Returns ``fraction`` * the
    total, or ``None`` (CPU backend / nothing queryable) so callers fall back to a fixed heuristic.
    ``fraction`` leaves headroom for kernel scratch beyond the stacked inputs."""
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
    """PRNG offset for a JAX kernel, stable across processes.

    Two runs of one config with one seed must produce one structure. Deriving the
    offset from `hash()` of a string broke that, because Python salts string
    hashing per process, so every run of a JAX-backed stage silently explored a
    different trajectory and `PYTHONHASHSEED=0` was needed to compare two runs.
    blake2b is stable across processes and machines.

    `src` distinguishes concurrent kernels, normally the active scope path.
    `base` mixes in the seed the DAG node carries, so the offset follows from
    `Seeded.seed` the way the numba path's already does. The odd multiplier
    spreads nearby seeds across the output range before the modulo.
    """
    import hashlib

    v = int.from_bytes(hashlib.blake2b(src.encode("utf-8"), digest_size=8).digest(), "big")
    if base is not None:
        v ^= (int(base) & 0xFFFFFFFFFFFFFFFF) * 0x9E3779B97F4A7C15
    return int(v % (2**31))


def jax_bucket_for(n: int, ladder: tuple[int, ...] = SHAPE_BUCKETS) -> int:
    """Smallest ladder bucket >= n, or n itself if it exceeds the top bucket."""
    for b in ladder:
        if n <= b:
            return b
    return n
