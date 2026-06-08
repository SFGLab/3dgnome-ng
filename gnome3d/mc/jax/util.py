import logging
import os
import threading

from gnome3d import log

# --- JAX GPU allocator preinit (MUST run before the first `import jax`, which is lazy in
# jax_is_available below) ---------------------------------------------------------------
# The genome runs dozens of differently-shaped kernels (arcs/smooth/checker at B=256..16384),
# which fragments JAX's default BFC pool: large-B allocations then OOM on a GPU with plenty
# of free-but-non-contiguous memory.  cuda_malloc_async coalesces (sidesteps BFC
# fragmentation); the 0.95 fraction gives more pool headroom.  setdefault so an explicit
# shell env still overrides.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

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


def jax_device_budget_bytes(fraction: float = 0.95) -> int | None:
    """Best-effort usable device memory (bytes) for sizing batched kernels.

    Returns `fraction` * the primary device's reported byte limit, or ``None``
    when it can't be determined - e.g. the CPU backend, or a platform whose
    ``Device.memory_stats()`` is unavailable/empty.  Callers fall back to a fixed
    heuristic on ``None``.  Uses the total limit (not free) since XLA preallocates
    its pool; ``fraction`` leaves headroom for kernel scratch beyond the stacked
    inputs."""
    if not jax_is_available():
        return None
    try:
        import jax  # type: ignore[import-not-found]

        stats = jax.devices()[0].memory_stats()  # pyright: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001 - any backend without memory_stats
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    if not limit:
        return None
    return int(int(limit) * fraction)


def jax_bucket_for(n: int, ladder: tuple[int, ...] = SHAPE_BUCKETS) -> int:
    """Smallest ladder bucket >= n, or n itself if it exceeds the top bucket."""
    for b in ladder:
        if n <= b:
            return b
    return n
