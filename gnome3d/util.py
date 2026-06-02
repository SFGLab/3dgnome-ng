"""
util functions for 3dgnome-ng.
"""

from __future__ import annotations

import math
import os
import random
import threading

import numpy as np

from gnome3d import log
from gnome3d.types import F32Array, F64Array

LOG = log.get("util")

# Thread-local RNG for the positioning noise.  The global `random` module is a
# single `random.Random` instance shared across threads, so a `ThreadedExecutor`
# running several stages at once would have them clobber each other's stream.  A
# per-thread `random.Random` is byte-identical for a given seed (the global
# module IS such an instance) but isolated per thread, so threaded stage
# execution stays deterministic.  Seed it with `seed_rng`; draw via the helpers.
_tls = threading.local()


def _rng() -> random.Random:
    r = getattr(_tls, "rng", None)
    if r is None:
        r = _tls.rng = random.Random()
    return r

def seed_rng(seed: int) -> None:
    """Seed the calling thread's positioning RNG (used by `random_vector_np`).
    Replaces a bare `random.seed`, so coarse seeding and per-stage seeding are
    thread-isolated; same seed -> same stream as the old global `random`."""
    _rng().seed(seed)


def genomic_length_to_distance(length_bp: int, base: float, scale: float, power: float) -> float:
    """Reference: genomicLengthToDistance(length) = base + scale * (length/1000)^power"""
    return base + scale * (length_bp / 1000.0) ** power


def freq_to_dist_heatmap(freq: float, scale: float, power: float) -> float:
    """Reference: freqToDistanceHeatmap(freq) = scale * freq^power"""
    return scale * (freq ** power)


def freq_to_dist_heatmap_inter(freq: float, scale_inter: float, power_inter: float) -> float:
    """Reference: freqToDistanceHeatmapInter(freq) = scale_inter * freq^power_inter"""
    return scale_inter * (freq ** power_inter)


def freq_to_distance(freq: int, a: float, scale: float, shift: float, base_level: float) -> float:
    """Reference: freqToDistance(freq) = base_level + scale / exp(a * (freq + shift))"""
    try:
        return base_level + scale / math.exp(a * (freq + shift))
    except OverflowError:  # Reference exp() returns inf -> scale/inf = 0
        return base_level


def random_vector_np(step: float, in_2d: bool = False) -> F32Array:
    """Uniform cube displacement: each component in [-step, step].
    Mirrors Reference displace() in lib/common.cpp.  When in_2d is True, the
    z component is forced to 0 (matches `Settings::use2D` branch).
    """
    r = _rng()
    z = 0.0 if in_2d else r.uniform(-step, step)
    return np.array(
        [
            r.uniform(-step, step),
            r.uniform(-step, step),
            z,
        ],
        dtype=np.float32,
    )


def calc_orientation(pos: F64Array, cind: int, n: int, char_orientation: str) -> F64Array:
    """
    Normalized orientation vector for bead at active-region index cind.
    """
    if cind == 0:
        orn = pos[cind + 1] - pos[cind]
    elif cind == n - 1:
        orn = pos[cind] - pos[cind - 1]
    else:
        orn = pos[cind + 1] - pos[cind - 1]
    if char_orientation == "L":
        orn = -orn
    norm = float(np.linalg.norm(orn))
    if norm > 1e-12:
        orn = orn / norm

    return np.asarray(orn, dtype=np.float64).copy()


_JAX_AVAILABLE: bool | None = None  # None = not yet probed
_init_lock = threading.Lock()


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


# Shape-bucket ladder.  When settings.jax_bucket_shapes is on, every kernel's
# bead count N is padded up to the next bucket so XLA compiles ~one program per
# bucket (8 total) instead of one per distinct region size.  Geometric x2 so
# worst-case padding waste is <2x compute.  N above the top bucket compiles at
# its exact size (rare).
_SHAPE_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Separate (finer/smaller) ladders for smooth orientation's anchor count and
# neighbor width — these scale below N, so reusing _SHAPE_BUCKETS would waste a
# lot at small sizes.
_ANCHOR_BUCKETS: tuple[int, ...] = (16, 64, 256, 1024, 4096, 16384)
_NBR_BUCKETS: tuple[int, ...] = (4, 8, 16, 32, 64)


def jax_bucket_for(n: int, ladder: tuple[int, ...] = _SHAPE_BUCKETS) -> int:
    """Smallest ladder bucket >= n, or n itself if it exceeds the top bucket."""
    for b in ladder:
        if n <= b:
            return b
    return n
