"""
util functions for 3dgnome-ng.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

from gnome3d import log
from gnome3d.types import BoolArray, F32Array, F64Array

LOG = log.get("util")

# Measured on a 36 core machine with OpenBLAS: the vector routines L-BFGS-B calls between
# evaluations cost 6 ms at 36 threads and 0.03 ms at 4 on 90,000 variables, and the maxent fit's
# objective is fastest at 8 from 350 to 9,195 anchors.
_OPTIMISER_BLAS_THREADS = 8


@contextmanager
def blas_threads() -> Generator[None]:
    """Hold the linear algebra library at up to eight threads for the block.

    scipy's L-BFGS-B does its own vector arithmetic through BLAS, and a library that wakes
    every core for a vector of some ten thousand numbers spends the call waking them. The
    limit is process wide while the block runs and never raises a lower count set by the
    environment.
    """
    from threadpoolctl import ThreadpoolController

    ctl: Any = ThreadpoolController()
    now = max(
        (int(lib.num_threads) for lib in ctl.lib_controllers if lib.user_api == "blas"), default=1
    )
    with ctl.limit(limits=min(_OPTIMISER_BLAS_THREADS, now), user_api="blas"):
        yield


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


def positioning_rng() -> random.Random:
    """The calling thread's positioning RNG, for a caller that draws its own shapes."""
    return _rng()


def seed_rng(seed: int) -> None:
    """Seed the calling thread's positioning RNG (used by `random_vector_np`).
    Replaces a bare `random.seed`, so coarse seeding and per-stage seeding are
    thread-isolated; same seed -> same stream as the old global `random`."""
    _rng().seed(seed)


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


def add_movable_noise_inplace(pos: F32Array, fixed: BoolArray | None, step: float) -> None:
    """Add a uniform-cube displacement (each component in [-step, step]) to every
    non-`fixed` bead of `pos`, in place.  `fixed=None` noises every bead (the
    arcs case, which has no fixed mask).

    Vectorised, byte-identical replacement for the per-bead loop
    ``for i in range(n): if not fixed[i]: pos[i] += random_vector_np(step)`` -
    same Mersenne-Twister stream, same float32 result.  The equivalence rests on:
      * the loop draws ``z, x, y`` per movable bead (index order), so a flat block
        of ``3*m`` draws reshaped ``(m, 3)`` reproduces the exact sequence;
      * ``r.uniform(-s, s) == -s + (2s)*r.random()``, and ``(2s)*x - s`` is the
        same IEEE-754 value (add is commutative), then rounded to float32.
    Draws the per-bead ``z`` component even though it lands in column 2, matching
    the 3D (non-`in_2d`) `random_vector_np` the callers use.
    """
    idx = np.arange(pos.shape[0]) if fixed is None else np.flatnonzero(~fixed)
    m = int(idx.size)
    if m == 0:
        return
    draw = _rng().random
    raw = np.fromiter((draw() for _ in range(3 * m)), dtype=np.float64, count=3 * m)
    disp = (2.0 * step) * raw.reshape(m, 3) - step  # columns: [z, x, y]
    pos[idx, 0] += disp[:, 1].astype(np.float32)
    pos[idx, 1] += disp[:, 2].astype(np.float32)
    pos[idx, 2] += disp[:, 0].astype(np.float32)


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
