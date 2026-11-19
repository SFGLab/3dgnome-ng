"""Helpers shared across the package."""

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

# The count at which the vector routines of L-BFGS-B and the maxent fit run fastest.
_OPTIMISER_BLAS_THREADS = 8


@contextmanager
def blas_threads() -> Generator[None]:
    """Holds BLAS at up to eight threads for the block.

    The limit is process wide and does not raise a lower count set by the environment.
    """
    from threadpoolctl import ThreadpoolController

    ctl: Any = ThreadpoolController()
    now = max(
        (int(lib.num_threads) for lib in ctl.lib_controllers if lib.user_api == "blas"), default=1
    )
    with ctl.limit(limits=min(_OPTIMISER_BLAS_THREADS, now), user_api="blas"):
        yield


# One `random.Random` per thread. A seed draws the stream the global `random` module draws.
_tls = threading.local()


def _rng() -> random.Random:
    r = getattr(_tls, "rng", None)
    if r is None:
        r = _tls.rng = random.Random()
    return r


def positioning_rng() -> random.Random:
    """Returns the positioning RNG of the calling thread."""
    return _rng()


def seed_rng(seed: int) -> None:
    """Seeds the positioning RNG of the calling thread."""
    _rng().seed(seed)


def random_vector_np(step: float, in_2d: bool = False) -> F32Array:
    """Returns a displacement with each component uniform in [-step, step]. z is 0 when `in_2d`."""
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
    """Adds a displacement uniform in [-step, step] to every bead not in `fixed`, in place.

    Draws z, x, y per bead in bead order, the stream of one `random_vector_np` call per bead.
    """
    idx = np.arange(pos.shape[0]) if fixed is None else np.flatnonzero(~fixed)
    m = int(idx.size)
    if m == 0:
        return
    draw = _rng().random
    raw = np.fromiter((draw() for _ in range(3 * m)), dtype=np.float64, count=3 * m)
    disp = (2.0 * step) * raw.reshape(m, 3) - step  # columns z, x, y
    pos[idx, 0] += disp[:, 1].astype(np.float32)
    pos[idx, 1] += disp[:, 2].astype(np.float32)
    pos[idx, 2] += disp[:, 0].astype(np.float32)


def calc_orientation(pos: F64Array, cind: int, n: int, char_orientation: str) -> F64Array:
    """Returns the unit orientation vector of the bead at index `cind`."""
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
