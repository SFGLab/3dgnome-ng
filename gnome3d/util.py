"""
util functions for 3dgnome-ng.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np

from .types import F32Array, F64Array

if TYPE_CHECKING:
    from .settings import Settings
    from .solver import Solver


# Solver factory — kept here (not in cli/simulate) so the entry points don't
# depend on the concrete solver implementation; the GPU path is selected for
# them without exposing it.


def make_solver(settings: Settings) -> Solver:
    """Return the right Solver for `settings`: a GPU region-batching `JaxSolver`
    when `jax_region_batch` is enabled on the JAX smooth backend (and small-IB
    boost is off — that varies springs per IB, which the batched kernel does not
    yet take per-chain from the solver), else the base serial `Solver`.

    Imports are function-local to avoid a util <-> solver import cycle."""
    from .solver import Solver
    from .solver_jax import JaxSolver

    use_batch = (
        str(settings.mc_backend).strip().lower() == "jax"
        and bool(settings.mc_backend_apply_to_smooth)
        and bool(getattr(settings, "jax_region_batch", False))
        and not bool(settings.use_small_ib_boost)
    )
    return JaxSolver(settings) if use_batch else Solver(settings)


# Distance conversion functions


def genomic_length_to_distance(length_bp: int, base: float, scale: float, power: float) -> float:
    """Reference: genomicLengthToDistance(length) = base + scale * (length/1000)^power"""
    return base + scale * (length_bp / 1000.0) ** power


def freq_to_dist_heatmap(freq: float, scale: float, power: float) -> float:
    """Reference: freqToDistanceHeatmap(freq) = scale * freq^power"""
    return scale * (freq**power)


def freq_to_dist_heatmap_inter(freq: float, scale_inter: float, power_inter: float) -> float:
    """Reference: freqToDistanceHeatmapInter(freq) = scale_inter * freq^power_inter"""
    return scale_inter * (freq**power_inter)


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
    z = 0.0 if in_2d else random.uniform(-step, step)
    return np.array(
        [
            random.uniform(-step, step),
            random.uniform(-step, step),
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
