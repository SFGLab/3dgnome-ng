"""
util functions for 3dgnome-ng.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .types import F32Array, F64Array


# Distance conversion functions

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
    z = 0.0 if in_2d else random.uniform(-step, step)
    return np.array([
        random.uniform(-step, step),
        random.uniform(-step, step),
        z,
    ], dtype=np.float32)


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
    if char_orientation == 'L':
        orn = -orn
    norm = float(np.linalg.norm(orn))
    if norm > 1e-12:
        orn = orn / norm
    return np.asarray(orn, dtype=np.float64).copy()
