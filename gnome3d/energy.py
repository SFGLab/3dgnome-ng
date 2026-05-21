"""
scoring / energy functions for 3dgnome-ng.
"""

import math
import random

import numpy as np


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


# Angle metric

def angle_metric(v1, v2) -> float:
    """
    1 - (dot(norm(v1), norm(v2)) + 1) / 2
    Returns value in [0, 1]: 0 = parallel, 1 = anti-parallel, 0.5 = orthogonal.
    Accepts Python sequences or numpy arrays.
    """
    l1 = math.sqrt(sum(x * x for x in v1))
    l2 = math.sqrt(sum(x * x for x in v2))
    if l1 < 1e-10 or l2 < 1e-10:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2)) / (l1 * l2)
    return 1.0 - (dot + 1.0) / 2.0


# Heatmap score (double-counted)

def score_heatmap(pos: np.ndarray, exp_dist: np.ndarray, diag_size: int) -> float:
    """
    Full heatmap energy score (double-counted, matching Reference calcScoreHeatmapActiveRegion(-1)).

    pos:      (N, 3) float array - bead positions
    exp_dist: (N, N) float array - expected pairwise distances
    diag_size: int - skip pairs within this diagonal band

    The Reference implementation calls calcScoreHeatmapActiveRegion(moved) for every
    moved index.  That inner function sums over all i != moved with
    |i - moved| >= diag_size and exp_dist[i][moved] >= 1e-6.
    Together, every pair (i, j) with i != j is counted twice - once when
    moved=i and once when moved=j.
    """
    pos = np.asarray(pos, dtype=np.float64)
    exp_dist = np.asarray(exp_dist, dtype=np.float64)
    n = pos.shape[0]

    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))

    idx = np.arange(n)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    zero_mask = exp_dist < 1e-6
    mask = diag_mask | zero_mask

    safe_exp = np.where(mask, 1.0, exp_dist)
    cerr = (d - safe_exp) / safe_exp
    contrib = cerr * cerr
    contrib[mask] = 0.0
    return float(contrib.sum())


# Arc spring score

def score_arcs(
    pos: np.ndarray,
    arcs: list,
    stretch_k: float,
    squeeze_k: float,
) -> float:
    """
    Arc spring energy (global, each arc counted once).

    pos:       (N, 3) float array
    arcs:      list of (i, j, exp_d) tuples
               exp_d < 0   -> repulsion term  1 / d
               exp_d < 1e-6 -> skip
               otherwise   -> spring term (d - exp_d)^2 / exp_d^2 * k
    stretch_k: spring constant when d > exp_d
    squeeze_k: spring constant when d < exp_d

    Matches Reference calcScoreDistancesActiveRegion() (global, i < j pairs).
    """
    pos = np.asarray(pos, dtype=np.float64)
    sc = 0.0
    for i, j, exp_d in arcs:
        d = _np_dist(pos[i], pos[j])
        if exp_d < 0.0:
            sc += 1.0 / max(d, 1e-10)
            continue
        if exp_d < 1e-6:
            continue
        rel = (d - exp_d) / exp_d
        sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


# Chain smoothness score

def score_smooth(
    pos: np.ndarray,
    dist_to_next: np.ndarray,
    stretch_k: float,
    squeeze_k: float,
    angular_k: float,
    w_dist: float,
    w_angle: float,
) -> float:
    """
    Chain smoothness energy: bond-length penalty + cubic angle penalty.

    pos:          (N, 3) float array - bead positions
    dist_to_next: (N-1,) float array - expected bond lengths

    sca = sum_{i=0..N-2} ((|v_i| - dtn_i) / dtn_i)^2 * k_stretch_or_squeeze
    scb = sum_{i=1..N-2} angle(v_{i-1}, v_i)^3 * angular_k

    Returns sca * w_dist + scb * w_angle.

    Matches Reference calcScoreStructureSmooth(true, true) [global].
    """
    pos = np.asarray(pos, dtype=np.float64)
    dist_to_next = np.asarray(dist_to_next, dtype=np.float64)
    n = pos.shape[0]
    sca = 0.0
    scb = 0.0
    v_prev = None
    for i in range(n - 1):
        v = pos[i] - pos[i + 1]
        vlen = float(np.sqrt(v.dot(v)))
        dtn = float(dist_to_next[i]) if i < dist_to_next.shape[0] else 1.0
        if dtn < 1e-6:
            dtn = 1e-6
        diff = (vlen - dtn) / dtn
        sca += diff * diff * (stretch_k if diff >= 0.0 else squeeze_k)
        if v_prev is not None:
            ang = angle_metric(v, v_prev)
            scb += ang * ang * ang * angular_k
        v_prev = v
    return sca * w_dist + scb * w_angle


# Metropolis acceptance probability

def metropolis_prob(
    jump_scale: float,
    jump_coef: float,
    score_curr: float,
    score_prev: float,
    T: float,
) -> float:
    """
    Metropolis acceptance probability.

    Reference: tempJumpScale * exp(-tempJumpCoef * (score_curr / score_prev) / T)

    Note: jump_scale can exceed 1.0 (default 50), so the result can be > 1.
    The caller decides whether to accept by comparing against rand().
    """
    if T <= 0.0:
        return 0.0
    return jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)


# CTCF orientation energy functions

def calc_orientation(pos: np.ndarray, cind: int, n: int, char_orientation: str) -> np.ndarray:
    """
    Normalized orientation vector for bead at active-region index cind.

    Matches Reference LooperSolver::calcOrientation(cind):
      - endpoints: one-sided difference
      - interior: central difference (pos[cind+1] - pos[cind-1])
      - 'L' motif: negate
      - normalize to unit length
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
    return orn.copy()


def score_orientation(
    anchor_orientations: list,
    neighbors: dict,
    neighbor_weights: dict,
    motif_weight: float,
    motifs_symmetric: bool = True,
) -> float:
    """
    Full CTCF orientation score (uses arc weights, double-counts each arc pair).
    Matches Reference calcScoreOrientation(const vector<vector3>& orientation).

    anchor_orientations: list of (3,) arrays indexed by anchor list position
    neighbors:  {anchor_i: [anchor_j, ...]}
    neighbor_weights: {anchor_i: [float, ...]}  (sqrt(arc.score) per arc)
    """
    err = 0.0
    for i, nbrs in neighbors.items():
        ws = neighbor_weights[i]
        for k, j in enumerate(nbrs):
            if motifs_symmetric:
                ang = angle_metric(anchor_orientations[i], anchor_orientations[j])
            else:
                ang = angle_metric(anchor_orientations[i], -anchor_orientations[j])
            err += ang * ang * ws[k]
    return err * motif_weight


def local_score_orientation(
    anchor_orientations: list,
    anchor_index: int,
    neighbors: dict,
    motif_weight: float,
    motifs_symmetric: bool = True,
) -> float:
    """
    Local CTCF orientation score for one anchor - UNWEIGHTED.
    Mirrors Reference calcScoreOrientation(orn, anchor_index), used as the harness
    reference (test_orientation in compare.py) for bit-equivalence with Reference.

    NOT used by the actual MC loop. The MC kernel uses _local_score_orientation_nb
    in mc.py which is WEIGHTED (drift-free incremental update); see
    [[project-orientation-mc-fix]].
    """
    err = 0.0
    for j in neighbors[anchor_index]:
        if motifs_symmetric:
            ang = angle_metric(anchor_orientations[anchor_index], anchor_orientations[j])
        else:
            ang = angle_metric(anchor_orientations[anchor_index], -anchor_orientations[j])
        err += ang * ang
    return err * motif_weight


# Fast NumPy-based scoring for use inside the MC loop
# These replicate the Reference "local" scoring functions used during MC steps.

def _np_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(d.dot(d)))


def local_score_heatmap_np(
    pos: np.ndarray,
    exp_dist: np.ndarray,
    diag_size: int,
    moved: int,
    st: int = 0,
    end: int = -1,
) -> float:
    """
    Local heatmap score for bead `moved`.
    Matches Reference calcScoreHeatmapActiveRegion(moved).
    pos: (N, 3), exp_dist: (N, N)
    """
    n = pos.shape[0]
    if end < 0:
        end = n - 1
    err = 0.0
    for i in range(st, end + 1):
        if abs(i - moved) < diag_size:
            continue
        e = exp_dist[i, moved]
        if e < 1e-6:
            continue
        d = _np_dist(pos[i], pos[moved])
        cerr = (d - e) / e
        err += cerr * cerr
    return err


def global_score_heatmap_np(
    pos: np.ndarray,
    exp_dist: np.ndarray,
    diag_size: int,
) -> float:
    """Full double-counted heatmap score (NumPy, for MC initialisation)."""
    n = pos.shape[0]
    total = 0.0
    for moved in range(n):
        total += local_score_heatmap_np(pos, exp_dist, diag_size, moved)
    return total


def local_score_arcs_np(
    pos: np.ndarray,
    exp_dist_mat: np.ndarray,
    moved: int,
    stretch_k: float,
    squeeze_k: float,
) -> float:
    """
    Local arc score for bead `moved`.
    Reference: calcScoreDistancesActiveRegion(cluster_moved).
    exp_dist_mat: (N, N) where -1 means repulsion, 0 means no arc.
    """
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if i == moved:
            continue
        e = exp_dist_mat[i, moved]
        d = _np_dist(pos[moved], pos[i])
        if e < 0.0:
            sc += 1.0 / max(d, 1e-10)
            continue
        if e < 1e-6:
            continue
        rel = (d - e) / e
        sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


def global_score_arcs_np(
    pos: np.ndarray,
    exp_dist_mat: np.ndarray,
    stretch_k: float,
    squeeze_k: float,
) -> float:
    """Full arc spring score - sums i < j pairs once each (not double-counted)."""
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            e = exp_dist_mat[i, j]
            d = _np_dist(pos[i], pos[j])
            if e < 0.0:
                sc += 1.0 / max(d, 1e-10)
                continue
            if e < 1e-6:
                continue
            rel = (d - e) / e
            sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


def random_vector_np(step: float) -> np.ndarray:
    """Uniform cube displacement: each component in [-step, step]."""
    return np.array([
        random.uniform(-step, step),
        random.uniform(-step, step),
        random.uniform(-step, step),
    ], dtype=np.float32)
