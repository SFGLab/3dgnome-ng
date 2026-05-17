"""
src/energy.py  —  scoring / energy functions for 3dgnome-torch.

All functions must match the C++ reference exactly (within 1e-6 absolute
tolerance).  The harness in harness/compare.py tests each one independently.

Non-obvious details (see AGENTS.md):
  - angle_metric() is NOT acos: it is  1 - (dot(n1, n2) + 1) / 2
  - score_heatmap() double-counts: every pair (i,j) contributes twice
  - score_arcs() counts each arc once (i < j convention in global score)
  - metropolis_prob() uses ratio, not difference: exp(-coef * curr/prev / T)
  - random_vector() is uniform in a cube: each component in [-step, step]
"""

import math
import random

import torch


def get_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Distance conversion functions
# (signatures match compare.py _try_import calls exactly)

def genomic_length_to_distance(length_bp: int, base: float, scale: float, power: float) -> float:
    """C++: genomicLengthToDistance(length) = base + scale * (length/1000)^power"""
    return base + scale * (length_bp / 1000.0) ** power


def freq_to_dist_heatmap(freq: float, scale: float, power: float) -> float:
    """C++: freqToDistanceHeatmap(freq) = scale * freq^power"""
    return scale * (freq ** power)


def freq_to_dist_heatmap_inter(freq: float, scale_inter: float, power_inter: float) -> float:
    """C++: freqToDistanceHeatmapInter(freq) = scale_inter * freq^power_inter"""
    return scale_inter * (freq ** power_inter)


def freq_to_distance(freq: int, a: float, scale: float, shift: float, base_level: float) -> float:
    """C++: freqToDistance(freq) = base_level + scale / exp(a * (freq + shift))"""
    return base_level + scale / math.exp(a * (freq + shift))


# ---------------------------------------------------------------------------
# Angle metric
# NOT acos — this is the linear dissimilarity used throughout 3dgnome.

def angle_metric(v1, v2):
    """
    1 - (dot(norm(v1), norm(v2)) + 1) / 2
    Returns value in [0, 1]: 0 = parallel, 1 = anti-parallel, 0.5 = orthogonal.
    Works with Python sequences (returns float) or torch Tensors (returns Tensor).
    """
    if isinstance(v1, torch.Tensor):
        l1 = v1.norm()
        l2 = v2.norm()
        if l1 < 1e-10 or l2 < 1e-10:
            return v1.new_zeros(())
        dot = (v1 / l1).dot(v2 / l2)
        return 1.0 - (dot + 1.0) / 2.0
    # Python sequence path
    l1 = math.sqrt(sum(x * x for x in v1))
    l2 = math.sqrt(sum(x * x for x in v2))
    if l1 < 1e-10 or l2 < 1e-10:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2)) / (l1 * l2)
    return 1.0 - (dot + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Heatmap score (double-counted)

def score_heatmap(pos: torch.Tensor, exp_dist: torch.Tensor, diag_size: int) -> torch.Tensor:
    """
    Full heatmap energy score (double-counted, matching C++ calcScoreHeatmapActiveRegion(-1)).

    pos:      (N, 3) float tensor — bead positions
    exp_dist: (N, N) float tensor — expected pairwise distances
    diag_size: int — skip pairs within this diagonal band

    The C++ implementation calls calcScoreHeatmapActiveRegion(moved) for every
    moved index.  That inner function sums over all i != moved with
    |i - moved| >= diag_size and exp_dist[i][moved] >= 1e-6.
    Together, every pair (i, j) with i != j is counted twice — once when
    moved=i and once when moved=j.

    Returns a scalar Tensor.
    """
    n = pos.shape[0]
    # d[i, j] = ||pos[i] - pos[j]||
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)   # (n, n, 3)
    d = diff.norm(dim=2)                          # (n, n)

    idx = torch.arange(n, device=pos.device, dtype=torch.long)
    diag_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() < diag_size
    zero_mask = exp_dist < 1e-6
    mask = diag_mask | zero_mask

    safe_exp = exp_dist.clone()
    safe_exp[mask] = 1.0
    cerr = (d - safe_exp) / safe_exp
    return (cerr * cerr).masked_fill(mask, 0.0).sum()


# ---------------------------------------------------------------------------
# Arc spring score

def score_arcs(
    pos: torch.Tensor,
    arcs: list,
    stretch_k: float,
    squeeze_k: float,
) -> torch.Tensor:
    """
    Arc spring energy (global, each arc counted once).

    pos:       (N, 3) float tensor
    arcs:      list of (i, j, exp_d) tuples
               exp_d < 0   → repulsion term  1 / d
               exp_d < 1e-6 → skip
               otherwise   → spring term (d - exp_d)^2 / exp_d^2 * k
    stretch_k: spring constant when d > exp_d
    squeeze_k: spring constant when d < exp_d

    Matches C++ calcScoreDistancesActiveRegion() (global, i < j pairs).
    """
    sc = pos.new_zeros(())
    for i, j, exp_d in arcs:
        d = (pos[i] - pos[j]).norm()
        if exp_d < 0.0:
            sc = sc + 1.0 / d.clamp(min=1e-10)
            continue
        if exp_d < 1e-6:
            continue
        rel = (d - exp_d) / exp_d
        k = stretch_k if rel.item() >= 0.0 else squeeze_k
        sc = sc + rel * rel * k
    return sc


# ---------------------------------------------------------------------------
# Chain smoothness score

def score_smooth(
    pos: torch.Tensor,
    dist_to_next: torch.Tensor,
    stretch_k: float,
    squeeze_k: float,
    angular_k: float,
    w_dist: float,
    w_angle: float,
) -> torch.Tensor:
    """
    Chain smoothness energy: bond-length penalty + cubic angle penalty.

    pos:          (N, 3) float tensor — bead positions
    dist_to_next: (N-1,) float tensor — expected bond lengths

    sca = Σ_{i=0..N-2} ((|v_i| - dtn_i) / dtn_i)^2 * k_stretch_or_squeeze
    scb = Σ_{i=1..N-2} angle(v_{i-1}, v_i)^3 * angular_k

    Returns sca * w_dist + scb * w_angle.

    Matches C++ calcScoreStructureSmooth(true, true) [global].
    """
    n = pos.shape[0]
    sca = pos.new_zeros(())
    scb = pos.new_zeros(())
    v_prev = None
    for i in range(n - 1):
        v = pos[i] - pos[i + 1]
        vlen = v.norm()
        dtn = dist_to_next[i].item() if i < len(dist_to_next) else 1.0
        if dtn < 1e-6:
            dtn = 1e-6
        diff = (vlen - dtn) / dtn
        k = stretch_k if diff.item() >= 0.0 else squeeze_k
        sca = sca + diff * diff * k
        if v_prev is not None:
            ang = angle_metric(v, v_prev)
            scb = scb + ang * ang * ang * angular_k
        v_prev = v
    return sca * w_dist + scb * w_angle


# ---------------------------------------------------------------------------
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

    C++: tempJumpScale * exp(-tempJumpCoef * (score_curr / score_prev) / T)

    Note: jump_scale can exceed 1.0 (default 50), so the result can be > 1.
    The caller decides whether to accept by comparing against rand().
    """
    if T <= 0.0:
        return 0.0
    return jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)


# ---------------------------------------------------------------------------
# Fast NumPy-based scoring for use inside the MC loop
# These replicate the C++ "local" scoring functions used during MC steps.

import numpy as np


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
    Matches C++ calcScoreHeatmapActiveRegion(moved).
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
    Matches C++ calcScoreDistancesActiveRegion(cluster_moved).
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
    """Full arc spring score — sums i < j pairs once each (not double-counted)."""
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
