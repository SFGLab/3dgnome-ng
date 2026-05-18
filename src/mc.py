"""
src/mc.py  -  Monte Carlo simulation loops for 3dgnome-ng.

Mirrors C++ LooperSolver::MonteCarloHeatmap(), MonteCarloArcs(), and
MonteCarloArcsSmooth().

All three loops run entirely inside Numba @njit batch functions.
On first import the JIT functions compile (~10–30 s); subsequent runs
load from __pycache__ (cache=True).

Acceptance criterion (all loops):
    ok = (score_new <= score_curr)
      or rand() < jump_scale * exp(-jump_coef * score_new/score_curr / T)
"""

import math

import numpy as np
from numba import njit as _njit


# ---------------------------------------------------------------------------
# Smooth MC helpers

@_njit(cache=True)
def _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w):
    dx = pos[i, 0] - pos[i + 1, 0]
    dy = pos[i, 1] - pos[i + 1, 1]
    dz = pos[i, 2] - pos[i + 1, 2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    e = dtn[i]
    if e < 1e-6:
        e = 1e-6
    rel = (d - e) / e
    k = stretch_k if rel >= 0.0 else squeeze_k
    return rel * rel * k * dist_w


@_njit(cache=True)
def _smooth_ang_nb(pos, i, ang_k, ang_w):
    v1x = pos[i, 0] - pos[i + 1, 0]
    v1y = pos[i, 1] - pos[i + 1, 1]
    v1z = pos[i, 2] - pos[i + 1, 2]
    v2x = pos[i + 1, 0] - pos[i + 2, 0]
    v2y = pos[i + 1, 1] - pos[i + 2, 1]
    v2z = pos[i + 1, 2] - pos[i + 2, 2]
    n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
    if cos_a > 1.0: cos_a = 1.0
    if cos_a < -1.0: cos_a = -1.0
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return ang * ang * ang * ang_k * ang_w


@_njit(cache=True)
def _local_smooth_nb(pos, dtn, p, n, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    sc = 0.0
    i = p - 1
    if 0 <= i < n - 1:
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    if 0 <= p < n - 1:
        sc += _smooth_len_nb(pos, dtn, p, stretch_k, squeeze_k, dist_w)
    for off in range(-2, 1):
        i = p + off
        if 0 <= i < n - 2:
            sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


@_njit(cache=True)
def _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n - 1):
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    for i in range(n - 2):
        sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


@_njit(cache=True)
def _batch_smooth_nb(pos, dtn, movable, step_size, T, dt,
                     jump_scale, jump_coef, n_steps,
                     stretch_k, squeeze_k, ang_k, dist_w, ang_w, score):
    """Run n_steps smooth-MC steps.  Returns (T_out, score_out, n_ok)."""
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p = movable[np.random.randint(0, n_mov)]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_prev = _local_smooth_nb(pos, dtn, p, n,
                                    stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        pos[p, 0] += dx;
        pos[p, 1] += dy;
        pos[p, 2] += dz
        loc_curr = _local_smooth_nb(pos, dtn, p, n,
                                    stretch_k, squeeze_k, ang_k, dist_w, ang_w)

        score_new = score - loc_prev + loc_curr
        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = (np.random.random() <
                  jump_scale * math.exp(-jump_coef * (score_new / score) / T))
        if ok:
            n_ok += 1
            score = score_new
        else:
            pos[p, 0] -= dx;
            pos[p, 1] -= dy;
            pos[p, 2] -= dz
        T *= dt
    return T, score, n_ok


# ---------------------------------------------------------------------------
# Orientation MC helpers (JIT-compiled)

@_njit(cache=True)
def _calc_orientation_nb(pos, cind, n, is_L):
    """Returns (ox, oy, oz) normalized orientation vector for anchor at cind."""
    if cind == 0:
        ox = pos[cind + 1, 0] - pos[cind, 0]
        oy = pos[cind + 1, 1] - pos[cind, 1]
        oz = pos[cind + 1, 2] - pos[cind, 2]
    elif cind == n - 1:
        ox = pos[cind, 0] - pos[cind - 1, 0]
        oy = pos[cind, 1] - pos[cind - 1, 1]
        oz = pos[cind, 2] - pos[cind - 1, 2]
    else:
        ox = pos[cind + 1, 0] - pos[cind - 1, 0]
        oy = pos[cind + 1, 1] - pos[cind - 1, 1]
        oz = pos[cind + 1, 2] - pos[cind - 1, 2]
    if is_L:
        ox = -ox;
        oy = -oy;
        oz = -oz
    nm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if nm > 1e-12:
        ox /= nm;
        oy /= nm;
        oz /= nm
    return ox, oy, oz


@_njit(cache=True)
def _score_orientation_full_nb(anchor_orn, nbr_offsets, nbr_indices, nbr_weights,
                               motif_weight, symmetric):
    """Global orientation score with arc weights; double-counts each arc pair."""
    n_anchors = anchor_orn.shape[0]
    err = 0.0
    for i in range(n_anchors):
        for ki in range(nbr_offsets[i], nbr_offsets[i + 1]):
            j = nbr_indices[ki]
            w = nbr_weights[ki]
            ax = anchor_orn[i, 0];
            ay = anchor_orn[i, 1];
            az = anchor_orn[i, 2]
            bx = anchor_orn[j, 0];
            by = anchor_orn[j, 1];
            bz = anchor_orn[j, 2]
            if not symmetric:
                bx = -bx
                by = -by
                bz = -bz
            dot = ax * bx + ay * by + az * bz
            ang = 1.0 - (dot + 1.0) * 0.5
            err += ang * ang * w
    return err * motif_weight


@_njit(cache=True)
def _batch_smooth_orientation_nb(
    pos, dtn, movable, orn_is_L, anchor_ar,
    nbr_offsets, nbr_indices, nbr_weights,
    anchor_orn, bead_to_anchor_k,
    step_size, T, dt, jump_scale, jump_coef, n_steps,
    stretch_k, squeeze_k, ang_k, dist_w, ang_w,
    motif_weight, symmetric, score,
):
    """Smooth MC with CTCF orientation energy.
    Full structure + orientation score recomputed every step, matching C++.
    anchor_orn (n_anchors, 3) is updated in-place across calls.
    Returns (T_out, score_out, n_ok).
    """
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p = movable[np.random.randint(0, n_mov)]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        orn_k = bead_to_anchor_k[p]
        prev_ox = 0.0
        prev_oy = 0.0
        prev_oz = 0.0
        if orn_k >= 0:
            prev_ox = anchor_orn[orn_k, 0]
            prev_oy = anchor_orn[orn_k, 1]
            prev_oz = anchor_orn[orn_k, 2]

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        if orn_k >= 0:
            ar = anchor_ar[orn_k]
            ox, oy, oz = _calc_orientation_nb(pos, ar, n, orn_is_L[ar])
            anchor_orn[orn_k, 0] = ox
            anchor_orn[orn_k, 1] = oy
            anchor_orn[orn_k, 2] = oz

        score_new = (
            _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w)
            + _score_orientation_full_nb(anchor_orn, nbr_offsets, nbr_indices,
                                         nbr_weights, motif_weight, symmetric)
        )

        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = (np.random.random() <
                  jump_scale * math.exp(-jump_coef * (score_new / score) / T))
        if ok:
            n_ok += 1
            score = score_new
        else:
            pos[p, 0] -= dx;
            pos[p, 1] -= dy;
            pos[p, 2] -= dz
            if orn_k >= 0:
                anchor_orn[orn_k, 0] = prev_ox
                anchor_orn[orn_k, 1] = prev_oy
                anchor_orn[orn_k, 2] = prev_oz
        T *= dt
    return T, score, n_ok


# ---------------------------------------------------------------------------
# Arcs MC helpers

@_njit(cache=True)
def _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if i == p:
            continue
        e = exp[i, p]
        dx = pos[p, 0] - pos[i, 0]
        dy = pos[p, 1] - pos[i, 1]
        dz = pos[p, 2] - pos[i, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if e < 0.0:
            sc += 1.0 / (d if d > 1e-10 else 1e-10)
        elif e >= 1e-6:
            rel = (d - e) / e
            sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


@_njit(cache=True)
def _init_arcs_nb(pos, exp, stretch_k, squeeze_k):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            e = exp[i, j]
            if -1e-10 < e < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if e < 0.0:
                sc += 1.0 / (d if d > 1e-10 else 1e-10)
            else:
                rel = (d - e) / e
                sc += rel * rel * (stretch_k if rel >= 0.0 else squeeze_k)
    return sc


@_njit(cache=True)
def _batch_arcs_nb(pos, exp, step_size, T, dt, jump_scale, jump_coef,
                   n_steps, stretch_k, squeeze_k, score):
    """Run n_steps arc-MC steps.  Returns (T_out, score_out, n_ok)."""
    n = pos.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_prev = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)
        pos[p, 0] += dx;
        pos[p, 1] += dy;
        pos[p, 2] += dz
        loc_curr = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)

        score_new = score - loc_prev + loc_curr
        ok = score_new <= score
        if not ok and score > 0.0 and T > 0.0:
            ok = (np.random.random() <
                  jump_scale * math.exp(-jump_coef * (score_new / score) / T))
        if ok:
            n_ok += 1
            score = score_new
        else:
            pos[p, 0] -= dx;
            pos[p, 1] -= dy;
            pos[p, 2] -= dz
        T *= dt
    return T, score, n_ok


# ---------------------------------------------------------------------------
# Heatmap MC helpers

@_njit(cache=True)
def _local_heatmap_nb(pos, exp_safe, skip_col, p):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if skip_col[i]:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        e = exp_safe[i, p]
        err = (d - e) / e
        sc += err * err
    return sc


@_njit(cache=True)
def _init_heatmap_nb(pos, exp_safe, skip):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        for j in range(n):
            if skip[i, j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            e = exp_safe[i, j]
            err = (d - e) / e
            sc += err * err
    return sc


@_njit(cache=True)
def _batch_heatmap_nb(pos, exp_safe, skip, step_size, T, dt,
                      jump_scale, jump_coef, n_steps, score):
    """Run n_steps heatmap-MC steps.  Returns (T_out, score_out, n_ok)."""
    n = pos.shape[0]
    n_ok = 0
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_prev = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)
        pos[p, 0] += dx;
        pos[p, 1] += dy;
        pos[p, 2] += dz
        loc_curr = _local_heatmap_nb(pos, exp_safe, skip[:, p], p)

        # heatmap score double-counts: factor 2
        score_new = score + 2.0 * (loc_curr - loc_prev)
        ok = score_new <= score
        if not ok and T > 0.0 and score > 0.0:
            ok = (np.random.random() <
                  jump_scale * math.exp(-jump_coef * (score_new / score) / T))
        if ok:
            n_ok += 1
            score = score_new
        else:
            pos[p, 0] -= dx;
            pos[p, 1] -= dy;
            pos[p, 2] -= dz
        T *= dt
    return T, score, n_ok


# ---------------------------------------------------------------------------
# Shared helper

def _as_f64(arr):
    return np.ascontiguousarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public MC loops

def mc_heatmap(
    pos: np.ndarray,  # (N, 3) float32 - modified in place
    exp_dist: np.ndarray,  # (N, N) - expected pairwise distances
    diag_size: int,
    step_size: float,
    settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """
    MonteCarloHeatmap: simulated annealing using heatmap distance energy.

    Global score is double-counted, so the MC update rule is:
        score += 2 * (local_curr - local_prev)

    Mirrors C++ LooperSolver::MonteCarloHeatmap().  Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    idx = np.arange(n)
    diag_mask = np.abs(idx[:, None] - idx[None, :]) < diag_size
    skip = diag_mask | (exp_dist < 1e-6)
    exp_safe = np.where(skip, 1.0, exp_dist)

    T = float(settings.max_temp_heatmap)
    dt = float(settings.dt_temp_heatmap)
    jump_scale = float(settings.jump_scale_heatmap)
    jump_coef = float(settings.jump_coef_heatmap)
    stop_steps = int(settings.mc_stop_steps_heatmap)
    stop_improvement = float(settings.mc_stop_improvement_heatmap)
    stop_successes = int(settings.mc_stop_successes_heatmap)

    prefix = f"    [{label}] " if label else "    "

    pw = _as_f64(pos)
    es64 = _as_f64(exp_safe)
    skip_b = np.ascontiguousarray(skip, dtype=np.bool_)
    score = float(_init_heatmap_nb(pw, es64, skip_b))

    ms_score = score
    step_i = 0
    while True:
        T, score, n_ok = _batch_heatmap_nb(
            pw, es64, skip_b, float(step_size), T, dt,
            jump_scale, jump_coef, stop_steps, score)
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < 1e-6
        )
        if verbose:
            print(f"{prefix}step {step_i:>7,}  score={score:.4f}"
                  f"  ratio={ratio:.4f}  ok={n_ok}/{stop_steps}"
                  + ("  [done]" if converged else ""), flush=True)
        if converged:
            break
        ms_score = score

    pos[:] = pw.astype(pos.dtype)
    return score


def mc_arcs(
    pos: np.ndarray,  # (N, 3) float32 - modified in place
    exp_dist_mat: np.ndarray,  # (N, N) - -1=repulsion, 0=none, >0=spring distance
    step_size: float,
    settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """
    MonteCarloArcs: simulated annealing using arc spring energy.

    Global score counts i < j pairs once.  Local score sums all other beads,
    so the MC update rule is:
        score = score - local_prev + local_curr   (no factor 2)

    Mirrors C++ LooperSolver::MonteCarloArcs().  Returns final score.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0

    T = float(settings.max_temp)
    dt = float(settings.dt_temp)
    jump_scale = float(settings.jump_scale)
    jump_coef = float(settings.jump_coef)
    stop_steps = int(settings.mc_stop_steps)
    stop_improvement = float(settings.mc_stop_improvement)
    stop_successes = int(settings.mc_stop_successes)
    stretch_k = float(settings.spring_stretch_arcs)
    squeeze_k = float(settings.spring_squeeze_arcs)

    prefix = f"    [{label}] " if label else "    "

    pw = _as_f64(pos)
    exp64 = _as_f64(exp_dist_mat)
    score = float(_init_arcs_nb(pw, exp64, stretch_k, squeeze_k))

    ms_score = score
    step_i = 0
    while True:
        T, score, n_ok = _batch_arcs_nb(
            pw, exp64, float(step_size), T, dt, jump_scale, jump_coef,
            stop_steps, stretch_k, squeeze_k, score)
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < 1e-5 or ratio > 0.9999
        )
        if verbose:
            print(f"{prefix}step {step_i:>7,}  score={score:.4f}"
                  f"  ratio={ratio:.4f}  ok={n_ok}/{stop_steps}"
                  + ("  [done]" if converged else ""), flush=True)
        if converged:
            break
        ms_score = score

    pos[:] = pw.astype(pos.dtype)
    return score


def mc_smooth(
    pos: np.ndarray,  # (N, 3) float32 - modified in place; anchors are fixed
    dtn: np.ndarray,  # (N-1,) expected distances between consecutive beads
    fixed: np.ndarray,  # (N,) bool - True for anchor beads (never moved)
    step_size: float,
    settings,
    char_orientations: np.ndarray = None,  # (N,) CTCF orientation chars; None = no motif
    anchor_neighbors: dict = None,  # {anchor_k: [anchor_j, ...]}
    anchor_neighbor_weights: dict = None,  # {anchor_k: [float, ...]}
    label: str = "",
    verbose: bool = False,
) -> float:
    """
    MonteCarloArcsSmooth: chain connectivity + angle MC.

    When char_orientations is provided and settings.use_ctcf_motif is True,
    also optimises CTCF orientation energy via a pure-Python fallback loop
    (Mirrors C++ MonteCarloArcsSmooth with useCTCFMotifOrientation=True).

    Anchor beads (fixed=True) are never moved.  Returns final score.
    """
    n = pos.shape[0]
    if n <= 2:
        return 0.0

    T = float(settings.max_temp_smooth)
    dt = float(settings.dt_temp_smooth)
    jump_scale = float(settings.jump_scale_smooth)
    jump_coef = float(settings.jump_coef_smooth)
    stop_steps = int(settings.mc_stop_steps_smooth)
    stop_improvement = float(settings.mc_stop_improvement_smooth)
    stop_successes = int(settings.mc_stop_successes_smooth)
    stretch_k = float(settings.spring_stretch)
    squeeze_k = float(settings.spring_squeeze)
    ang_k = float(settings.spring_angular)
    dist_w = float(settings.smooth_dist_weight)
    ang_w = float(settings.smooth_angle_weight)

    use_orn = (
        char_orientations is not None
        and anchor_neighbors is not None
        and getattr(settings, "use_ctcf_motif", False)
    )
    motif_weight = float(getattr(settings, "motif_weight", 1.0))
    motifs_symmetric = bool(getattr(settings, "motifs_symmetric", True))

    movable = np.where(~fixed)[0]
    if len(movable) == 0:
        return 0.0

    prefix = f"    [{label}] " if label else "    "

    pw = _as_f64(pos)
    dtn64 = _as_f64(dtn)

    if use_orn:
        from .energy import calc_orientation as _calc_orn
        # Build CSR neighbor arrays (computed once before the MC loop)
        anchor_ar = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = len(anchor_ar)
        nbr_offsets = np.zeros(n_anchors + 1, dtype=np.int32)
        for _k in range(n_anchors):
            nbr_offsets[_k + 1] = nbr_offsets[_k] + len(anchor_neighbors.get(_k, []))
        _total = int(nbr_offsets[n_anchors])
        nbr_indices = np.empty(_total, dtype=np.int32)
        nbr_weights_arr = np.empty(_total, dtype=np.float64)
        for _k in range(n_anchors):
            for _ki, (_j, _w) in enumerate(zip(anchor_neighbors.get(_k, []),
                                               anchor_neighbor_weights.get(_k, []))):
                _off = nbr_offsets[_k] + _ki
                nbr_indices[_off] = _j
                nbr_weights_arr[_off] = _w
        # bool flag per bead: True if orientation is 'L'
        orn_is_L = np.array([c == 'L' for c in char_orientations], dtype=np.bool_)
        # bead_to_anchor_k[i] = k if bead i is adjacent to anchor k, else -1
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        for _k in range(n_anchors):
            _ar = int(anchor_ar[_k])
            if _ar > 0:
                bead_to_anchor_k[_ar - 1] = _k
            if _ar + 1 < n:
                bead_to_anchor_k[_ar + 1] = _k
        # Initial anchor orientation matrix (updated in-place across batches)
        anchor_orn = np.zeros((n_anchors, 3), dtype=np.float64)
        for _k in range(n_anchors):
            _ar = int(anchor_ar[_k])
            anchor_orn[_k] = _calc_orn(pw, _ar, n, char_orientations[_ar])
        mov_orn = np.ascontiguousarray(movable, dtype=np.int64)
        score = (float(_init_smooth_nb(pw, dtn64, stretch_k, squeeze_k, ang_k, dist_w, ang_w))
                 + float(_score_orientation_full_nb(anchor_orn, nbr_offsets, nbr_indices,
                                                    nbr_weights_arr, motif_weight, motifs_symmetric)))
    else:
        mov64 = np.ascontiguousarray(movable, dtype=np.int64)
        score = float(_init_smooth_nb(pw, dtn64, stretch_k, squeeze_k,
                                      ang_k, dist_w, ang_w))

    ms_score = score
    step_i = 0
    while True:
        if use_orn:
            T, score, n_ok = _batch_smooth_orientation_nb(
                pw, dtn64, mov_orn, orn_is_L, anchor_ar,
                nbr_offsets, nbr_indices, nbr_weights_arr,
                anchor_orn, bead_to_anchor_k,
                float(step_size), T, dt, jump_scale, jump_coef, stop_steps,
                stretch_k, squeeze_k, ang_k, dist_w, ang_w,
                motif_weight, motifs_symmetric, score)
        else:
            T, score, n_ok = _batch_smooth_nb(
                pw, dtn64, mov64, float(step_size), T, dt,
                jump_scale, jump_coef, stop_steps,
                stretch_k, squeeze_k, ang_k, dist_w, ang_w, score)
        step_i += stop_steps
        ratio = score / ms_score if ms_score > 0 else 1.0
        converged = (
            (score > stop_improvement * ms_score and n_ok < stop_successes)
            or score < 1e-6
        )
        if verbose:
            print(f"{prefix}step {step_i:>7,}  score={score:.4f}"
                  f"  ratio={ratio:.4f}  ok={n_ok}/{stop_steps}"
                  + ("  [done]" if converged else ""), flush=True)
        if converged:
            break
        ms_score = score

    pos[:] = pw.astype(pos.dtype)
    return score
