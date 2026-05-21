"""
Monte Carlo simulation loops for 3dgnome-ng.

Mirrors Reference LooperSolver::MonteCarloHeatmap(), MonteCarloArcs(), and
MonteCarloArcsSmooth().

On first import the JIT functions compile (~10-30 s); subsequent runs
load from cache

Acceptance criterion (all loops):
    ok = (score_new <= score_curr)
      or rand() < jump_scale * exp(-jump_coef * score_new/score_curr / T)
"""

import math

import numpy as np
from numba import njit


# Smooth MC helpers

@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _init_smooth_nb(pos, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w):
    n = pos.shape[0]
    sc = 0.0
    for i in range(n - 1):
        sc += _smooth_len_nb(pos, dtn, i, stretch_k, squeeze_k, dist_w)
    for i in range(n - 2):
        sc += _smooth_ang_nb(pos, i, ang_k, ang_w)
    return sc


# Excluded-volume helpers (harmonic soft repulsion, cutoff at r0)
#
#   E_pair(d) = weight * ((r0 - d) / r0)^2   if d < r0
#             = 0                            otherwise
#
# Normalized by r0 so `weight` is dimensionally comparable to spring constants.
# Global score double-counts pairs (matches the heat-energy convention):
# sum_{i != j, |i-j| > skip} E_pair(d_ij). Delta is 2 * (local_curr - local_prev).

@njit(cache=True)
def _excl_pair_nb(d, r0, weight):
    if d >= r0:
        return 0.0
    rel = (r0 - d) / r0
    return weight * rel * rel


@njit(cache=True)
def _local_excl_nb(pos, p, r0, weight, skip):
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        diff = i - p
        if diff < 0:
            diff = -diff
        if diff <= skip:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        err += _excl_pair_nb(d, r0, weight)
    return err


@njit(cache=True)
def _init_excl_nb(pos, r0, weight, skip):
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            if diff <= skip:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            err += _excl_pair_nb(d, r0, weight)
    return err


@njit(cache=True)
def _batch_smooth_kernel_nb(
    pos, dtn, movable, step_size, T, dt, jump_scale, jump_coef, n_steps,
    stretch_k, squeeze_k, ang_k, dist_w, ang_w,
    use_heat, heat_dist, heat_weight,
    use_orn, orn_is_L, anchor_ar,
    nbr_offsets, nbr_indices, nbr_weights,
    anchor_orn, bead_to_anchor_k,
    motif_weight, symmetric,
    use_excl, excl_r0, excl_weight, excl_skip,
    score_struct, score_orn, score_heat, score_excl,
):
    """Unified smooth-MC kernel.  Energy terms (toggled by flags):
      * structure   (always on): incremental delta via _local_smooth_nb
      * heat        (use_heat):  incremental delta via _local_heat_nb, 2x factor
      * orientation (use_orn):   incremental delta via weighted local, 2x factor
      * excl. vol   (use_excl):  incremental delta via _local_excl_nb, 2x factor

    Returns (T_out, score_struct, score_orn, score_heat, score_excl, n_ok).
    Disabled-term arrays must still be valid-typed (any shape) - they are not
    indexed when their flag is False.
    """
    n = pos.shape[0]
    n_mov = movable.shape[0]
    n_ok = 0
    score = score_struct + score_orn + score_heat + score_excl
    for _ in range(n_steps):
        p = movable[np.random.randint(0, n_mov)]
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_struct_prev = _local_smooth_nb(pos, dtn, p, n,
                                           stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        loc_heat_prev = 0.0
        if use_heat:
            loc_heat_prev = _local_heat_nb(pos, heat_dist, p, heat_weight)

        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        orn_k = -1
        prev_ox = 0.0
        prev_oy = 0.0
        prev_oz = 0.0
        loc_orn_prev = 0.0
        if use_orn:
            orn_k = bead_to_anchor_k[p]
            if orn_k >= 0:
                prev_ox = anchor_orn[orn_k, 0]
                prev_oy = anchor_orn[orn_k, 1]
                prev_oz = anchor_orn[orn_k, 2]
                loc_orn_prev = _local_score_orientation_nb(
                    anchor_orn, orn_k, nbr_offsets, nbr_indices, nbr_weights,
                    motif_weight, symmetric)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_struct_curr = _local_smooth_nb(pos, dtn, p, n,
                                           stretch_k, squeeze_k, ang_k, dist_w, ang_w)
        score_struct_new = score_struct - loc_struct_prev + loc_struct_curr

        score_heat_new = score_heat
        if use_heat:
            loc_heat_curr = _local_heat_nb(pos, heat_dist, p, heat_weight)
            score_heat_new = score_heat + 2.0 * (loc_heat_curr - loc_heat_prev)

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_orn_new = score_orn
        if use_orn and orn_k >= 0:
            ar = anchor_ar[orn_k]
            ox, oy, oz = _calc_orientation_nb(pos, ar, n, orn_is_L[ar])
            anchor_orn[orn_k, 0] = ox
            anchor_orn[orn_k, 1] = oy
            anchor_orn[orn_k, 2] = oz
            loc_orn_curr = _local_score_orientation_nb(
                anchor_orn, orn_k, nbr_offsets, nbr_indices, nbr_weights,
                motif_weight, symmetric)
            score_orn_new = score_orn + 2.0 * (loc_orn_curr - loc_orn_prev)

        score_new = score_struct_new + score_orn_new + score_heat_new + score_excl_new

        ok = score_new < score
        if not ok and T > 0.0 and score > 0.0:
            ok = (np.random.random() <
                  jump_scale * math.exp(-jump_coef * (score_new / score) / T))
        if ok:
            n_ok += 1
            score = score_new
            score_struct = score_struct_new
            score_orn = score_orn_new
            score_heat = score_heat_new
            score_excl = score_excl_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
            if use_orn and orn_k >= 0:
                anchor_orn[orn_k, 0] = prev_ox
                anchor_orn[orn_k, 1] = prev_oy
                anchor_orn[orn_k, 2] = prev_oz
        T *= dt
    return T, score_struct, score_orn, score_heat, score_excl, n_ok


# Orientation MC helpers

@njit(cache=True)
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
        ox = -ox
        oy = -oy
        oz = -oz
    nm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if nm > 1e-12:
        ox /= nm
        oy /= nm
        oz /= nm
    return ox, oy, oz


@njit(cache=True)
def _score_orientation_full_nb(anchor_orn, nbr_offsets, nbr_indices, nbr_weights,
                               motif_weight, symmetric):
    """Global orientation score with arc weights; used for initialisation only."""
    n_anchors = anchor_orn.shape[0]
    err = 0.0
    for i in range(n_anchors):
        for ki in range(nbr_offsets[i], nbr_offsets[i + 1]):
            j = nbr_indices[ki]
            w = nbr_weights[ki]
            ax = anchor_orn[i, 0]
            ay = anchor_orn[i, 1]
            az = anchor_orn[i, 2]
            bx = anchor_orn[j, 0]
            by = anchor_orn[j, 1]
            bz = anchor_orn[j, 2]
            if not symmetric:
                bx = -bx
                by = -by
                bz = -bz
            dot = ax * bx + ay * by + az * bz
            ang = 1.0 - (dot + 1.0) * 0.5
            err += ang * ang * w
    return err * motif_weight


@njit(cache=True)
def _local_score_orientation_nb(anchor_orn, k, nbr_offsets, nbr_indices,
                                nbr_weights, motif_weight, symmetric):
    """Local orientation score for anchor k, WEIGHTED by per-arc weights.
    Used for the incremental update: score_orn += 2*(local_curr - local_prev).
    The weights make this delta exact w.r.t. _score_orientation_full_nb - no drift.
    Diverges from Reference calcScoreOrientation(orn, anchor_index), which is unweighted
    and therefore drifts; see [[project-orientation-mc-fix]].
    """
    err = 0.0
    for ki in range(nbr_offsets[k], nbr_offsets[k + 1]):
        j = nbr_indices[ki]
        w = nbr_weights[ki]
        ax = anchor_orn[k, 0]
        ay = anchor_orn[k, 1]
        az = anchor_orn[k, 2]
        bx = anchor_orn[j, 0]
        by = anchor_orn[j, 1]
        bz = anchor_orn[j, 2]
        if not symmetric:
            bx = -bx
            by = -by
            bz = -bz
        dot = ax * bx + ay * by + az * bz
        ang = 1.0 - (dot + 1.0) * 0.5
        err += ang * ang * w
    return err * motif_weight


@njit(cache=True)
def _local_heat_nb(pos, heat_dist, p, heat_weight):
    """Local heat score for bead p vs all others.
    Mirrors Reference calcScoreSubanchorHeatmap(int moved) - sums all i != p.
    """
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        if i == p:
            continue
        exp_d = heat_dist[i, p]
        if exp_d < 1e-6:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (d - exp_d) / exp_d
        err += rel * rel
    return err * heat_weight


@njit(cache=True)
def _init_heat_nb(pos, heat_dist, heat_weight):
    """Global heat score (double-counts pairs, matching Reference calcScoreSubanchorHeatmap())."""
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            exp_d = heat_dist[i, j]
            if exp_d < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            rel = (d - exp_d) / exp_d
            err += rel * rel
    return err * heat_weight


# Arcs MC helpers

@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _batch_arcs_nb(pos, exp, step_size, T, dt, jump_scale, jump_coef,
                   n_steps, stretch_k, squeeze_k,
                   use_excl, excl_r0, excl_weight, excl_skip,
                   score_arcs, score_excl):
    """Run n_steps arc-MC steps with optional excluded-volume energy.

    Returns (T_out, score_arcs, score_excl, n_ok).
    Arc spring score uses (curr - prev) delta; excl uses 2*(curr - prev).
    """
    n = pos.shape[0]
    n_ok = 0
    score = score_arcs + score_excl
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dz = np.random.uniform(-step_size, step_size)

        loc_arc_prev = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)
        loc_excl_prev = 0.0
        if use_excl:
            loc_excl_prev = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)

        pos[p, 0] += dx
        pos[p, 1] += dy
        pos[p, 2] += dz

        loc_arc_curr = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)
        score_arcs_new = score_arcs - loc_arc_prev + loc_arc_curr

        score_excl_new = score_excl
        if use_excl:
            loc_excl_curr = _local_excl_nb(pos, p, excl_r0, excl_weight, excl_skip)
            score_excl_new = score_excl + 2.0 * (loc_excl_curr - loc_excl_prev)

        score_new = score_arcs_new + score_excl_new
        ok = score_new <= score
        if not ok and score > 0.0 and T > 0.0:
            ok = (np.random.random() < jump_scale * math.exp(-jump_coef * (score_new / score) / T))

        if ok:
            n_ok += 1
            score = score_new
            score_arcs = score_arcs_new
            score_excl = score_excl_new
        else:
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score_arcs, score_excl, n_ok


# Heatmap MC helpers

@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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
        pos[p, 0] += dx
        pos[p, 1] += dy
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
            pos[p, 0] -= dx
            pos[p, 1] -= dy
            pos[p, 2] -= dz
        T *= dt
    return T, score, n_ok


# Shared helper

def _as_f64(arr):
    return np.ascontiguousarray(arr, dtype=np.float64)


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

    Mirrors Reference LooperSolver::MonteCarloHeatmap().  Returns final score.
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

    Mirrors Reference LooperSolver::MonteCarloArcs().  Returns final score.
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

    use_excl = (bool(getattr(settings, "use_excluded_volume", False))
                and bool(getattr(settings, "exclusion_apply_to_arcs", False)))
    excl_r0 = float(getattr(settings, "exclusion_radius", 1.0))
    excl_weight = float(getattr(settings, "exclusion_weight", 1.0))
    excl_skip = int(getattr(settings, "exclusion_skip_neighbors", 1))

    prefix = f"    [{label}] " if label else "    "

    pw = _as_f64(pos)
    exp64 = _as_f64(exp_dist_mat)
    score_arcs = float(_init_arcs_nb(pw, exp64, stretch_k, squeeze_k))
    score_excl = float(_init_excl_nb(pw, excl_r0, excl_weight, excl_skip)) if use_excl else 0.0
    score = score_arcs + score_excl

    ms_score = score
    step_i = 0
    while True:
        T, score_arcs, score_excl, n_ok = _batch_arcs_nb(
            pw, exp64, float(step_size), T, dt, jump_scale, jump_coef,
            stop_steps, stretch_k, squeeze_k,
            use_excl, excl_r0, excl_weight, excl_skip,
            score_arcs, score_excl)
        score = score_arcs + score_excl
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
    heat_dist: np.ndarray = None,  # (N, N) subanchor heat expected distances; None = disabled
    label: str = "",
    verbose: bool = False,
) -> float:
    """
    MonteCarloArcsSmooth: chain connectivity + angle MC.

    Optionally adds CTCF orientation energy (char_orientations) and/or
    subanchor heat energy (heat_dist).  Mirrors Reference MonteCarloArcsSmooth
    with useCTCFMotifOrientation and use_subanchor_heatmap flags.

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
    use_heat = heat_dist is not None
    motif_weight = float(getattr(settings, "motif_weight", 1.0))
    motifs_symmetric = bool(getattr(settings, "motifs_symmetric", True))
    heat_weight = float(getattr(settings, "subanchor_heatmap_dist_weight", 0.01))

    use_excl = (bool(getattr(settings, "use_excluded_volume", False))
                and bool(getattr(settings, "exclusion_apply_to_smooth", False)))
    excl_r0 = float(getattr(settings, "exclusion_radius", 1.0))
    excl_weight = float(getattr(settings, "exclusion_weight", 1.0))
    excl_skip = int(getattr(settings, "exclusion_skip_neighbors", 1))

    movable = np.where(~fixed)[0]
    if len(movable) == 0:
        return 0.0

    prefix = f"    [{label}] " if label else "    "

    pw = _as_f64(pos)
    dtn64 = _as_f64(dtn)
    mov64 = np.ascontiguousarray(movable, dtype=np.int64)

    # Heat state (dummy when disabled - never indexed inside the kernel)
    if use_heat:
        heat64 = _as_f64(heat_dist)
        score_heat = float(_init_heat_nb(pw, heat64, heat_weight))
    else:
        heat64 = np.zeros((1, 1), dtype=np.float64)
        score_heat = 0.0

    # Orientation state (dummy when disabled)
    if use_orn:
        from .energy import calc_orientation as _calc_orn
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
        orn_is_L = np.array([c == 'L' for c in char_orientations], dtype=np.bool_)
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        for _k in range(n_anchors):
            _ar = int(anchor_ar[_k])
            if _ar > 0:
                bead_to_anchor_k[_ar - 1] = _k
            if _ar + 1 < n:
                bead_to_anchor_k[_ar + 1] = _k
        anchor_orn = np.zeros((n_anchors, 3), dtype=np.float64)
        for _k in range(n_anchors):
            _ar = int(anchor_ar[_k])
            anchor_orn[_k] = _calc_orn(pw, _ar, n, char_orientations[_ar])
        score_orn = float(_score_orientation_full_nb(
            anchor_orn, nbr_offsets, nbr_indices, nbr_weights_arr,
            motif_weight, motifs_symmetric))
    else:
        anchor_ar = np.zeros(1, dtype=np.int32)
        nbr_offsets = np.zeros(2, dtype=np.int32)
        nbr_indices = np.zeros(1, dtype=np.int32)
        nbr_weights_arr = np.zeros(1, dtype=np.float64)
        orn_is_L = np.zeros(1, dtype=np.bool_)
        bead_to_anchor_k = np.full(n, -1, dtype=np.int32)
        anchor_orn = np.zeros((1, 3), dtype=np.float64)
        score_orn = 0.0

    score_struct = float(_init_smooth_nb(pw, dtn64, stretch_k, squeeze_k,
                                         ang_k, dist_w, ang_w))
    score_excl = float(_init_excl_nb(pw, excl_r0, excl_weight, excl_skip)) if use_excl else 0.0
    score = score_struct + score_orn + score_heat + score_excl

    ms_score = score
    step_i = 0
    while True:
        T, score_struct, score_orn, score_heat, score_excl, n_ok = _batch_smooth_kernel_nb(
            pw, dtn64, mov64, float(step_size), T, dt, jump_scale, jump_coef,
            stop_steps, stretch_k, squeeze_k, ang_k, dist_w, ang_w,
            use_heat, heat64, heat_weight,
            use_orn, orn_is_L, anchor_ar,
            nbr_offsets, nbr_indices, nbr_weights_arr,
            anchor_orn, bead_to_anchor_k,
            motif_weight, motifs_symmetric,
            use_excl, excl_r0, excl_weight, excl_skip,
            score_struct, score_orn, score_heat, score_excl,
        )
        score = score_struct + score_orn + score_heat + score_excl
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
