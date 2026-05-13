"""
Monte Carlo optimisation phases — cudaMMC parity rewrite (Phase 4).

  * monte_carlo_heatmap        ←  LooperSolver.cpp:421-518   MonteCarloHeatmap
  * monte_carlo_arcs           ←  LooperSolver.cpp:3058-3159 MonteCarloArcs
  * monte_carlo_arcs_smooth    ←  LooperSolver.cpp:3161-3390 MonteCarloArcsSmooth

Every algorithmic line carries a ``# cudaMMC LINE`` annotation.  The three
loops follow upstream byte-for-byte:

  * Heatmap MC:   **full recompute every iteration** (cpp:445, cpp:468); no
                  delta tracking, no milestone resync.  Metropolis branch
                  gated on ``T > 0`` (cpp:471).  Stop = ``(score > impr*ms
                  && succ < min) || score < 1e-6`` (cpp:501-504).
  * Arcs MC:      **delta tracking** ``score_curr += local_curr - local_prev``
                  (cpp:3109).  Metropolis UN-gated on T (cpp:3115 has no
                  ``T > 0`` check).  Stop adds the direction-agnostic
                  ``score_curr / milestone > 0.9999`` (cpp:3146) and
                  ``score < 1e-5``.
  * Smooth MC:    delta tracking with separate orientation factor 2×
                  (cpp:3311-3313).  Metropolis gated on ``T > 0``
                  (cpp:3331).  Strict ``<`` accept (cpp:3329).  Stop =
                  ``(score > impr*ms && succ < min) || score < 1e-6``
                  (cpp:3372-3376) — **no** ratio>0.9999.

Random-displacement helper uses uniform ``(2u-1)*step`` per axis
(common.cpp:14-25; ParallelMonteCarloHeatmap.cu:75-80).
"""

import math
import random
import time
from typing import List, Optional

import torch

from .scores import (
    score_heatmap_chunked,
    score_arcs,
    score_arcs_single,
    score_distances_active_region,
    score_distances_active_region_single,
    score_chain_single,
    score_structure_smooth,
    score_orientation,
    score_orientation_single,
    _calc_orientation_vectors,
)
from .settings import Settings


# ── Helpers ─────────────────────────────────────────────────────────────────

def _random_displacement(step_size: float, use_2d: bool,
                          device: torch.device) -> torch.Tensor:
    """Uniform ``(2u-1)*step`` per axis (cudaMMC common.cpp:14-25)."""
    v = (torch.rand(3, device=device) * 2.0 - 1.0) * step_size
    if use_2d:
        v[2] = 0.0
    return v


def _accept_metropolis(jump_scale: float, jump_coef: float,
                       score_curr: float, score_prev: float,
                       T: float, rng: random.Random) -> bool:
    """Ratio-based Metropolis (cudaMMC cpp:3114-3116 / cpp:471-475).

    ``tp = jump_scale * exp(-jump_coef * (score_curr / score_prev) / T)``
    Caller enforces the upstream ``T > 0`` gate where it applies (heatmap,
    smooth) or omits it (arcs).  **No greedy fallback** and **no
    ``score_prev > 0`` guard** — cudaMMC has neither (AUDIT §D1, §F8;
    cpp:471-475, 3113-3117, 3331-3335).  The upstream score is a sum of
    non-negative spring/heatmap terms, so ``score_prev`` cannot legitimately
    be ≤ 0; if a port bug ever drives it to 0, the resulting ``inf`` / ``nan``
    will be rejected by the ``<`` accept check immediately above this call.
    Only the underflow guard ``arg < -700`` is kept to avoid ``OverflowError``.
    """
    arg = -jump_coef * (score_curr / score_prev) / T
    if arg < -700.0:
        return False
    tp = jump_scale * math.exp(arg)
    return rng.random() < tp


def _random_displacements(n: int, step_size: float, use_2d: bool,
                           device: torch.device) -> torch.Tensor:
    """Vectorised ``(2u-1)*step`` per axis for ``n`` beads at once.

    Mirrors cudaMMC ``random_vector`` (common.cpp:14-25) — same uniform
    distribution as :func:`_random_displacement`, just batched for the
    per-restart re-noising sites in :mod:`solver` (AUDIT §G8).
    """
    v = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * step_size
    if use_2d:
        v[:, 2] = 0.0
    return v


# ============================================================================
# Phase 1 — MonteCarloHeatmap   (LooperSolver.cpp:421-518)
# ============================================================================
# CPU reference; the GPU production path is ParallelMonteCarloHeatmap.cu.
# cudaMMC recomputes the FULL double-counted heatmap score every iteration
# (cpp:445 + cpp:468) — no delta tracking, no milestone resync.  Our
# ``_full_score`` already multiplies the upper triangle by 2 to match
# cudaMMC's double-counted scale, so the absolute stop threshold ports
# verbatim as ``< 1e-6`` (cpp:504, AUDIT §D4).

def monte_carlo_heatmap(
    pos: torch.Tensor,            # (N, 3) float32
    expected: torch.Tensor,       # (N, N) fp32 expected-distance matrix
    fixed_mask: torch.Tensor,     # (N,) bool
    settings: Settings,
    same_chr_mask: Optional[torch.Tensor] = None,
    *,
    step_size: Optional[float] = None,
    diagonal_size: int = 1,
    verbose: bool = True,
) -> torch.Tensor:
    """Sequential single-bead heatmap MC — cpp:421-518."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    if N <= 1:                                              # cpp:441-442
        return pos
    rng = random.Random()

    # cpp:423: step_size *= 0.5
    step = (step_size if step_size is not None else s.step_size_heatmap) * 0.5
    # cpp:425: T = Settings::maxTempHeatmap
    T = s.max_temp_heatmap

    # cpp:445: score_curr = calcScoreHeatmapActiveRegion()
    # We sum upper-triangle and double → cudaMMC's double-counted total.
    def _full_score() -> float:
        return 2.0 * score_heatmap_chunked(pos, expected, diagonal_size,
                                           same_chr_mask).item()

    score_curr = _full_score()
    score_prev = score_curr                                 # cpp:448
    milestone_score = score_curr                            # cpp:449
    milestone_success = 0                                   # cpp:436
    i = 0

    if verbose:
        print(f"  [heatmap MC] N={N}  T={T:.1f}  step={step:.4f}  "
              f"score={score_curr:.6f}")

    _t_start = time.time()
    _t_last = _t_start
    # cpp:456: while (true)
    while True:
        p = rng.randrange(N)                                 # cpp:459
        if fixed_mask[p]:                                    # cpp:462-463
            continue

        # cpp:465-467: displacement; pos += disp
        disp = _random_displacement(step, s.use_2d, device)
        pos[p] += disp

        # cpp:468: FULL recompute — no delta tracking.
        score_curr = _full_score()

        # cpp:469: ok = score_curr <= score_prev
        ok = score_curr <= score_prev

        # cpp:471-475: Metropolis branch, gated on T > 0
        if not ok and T > 0.0:
            ok = _accept_metropolis(s.temp_jump_scale_heatmap,
                                    s.temp_jump_coef_heatmap,
                                    score_curr, score_prev, T, rng)

        if ok:                                               # cpp:477-480
            milestone_success += 1
        else:                                                # cpp:481-484
            pos[p] -= disp
            score_curr = score_prev                          # cpp:483

        T *= s.dt_temp_heatmap                               # cpp:486
        i += 1                                               # cpp:514

        # ── heartbeat: print progress every ~5 s wall time even if the
        # milestone interval is far off.  Helps tell a slow loop from a
        # truly stuck one; does NOT alter the algorithm.
        if verbose and (time.time() - _t_last) > 5.0:
            _t_last = time.time()
            its = i / max(_t_last - _t_start, 1e-9)
            print(f"  [heatmap MC] ..i={i:7d}  T={T:.5f}  "
                  f"score={score_curr:.6f}  {its:.0f} it/s")

        # cpp:488-509: milestone check
        if i % s.mc_stop_steps_heatmap == 0:
            ratio = score_curr / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [heatmap MC] i={i:7d}  T={T:.5f}  score={score_curr:.6f}  "
                      f"ratio={ratio:.5f}  ms_succ={milestone_success}")
            # cpp:501-504: ONLY these two stop clauses
            if ((score_curr > s.mc_stop_improvement_heatmap * milestone_score
                    and milestone_success < s.mc_stop_min_successes_heatmap)
                    or score_curr < 1e-6):
                break
            milestone_score = score_curr                     # cpp:507
            milestone_success = 0                            # cpp:508

        score_prev = score_curr                              # cpp:513

    return pos


# ============================================================================
# Phase 2 — MonteCarloArcs   (LooperSolver.cpp:3058-3159)
# ============================================================================
# Delta tracking: ``score_curr += local_curr - local_prev`` (cpp:3109).
# Metropolis UN-gated on T (cpp:3115).  Stop adds the direction-agnostic
# ``score_curr / milestone > 0.9999`` clause (cpp:3146).
#
# AUDIT §G1: the score function is :func:`score_distances_active_region`
# operating on the **dense per-IB expected-distance matrix** (cpp:1919-1984).
# The dense matrix has ``-1`` at non-arc pairs (repulsion sentinel,
# cpp:1932-1934) so every (i,j) contributes either a spring (arc-connected),
# a ``1/d`` repulsion (full score only — cudaMMC's single-bead variant
# *comments out* the repulsion, cpp:1967-1969), or is skipped (``< 1e-6``).

def monte_carlo_arcs(
    pos: torch.Tensor,
    expected: torch.Tensor,         # (N, N) fp32 dense matrix, -1 = repulsion
    fixed_mask: torch.Tensor,
    settings: Settings,
    *,
    step_size: Optional[float] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Single-bead arc MC on the dense per-IB matrix — cpp:3058-3159, AUDIT §G1.

    ``expected`` is the ``heatmap_exp_dist_anchor`` matrix built by
    :meth:`LooperSolver._build_anchor_expected_dist_ib` (mirror of
    ``calcAnchorExpectedDistancesHeatmap``, cpp:3837-3916): dense ``N×N``,
    defaulted to ``-1`` (repulsion), diagonal cleared, arc pairs overwritten
    with ``freqToDistance(freq)``.
    """
    s = settings
    device = pos.device
    N = pos.shape[0]
    if N <= 1:
        return pos
    rng = random.Random()

    # cpp:3062-3064
    T = s.max_temp_arcs
    step = step_size if step_size is not None else s.step_size_arcs

    k_stretch = s.spring_constant_stretch_arcs
    k_squeeze = s.spring_constant_squeeze_arcs

    def _full_total() -> float:
        return score_distances_active_region(
            pos, expected,
            k_stretch=k_stretch, k_squeeze=k_squeeze,
            k_repulsion=1.0,
        ).item()

    def _single(idx: int) -> float:
        # cpp:1967-1969: single-bead repulsion is commented out upstream.
        return score_distances_active_region_single(
            pos, idx, expected,
            k_stretch=k_stretch, k_squeeze=k_squeeze,
            include_repulsion=False,
        ).item()

    score_curr = _full_total()                              # cpp:3086
    score_prev = score_curr                                 # cpp:3088
    milestone_score = score_curr                            # cpp:3089
    milestone_success = 0                                   # cpp:3077
    i = 0

    if verbose:
        print(f"  [arcs MC] N={N}  T={T:.1f}  step={step:.4f}  "
              f"score={score_curr:.6f}")

    _t_start = time.time()
    _t_last = _t_start
    # cpp:3094: while (true)
    while True:
        p = rng.randrange(N)                                 # cpp:3096
        # cpp:3099-3100: cudaMMC aborts on a fixed bead at arcs level.
        # We silently skip for now — Phase 5 will switch to strict abort.
        if fixed_mask[p]:
            continue

        local_prev = _single(p)                              # cpp:3102

        disp = _random_displacement(step, s.use_2d, device)  # cpp:3104
        pos[p] += disp                                       # cpp:3105

        local_curr = _single(p)                              # cpp:3107

        score_curr = score_prev - local_prev + local_curr    # cpp:3109

        ok = score_curr <= score_prev                        # cpp:3111

        # cpp:3113-3117: Metropolis — NO `T > 0` gate in arcs phase.
        if not ok:
            ok = _accept_metropolis(s.temp_jump_scale_arcs,
                                    s.temp_jump_coef_arcs,
                                    score_curr, score_prev, T, rng)

        if ok:                                                  # cpp:3119-3124
            milestone_success += 1
        else:                                                   # cpp:3125-3128
            pos[p] -= disp
            score_curr = score_prev

        T *= s.dt_temp_arcs                                     # cpp:3130
        i += 1                                                  # cpp:3155

        # heartbeat: see monte_carlo_heatmap
        if verbose and (time.time() - _t_last) > 5.0:
            _t_last = time.time()
            its = i / max(_t_last - _t_start, 1e-9)
            print(f"  [arcs MC] ..i={i:7d}  T={T:.5f}  "
                  f"score={score_curr:.6f}  {its:.0f} it/s")

        # Python-only safety cap (Settings.mc_max_iters_arcs /
        # mc_max_seconds_arcs).  0 ⇒ disabled, matching cudaMMC exactly.
        if ((s.mc_max_iters_arcs > 0 and i >= s.mc_max_iters_arcs)
                or (s.mc_max_seconds_arcs > 0.0
                    and time.time() - _t_start > s.mc_max_seconds_arcs)):
            if verbose:
                print(f"  [arcs MC] stopping early (limit reached) "
                      f"i={i} elapsed={time.time()-_t_start:.1f}s "
                      f"score={score_curr:.4f}")
            return pos

        # cpp:3133-3151: milestone check
        if i % s.mc_stop_steps_arcs == 0:
            ratio = score_curr / max(milestone_score, 1e-30)
            if verbose:
                # ``score_curr`` is delta-tracked (cpp:3109) — the single-bead
                # variant of the score function has its 1/d repulsion branch
                # commented out (cpp:1966-1969), so any change in repulsion
                # energy is silently ignored by the delta.  We recompute the
                # true full score here (cheap, only every stop_steps iters)
                # so the log isn't dominated by stale initial-packing
                # repulsion baked into ``score_curr``.
                true_score = score_distances_active_region(
                    pos, expected,
                    k_stretch=k_stretch, k_squeeze=k_squeeze,
                    k_repulsion=1.0,
                ).item()
                print(f"  [arcs MC] i={i:7d}  T={T:.5f}  "
                      f"score={score_curr:.6f}  ratio={ratio:.5f}  "
                      f"ms_succ={milestone_success}  "
                      f"true_full={true_score:.2f}")
            # cpp:3143-3146: three stop clauses (ratio>0.9999 is unconditional)
            if ((score_curr > s.mc_stop_improvement_arcs * milestone_score
                    and milestone_success < s.mc_stop_min_successes_arcs)
                    or score_curr < 1e-5
                    or ratio > 0.9999):
                return pos
            milestone_score = score_curr                        # cpp:3149
            milestone_success = 0                               # cpp:3150

        score_prev = score_curr                                 # cpp:3153

    return pos


def monte_carlo_arcs_sparse(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,
    fixed_mask: torch.Tensor,
    settings: Settings,
    *,
    step_size: Optional[float] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """**Legacy sparse-arc** MC kept only for backward-compat callers.

    AUDIT §G1: this code path is missing the ``-1`` repulsion energy that
    cudaMMC's dense matrix encodes for every non-arc pair, so structures
    collapse.  Phase-5 callers must use :func:`monte_carlo_arcs` with a
    dense expected matrix from
    :meth:`LooperSolver._build_anchor_expected_dist_ib`.
    """
    del chain_lengths   # arcs phase doesn't score chain lengths
    s = settings
    device = pos.device
    N = pos.shape[0]
    if N <= 1:
        return pos
    rng = random.Random()

    T = s.max_temp_arcs
    step = step_size if step_size is not None else s.step_size_arcs

    def _arc_total() -> float:
        return score_arcs(pos, arc_starts, arc_ends, arc_expected,
                          s.spring_constant_stretch_arcs, 1.0).item()

    score_curr = _arc_total()
    score_prev = score_curr
    milestone_score = score_curr
    milestone_success = 0
    i = 0

    if verbose:
        print(f"  [arcs MC sparse] N={N}  T={T:.1f}  step={step:.4f}  "
              f"score={score_curr:.6f}")

    while True:
        p = rng.randrange(N)
        if fixed_mask[p]:
            continue
        local_prev = score_arcs_single(pos, p, arc_starts, arc_ends,
                                        arc_expected,
                                        s.spring_constant_stretch_arcs,
                                        1.0).item()
        disp = _random_displacement(step, s.use_2d, device)
        pos[p] += disp
        local_curr = score_arcs_single(pos, p, arc_starts, arc_ends,
                                        arc_expected,
                                        s.spring_constant_stretch_arcs,
                                        1.0).item()
        score_curr = score_prev - local_prev + local_curr
        ok = score_curr <= score_prev
        if not ok:
            ok = _accept_metropolis(s.temp_jump_scale_arcs,
                                    s.temp_jump_coef_arcs,
                                    score_curr, score_prev, T, rng)
        if ok:
            milestone_success += 1
        else:
            pos[p] -= disp
            score_curr = score_prev
        T *= s.dt_temp_arcs
        i += 1
        if i % s.mc_stop_steps_arcs == 0:
            ratio = score_curr / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [arcs MC sparse] i={i:7d}  T={T:.5f}  score={score_curr:.6f}  "
                      f"ratio={ratio:.5f}  ms_succ={milestone_success}")
            if ((score_curr > s.mc_stop_improvement_arcs * milestone_score
                    and milestone_success < s.mc_stop_min_successes_arcs)
                    or score_curr < 1e-5
                    or ratio > 0.9999):
                return pos
            milestone_score = score_curr
            milestone_success = 0
        score_prev = score_curr
    return pos


# ============================================================================
# Phase 3 — MonteCarloArcsSmooth   (LooperSolver.cpp:3161-3390)
# ============================================================================
# Tracks structure + orientation + (optional) subanchor-heatmap separately;
# orientation contributes with factor 2 to the delta (cpp:3311-3313) because
# ``curr_score_orientation`` is the FULL pairwise sum while
# ``calcScoreOrientation(orn_index)`` returns only the half touching that
# anchor.  Strict ``<`` accept (cpp:3329).  Metropolis gated on ``T > 0``
# (cpp:3331).

def _build_cluster_type(N: int,
                        is_anchor: Optional[torch.Tensor]) -> List[int]:
    """Mirror of cudaMMC cpp:3214-3240 cluster_type construction.

    For every active-region bead:

        cluster_type[i]     = 3 + anchor_index   (i is an anchor)
        cluster_type[i-1]   = 1                  (left neighbour)
        cluster_type[i+1]   = 2                  (right neighbour)
        cluster_type[i-2]   = -last_anchor       (← negative pointer)
        cluster_type[last_anchor+2] = -i         (→ negative pointer)

    When ``is_anchor`` is None we treat **every** bead as an anchor (matches
    the legacy Phase-4 input which is anchors-only; subanchors land in Phase 5).
    """
    ct = [0] * N
    last_anchor = -1
    anchor_ind = 0
    for i in range(N):
        if is_anchor is None or bool(is_anchor[i]):
            ct[i] = 3 + anchor_ind                          # cpp:3221
            if i > 0 and ct[i - 1] == 0:
                ct[i - 1] = 1                               # cpp:3223
            if i + 1 < N:
                ct[i + 1] = 2                               # cpp:3225
            if last_anchor >= 0:
                if i - 2 >= 0:
                    ct[i - 2] = -last_anchor                # cpp:3229
                if last_anchor + 2 < N:
                    ct[last_anchor + 2] = -i                # cpp:3230
            last_anchor = i
            anchor_ind += 1
    return ct


def monte_carlo_arcs_smooth(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,           # legacy compat (unused here)
    chain_lengths: torch.Tensor,
    orientations: List[str],
    fixed_mask: torch.Tensor,
    settings: Settings,
    *,
    step_size: Optional[float] = None,
    is_anchor_mask: Optional[torch.Tensor] = None,
    subanchor_expected: Optional[torch.Tensor] = None,
    subanchor_diag: int = 1,
    verbose: bool = True,
) -> torch.Tensor:
    """Smooth MC — cpp:3161-3390.

    ``arc_expected`` is accepted only for legacy caller compatibility; cudaMMC
    smooth MC scores structure (chain + angle) + orientation + subanchor
    heatmap — **never** arc springs.

    ``is_anchor_mask`` (bool tensor, length N) — drives the ``cluster_type``
    machinery so the orientation 2× delta only fires when an anchor or
    anchor-neighbour moves.  None ⇒ treat every bead as an anchor (legacy
    anchors-only input).

    ``subanchor_expected`` activates ``calcScoreSubanchorHeatmap``
    (cpp:3316-3320) when set together with ``settings.use_subanchor_heatmap``;
    currently a flag-gated stub that raises until Phase 5 densification lands.
    """
    del arc_expected   # arcs are scored by Phase 2, not here.
    s = settings
    device = pos.device
    N = pos.shape[0]
    if N <= 1:
        return pos
    rng = random.Random()

    # cpp:3166-3168
    T = s.max_temp_smooth
    step = step_size if step_size is not None else s.step_size_smooth

    use_ctcf = bool(s.use_ctcf_motif_orientation)
    use_subheat = (bool(s.use_subanchor_heatmap)
                    and subanchor_expected is not None)
    if use_subheat:
        # AUDIT §F7 / §G6 — flag-gated stub; Phase 5 lands real implementation.
        from .scores import score_subanchor_heatmap
        score_subanchor_heatmap(pos, subanchor_expected,
                                diagonal_size=subanchor_diag, enabled=True)

    # cpp:3214-3240: cluster_type machinery (only when CTCF flag is on)
    cluster_type = (_build_cluster_type(N, is_anchor_mask)
                    if use_ctcf else [0] * N)

    # cpp:3242-3246: initial scores
    curr_struct = score_structure_smooth(
        pos, chain_lengths,
        k_stretch=s.spring_constant_stretch,
        k_squeeze=s.spring_constant_squeeze,
        angular_k=s.spring_angular_constant,
        weight_dist=s.weight_dist_smooth,
        weight_angle=s.weight_angle_smooth,
    ).item()
    curr_orient = (score_orientation(
                       pos, orientations, arc_starts, arc_ends,
                       weight=s.motif_orientation_weight,
                       motifs_symmetric=bool(s.motifs_symmetric),
                       use_ctcf=use_ctcf).item()
                   if use_ctcf else 0.0)
    curr_heat = 0.0    # subanchor-heatmap not yet implemented
    score_curr = curr_struct + curr_orient + curr_heat      # cpp:3249

    prev_struct = curr_struct                               # cpp:3252-3254
    prev_orient = curr_orient
    prev_heat = curr_heat
    score_prev = score_curr                                 # cpp:3255
    milestone_score = score_curr                            # cpp:3256
    milestone_success = 0
    i = 0

    if verbose:
        print(f"  [smooth MC] N={N}  T={T:.1f}  step={step:.4f}  "
              f"score={score_curr:.6f} (struct={curr_struct:.4f}, "
              f"orient={curr_orient:.4f})")

    _t_start = time.time()
    _t_last = _t_start
    # cpp:3264: while (true)
    while True:
        p = rng.randrange(N)                                # cpp:3266
        if fixed_mask[p]:                                   # cpp:3269-3270
            continue

        # cpp:3273-3296: anchor / neighbour orientation handling
        orn_index = -1
        anchor_index = -1
        local_prev_orient = 0.0
        if use_ctcf and cluster_type[p] > 0:
            # cpp:3276-3284: resolve anchor index in active_region
            anchor_index = p
            if cluster_type[p] == 1:
                anchor_index = p + 1
            elif cluster_type[p] == 2:
                anchor_index = p - 1
            if (0 <= anchor_index < N
                    and 0 <= anchor_index < len(cluster_type)
                    and cluster_type[anchor_index] >= 3):
                orn_index = cluster_type[anchor_index] - 3   # cpp:3288
                local_prev_orient = score_orientation_single(
                    pos, anchor_index, orientations,
                    arc_starts, arc_ends,
                    weight=s.motif_orientation_weight,
                    motifs_symmetric=bool(s.motifs_symmetric),
                    use_ctcf=True).item()                    # cpp:3293-3294

        # cpp:3298: local_prev_structure = calcScoreStructureSmooth(p, true, true)
        local_prev_struct = score_chain_single(
            pos, p, chain_lengths,
            k_stretch=s.spring_constant_stretch,
            k_squeeze=s.spring_constant_squeeze,
            angular_k=s.spring_angular_constant,
        ).item()

        disp = _random_displacement(step, s.use_2d, device)  # cpp:3302
        pos[p] += disp                                       # cpp:3303

        local_curr_struct = score_chain_single(
            pos, p, chain_lengths,
            k_stretch=s.spring_constant_stretch,
            k_squeeze=s.spring_constant_squeeze,
            angular_k=s.spring_angular_constant,
        ).item()                                             # cpp:3305

        if orn_index >= 0:
            # cpp:3308-3309
            local_curr_orient = score_orientation_single(
                pos, anchor_index, orientations,
                arc_starts, arc_ends,
                weight=s.motif_orientation_weight,
                motifs_symmetric=bool(s.motifs_symmetric),
                use_ctcf=True).item()
            # cpp:3311-3313: factor 2× on orientation delta
            curr_orient = curr_orient + 2.0 * (local_curr_orient - local_prev_orient)

        # cpp:3322-3323: factor 1× on structure delta
        curr_struct = curr_struct - local_prev_struct + local_curr_struct
        # cpp:3326-3327
        score_curr = curr_struct + curr_orient + curr_heat

        ok = score_curr < score_prev                         # cpp:3329 strict <

        # cpp:3331-3335: Metropolis gated on T > 0
        if not ok and T > 0.0:
            ok = _accept_metropolis(s.temp_jump_scale_smooth,
                                    s.temp_jump_coef_smooth,
                                    score_curr, score_prev, T, rng)

        if ok:                                               # cpp:3338-3343
            milestone_success += 1
            score_prev = score_curr
            prev_struct = curr_struct
            prev_orient = curr_orient
            prev_heat = curr_heat
        else:                                                # cpp:3345-3358
            pos[p] -= disp
            score_curr = score_prev
            curr_struct = prev_struct
            curr_heat = prev_heat
            if orn_index >= 0:
                curr_orient = prev_orient
                # cpp:3357: orientation is re-derived from pos every call,
                # so rolling pos back is sufficient — no stored vector to
                # restore in this Python port.

        i += 1

        # heartbeat: see monte_carlo_heatmap
        if verbose and (time.time() - _t_last) > 5.0:
            _t_last = time.time()
            its = i / max(_t_last - _t_start, 1e-9)
            print(f"  [smooth MC py] ..i={i:8d}  T={T:.5f}  "
                  f"score={score_curr:.6f}  {its:.0f} it/s")

        # Python-only safety cap, see monte_carlo_arcs.
        if ((s.mc_max_iters_smooth > 0 and i >= s.mc_max_iters_smooth)
                or (s.mc_max_seconds_smooth > 0.0
                    and time.time() - _t_start > s.mc_max_seconds_smooth)):
            if verbose:
                print(f"  [smooth MC py] stopping early (limit reached) "
                      f"i={i} elapsed={time.time()-_t_start:.1f}s "
                      f"score={score_curr:.4f}")
            break
                  f"(struct={curr_struct:.2f}, orient={curr_orient:.2f})")

        # Python-only safety cap, see monte_carlo_arcs.
        if ((s.mc_max_iters_smooth > 0 and i >= s.mc_max_iters_smooth)
                or (s.mc_max_seconds_smooth > 0.0
                    and time.time() - _t_start > s.mc_max_seconds_smooth)):
            if verbose:
                print(f"  [smooth MC] stopping early (limit reached) "
                      f"i={i} elapsed={time.time()-_t_start:.1f}s "
                      f"score={score_curr:.4f}")
            return pos

        # cpp:3361-3380: milestone check
        if i % s.mc_stop_steps_smooth == 0:
            ratio = score_curr / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [smooth MC] i={i:7d}  T={T:.5f}  score={score_curr:.6f}  "
                      f"ratio={ratio:.5f}  ms_succ={milestone_success}  "
                      f"(struct={curr_struct:.4f}, orient={curr_orient:.4f})")
            # cpp:3372-3376: NO ratio>0.9999 clause in smooth phase
            if ((score_curr > s.mc_stop_improvement_smooth * milestone_score
                    and milestone_success < s.mc_stop_min_successes_smooth)
                    or score_curr < 1e-6):
                return pos
            milestone_score = score_curr                     # cpp:3378
            milestone_success = 0                            # cpp:3379

        T *= s.dt_temp_smooth                                # cpp:3382

    return pos

