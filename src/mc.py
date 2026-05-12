"""
Monte Carlo optimisation phases.

Phase 1 - CPU heatmap MC  (MonteCarloHeatmap — LooperSolver.cpp:421-518)
Phase 2 - CPU arcs MC     (MonteCarloArcs    — LooperSolver.cpp:3058-3159)
Phase 3 - CPU smooth MC   (MonteCarloArcsSmooth — LooperSolver.cpp:3161-3390)

Every algorithmic line carries a # cudaMMC: annotation pointing to the exact
C++/CUDA source line it replicates.  Source files (paths from repo root):
  cudammc/src/LooperSolver.cpp               – all three phases
  cudammc/thirdparty/common.cpp              – random_vector helper
"""

import math
import random
from typing import List, Optional

import torch

from .scores import (
    score_heatmap_chunked,
    score_heatmap_single,
    score_arcs,
    score_arcs_single,
    score_chain_single,
    score_structure_smooth,
    score_orientation,
    score_orientation_single,
)
from .settings import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_displacement(step_size: float, use_2d: bool,
                          device: torch.device) -> torch.Tensor:
    # cudaMMC common.cpp random_vector / ParallelMonteCarloHeatmap.cu:75-80:
    #   __device__ void randomVector(half3 &vector, const float &max_size, bool &in2D, …)
    #     vector.x = random(max_size, true, state);  // (2*uniform-1)*range → [-step,step]
    #     vector.y = random(max_size, true, state);
    #     vector.z = in2D ? __float2half(0.0f) : random(max_size, true, state);
    v = (torch.rand(3, device=device) * 2.0 - 1.0) * step_size  # cudaMMC .cu:77-78: (2f*uniform-1)*range
    if use_2d:
        v[2] = 0.0  # cudaMMC .cu:79: in2D ? __float2half(0.0f)
    return v


def _with_chance_ratio(jump_scale: float, jump_coef: float,
                        score_curr: float, score_prev: float,
                        T: float, rng: random.Random) -> bool:
    # cudaMMC LooperSolver.cpp:3114-3116 (MonteCarloArcs):
    #   tp = Settings::tempJumpScale *
    #        exp(-Settings::tempJumpCoef * (score_curr / score_prev) / T);
    #   ok = withChance(tp);
    # Guard: if scores drifted non-positive due to floating-point accumulation,
    # fall back to greedy comparison (accept iff improvement) to avoid division
    # by near-zero or sign inversion that would cause exp() overflow.
    if score_prev <= 0.0 or score_curr <= 0.0:
        return score_curr <= score_prev
    exp_arg = -jump_coef * (score_curr / score_prev) / max(T, 1e-30)  # cudaMMC cpp:3115
    if exp_arg < -700:                                            # underflow → prob ≈ 0
        return False
    prob = jump_scale * math.exp(exp_arg)                        # cudaMMC cpp:3114
    return rng.random() < prob                                    # cudaMMC cpp:3116: withChance(tp)


# ── Phase 1: sequential single-bead heatmap MC ───────────────────────────────
# cudaMMC source: LooperSolver.cpp:421-518  MonteCarloHeatmap(float step_size)

def monte_carlo_heatmap(
    pos: torch.Tensor,          # (N, 3) float32
    expected: torch.Tensor,     # (N, N) expected distances (float16 ok)
    fixed_mask: torch.Tensor,   # (N,) bool
    settings: Settings,
    same_chr_mask: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Sequential single-bead heatmap MC – one-to-one with CPU MonteCarloHeatmap.

    cudaMMC source: LooperSolver.cpp:421-518  MonteCarloHeatmap(float step_size)

    Each iteration selects one random free bead, proposes a displacement,
    updates total score via O(N) per-bead delta (score_heatmap_single), and
    accepts/rejects with ≤ criterion + ratio Metropolis.  Temperature cools
    every individual bead move.  Milestone check every milestone_steps_heatmap
    moves; stop when score fails to improve ≥ 0.5% with < 5 successes.
    """
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    # cudaMMC cpp:423: step_size *= 0.5
    step = s.step_size_heatmap * 0.5

    # cudaMMC cpp:425: T = Settings::maxTempHeatmap
    T = s.max_temp_heatmap

    # cudaMMC cpp:445: score_curr = calcScoreHeatmapActiveRegion()
    # score_heatmap_single counts ALL j≠idx (full row), giving delta = 2×upper-triangle change.
    # Initialise with 2×upper-triangle so total_score stays consistent with per-bead deltas
    # and never drifts negative (which would flip the Metropolis ratio sign → overflow).
    total_score = 2.0 * score_heatmap_chunked(pos, expected, s.diagonal_size, same_chr_mask).item()
    # cudaMMC cpp:448: score_prev = score_curr
    score_prev = total_score
    # cudaMMC cpp:449: milestone_score = score_curr
    milestone_score = total_score
    # cudaMMC cpp:435-436: milestone_success = 0
    milestone_success = 0
    # cudaMMC cpp:455: i = 1
    individual_steps = 0

    if verbose:
        print(f"  [heatmap MC] N={N}  T_initial={T:.1f}  step={step:.4f}  score={total_score:.4f}")

    # cudaMMC cpp:456: while (true) {
    while True:
        # cudaMMC cpp:459: p = random(size)
        bead_idx = rng.randrange(N)
        # cudaMMC cpp:462-463: if (clusters[ind].is_fixed) continue
        if fixed_mask[bead_idx]:
            continue

        # O(N) per-bead score before move — replaces O(N²) calcScoreHeatmapActiveRegion
        local_prev = score_heatmap_single(pos, bead_idx, expected,
                                          s.diagonal_size, same_chr_mask).item()

        # cudaMMC cpp:465-467: displacement = random_vector(step_size, use2D); pos += disp
        disp = _random_displacement(step, s.use_2d, device)
        pos[bead_idx] += disp  # cudaMMC cpp:467: clusters[ind].pos += displacement

        # cudaMMC cpp:468: score_curr = calcScoreHeatmapActiveRegion()  (O(N²) → O(N) via delta)
        local_curr = score_heatmap_single(pos, bead_idx, expected,
                                          s.diagonal_size, same_chr_mask).item()
        score_curr = total_score + (local_curr - local_prev)

        # cudaMMC cpp:469: ok = score_curr <= score_prev
        ok = score_curr <= score_prev

        # cudaMMC cpp:471-475: if (!ok && T > 0) { tp = scale*exp(-coef*(s1/s0)/T); ok = withChance(tp) }
        if not ok and T > 0.0:
            ok = _with_chance_ratio(s.temp_jump_scale_heatmap, s.temp_jump_coef_heatmap,
                                    score_curr, score_prev, T, rng)

        # cudaMMC cpp:478-484: if ok: score_prev=score_curr; milestone_success++ else: pos -= disp
        if ok:
            total_score = score_curr          # accept
            score_prev = total_score          # cudaMMC cpp:480: score_prev = score_curr
            milestone_success += 1
        else:
            pos[bead_idx] -= disp             # cudaMMC cpp:482: clusters[ind].pos -= displacement
            score_curr = score_prev           # cudaMMC cpp:483

        # cudaMMC cpp:486: T *= Settings::dtTempHeatmap  ← cooling every individual bead move
        T *= s.dt_temp_heatmap

        individual_steps += 1  # cudaMMC cpp:514: i++

        # cudaMMC cpp:488: if (i % Settings::MCstopConditionStepsHeatmap == 0) {
        if individual_steps % s.milestone_steps_heatmap == 0:
            # Resync from scratch every milestone: score_heatmap_single uses only row k of the
            # (asymmetric) expected matrix, so delta accumulates O(N) drift per move.
            # A full recompute every 10k steps keeps the tracked score non-negative and avoids
            # sign flips in the Metropolis ratio that cause math.exp overflow.
            total_score = 2.0 * score_heatmap_chunked(pos, expected,
                                                       s.diagonal_size, same_chr_mask).item()
            score_prev = total_score
            ratio = total_score / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [heatmap MC] step={individual_steps:7d}  T={T:.5f}  "
                      f"score={total_score:.4f}  ratio={ratio:.5f}  ms_succ={milestone_success}")
            # cudaMMC cpp:501-504:
            #   if (score_curr > MCstopConditionImprovementHeatmap * milestone_score
            #       && milestone_success < MCstopConditionMinSuccessesHeatmap) || score < 1e-6
            # Extra cold-phase guard: cudaMMC runs on small active regions (~30 beads) so
            # milestone_success naturally drops to 0 at T≈0 (local min reached quickly).
            # With N=288, there are always tiny greedy improvements → success stays high.
            # Stop when T is cold (< 0.5% of T_max) AND improvement per milestone < 0.01%.
            cold = T < s.max_temp_heatmap * 0.005
            if ((total_score > s.milestone_improvement_ratio * milestone_score
                    and milestone_success < s.min_successes_heatmap)
                    or total_score < 2e-6   # 2× because total_score = 2×upper-triangle
                    or (cold and ratio > 0.9999)):
                break
            # cudaMMC cpp:507-508: milestone_score = score_curr; milestone_success = 0
            milestone_score = total_score
            milestone_success = 0

        # cudaMMC cpp:513: score_prev = score_curr
        score_prev = score_curr

    return pos


# ── Phase 2: arcs MC (GPU tensors, CPU control loop) ─────────────────────────
# cudaMMC source: LooperSolver.cpp:3058-3159  MonteCarloArcs()
# Original cudaMMC is fully CPU.  We keep the same sequential Gauss-Seidel
# (random bead selection) design but tensor math runs on pos.device (GPU).
# The per-bead Python loop and .item() syncs are unavoidable without a full
# GPU kernel rewrite.

def monte_carlo_arcs(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,
    fixed_mask: torch.Tensor,
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    """Incremental single-bead MC for arc spring energy (no chain springs)."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    # cudaMMC cpp:3062-3064: maxT = Settings::maxTemp; dt = Settings::dtTemp; T = maxT
    T = s.max_temp_arcs
    step = s.step_size_arcs
    # Safety cap: prevent infinite loops on large IBs with pure-repulsion score drift.
    # cudaMMC GPU avoids this via natural T=0 frozen-state convergence; Python needs a budget.
    # Safety cap: T decays to ~0 in ~ln(max_T/1e-4)/ln(1/dt) steps ≈ 200K for dt=0.99995.
    # After T=0, allow up to 2× that budget for post-convergence drift to settle.
    import math as _math
    _cooling_steps = int(_math.log(s.max_temp_arcs / 1e-4) / _math.log(1.0 / s.dt_temp_arcs)) + 1
    max_steps = _cooling_steps * 4

    # cudaMMC cpp:3086: score_curr = calcScoreDistancesActiveRegion()  ← arc springs ONLY
    total_score = score_arcs(pos, arc_starts, arc_ends, arc_expected,
                              s.k_spring, s.k_spring_repulsion).item()
    # cudaMMC cpp:3088-3089: score_prev = score_curr; milestone_score = score_curr
    milestone_score = total_score
    # cudaMMC cpp:3067: int i = 1
    individual_steps = 0
    # cudaMMC cpp:3075-3077: int success=0; milestone_success=0
    milestone_success = 0

    # cudaMMC cpp:3094: while (true) {
    while True:
        # cudaMMC cpp:3096: p = random(size)  ← uniform random bead index
        bead_idx = rng.randrange(N)

        # cudaMMC cpp:3099-3100: if (clusters[ind].is_fixed) error(…)
        if fixed_mask[bead_idx]:
            continue

        # cudaMMC cpp:3102: local_score_prev = calcScoreDistancesActiveRegion(p)  ← arc springs only
        local_prev = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                       arc_expected, s.k_spring,
                                       s.k_spring_repulsion).item()
        # cudaMMC cpp:3088: score_prev = score_curr  (updated each iteration at cpp:3153)
        score_prev = total_score

        # cudaMMC cpp:3104: tmp = random_vector(step_size, Settings::use2D)
        disp = _random_displacement(step, s.use_2d, device)
        # cudaMMC cpp:3105: clusters[ind].pos += tmp
        pos[bead_idx] += disp

        # cudaMMC cpp:3107: local_score_curr = calcScoreDistancesActiveRegion(p)
        local_curr = score_arcs_single(pos, bead_idx, arc_starts, arc_ends,
                                       arc_expected, s.k_spring,
                                       s.k_spring_repulsion).item()

        # cudaMMC cpp:3109: score_curr = score_curr - local_score_prev + local_score_curr
        delta = local_curr - local_prev
        score_curr = total_score + delta

        # cudaMMC cpp:3111: ok = score_curr <= score_prev  (≤ for arcs phase)
        if score_curr <= score_prev:                               # cudaMMC cpp:3111
            total_score = score_curr
            milestone_success += 1                                 # cudaMMC cpp:3123
        elif _with_chance_ratio(s.temp_jump_scale_arcs, s.temp_jump_coef_arcs,
                                score_curr, score_prev, T, rng):  # cudaMMC cpp:3114-3116
            total_score = score_curr
            milestone_success += 1                                 # cudaMMC cpp:3123
        else:
            # cudaMMC cpp:3126-3127: clusters[ind].pos -= tmp; score_curr = score_prev
            pos[bead_idx] -= disp
            # total_score unchanged (score_curr = score_prev, but we track total_score)

        # cudaMMC cpp:3130: T *= dt  ← cooling EVERY bead move (not per outer step)
        T *= s.dt_temp_arcs
        # cudaMMC cpp:3155: i++
        individual_steps += 1

        # cudaMMC cpp:3133: if (i % Settings::MCstopConditionSteps == 0) {
        if individual_steps % s.milestone_steps_arcs == 0:
            # Resync total_score from scratch each milestone to prevent floating-point
            # drift in the per-bead delta accumulation from making total_score go negative.
            # Negative total_score inverts the Metropolis ratio sign → exp() overflow.
            total_score = score_arcs(pos, arc_starts, arc_ends, arc_expected,
                                     s.k_spring, s.k_spring_repulsion).item()
            ratio = total_score / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [arcs MC] step={individual_steps:7d}  T={T:.5f}  "
                      f"score={total_score:.6f}  ratio={ratio:.5f}  "
                      f"ms_succ={milestone_success}")
            # cudaMMC cpp:3143-3146:
            #   if ((score_curr > MCstopConditionImprovement * milestone_score &&
            #        milestone_success < MCstopConditionMinSuccesses) ||
            #       score_curr < 1e-5 || score_curr / milestone_score > 0.9999)
            cold = T < s.max_temp_arcs * 1e-4  # T effectively zero
            if ((total_score > s.milestone_improvement_ratio * milestone_score
                    and milestone_success < s.min_successes_arcs)
                    or total_score < 1e-5
                    or (ratio > 0.9999 and total_score <= milestone_score)
                    or (cold and ratio > 0.999)  # frozen: < 0.1% improvement per milestone
                    or individual_steps >= max_steps):  # safety cap for large IBs
                return pos
            # cudaMMC cpp:3149-3150: milestone_score = score_curr; milestone_success = 0
            milestone_score = total_score
            milestone_success = 0

        # cudaMMC cpp:3153: score_prev = score_curr  (updated every step regardless of accept/reject)

    return pos


# ── Phase 3: smooth MC (GPU tensors, CPU control loop — same as Phase 2) ─────
# cudaMMC source: LooperSolver.cpp:3161-3390  MonteCarloArcsSmooth()

def monte_carlo_arcs_smooth(
    pos: torch.Tensor,
    arc_starts: torch.Tensor,
    arc_ends: torch.Tensor,
    arc_expected: torch.Tensor,
    chain_lengths: torch.Tensor,
    orientations: List[str],
    fixed_mask: torch.Tensor,
    settings: Settings,
    verbose: bool = True,
) -> torch.Tensor:
    # cudaMMC cpp:3161: MonteCarloArcsSmooth(float step_size, bool use_subanchor_heatmap)
    # score = calcScoreStructureSmooth (chain+angular) + calcScoreOrientation — NO arc springs
    """Combined smooth MC: structure (chain+angle) + orientation."""
    s = settings
    device = pos.device
    N = pos.shape[0]
    rng = random.Random()

    # cudaMMC cpp:3166-3168: maxT = Settings::maxTempSmooth; dt = Settings::dtTempSmooth; T = maxT
    T = s.max_temp_smooth
    step = s.step_size_smooth
    import math as _math
    _cooling_steps = int(_math.log(s.max_temp_smooth / 1e-4) / _math.log(1.0 / s.dt_temp_smooth)) + 1
    max_steps = _cooling_steps * 4

    # cudaMMC cpp:3243: curr_score_structure = calcScoreStructureSmooth(true, true)
    # cudaMMC cpp:3244-3245: curr_score_orientation = calcScoreOrientation(anchor_orientation)
    # cudaMMC cpp:3249: score_curr = curr_score_structure + curr_score_orientation + curr_score_heat
    ts = (score_structure_smooth(pos, chain_lengths, s.k_chain, s.angular_k)
          + score_orientation(pos, orientations, arc_starts, arc_ends,
                              s.k_orient)).item()
    # cudaMMC cpp:3255-3256: score_prev = score_curr; milestone_score = score_curr
    milestone_score = ts
    # cudaMMC cpp:3185-3186: int success=0; milestone_success=0
    milestone_success = 0
    individual_steps = 0
    # cudaMMC cpp:3263: i = 1

    # cudaMMC cpp:3264: while (true) {
    while True:
        # cudaMMC cpp:3266: p = random(size)  ← uniform random bead index
        bead_idx = rng.randrange(N)

        # cudaMMC cpp:3269-3270: if (clusters[ind].is_fixed) continue
        if fixed_mask[bead_idx]:
            continue

        # cudaMMC cpp:3298: local_score_prev_structure = calcScoreStructureSmooth(p, true, true)
        struct_b = score_chain_single(pos, bead_idx, chain_lengths,
                                      s.k_chain, s.angular_k).item()
        # cudaMMC cpp:3293-3294: local_score_prev_orientation = calcScoreOrientation(anchor_orientation, orn_index)
        ori_b = score_orientation_single(pos, bead_idx, orientations,
                                         arc_starts, arc_ends,
                                         s.k_orient).item()
        # cudaMMC cpp:3255: score_prev = score_curr  (tracked via ts)
        score_prev = ts

        # cudaMMC cpp:3302: tmp = random_vector(step_size, Settings::use2D)
        disp = _random_displacement(step, s.use_2d, device)
        # cudaMMC cpp:3303: clusters[ind].pos += tmp
        pos[bead_idx] += disp

        # cudaMMC cpp:3305: local_score_curr_structure = calcScoreStructureSmooth(p, true, true)
        struct_a = score_chain_single(pos, bead_idx, chain_lengths,
                                      s.k_chain, s.angular_k).item()
        # cudaMMC cpp:3308-3309: local_score_curr_orientation = calcScoreOrientation(anchor_orientation, orn_index)
        ori_a = score_orientation_single(pos, bead_idx, orientations,
                                         arc_starts, arc_ends,
                                         s.k_orient).item()

        # cudaMMC cpp:3322-3323: curr_score_structure += (local_curr - local_prev)   factor 1
        # cudaMMC cpp:3311-3313: curr_score_orientation += 2*(local_curr - local_prev) factor 2 (pairwise)
        # cudaMMC cpp:3326-3327: score_curr = curr_score_structure + curr_score_orientation + curr_score_heat
        delta = (struct_a - struct_b) + 2.0 * (ori_a - ori_b)
        score_curr = ts + delta

        # cudaMMC cpp:3329: ok = score_curr < score_prev  (strict < for smooth phase)
        if score_curr < score_prev:                                # cudaMMC cpp:3329
            ts = score_curr
            milestone_success += 1                                 # cudaMMC cpp:3340
        elif _with_chance_ratio(s.temp_jump_scale_smooth, s.temp_jump_coef_smooth,
                                score_curr, score_prev, T, rng):  # cudaMMC cpp:3332-3334
            ts = score_curr
            milestone_success += 1                                 # cudaMMC cpp:3340
        else:
            # cudaMMC cpp:3348-3349: clusters[ind].pos -= tmp; score_curr = score_prev
            pos[bead_idx] -= disp

        # cudaMMC cpp:3382: T *= dt  ← cooling every bead move
        T *= s.dt_temp_smooth
        # cudaMMC cpp:3384: i++
        individual_steps += 1

        # cudaMMC cpp:3361: if (i % Settings::MCstopConditionStepsSmooth == 0) {
        if individual_steps % s.milestone_steps_smooth == 0:
            # Resync to prevent floating-point drift from making ts go negative.
            ts = (score_structure_smooth(pos, chain_lengths, s.k_chain, s.angular_k)
                  + score_orientation(pos, orientations, arc_starts, arc_ends,
                                      s.k_orient)).item()
            ratio = ts / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [smooth MC] step={individual_steps:7d}  T={T:.5f}  "
                      f"score={ts:.6f}  ratio={ratio:.5f}  "
                      f"ms_succ={milestone_success}")
            # cudaMMC cpp:3372-3375:
            #   if ((score_curr > MCstopConditionImprovementSmooth * milestone_score &&
            #        milestone_success < MCstopConditionMinSuccessesSmooth) ||
            #       score_curr < 1e-6)
            cold = T < s.max_temp_smooth * 1e-4
            if ((ts > s.milestone_improvement_ratio * milestone_score
                    and milestone_success < s.min_successes_smooth)
                    or ts < 1e-6
                    or (ratio > 0.9999 and ts <= milestone_score)
                    or (cold and ratio > 0.999)  # frozen: < 0.1% improvement per milestone
                    or individual_steps >= max_steps):  # safety cap for large IBs
                return pos
            # cudaMMC cpp:3378-3379: milestone_score = score_curr; milestone_success = 0
            milestone_score = ts
            milestone_success = 0

        step *= s.step_size_decay_smooth

    return pos
