"""
Monte Carlo optimisation phases.

Phase 1 - GPU heatmap MC  (ParallelMonteCarloHeatmap equivalent)
Phase 2 - CPU arcs MC     (MonteCarloArcs equivalent)
Phase 3 - CPU smooth MC   (MonteCarloArcsSmooth equivalent)

Every algorithmic line carries a # cudaMMC: annotation pointing to the exact
C++/CUDA source line it replicates.  Source files (paths from repo root):
  cudammc/src/ParallelMonteCarloHeatmap.cu   – Phase 1 GPU kernel
  cudammc/src/LooperSolver.cpp               – Phases 2 & 3 CPU MC
  cudammc/thirdparty/common.cpp              – random_vector helper
"""

import math
import random
from typing import List, Optional

import torch

from .scores import (
    score_heatmap_chunked,
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
    prob = jump_scale * math.exp(                                 # cudaMMC cpp:3114: tempJumpScale *
        -jump_coef * (score_curr / max(score_prev, 1e-30)) / max(T, 1e-30))  # cudaMMC cpp:3115: exp(-coef*(s1/s0)/T)
    return rng.random() < prob                                    # cudaMMC cpp:3116: withChance(tp)


# ── Per-bead score delta (vectorised over all beads simultaneously) ───────────

def _bead_score_delta(
    pos: torch.Tensor,       # (N, 3) current positions
    new_pos: torch.Tensor,   # (N, 3) proposed positions
    expected: torch.Tensor,  # (N, N) fp16 or fp32
    diagonal_size: int,
    same_chr_mask: Optional[torch.Tensor],
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (delta, score_old) each (N,):
      delta[i]     = S_new[i] - S_old[i]
      score_old[i] = S_old[i]  (per-bead score before move)

    S[i] = sum_{j: j≠i, |i-j|>=diag, exp>1e-3} (dist(pos_i, pos_j)/exp[i,j] - 1)^2

    Jacobi: S_new[i] uses new_pos[i] vs the CURRENT pos[j] for all j.
    Both outputs are needed for the ratio-based Metropolis criterion.

    Mirrors cudaMMC ParallelMonteCarloHeatmap.cu:128-165
    calcScoreHeatmapSingleActiveRegion(moved, clusters_positions, …, curr_vector, warpIdx)
    """
    N = pos.shape[0]
    device = pos.device
    delta = torch.zeros(N, device=device)     # cudaMMC .cu:135: double err = 0.0 (per bead accumulator)
    score_old = torch.zeros(N, device=device)  # same; second accumulator for old position

    pos_sq = (pos ** 2).sum(dim=1)            # precomputed ||pos_j||² for all j
    new_sq = (new_pos ** 2).sum(dim=1)        # precomputed ||new_pos_i||² for all i

    for i0 in range(0, N, chunk_size):
        i1 = min(i0 + chunk_size, N)

        p_old = pos[i0:i1]             # (C, 3)   old positions for beads i0..i1-1
        p_new = new_pos[i0:i1]         # (C, 3)   proposed positions for beads i0..i1-1
        sq_o = pos_sq[i0:i1]           # (C,)
        sq_n = new_sq[i0:i1]           # (C,)

        dot_o = p_old @ pos.t()        # (C, N)   p_old · pos_j  (dot product trick)
        # cudaMMC .cu:157-159: subtractVectors(temp_one, *(clusters_positions+i), curr_vector)
        # cudaMMC .cu:102-106: magnitude = sqrt(x²+y²+z²)
        d_old = (sq_o[:, None] + pos_sq[None, :] - 2 * dot_o).clamp(0).sqrt()  # (C, N) Euclidean d_old[c,j]

        dot_n = p_new @ pos.t()        # (C, N)   p_new · pos_j
        d_new = (sq_n[:, None] + pos_sq[None, :] - 2 * dot_n).clamp(0).sqrt()  # (C, N) Euclidean d_new[c,j]

        exp_c = expected[i0:i1].float()   # (C, N)  expected distances; cast fp16→fp32 if needed

        i_idx = torch.arange(i0, i1, device=device)[:, None]   # (C, 1)  bead indices i
        j_idx = torch.arange(N, device=device)[None, :]         # (1, N)  bead indices j

        # cudaMMC .cu:152:   if (abs(i - moved) >= heatmapDiagonalSize)
        # cudaMMC .cu:153:   if (i == moved || helper < 1e-3) continue   (skip self + zero entries)
        # cudaMMC .cu:145-149: getChromosomeHeatmapBoundary → restricts to same chromosome
        valid = ((i_idx != j_idx)                                # skip self (i == moved handled by zero displacement)
                 & ((i_idx - j_idx).abs() >= diagonal_size)     # cudaMMC .cu:152: abs(i-moved)>=heatmapDiagonalSize
                 & (exp_c > 1e-3))                               # cudaMMC .cu:153: heatmap_dist[…] < 1e-3 → skip
        if same_chr_mask is not None:
            valid = valid & same_chr_mask[i0:i1]                # cudaMMC .cu:145-149: chromosome boundary guard

        safe_e = exp_c.clamp(min=1e-9)
        # cudaMMC .cu:160: helper = magnitude(temp_one) / helper - 1   (ratio - 1)
        r_old = torch.where(valid, d_old / safe_e - 1.0, torch.zeros_like(d_old))
        r_new = torch.where(valid, d_new / safe_e - 1.0, torch.zeros_like(d_new))

        # cudaMMC .cu:161: err += helper * helper   (squared error)
        score_old[i0:i1] = (r_old ** 2).sum(dim=1)
        delta[i0:i1] = (r_new ** 2 - r_old ** 2).sum(dim=1)

    return delta, score_old


# ── Phase 1: vectorised GPU heatmap MC ───────────────────────────────────────

def monte_carlo_heatmap(
    pos: torch.Tensor,          # (N, 3) float32, on GPU
    expected: torch.Tensor,     # (N, N) expected distances (float16 ok), on GPU
    fixed_mask: torch.Tensor,   # (N,) bool
    settings: Settings,
    same_chr_mask: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Vectorised Jacobi heatmap MC – mirrors the CUDA warp-parallel kernel.

    cudaMMC source: ParallelMonteCarloHeatmap.cu
      Host setup:  ParallelMonteCarloHeatmap()  lines 327-429
      GPU kernel:  MonteCarloHeatmapKernel()     lines 194-325

    Each inner step proposes a displacement for ALL beads simultaneously,
    computes per-bead score deltas in one chunked GPU pass (no Python loop
    over beads, no N×N allocation), and accepts/rejects per-bead with a
    vectorised Metropolis criterion.

    Outer step = mc_inner_steps Jacobi rounds, followed by temperature
    cooling and a milestone check.
    """
    s = settings
    device = pos.device
    N = pos.shape[0]

    # cudaMMC .cu:331:  double T = Settings::maxTempHeatmap
    T = s.max_temp_heatmap
    # cudaMMC .cu:402:  kernel receives 0.75f * step_size (caller passes 0.75*step)
    step = s.step_size_heatmap
    free = ~fixed_mask                  # (N,) mask of movable beads
    # cudaMMC .cu:228-229: if (clusters_fixed[warpIdx]) return;

    # cudaMMC .cu:392:  score_curr = calcScoreHeatmapActiveRegion()  (initial total score)
    total_score = score_heatmap_chunked(pos, expected,
                                         s.diagonal_size, same_chr_mask).item()
    best_score = total_score
    # cudaMMC .cu:205:  int improvementMisses = 0
    milestone_fails = 0
    outer_step = 0

    effective_inner = s.mc_inner_steps  # cudaMMC .cu:225: #define N 512 (inner iterations per warp)

    # Don't check milestones until T drops to ≤10% of initial — at high T,
    # score fluctuates randomly and milestone_fails would trigger after ~3 steps.
    milestone_temp_threshold = T * 0.1
    # cudaMMC .cu:210:  float milestoneScore = score_curr  (reset on each outer step)

    if verbose:
        print(f"  [heatmap MC] N={N}  T_initial={T:.1f}  "
              f"milestone starts at T={milestone_temp_threshold:.2f}")

    # cudaMMC .cu:222: while (true) {  — outer annealing loop
    while milestone_fails < s.milestone_fails_threshold:
        outer_step += 1

        # cudaMMC .cu:226-250: for (int i = 0; i < N; ++i) { … }  — inner Jacobi loop
        for _ in range(effective_inner):
            # cudaMMC .cu:231: randomVector(displacement, step_size, settings.use2D, &localState)
            disp = torch.zeros(N, 3, device=device)
            disp[free] = (torch.rand(free.sum(), 3, device=device) * 2.0 - 1.0) * step
            # cudaMMC .cu:79: in2D ? __float2half(0.0f)
            if s.use_2d:
                disp[:, 2] = 0.0

            # cudaMMC .cu:232: addToVector(curr_vector, displacement)
            new_pos = pos + disp   # (N, 3); zeros for fixed beads → no change

            # cudaMMC .cu:234-237: score_curr = calcScoreHeatmapSingleActiveRegion(warpIdx, …, curr_vector, warpIdx)
            delta, score_old = _bead_score_delta(pos, new_pos, expected,
                                                  s.diagonal_size, same_chr_mask)

            # cudaMMC .cu:239: if ((score_curr <= score_prev) || (T > 0.0f && withChance(…)))
            # cudaMMC .cu:241-244: withChance(tempJumpScaleHeatmap * expf(-tempJumpCoefHeatmap * (score_curr/score_prev) * (1/T)))
            score_new = score_old + delta                            # (N,) per-bead proposed score
            T_safe = max(T, 1e-30)
            log_thresh = (math.log(max(s.temp_jump_scale_heatmap, 1e-30))   # log(scale)
                          - s.temp_jump_coef_heatmap                         # -coef *
                            * (score_new / score_old.clamp(min=1e-30)) / T_safe)  # (s1/s0)/T
            rand_log = torch.rand(N, device=device).log()           # log(uniform) for log-space comparison
            # cudaMMC .cu:239: score_curr<=score_prev  OR  withChance(…)
            accept = free & ((delta <= 0) | (rand_log < log_thresh))

            # cudaMMC .cu:245: score_prev = score_curr; continue  (accept)
            # cudaMMC .cu:249: subtractValueFromVector(curr_vector, displacement)  (reject)
            pos[accept] = new_pos[accept]
            total_score += delta[accept].sum().item()

        # cudaMMC .cu:252: T *= settings.dtTempHeatmap
        T *= s.dt_temp_heatmap
        # cudaMMC .cu:253: step_size *= 0.95
        step *= s.step_size_decay_heatmap

        # Recompute true score — Jacobi delta accumulation drifts from reality
        # because deltas are computed against stale neighbour positions.
        # cudaMMC .cu:303-307: score_curr = calcScoreHeatmapActiveRegion(-1, …)  (thread 0 recomputes global score)
        total_score = score_heatmap_chunked(pos, expected,
                                             s.diagonal_size, same_chr_mask).item()

        annealing = T <= milestone_temp_threshold
        log_interval = 10 if annealing else 100
        if verbose and outer_step % log_interval == 0:
            phase = "anneal" if annealing else "explore"
            print(f"  [heatmap MC] step={outer_step:5d}  T={T:.4f}  "
                  f"score={total_score:.4f}  fails={milestone_fails}  [{phase}]")

        # cudaMMC .cu:310-312:
        #   if (score_curr > settings.MCstopConditionImprovementHeatmap * milestoneScore)
        #     ++improvementMisses;
        # cudaMMC .cu:314-316:
        #   if (improvementMisses >= settings.milestoneFailsThreshold || score_curr < 1e-04)
        #     *isDone = true;
        # cudaMMC .cu:318: milestoneScore = score_curr;
        if T <= milestone_temp_threshold:
            if total_score < s.milestone_improvement_ratio * best_score:  # cudaMMC: score improved ≥ 0.5%
                best_score = total_score
                milestone_fails = 0                                       # cudaMMC: resets on improvement
            else:
                milestone_fails += 1                                      # cudaMMC: ++improvementMisses
        else:
            best_score = min(best_score, total_score)

        # cudaMMC .cu:315: score_curr < 1e-04  → stop
        if total_score < 1e-4:
            break

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
            ratio = total_score / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [arcs MC] step={individual_steps:7d}  T={T:.5f}  "
                      f"score={total_score:.6f}  ratio={ratio:.5f}  "
                      f"ms_succ={milestone_success}")
            # cudaMMC cpp:3143-3146:
            #   if ((score_curr > MCstopConditionImprovement * milestone_score &&
            #        milestone_success < MCstopConditionMinSuccesses) ||
            #       score_curr < 1e-5 || score_curr / milestone_score > 0.9999)
            if ((total_score > s.milestone_improvement_ratio * milestone_score
                    and milestone_success < s.min_successes_arcs)
                    or total_score < 1e-5
                    or ratio > 0.9999):
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
            ratio = ts / max(milestone_score, 1e-30)
            if verbose:
                print(f"  [smooth MC] step={individual_steps:7d}  T={T:.5f}  "
                      f"score={ts:.6f}  ratio={ratio:.5f}  "
                      f"ms_succ={milestone_success}")
            # cudaMMC cpp:3372-3375:
            #   if ((score_curr > MCstopConditionImprovementSmooth * milestone_score &&
            #        milestone_success < MCstopConditionMinSuccessesSmooth) ||
            #       score_curr < 1e-6)
            if ((ts > s.milestone_improvement_ratio * milestone_score
                    and milestone_success < s.min_successes_smooth)
                    or ts < 1e-6
                    or ratio > 0.9999):
                return pos
            # cudaMMC cpp:3378-3379: milestone_score = score_curr; milestone_success = 0
            milestone_score = ts
            milestone_success = 0

        step *= s.step_size_decay_smooth

    return pos
