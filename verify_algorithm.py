"""
Algorithm verification harness: cudaMMC (C++/CUDA) vs Python reimplementation.

For each phase this file provides:
  1.  Annotated reference implementations transcribed from the C++ source with
      inline citations so every Python line maps to a C++ source line.
  2.  A DISCREPANCY TABLE at the top of each section.
  3.  Numerical sanity tests that can be run independently.

Source files (paths relative to repo root):
  cudaMMC/src/ParallelMonteCarloHeatmap.cu  – Phase 1 GPU kernel
  cudaMMC/src/LooperSolver.cpp              – Phases 2 & 3 CPU MC
  cudaMMC/src/Settings.cpp                  – Default parameter values
  cudaMMC/thirdparty/common.cpp             – random_vector helper

Run:  python verify_algorithm.py
"""

from __future__ import annotations
import math
import random
import sys
from typing import List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# DISCREPANCY MASTER TABLE
# ─────────────────────────────────────────────────────────────────────────────
# Legend: ✗ = mismatch confirmed, ✓ = matches, ~ = approximate match
#
# ┌───────────────────────────────┬────────────────────────────┬──────────────────────────┐
# │ Aspect                        │ cudaMMC (C++/CUDA)         │ Python (mc.py)           │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ METROPOLIS FORMULA            │ ratio-based                │ delta-based              │ ✗
# │   Phase 1 (heatmap GPU)       │ scale*exp(-coef*s1/s0/T)   │ scale*exp(-δ/(T*coef))   │
# │   Phase 2 (arcs)              │ same                       │ same as Phase 1          │
# │   Phase 3 (smooth)            │ same                       │ same as Phase 1          │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ RANDOM DISPLACEMENT           │ uniform[-s,s]³             │ Gaussian, normalised to s│ ✗
# │   (all phases)                │ random(step_size, true)    │ randn(3)*s/‖v‖           │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ T / STEP DECAY TIMING         │ per individual bead move   │ per outer step (N beads) │ ✗
# │   Phase 2 & 3                 │ T *= dt every bead move    │ T *= dt every outer step │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ MILESTONE CRITERION           │ ratio: s1 > 0.995*s0       │ absolute: Δs < 1e-4      │ ✗
# │   (all phases)                │ checked every 10000 steps  │ checked every outer step │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ BEAD SELECTION (Phase 2 & 3)  │ random pick per step       │ sequential 0…N-1         │ ✗
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ PHASE 1 STEP DECAY            │ 0.95 hardcoded / 512 inner │ settings.step_size_decay │ ~
# │                               │ iterations                 │ (0.999 per outer step)   │
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ SCORE TERMS – Phase 2         │ arc springs ONLY           │ arc + chain springs      │ ✗
# │              Phase 3          │ chain+angular+orientation  │ chain+arc+orientation    │ ✗
# ├───────────────────────────────┼────────────────────────────┼──────────────────────────┤
# │ PARAMETERS (Settings.cpp vs settings.py defaults)                                    │
# │   max_temp_arcs               │ 20.0                       │ 10.0                     │ ✗
# │   max_temp_smooth             │ 20.0                       │ 5.0                      │ ✗
# │   dt_temp_arcs                │ 0.99995                    │ 0.9999                   │ ✗
# │   dt_temp_smooth              │ 0.99995                    │ 0.9999                   │ ✗
# │   temp_jump_coef_arcs         │ 20.0                       │ 10.0                     │ ✗
# │   temp_jump_scale_arcs        │ 50.0                       │ 20.0                     │ ✗
# │   temp_jump_coef_smooth       │ 20.0                       │ 5.0                      │ ✗
# │   temp_jump_scale_smooth      │ 50.0                       │ 10.0                     │ ✗
# │   min_successes_arcs/smooth   │ 5                          │ 10                       │ ✗
# │   milestone_steps             │ 10000 (individual moves)   │ 1 (outer step)           │ ✗
# │   improvement_ratio           │ 0.995 (ratio, per phase)   │ 1e-4 (absolute)          │ ✗
# │   k_chain / springConstant    │ squeeze=stretch=0.1        │ k_chain=1.0              │ ✗
# │   genomic_dist_power          │ 0.5                        │ 0.75                     │ ✗
# │   genomic_dist_scale          │ 1.0                        │ 0.5                      │ ✗
# │   genomic_dist_base           │ 0.0                        │ 1.0                      │ ✗
# │   freq_dist_scale (heatmap)   │ 100.0                      │ 25.0                     │ ✗
# │   freq_dist_power (heatmap)   │ -0.333                     │ -0.6                     │ ✗
# └───────────────────────────────┴────────────────────────────┴──────────────────────────┘


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – RANDOM DISPLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/thirdparty/common.cpp:
#   float random(float range, bool negative) {
#     if (negative)
#       return (2.0f * random_uniform() - 1.0f) * range;  // uniform in [-range, range]
#     return range * random_uniform();
#   }
#   vector3 random_vector(float max_size, bool in2D) {
#     if (in2D)
#       return vector3(random(max_size, true), random(max_size, true), 0.0);
#     return vector3(random(max_size, true),   // ← uniform per-component
#                   random(max_size, true),
#                   random(max_size, true));
#   }
#
# Python mc.py (INCORRECT):
#   v = torch.randn(3, device=device)          # Gaussian, not uniform
#   if use_2d: v[2] = 0.0
#   return v * (step_size / v.norm().clamp(min=1e-9))  # normalised to exact length
#
# DIFF: cudaMMC draws each component independently from Uniform[-step,step].
#   The Python version draws from a sphere surface (constant magnitude).
#   These have different distributions: the C++ cube can have magnitude
#   up to step*sqrt(3) while always landing exactly at step_size in Python.

def ref_random_displacement(step_size: float, use_2d: bool) -> list:
    """Reference: cudaMMC random_vector(step_size, use2D)."""
    x = (2 * random.random() - 1) * step_size
    y = (2 * random.random() - 1) * step_size
    z = 0.0 if use_2d else (2 * random.random() - 1) * step_size
    return [x, y, z]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – METROPOLIS ACCEPTANCE
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/src/LooperSolver.cpp (MonteCarloArcs, line 3114):
#   tp = Settings::tempJumpScale *
#        exp(-Settings::tempJumpCoef * (score_curr / score_prev) / T);
#   ok = withChance(tp);
#
# cudaMMC/src/ParallelMonteCarloHeatmap.cu (GPU kernel):
#   withChance(settings.tempJumpScaleHeatmap *
#              expf(-settings.tempJumpCoefHeatmap *
#                   (score_curr * (1/score_prev)) * (1/T)), &localState)
#
# Python mc.py _with_chance (INCORRECT):
#   prob = jump_scale * math.exp(-delta / max(temp, 1e-10))
#   where temp = T * jump_coef
#   → prob = jump_scale * exp(-delta / (T * coef))
#
# DIFF:
#   cudaMMC: prob = scale * exp(-coef * ratio / T)   where ratio = score_curr / score_prev
#   Python:  prob = scale * exp(-delta / (T * coef)) where delta = score_curr - score_prev
#
#   For small δ: ratio ≈ 1 + δ/S, so cudaMMC ≈ scale*exp(-coef/T)*exp(-coef*δ/(T*S))
#   Python:                                           scale*exp(-δ/(T*coef))
#
#   The cudaMMC formula normalises δ by the current total score S, making acceptance
#   probability adaptive. Python uses absolute delta — fundamentally different dynamics.

def ref_metropolis_accept(score_curr: float, score_prev: float,
                          jump_scale: float, jump_coef: float, T: float,
                          rng: random.Random) -> bool:
    """Reference: cudaMMC Metropolis acceptance (ratio-based).
    cudaMMC/src/LooperSolver.cpp line 3114–3116
    """
    if score_curr <= score_prev:
        return True
    prob = jump_scale * math.exp(-jump_coef * (score_curr / max(score_prev, 1e-30)) / T)
    return rng.random() < prob


def py_metropolis_accept_buggy(delta: float, jump_scale: float,
                                jump_coef: float, T: float,
                                rng: random.Random) -> bool:
    """Current Python implementation (delta-based — does NOT match cudaMMC)."""
    if delta <= 0:
        return True
    prob = jump_scale * math.exp(-delta / max(T * jump_coef, 1e-10))
    return rng.random() < prob


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – MILESTONE / STOPPING CRITERION
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/src/LooperSolver.cpp MonteCarloArcs (line 3133–3151):
#   if (i % Settings::MCstopConditionSteps == 0) {
#     if ((score_curr > MCstopConditionImprovement * milestone_score &&
#          milestone_success < MCstopConditionMinSuccesses) ||
#         score_curr < 1e-5 || score_curr / milestone_score > 0.9999)
#       break;
#     milestone_score = score_curr;
#     milestone_success = 0;
#   }
#   score_prev = score_curr;   // updated every step
#
# Defaults (Settings.cpp lines 236–238):
#   MCstopConditionImprovement    = 0.995   ← ratio test (must improve > 0.5%)
#   MCstopConditionMinSuccesses   = 5
#   MCstopConditionSteps          = 10000   ← individual bead moves
#
# Python mc.py monte_carlo_arcs (INCORRECT):
#   milestone_improvement = prev_milestone - total_score   # absolute delta
#   if (milestone_improvement < s.improvement_threshold_arcs  # = 1e-4
#           and successes < s.min_successes_arcs):             # = 10
#       break
#   # checked every outer step (= N bead moves), not every 10000 steps
#
# DIFF (3 separate issues):
#   1. Criterion type: ratio (0.995) vs absolute (1e-4)
#   2. Check frequency: every 10000 individual moves vs every outer step
#   3. Min successes: 5 vs 10

def ref_milestone_should_stop(score_curr: float, milestone_score: float,
                               milestone_success: int,
                               improvement: float = 0.995,
                               min_successes: int = 5) -> bool:
    """Reference: cudaMMC stopping criterion (ratio-based).
    cudaMMC/src/LooperSolver.cpp line 3143–3147
    """
    return ((score_curr > improvement * milestone_score and
             milestone_success < min_successes) or
            score_curr < 1e-5 or
            score_curr / max(milestone_score, 1e-30) > 0.9999)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – PHASE 1 GPU HEATMAP MC
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/src/ParallelMonteCarloHeatmap.cu:
#
#   __global__ void monteCarloHeatmapKernel(…) {
#     // Each thread = one bead (warp-parallel = Jacobi update)
#     // N_WARP = blockDim.x = 512 threads = 512 beads simultaneously
#
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;  // bead index
#
#     for (int inner = 0; inner < mc_inner_steps; inner++) {
#       // random_vector: uniform[-step, step] per component (not normalised)
#       vector.x = random(max_size, true, state);
#       vector.y = random(max_size, true, state);
#       vector.z = settings.use2D ? 0.0f : random(max_size, true, state);
#
#       newpos = pos[idx] + vector;
#
#       // per-bead score delta vs ALL other beads
#       score_prev = calcScoreHeatmapBead(idx, pos, …);
#       score_curr = calcScoreHeatmapBead(idx, newpos, pos, …);
#
#       // Metropolis: RATIO-based
#       if (score_curr < score_prev ||
#           withChance(scale * expf(-coef * score_curr/score_prev / T), state))
#         pos[idx] = newpos;
#     }
#
#     // after mc_inner_steps Jacobi rounds:
#     T     *= settings.dtTempHeatmap;
#     step  *= 0.95f;   // ← hardcoded step decay per 512-iteration block
#   }
#
# Python mc.py monte_carlo_heatmap (correspondence):
#   ✓ Jacobi (all beads simultaneously)
#   ✗ displacement: randn/normalise  ← should be uniform[-step,step]
#   ✗ Metropolis: delta-based        ← should be ratio-based
#   ✓ T *= dt_temp per outer step
#   ✗ step decay: 0.999 (setting)    ← should be 0.95 per 512-iter block
#   ✗ milestone: absolute / per step ← should be ratio / per 10000 moves
#   ✓ recompute true score periodically (added in this session)
#
# cudaMMC GPU per-bead score (heatmap):
#   score_i = sum_{j≠i, |i-j|≥diag, exp[i,j]>1e-3}
#             ((dist(pos_i, pos_j) / exp[i,j]) - 1)^2
#   ← matches Python _bead_score_delta exactly ✓


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – PHASE 2 ARCS MC
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/src/LooperSolver.cpp MonteCarloArcs (lines 3058–3159):
#
#   score_curr = calcScoreDistancesActiveRegion();  // arc springs only, no chain!
#   score_prev = score_curr;
#   milestone_score = score_curr;
#   i = 1;
#   while (true) {
#     p = random(size);         // ← RANDOM bead (Gauss-Seidel), not sequential
#     ind = active_region[p];
#     if (clusters[ind].is_fixed) error("…");
#
#     local_score_prev = calcScoreDistancesActiveRegion(p);  // arc springs only
#     tmp = random_vector(step_size, use2D);                 // uniform cube
#     clusters[ind].pos += tmp;
#
#     local_score_curr = calcScoreDistancesActiveRegion(p);
#     score_curr = score_curr - local_score_prev + local_score_curr;
#
#     ok = score_curr <= score_prev;
#     if (!ok) {
#       tp = tempJumpScale * exp(-tempJumpCoef * (score_curr/score_prev) / T);
#       ok = withChance(tp);
#     }
#     if (ok) { success++; milestone_success++; }
#     else { clusters[ind].pos -= tmp; score_curr = score_prev; }
#
#     T *= dt;   // ← cooling EVERY bead move, not every outer step
#
#     if (i % MCstopConditionSteps == 0) {  // every 10000 moves
#       if (score_curr > 0.995*milestone_score && milestone_success < 5) break;
#       milestone_score = score_curr; milestone_success = 0;
#     }
#     score_prev = score_curr;
#     i++;
#   }
#
# Python mc.py monte_carlo_arcs (correspondence):
#   ✗ bead selection: sequential 0…N-1   ← should be random
#   ✗ score: arc + chain                 ← should be arc springs ONLY
#   ✗ Metropolis: delta-based            ← should be ratio-based
#   ✗ T decay: per outer step (N moves)  ← should be per bead move
#   ✗ milestone: absolute / per step     ← ratio / every 10000 moves
#   ✗ parameters: wrong defaults (see table above)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – PHASE 3 SMOOTH MC
# ─────────────────────────────────────────────────────────────────────────────
# cudaMMC/src/LooperSolver.cpp MonteCarloArcsSmooth (lines 3161–3390):
#
#   // score = structure (chain+angular) + orientation + optional subanchor heatmap
#   // NO direct arc springs in this phase!
#   curr_score_structure   = calcScoreStructureSmooth(true, true);
#   curr_score_orientation = calcScoreOrientation(anchor_orientation);
#   score_curr = curr_score_structure + curr_score_orientation + curr_score_heat;
#
#   while (true) {
#     p = random(size);   // random bead
#     ind = active_region[p];
#     if (clusters[ind].is_fixed) continue;
#
#     local_score_prev_structure = calcScoreStructureSmooth(p, true, true);
#     // (orientation score per bead, if anchor neighbor)
#     tmp = random_vector(step_size, use2D);    // uniform cube
#     clusters[ind].pos += tmp;
#
#     local_score_curr_structure = calcScoreStructureSmooth(p, true, true);
#     // orientation updated and scaled by factor 2:
#     curr_score_orientation += 2*(local_curr_orient - local_prev_orient);
#     curr_score_structure   += (local_curr_struct  - local_prev_struct);
#     // arc score: NOT included
#
#     ok = score_curr < score_prev;
#     if (!ok && T > 0) {
#       tp = tempJumpScaleSmooth * exp(-tempJumpCoefSmooth * (s1/s0) / T);
#       ok = withChance(tp);
#     }
#     if (ok) { success++; milestone_success++; score_prev = score_curr; }
#     else { clusters[ind].pos -= tmp; score_curr = score_prev; /* restore */ }
#
#     if (i % MCstopConditionStepsSmooth == 0) { /* ratio milestone */ }
#     T *= dt;   // cooling every bead move
#     i++;
#   }
#
# Python mc.py monte_carlo_arcs_smooth (correspondence):
#   ✗ bead selection: sequential           ← should be random
#   ✗ score: chain + ARC + orientation     ← should be chain+angular+orientation ONLY
#   ✗ Metropolis: delta-based              ← ratio-based
#   ✗ T decay: per outer step              ← per bead move
#   ✗ milestone: absolute / per step       ← ratio / every 10000 moves
#   ✓ factor 2 on orientation/arc (pairwise) ← matches (arc is wrong term but factor matches)
#   ✗ parameters: wrong defaults


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL SANITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_metropolis_formulas():
    """Show difference between ratio and delta Metropolis formulas."""
    rng1 = random.Random(42)
    rng2 = random.Random(42)

    jump_scale = 50.0
    jump_coef = 20.0
    T = 10.0

    score_prev = 5.0
    deltas = [0.01, 0.1, 0.5, 1.0, 5.0]

    print("Metropolis formula comparison (jump_scale=50, jump_coef=20, T=10, score_prev=5)")
    print(f"{'delta':>8} {'score_curr':>12} {'ratio prob':>12} {'delta prob':>12} {'ratio/delta':>12}")
    print("-" * 60)
    for delta in deltas:
        score_curr = score_prev + delta
        ratio_prob = jump_scale * math.exp(
            -jump_coef * (score_curr / score_prev) / T)
        delta_prob = jump_scale * math.exp(
            -delta / max(T * jump_coef, 1e-10))
        print(f"{delta:8.3f} {score_curr:12.3f} {ratio_prob:12.6f} {delta_prob:12.6f} "
              f"{ratio_prob/delta_prob:12.3f}x")
    print()


def test_milestone_criterion():
    """Show when ratio vs absolute milestone fires."""
    print("Milestone stopping criterion comparison")
    print("(cudaMMC: ratio 0.995; Python: absolute delta 1e-4)")
    print()
    score_sequences = [
        ("Rapid improvement",  [100.0, 80.0, 60.0, 50.0, 49.9]),
        ("Slow improvement",   [100.0, 99.9, 99.8, 99.7, 99.6]),
        ("Tiny improvement",   [100.0, 100.0-1e-3, 100.0-2e-3]),
        ("Near-zero score",    [0.01,  0.009, 0.008]),
    ]
    for name, seq in score_sequences:
        print(f"  {name}:")
        for i in range(1, len(seq)):
            s0, s1 = seq[i-1], seq[i]
            delta = s0 - s1
            ratio_stop = ref_milestone_should_stop(s1, s0, milestone_success=4)
            abs_stop   = delta < 1e-4
            print(f"    {s0:.4f}→{s1:.4f}  "
                  f"ratio_stop={ratio_stop}  abs_stop={abs_stop}")
        print()


def test_score_formula_heatmap():
    """Verify Python score matches the cudaMMC heatmap score formula."""
    import sys
    sys.path.insert(0, ".")
    try:
        from src.scores import score_heatmap_chunked
    except ImportError:
        print("SKIP test_score_formula_heatmap: src not importable")
        return

    # 3 beads, explicit expected distances
    pos = torch.tensor([[0.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0],
                         [4.0, 0.0, 0.0]])
    # expected[i,j] = 2.0 for all pairs
    exp = torch.full((3, 3), 2.0)

    # cudaMMC heatmap score formula (from ParallelMonteCarloHeatmap.cu):
    # score_i = sum_{j≠i, |i-j|≥diag} ((dist(i,j)/exp[i,j]) - 1)^2
    # total = sum_i score_i, but each pair (i,j) counted TWICE (once from i, once from j)
    #
    # NOTE: Python score_heatmap_chunked counts each pair ONCE.
    # Check whether the factor matters for the per-bead delta.
    diag = 2  # exclude |i-j| < 2
    # For diagonal_size=2: only pair (0,2) is included, |0-2|=2 >= 2
    d02 = 4.0
    e02 = 2.0
    expected_score_per_pair = ((d02/e02) - 1)**2  # = (2-1)^2 = 1

    py_score = score_heatmap_chunked(pos, exp, diag, None).item()
    print(f"test_score_formula_heatmap:")
    print(f"  expected (one pair, counted once): {expected_score_per_pair:.4f}")
    print(f"  Python score_heatmap_chunked:      {py_score:.4f}")
    print(f"  {'✓' if abs(py_score - expected_score_per_pair) < 1e-5 else '✗'} match\n")


def test_displacement_distribution():
    """Compare magnitude statistics of uniform-cube vs normalized-Gaussian displacement."""
    import numpy as np
    step = 1.0
    N = 100_000

    # cudaMMC: uniform per-component in [-step, step]
    cube = np.random.uniform(-step, step, (N, 3))
    cube_mag = np.linalg.norm(cube, axis=1)

    # Python: Gaussian, normalised to step
    gauss = np.random.randn(N, 3)
    gauss_mag = np.linalg.norm(gauss, axis=1, keepdims=True)
    sphere = gauss / (gauss_mag + 1e-9) * step
    sphere_mag = np.linalg.norm(sphere, axis=1)

    print("Displacement distribution comparison (step_size=1.0, N=100k samples):")
    print(f"  cudaMMC uniform-cube: mean_mag={cube_mag.mean():.3f}  "
          f"max_mag={cube_mag.max():.3f}  std_mag={cube_mag.std():.3f}")
    print(f"  Python sphere-surf:  mean_mag={sphere_mag.mean():.3f}  "
          f"max_mag={sphere_mag.max():.3f}  std_mag={sphere_mag.std():.3f}")
    print(f"  NOTE: cube magnitude varies {cube_mag.min():.3f}–{cube_mag.max():.3f}; "
          f"sphere is constant 1.000\n")


def run_all_tests():
    print("=" * 70)
    print("cudaMMC vs Python algorithm verification")
    print("=" * 70)
    print()
    test_metropolis_formulas()
    test_milestone_criterion()
    test_score_formula_heatmap()
    test_displacement_distribution()
    print("=" * 70)
    print("Discrepancy summary: see DISCREPANCY MASTER TABLE at top of file")
    print("Most critical (affects output quality, in priority order):")
    print("  1. Metropolis formula: ratio-based vs delta-based")
    print("  2. Parameter values: max_temp, dt_temp, jump_coef, jump_scale")
    print("  3. Milestone criterion: ratio 0.995 vs absolute 1e-4")
    print("  4. Bead selection: random vs sequential (Gauss-Seidel vs round-robin)")
    print("  5. Phase score terms: arcs-phase should omit chain; smooth-phase should omit arcs")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
