# 3dgnome-ng: Agent Guide

## Project Goal

Python reimplementation of the Monte Carlo (MC) core of **3dgnome**. The original implementation is a ~6,400-line C++ simulation (`3dnome/MC/`) that predicts 3D chromosome structure from Hi-C contact frequency data. The reimplementation lives in `gnome3d/` and reproduced the C++ algorithm behavior as its starting point. MC loops run on CPU via Numba JIT; torch is used only for GPU device detection and the reference scoring functions in `gnome3d/energy.py`.

Do **not** modify anything inside `3dnome/`. That directory is the reference implementation - read it, never change it.

### Status: post-parity, feature-extension phase

The Python port reached algorithmic parity with the C++ reference; that work is documented and frozen. **New work no longer requires matching C++.** Features added from here on (biophysics extensions, new energy terms, scheduling tweaks, etc.) are expected to diverge intentionally from `3dnome/`.

Rules for new feature work:

- **All new features must be opt-in via `gnome3d/settings.py`.** Default-off so existing configs continue to reproduce the parity-era behavior.
- **Document divergences** in the "Python divergences from reference" section below — what changed, why, and which setting toggles it.
- The C++ reference and `harness/compare.py` / `harness/integration.py` remain authoritative for the **parity baseline** (feature flags off). They are not authoritative for new features.
- Inside a feature's own code, document any non-obvious behavior; project memory (`[[…]]` links) carries the longer-form rationale.

---

## Repository Layout

```
3dgnome-torch/
├── 3dnome/MC/                  # Reference C++ implementation (READ ONLY)
│   ├── LooperSolver.cpp/h      # Main solver - all MC loops live here
│   ├── Chromosome.cpp/h        # 3D structure (list of bead positions)
│   ├── HierarchicalChromosome.cpp/h  # Multi-level representation
│   ├── Heatmap.cpp/h           # 2D contact/frequency matrices
│   ├── InteractionArcs.cpp/h   # Pairwise arc/interaction management
│   ├── Cluster.cpp/h           # Single bead definition
│   └── lib/                    # mtxlib (vec3/mat44), RNG, RMSD utilities
├── src/                        # New PyTorch implementation (write here)
├── data/                       # Input datasets (GM12878, H1ESC, HFFC6)
│   └── GM12878/config.ini      # Example config with all parameters
├── pyproject.toml              # gnome3d-torch, entry point: main:main
└── AGENTS.md                   # This file
```

Python environment: `.venv/bin/python` (Python 3.11, torch >= 2.0, numpy >= 1.24).

```
3dgnome-torch/
├── harness/
│   ├── scorer.cpp      # C++ reference scorer compiled against real 3dnome sources
│   ├── compare.py      # Unit-level correctness harness (energy functions)
│   └── integration.py  # Integration test: run full MC on a region, compare distributions
```

---

## Algorithm Overview

3dgnome solves chromosome structure as a **coarse-to-fine hierarchical Monte Carlo** problem. There are four nested levels, each feeding the next:

```
Level 1 (Chromosome)  - whole chromosomes as single beads, inter-chr contacts
Level 2 (Segment)     - ~100kb–1Mb segments within each chromosome
Level 3 (Anchor)      - ~5–50kb ChIA-PET loop anchor regions
Level 4 (Subanchor)   - ~1–10kb fine-resolution loop bases
```

Each level runs a simulated annealing MC loop, then passes the resulting 3D positions down as constraints for the next level.

### Monte Carlo Loop (all levels)

```
initialize positions (random or interpolated from parent level)
T = T_start
while not converged:
    i = random bead in active region
    score_before = local_score(i)
    apply random displacement to i
    score_after  = local_score(i)
    delta = score_after - score_before
    if delta < 0 or rand() < exp(-tempJumpCoef * (score_after/score_before) / T):
        accept move  (update global_score += delta)
    else:
        reject move  (restore position)
    T *= dtTemp
    if milestones_without_improvement >= convergence_threshold:
        break
```

Reference: `LooperSolver.cpp` lines 329–405 (`MonteCarloHeatmap`), 2304–2550 (`MonteCarloArcs`, `MonteCarloArcsSmooth`).

---

## Data Structures

### Cluster (bead)
```python
@dataclass
class Cluster:
    pos: Tensor          # shape (3,), float32 - 3D position
    genomic_pos: int
    start: int           # genomic range start
    end: int             # genomic range end
    orientation: str     # 'L' | 'R' | 'N'  - CTCF motif direction
    parent: int          # index into parent-level cluster list
    level: int
    arcs: list[int]      # indices into arc list
    children: list[int]
    is_fixed: bool
    dist_to_next: float  # expected distance to next bead in chain
```

### Heatmap
```python
# shape (N, N), float32
# v[i][j] = contact frequency (or derived expected distance)
# diagonal_size: strip around diagonal to ignore
```

### InteractionArc
```python
@dataclass
class InteractionArc:
    start: int           # cluster index
    end: int             # cluster index
    score: int           # PET count or frequency
    eff_score: int       # aggregated/effective score
    factor: int          # protein factor (e.g. CTCF)
```

---

## Energy Functions

All scoring functions return a scalar; lower is better. Each level uses a subset.

### 1. Heatmap score (levels 1 & 2)
Compares pairwise distances against expected distances derived from contact frequency.

```
score = sum_{|i-j| in diagonal band} ((d_actual - d_expected)² / d_expected²)
```

Reference: `calcScoreHeatmapActiveRegion()` line 1752.

### 2. Arc spring score (level 3)
Spring energy for each pairwise arc interaction:

```
score = sum_arcs spring_k * ((d_actual - d_expected)² / d_expected²)

spring_k = stretchConstant  if d_actual > d_expected
           squeezeConstant  if d_actual < d_expected
```

Reference: `calcScoreDistancesActiveRegion()` line 1548.

### 3. Structure smoothness score (level 4)
Penalizes sharp bends and length deviations along the bead chain:

```
score = w_dist  * sum |d_actual - d_expected| / d_expected
      + w_angle * sum angle(bond_i, bond_{i+1})³
```

Reference: `calcScoreStructureSmooth()` line 1637.

### 4. CTCF orientation score (level 4)
For adjacent anchors connected by arcs, penalizes non-convergent CTCF orientations:

```
score = w_motif * sum angle(orientation_i, orientation_j)²
```

Reference: `calcScoreOrientation()` line 1673.

### 5. Subanchor heatmap score (level 4)
Same formula as heatmap score (#1) but applied at fine resolution.

Reference: `calcScoreSubanchorHeatmap()` line 1782.

---

## Move Types

| Level | Move | Step size |
|-------|------|-----------|
| Chromosome (1) | Random 3D displacement | `avg_distance * noiseCoefficientLevelChr` |
| Segment (2) | Random 3D displacement | `avg_distance * noiseCoefficientLevelSegment` |
| Anchor (3) | Random 3D displacement | `~0.01 × avg_distance` |
| Subanchor (4) | Random 3D displacement | `~5.0` (absolute) |
| Subanchor (4) | CTCF orientation update | triggered when near-anchor bead moves |

Displacement vectors are sampled from a 3D Gaussian (or uniform sphere). The C++ implementation uses `randBall()` / `randBallGaussian()` in `lib/common.h`.

---

## Key Parameters (from config.ini)

| Parameter | Meaning |
|-----------|---------|
| `tempJumpCoef` | Scale factor in Metropolis criterion |
| `dtTemp` | Temperature cooling multiplier per step |
| `noiseCoefficientLevel*` | Step size multiplier per level |
| `stretchConstant` / `squeezeConstant` | Spring asymmetry |
| `weightDist` / `weightAngle` | Smoothness energy weights |
| `motifWeight` | CTCF orientation energy weight |
| `subanchorHeatmapDistWeight` | Fine-resolution heatmap energy weight |
| `loopDensity` | Number of interpolated beads between anchors |
| `convergenceThreshold` | Milestones without improvement to stop |

Full list: `3dnome/MC/Settings.cpp` and example `data/GM12878/config.ini`.

---

## Implementation Plan for `src/`

Suggested module layout:

```
src/
├── __init__.py
├── data_structures.py   # Cluster, Heatmap, InteractionArc dataclasses
├── io.py                # Load anchors, singletons, arcs from files
├── distance.py          # genomicLengthToDistance() and heatmap -> expected distance
├── energy.py            # All five scoring functions as torch operations
├── moves.py             # Random displacement sampling
├── mc.py                # MC loop (simulated annealing), convergence logic
├── hierarchy.py         # Multi-level orchestration: chr -> seg -> anchor -> subanchor
├── densify.py           # Bead densification between anchors
└── main.py              # Entry point (gnome3d CLI)
```

### PyTorch notes

- Store all bead positions as a `(N, 3)` float32 tensor on the target device.
- Energy functions should operate on tensor slices for the *active region* only (not all N beads), matching the C++ local-score pattern.
- The inner MC loop is inherently sequential (each step depends on the previous accept/reject), so do **not** try to batch proposals within a single chain. Batch across independent MC chains instead (ensemble generation).
- `torch.no_grad()` everywhere in the MC loop - we are doing stochastic search, not gradient descent.
- Use `torch.compile` or keep operations simple to avoid recompilation overhead inside the loop.

---

## Reference Files to Read First

When working on any piece of the rewrite, read the corresponding C++ reference first:

| Task | Reference |
|------|-----------|
| Overall MC structure | `LooperSolver.cpp` lines 329–405, 2304–2550 |
| All scoring functions | `LooperSolver.cpp` lines 1518–1810 |
| Densification | `LooperSolver.cpp` line 1811 (`densifyActiveRegion`) |
| Bead/cluster definition | `Cluster.h`, `Chromosome.h` |
| Heatmap format | `Heatmap.cpp/h` |
| Arc format | `InteractionArc.h`, `InteractionArcs.cpp/h` |
| Distance/frequency mapping | `LooperSolver.cpp` `genomicLengthToDistance()` |
| Random number generation | `lib/common.h` `randBall`, `randBallGaussian` |
| Config parameters | `Settings.cpp`, `data/GM12878/config.ini` |

---

## Correctness Harness

The harness compiles `harness/scorer.cpp` directly against the real 3dnome MC sources (`3dnome/MC/*.cpp`). It uses `#define private public` before including `LooperSolver.h` to expose private methods - access control is compile-time only, so the object layout and compiled method bodies are identical to production. The result is that every comparison runs the actual `calcScoreHeatmapActiveRegion()`, `calcScoreStructureSmooth()`, etc. - not a reimplementation.

### Quick start

```bash
# First build (auto-runs on first comparison too)
python harness/compare.py --build-only

# Print C++ reference values only - no Python impl needed
python harness/compare.py --reference

# Run all tests (skips anything not yet in src/)
python harness/compare.py

# Run a specific test group
python harness/compare.py distfns
python harness/compare.py heatmap arcs smooth
```

### What is tested

| Group | What it checks | Python hook (src/energy.py) |
|-------|---------------|-----------------------------|
| `angle` | Custom angle metric (`1 - (dot+1)/2`, NOT acos) | `angle_metric(v1, v2)` |
| `distfns` | `genomicLengthToDistance`, `freqToDistanceHeatmap`, `freqToDistance` | `genomic_length_to_distance`, `freq_to_dist_heatmap`, `freq_to_distance` |
| `heatmap` | Full double-counted heatmap score | `score_heatmap(pos, exp_dist, diag)` |
| `arcs` | Arc spring score with repulsion branch | `score_arcs(pos, arcs, stretch_k, squeeze_k)` |
| `smooth` | Chain length + cubic angle penalty | `score_smooth(pos, dtn, stretch_k, squeeze_k, angular_k, w_dist, w_angle)` |
| `metropolis` | Acceptance probability `jump_scale * exp(-jump_coef * ratio / T)` | `metropolis_prob(js, jc, sc, sp, T)` |

### Non-obvious details captured in scorer.cpp

- **`angle()` is NOT `acos`**: `3dnome/MC/lib/common.cpp:40` defines it as `1 - (dot(norm(v1), norm(v2)) + 1) / 2`, a linear dissimilarity in [0, 1]. The smooth score's cubic penalty uses this.
- **Heatmap score double-counts**: the C++ computes `sum_moved sum_i err(i, moved)`, which counts every pair (i,j) twice. The Python must match this convention exactly so that the MC delta `2*(local_curr - local_prev)` is consistent.
- **Global score update**: `score_curr += 2.0 * (local_score_curr - local_score_prev)`. The factor 2 comes from the double-counting above.
- **Metropolis uses ratio, not difference**: acceptance probability is `jump_scale * exp(-jump_coef * (score_curr / score_prev) / T)`, and `jump_scale` (default 50) can push the probability above 1.
- **Random displacement is uniform in a cube**: `random_vector(step)` returns `(rand(±step), rand(±step), rand(±step))`, not a sphere or Gaussian.

---

## Integration Test

`harness/integration.py` runs the real C++ binary on a small ~2 Mb chr1 region (`chr1:18288319:20307135`, ~34 anchor beads) to produce an ensemble of structures, then runs the Python reimplementation on the same region and compares their bead-position distributions.

### What it measures

| Metric | Method |
|--------|--------|
| Radius of gyration | per-structure scalar; compare mean ± std |
| Pooled pairwise distances | all i<j pairs from all structures; 2-sample KS test |
| Consecutive bond lengths | chain bond distribution; 2-sample KS test |

PASS criteria: KS statistic ≤ 0.3 and p-value ≥ 0.05 for both pairwise and bond distributions.  The C++ run uses `-v 2` (heatmap + arc reconstruction), so leaf beads in the output are the ~34 anchor-level clusters.

### Interface Python must expose

```python
# src/simulate.py
def run_region(config_path: str, region: str, n_structures: int) -> list:
    """
    Run MC for the given region and return n_structures conformations.
    Each conformation is a list of (midpoint_bp, x, y, z) sorted by genomic position,
    matching the anchor-level beads output by C++ at -v 2.
    """
```

### Usage

```bash
# C++ reference run only (no Python required)
python harness/integration.py --cpp-only   # region: chr1:18288319-20307135 (dash, not colon)

# Full comparison (skips Python side gracefully if not implemented)
python harness/integration.py

# Faster run for CI (~5s/structure, lower quality)
python harness/integration.py --fast -n 3

# Keep output files for inspection
python harness/integration.py --keep
```

The test auto-skips the Python comparison if `src/simulate.py` is missing or raises `NotImplementedError` - it never fails just because Python is not yet implemented.

---

## Python divergences from reference

Tracked list of intentional deviations from `3dnome/MC/`. Each entry: what diverges, why, and how to toggle/restore parity. Keep this list current — new entries must be added when introducing diverging behavior, and removed when behavior is brought back into parity.

### Refactors (no behavior change at parity settings)

- **Unified smooth-MC kernel** ([gnome3d/mc.py](gnome3d/mc.py))
  C++ has separate `MonteCarloArcsSmooth` branches per feature combo. Python collapses the four prior specialized kernels (`_batch_smooth_nb`, `_batch_smooth_heat_nb`, `_batch_smooth_orientation_nb`, `_batch_smooth_orientation_heat_nb`) into one `_batch_smooth_kernel_nb` driven by `use_heat`/`use_orn` flags. Energy terms (struct, heat, orn) are tracked as independent components (`score_struct`, `score_orn`, `score_heat`) and combined per step.
  Side benefit: heat/orn paths previously did a full O(N) structure recompute every MC step — now incremental like the pure-smooth path. Verified drift stays at float-noise (~1e-9) under the codebase's reciprocal-neighbor invariant.

### Algorithm divergences

- **Orientation MC: weighted local scorer**
  Python uses a weighted local delta in `_local_score_orientation_nb` ([gnome3d/mc.py](gnome3d/mc.py)) so the incremental update is exact w.r.t. `_score_orientation_full_nb`. C++ uses an unweighted local scorer that drifts over many steps. See `[[project-orientation-mc-fix]]`. The reference scorer in `gnome3d/energy.py` stays unweighted so `harness/compare.py` still passes.

- **Singleton chr filter** (data loading)
  C++ bins inter-chromosomal singletons by position; Python correctly filters by chromosome. Smooth-MC heat scores diverge 3-5× as a result, but final structures still match. See `[[project-singleton-chr-filter-divergence]]`.

### New features (opt-in via settings, default-off)

- **Excluded volume** — `settings.use_excluded_volume = true` to enable.
  Soft harmonic repulsion preventing bead overlap. For pairs `(i, j)` with `|i - j| > exclusion_skip_neighbors` and `d_ij < exclusion_radius`, contributes `exclusion_weight * ((r0 - d)/r0)²` to the score. Implemented as an independent score component with the same `2 * (local_curr - local_prev)` incremental pattern as the heat term. Settings:
    - `use_excluded_volume` (master toggle)
    - `exclusion_radius` (r₀, position units; default 1.0)
    - `exclusion_weight` (k, comparable to spring constants; default 1.0)
    - `exclusion_apply_to_arcs` (gate for arc MC; default false)
    - `exclusion_apply_to_smooth` (gate for smooth MC; default true)
    - `exclusion_skip_neighbors` (skip pairs with `|i-j|` ≤ this; default 1, i.e. skip bonded)

  Why not in C++: 3dnome uses an inverse-distance repulsion only on pairs where the input arc matrix is marked negative — a sparse, conditional repulsion. This adds a global polymer-physics excluded-volume term independent of input data. Touches arc-MC (`_batch_arcs_nb`) and smooth-MC (`_batch_smooth_kernel_nb`).

- **Spherical confinement** — `settings.use_confinement = true` to enable.
  Soft envelope mimicking a nuclear boundary. For each bead at distance `r` from the per-MC-call centroid, contributes `confinement_weight * ((r - R)/R)²` if `r > R`, else 0. Per-bead (single-counted) — delta is `(local_curr - local_prev)` with no factor of 2. Settings:
    - `use_confinement` (master toggle)
    - `confinement_radius` (R; 0 = auto from bead count + bond length, see below)
    - `confinement_weight` (k; comparable to spring constants; default 1.0)
    - `confinement_packing_factor` (used only when radius=0; default 1.5 ≈ packing fraction ~7%)
    - `confinement_apply_to_arcs` (default true)
    - `confinement_apply_to_smooth` (default true)

  Auto-radius: `R = packing_factor × avg_bond × N^(1/3)` where `avg_bond` is the mean of positive expected distances (arcs) or `mean(dtn)` (smooth). Center is computed as the centroid of starting positions, so per-IB MC calls naturally confine around the IB centroid; global calls (segment level) confine around the chromosome centroid. Motivated by: small interaction blocks in full-chromosome runs were getting flung out as long stretched chains because chain-bond springs alone couldn't hold loosely-constrained anchors. Touches arc-MC and smooth-MC.

- **Small-IB spring boost** — `settings.use_small_ib_boost = true` to enable.
  When an IB has fewer anchors than `small_ib_threshold`, multiplies `spring_stretch_arcs`, `spring_squeeze_arcs`, `spring_stretch`, `spring_squeeze`, `spring_angular` by `small_ib_spring_multiplier` for that IB only. No kernel changes — implemented in `solver.py::_settings_for_ib()` by passing a `copy.copy(self.s)` clone with boosted values to `_reconstruct_cluster_arcs` / `_reconstruct_cluster_smooth` via an `s_override` parameter. Thread-safe (never mutates `self.s`). Settings:
    - `use_small_ib_boost`
    - `small_ib_threshold` (anchor count below which an IB is "small"; default 10)
    - `small_ib_spring_multiplier` (default 5.0)

  Why not in C++: complements confinement to prevent under-constrained small IBs from stretching out. The boost tightens chain and bond springs so the chain compresses against any repulsive/heatmap forces. Targeted: only affects small IBs, doesn't change behavior of large well-constrained IBs.

---

## Correctness Rules

These rules apply to the **parity baseline** (all new feature flags off). New feature work has its own rules in [Status: post-parity, feature-extension phase](#status-post-parity-feature-extension-phase) above.

1. When working on or near parity code, verify algorithmic choices against the C++ source. Do not invent behavior on the parity path.
2. If behavior in the C++ source is ambiguous or surprising, document it explicitly rather than working around it.
3. **Run `python harness/compare.py` after touching parity-baseline scoring code.** A function is not done until the harness reports PASS for its group.
4. **Run `python harness/integration.py` after touching parity-baseline MC code.** Bead-position distributions of C++ and Python ensembles must remain statistically compatible.
5. Parity-baseline scoring functions must produce numerically equivalent results to the C++ on the same inputs (within 1e-6 absolute tolerance).
6. New features are allowed and encouraged to diverge from C++. They must be opt-in via `gnome3d/settings.py`, documented in the divergences section above, and must not change behavior when their flag is off.
