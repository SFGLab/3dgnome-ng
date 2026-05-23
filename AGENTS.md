# 3dgnome-ng: Agent Guide

## Project Goal

Python reimplementation of the Monte Carlo (MC) core of **3dgnome**. The reference implementation (`3dnome/MC/`) is a ~6,400-line simulation that predicts 3D chromosome structure from Hi-C contact frequency data. The reimplementation lives in `gnome3d/` and reproduced the reference algorithm as its starting point. MC loops run on CPU via Numba JIT; torch is used only for GPU device detection and the reference scoring functions in `gnome3d/energy.py`.

Do **not** modify anything inside `3dnome/`. That directory is the reference implementation - read it, never change it.

### Terminology

Refer to `3dnome/MC/` as **"the reference"** (or "the reference implementation"), never as "C++". The language it happens to be written in is irrelevant to the design; the relevant fact is that it is the algorithmic source-of-truth we port from. This applies in code comments, docstrings, agent-visible documentation, commit messages, and conversations. When you cite a specific file or line, write `LooperSolver.cpp:1069-1104` — not "the C++ code at...".

### Status: post-parity, feature-extension phase

The Python port reached algorithmic parity with the reference; that work is documented and frozen. **New work no longer requires matching the reference.** Features added from here on (biophysics extensions, new energy terms, scheduling tweaks, etc.) are expected to diverge intentionally from `3dnome/`.

Rules for new feature work:

- **All new features must be opt-in via `gnome3d/settings.py`.** Default-off so existing configs continue to reproduce the parity-era behavior.
- **Document divergences** in the "Python divergences from reference" section below — what changed, why, and which setting toggles it.
- The reference and `harness/compare.py` / `harness/integration.py` remain authoritative for the **parity baseline** (feature flags off). They are not authoritative for new features.
- Inside a feature's own code, document any non-obvious behavior; project memory (`[[…]]` links) carries the longer-form rationale.

---

## Repository Layout

```
3dgnome-torch/
├── 3dnome/MC/                  # Reference implementation (READ ONLY)
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
│   ├── scorer.cpp      # reference scorer compiled against real 3dnome sources
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

Displacement vectors are sampled from a 3D Gaussian (or uniform sphere). The reference uses `randBall()` / `randBallGaussian()` in `lib/common.h`.

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
- Energy functions should operate on tensor slices for the *active region* only (not all N beads), matching the reference local-score pattern.
- The inner MC loop is inherently sequential (each step depends on the previous accept/reject), so do **not** try to batch proposals within a single chain. Batch across independent MC chains instead (ensemble generation).
- `torch.no_grad()` everywhere in the MC loop - we are doing stochastic search, not gradient descent.
- Use `torch.compile` or keep operations simple to avoid recompilation overhead inside the loop.

---

## Reference Files to Read First

When working on any piece of the rewrite, read the corresponding reference file first:

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

## Typechecking

The whole `gnome3d/` package is type-checked under **pyright strict mode**. Config lives in `pyproject.toml` under `[tool.pyright]`. Pyright is installed in the project venv:

```bash
.venv/bin/pyright              # check all of gnome3d/
.venv/bin/pyright gnome3d/mc.py  # check one file
```

Current state: **0 errors, 0 warnings**. New code is expected to maintain that. Run pyright before committing changes to `gnome3d/`.

### Conventions

- **Numpy array aliases** live in [gnome3d/types.py](gnome3d/types.py): `F32Array`, `F64Array`, `I32Array`, `I64Array`, `BoolArray`. Use these instead of bare `np.ndarray` so dtype intent is visible.
- **Semantic int aliases** also in `types.py`: `ClusterIndex`, `LocalArcIndex`, `GenomicPos`. They are `int` at runtime — they document the intent of an `int` parameter (a cluster index into `Solver.clusters` is not the same thing as a chr-relative arc index, even though both are `int`).
- **Collection aliases**: `AnchorMap`, `ArcMap`, `RawArcMap`, `BreakpointMap`, `ChrRootMap`, `ChrFirstClusterMap`, `ChrLevel`, `SingletonContact`, `BeadOut`. Prefer these over inline `dict[str, list[…]]` spellings.
- **Output tuples**: `BeadOut = tuple[int, int, float, float, float]` = `(genomic_start_bp, genomic_end_bp, x, y, z)`. First field is start so `sorted(beads, key=lambda b: b[0])` still gives left-to-right genomic order.
- **Dataclass defaults**: use a named factory function (e.g. `_empty_cluster_index_list`) instead of `field(default_factory=list)` — bare `list` leaks `list[Unknown]` through pyright in strict mode.

### Numba interop

`@njit` from numba has no type stubs, which would otherwise force every JIT-compiled function to be typed as `Any`. [gnome3d/mc.py](gnome3d/mc.py) wraps it with a typed identity decorator:

```python
def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))
    return decorator
```

Decorated functions keep their original signatures under pyright while still being JIT-compiled at runtime. **Do not use `typing.cast` inside `@njit` kernels** — numba's nopython frontend doesn't recognise it. For random-index lines like `int(np.random.randint(0, n))` where numpy stubs return `Any`, use a per-line `# pyright: ignore[reportUnknownArgumentType]` comment instead.

### Adding new code

- Annotate every function signature (params + return). Strict mode requires it.
- Avoid bare `list`, `dict`, `tuple` in annotations — always parameterise.
- For numpy arrays in non-kernel code, use the aliases (`F32Array`, etc.).
- Inside `@njit` kernels, parameter and local annotations are advisory only (numba ignores them) but still required for pyright. Keep them realistic — they document the contract.

---

## Correctness Harness

The harness compiles `harness/scorer.cpp` directly against the real 3dnome MC sources (`3dnome/MC/*.cpp`). It uses `#define private public` before including `LooperSolver.h` to expose private methods - access control is compile-time only, so the object layout and compiled method bodies are identical to production. The result is that every comparison runs the actual `calcScoreHeatmapActiveRegion()`, `calcScoreStructureSmooth()`, etc. - not a reimplementation.

### Quick start

```bash
# First build (auto-runs on first comparison too)
python harness/compare.py --build-only

# Print reference values only - no Python impl needed
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
- **Heatmap score double-counts**: the reference computes `sum_moved sum_i err(i, moved)`, which counts every pair (i,j) twice. The Python must match this convention exactly so that the MC delta `2*(local_curr - local_prev)` is consistent.
- **Global score update**: `score_curr += 2.0 * (local_score_curr - local_score_prev)`. The factor 2 comes from the double-counting above.
- **Metropolis uses ratio, not difference**: acceptance probability is `jump_scale * exp(-jump_coef * (score_curr / score_prev) / T)`, and `jump_scale` (default 50) can push the probability above 1.
- **Random displacement is uniform in a cube**: `random_vector(step)` returns `(rand(±step), rand(±step), rand(±step))`, not a sphere or Gaussian.

---

## Integration Test

`harness/integration.py` runs the reference binary on a small ~2 Mb chr1 region (`chr1:18288319:20307135`, ~34 anchor beads) to produce an ensemble of structures, then runs the Python reimplementation on the same region and compares their bead-position distributions.

### What it measures

| Metric | Method |
|--------|--------|
| Radius of gyration | per-structure scalar; compare mean ± std |
| Pooled pairwise distances | all i<j pairs from all structures; 2-sample KS test |
| Consecutive bond lengths | chain bond distribution; 2-sample KS test |

PASS criteria: KS statistic ≤ 0.3 and p-value ≥ 0.05 for both pairwise and bond distributions.  The reference run uses `-v 2` (heatmap + arc reconstruction), so leaf beads in the output are the ~34 anchor-level clusters.

### Interface Python must expose

```python
# src/simulate.py
def run_region(config_path: str, region: str, n_structures: int) -> list:
    """
    Run MC for the given region and return n_structures conformations.
    Each conformation is a list of (midpoint_bp, x, y, z) sorted by genomic position,
    matching the anchor-level beads output by the reference at -v 2.
    """
```

### Usage

```bash
# reference run only (no Python required)
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

### Settings hygiene

- **`noise_arcs` removed.** The reference declares `noiseCoefficientLevelAnchor` (read as `noise_arcs` in `[main]`) and multiplies it into a local `noise_size` variable in `LooperSolver.cpp:2085`, but the arc-MC call site on line 2136 passes a hardcoded `noise_size_small=0.005` instead — making the setting effectively dead in the reference. Python uses the same 0.005 hardcoded constant in [solver.py::_reconstruct_cluster_arcs](gnome3d/solver.py); the setting was dropped to avoid implying configurability.
- **`random_walk` ported.** Previously loaded-but-unused; now drives [solver.py::_random_walk_segment_level](gnome3d/solver.py), mirroring `LooperSolver.cpp:80-98` (chained 50.0-step walk per chromosome instead of segment-level heatmap MC). Honors `use_2d`.
- **`long_pet_*` ported.** Long-range arcs (gap > `max_pet_length`) are no longer discarded by [io.load_arcs](gnome3d/io.py) — they are carried on `ContactData.long_arcs` and folded into the segment heatmap by [solver.py::_add_long_pet_to_segment_heatmap](gnome3d/solver.py) as `long_pet_scale * arc.score ** long_pet_power`. Mirrors `LooperSolver.cpp:1069-1104`, including the asymmetric `h[st][end] += val` pattern (the downstream symmetrize step in `_normalize_heatmap` averages it to `val` on each side).
- **Chromosome-level MC ported.** Previously `steps_lvl1` / `noise_lvl1` were inert because Python had no chr-level reconstruction. Now [solver.py::_reconstruct_chromosome_level](gnome3d/solver.py) builds an n_chr × n_chr inter-chromosomal singleton heatmap, normalizes its first non-zero diagonal to 1.0, converts to expected distances, and runs `mc_heatmap` with `steps_lvl1` runs at `noise_lvl1 × avg_dist` step size. Mirrors `LooperSolver.cpp` lines 119-160 and 265-322. Triggered only when `len(chrs) > 1`; single-chr runs are unaffected. If the singleton input has no inter-chr contacts the chr roots are scattered randomly instead.
- **`normalizeHeatmapInter` ported.** [solver.py::_normalize_heatmap_inter](gnome3d/solver.py) was previously a no-op stub; now it multiplies the segment heatmap by `heatmap_inter_scaling` and divides intra-chromosome blocks back, matching `LooperSolver.cpp:1422-1459`. Net effect: intra-chr unchanged, inter-chr × scale. Active only on multi-chr runs (length-1 `current_level` short-circuits).
- **Multi-chromosome CLI / API surface.** [cli.py](gnome3d/cli.py) and [simulate.py::run_genome](gnome3d/simulate.py) accept the same syntax as the reference's `-c` flag: single chromosome, comma-separated list, `chrN-chrM` numeric range, or single sub-chromosomal region. Default (empty string) matches the reference: `chr1-chr22,chrX`. Parsing is centralized in [io.py::parse_chrs_arg](gnome3d/io.py). The CLI now writes one CIF per chromosome when more than one chr was requested (using a `_<chr>_` suffix in the filename); single-chr/region runs keep the old single-CIF behavior. `run_region` keeps its original single-region semantics for backward compatibility; `run_genome` returns `list[dict[str, list[BeadOut]]]` so per-chromosome bead lists are preserved.
- **`data_singletons_inter` loaded.** [data.py::ContactData.from_files](gnome3d/data.py) reads the optional second singletons file (config key `[data] singletons_inter`) and appends to `singletons` whenever `len(chrs) > 1`. Mirrors `LooperSolver.cpp:970` which adds inter-chromosomal singleton files for multi-chr runs. Single-chr runs skip this file (per-chr contacts in the main file are sufficient and the inter file is often a sparse stub).

### Refactors (no behavior change at parity settings)

- **Unified smooth-MC kernel** ([gnome3d/mc.py](gnome3d/mc.py))
  The reference has separate `MonteCarloArcsSmooth` branches per feature combo. Python collapses the four prior specialized kernels (`_batch_smooth_nb`, `_batch_smooth_heat_nb`, `_batch_smooth_orientation_nb`, `_batch_smooth_orientation_heat_nb`) into one `_batch_smooth_kernel_nb` driven by `use_heat`/`use_orn` flags. Energy terms (struct, heat, orn) are tracked as independent components (`score_struct`, `score_orn`, `score_heat`) and combined per step.
  Side benefit: heat/orn paths previously did a full O(N) structure recompute every MC step — now incremental like the pure-smooth path. Verified drift stays at float-noise (~1e-9) under the codebase's reciprocal-neighbor invariant.

### Algorithm divergences

- **Orientation MC: weighted local scorer**
  Python uses a weighted local delta in `_local_score_orientation_nb` ([gnome3d/mc.py](gnome3d/mc.py)) so the incremental update is exact w.r.t. `_score_orientation_full_nb`. The reference uses an unweighted local scorer that drifts over many steps. See `[[project-orientation-mc-fix]]`. The reference scorer in `gnome3d/energy.py` stays unweighted so `harness/compare.py` still passes.

- **Singleton chr filter** (data loading)
  The reference bins inter-chromosomal singletons by position; Python correctly filters by chromosome. Smooth-MC heat scores diverge 3-5× as a result, but final structures still match. See `[[project-singleton-chr-filter-divergence]]`.

### New features (opt-in via settings, default-off)

- **Excluded volume** — `settings.use_excluded_volume = true` to enable.
  Soft harmonic repulsion preventing bead overlap. For pairs `(i, j)` with `|i - j| > exclusion_skip_neighbors` and `d_ij < exclusion_radius`, contributes `exclusion_weight * ((r0 - d)/r0)²` to the score. Implemented as an independent score component with the same `2 * (local_curr - local_prev)` incremental pattern as the heat term. Settings:
    - `use_excluded_volume` (master toggle)
    - `exclusion_radius` (r₀, position units; default 1.0)
    - `exclusion_weight` (k, comparable to spring constants; default 1.0)
    - `exclusion_apply_to_arcs` (gate for arc MC; default false)
    - `exclusion_apply_to_smooth` (gate for smooth MC; default true)
    - `exclusion_skip_neighbors` (skip pairs with `|i-j|` ≤ this; default 1, i.e. skip bonded)

  Why not in the reference: 3dnome uses an inverse-distance repulsion only on pairs where the input arc matrix is marked negative — a sparse, conditional repulsion. This adds a global polymer-physics excluded-volume term independent of input data. Touches arc-MC (`_batch_arcs_nb`) and smooth-MC (`_batch_smooth_kernel_nb`).

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

  Why not in the reference: complements confinement to prevent under-constrained small IBs from stretching out. The boost tightens chain and bond springs so the chain compresses against any repulsive/heatmap forces. Targeted: only affects small IBs, doesn't change behavior of large well-constrained IBs.

---

## Correctness Rules

These rules apply to the **parity baseline** (all new feature flags off). New feature work has its own rules in [Status: post-parity, feature-extension phase](#status-post-parity-feature-extension-phase) above.

1. When working on or near parity code, verify algorithmic choices against the reference source. Do not invent behavior on the parity path.
2. If behavior in the reference source is ambiguous or surprising, document it explicitly rather than working around it.
3. **Run `python harness/compare.py` after touching parity-baseline scoring code.** A function is not done until the harness reports PASS for its group.
4. **Run `python harness/integration.py` after touching parity-baseline MC code.** Bead-position distributions of reference and Python ensembles must remain statistically compatible.
5. Parity-baseline scoring functions must produce numerically equivalent results to the reference on the same inputs (within 1e-6 absolute tolerance).
6. New features are allowed and encouraged to diverge from the reference. They must be opt-in via `gnome3d/settings.py`, documented in the divergences section above, and must not change behavior when their flag is off.
