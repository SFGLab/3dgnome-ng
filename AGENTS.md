# 3dgnome-ng: Agent Guide

## Project Goal

Python reimplementation of the Monte Carlo core of **3dgnome**. The reference implementation in `3dnome/MC/` is a ~6,400 line simulation that predicts 3D chromosome structure from ChIA-PET and Hi-C contact frequency data. The reimplementation lives in `gnome3d/` and reproduced the reference algorithm as its starting point. MC kernels run on CPU via Numba. A jax path batches regions on the GPU for the genome scale run. Torch is used for GPU device detection.

Do **not** modify anything inside `3dnome/`. That directory is the reference implementation - read it, never change it.

### Terminology

Refer to `3dnome/MC/` as **"the reference"** (or "the reference implementation"), never as "C++". The language it happens to be written in is irrelevant to the design; the relevant fact is that it is the algorithmic source-of-truth we port from. This applies in code comments, docstrings, agent-visible documentation, commit messages, and conversations. When you cite a specific file or line, write `LooperSolver.cpp:1069-1104` — not "the C++ code at...".

### Status: post-parity, feature-extension phase

The Python port reached algorithmic parity with the reference; that work is documented and frozen. **New work no longer requires matching the reference.** Features added from here on (biophysics extensions, new energy terms, scheduling tweaks, etc.) are expected to diverge intentionally from `3dnome/`.

Rules for new feature work:

- **All new features must be opt-in via `gnome3d/settings.py`.** Default-off so existing configs continue to reproduce the parity-era behavior.
- **Document divergences** in the "Python divergences from reference" section below — what changed, why, and which setting toggles it.
- The reference and `harness/integration.py` remain authoritative for the **parity baseline** (feature flags off). They are not authoritative for new features.
- The parity gate is a byte-exactness diff of a real reconstruction, not a unit harness. See [Parity gate](#parity-gate).
- Inside a feature's own code, document any non-obvious behavior; project memory (`[[…]]` links) carries the longer-form rationale.

---

## Repository Layout

```
3dgnome-torch/
├── 3dnome/MC/                  # Reference implementation (READ ONLY)
│   ├── LooperSolver.cpp/h      # Main solver, all MC loops live here
│   ├── Chromosome.cpp/h        # 3D structure, a list of bead positions
│   ├── HierarchicalChromosome.cpp/h  # Multi-level representation
│   ├── Heatmap.cpp/h           # 2D contact/frequency matrices
│   ├── InteractionArcs.cpp/h   # Pairwise arc/interaction management
│   ├── Cluster.cpp/h           # Single bead definition
│   └── lib/                    # mtxlib (vec3/mat44), RNG, RMSD utilities
├── gnome3d/                    # Python implementation. MC kernels run on CPU via Numba
│   ├── simulate.py             # Public reconstruction entry point
│   ├── reconstruct.py          # The self-expanding task DAG
│   ├── pipeline/               # Per-stage kernels (coarse, ib)
│   ├── mc/                     # numba and jax MC kernels (numba/, jax/)
│   ├── settings.py data.py hierarchy.py io.py util.py types.py tracks.py skeleton.py
├── validation/                 # Validation package (studies, metrics, manifests)
├── harness/
│   └── integration.py          # Full-MC integration test and the reference runner
├── data/                       # Input datasets (GM12878, H1ESC, HFFC6). _hic holds 4DN mcools
├── docs/                       # Local-only notes, gitignored and not part of the repo
├── pyproject.toml              # gnome3d-torch, [validation] extra for the validation package
└── AGENTS.md                   # This file
```

Python environment: `.venv/bin/python`. Use it rather than the system python.

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

## Implementation notes

- Bead positions are `(N, 3)` float64 arrays. Energy functions score the active region only, not all N beads, matching the reference local-score pattern.
- The inner MC loop is sequential. Each step depends on the previous accept or reject, so a single chain cannot batch its proposals. Parallelism comes from running independent chains or regions at once. The numba path runs one chain per thread with a thread local RNG and stays byte exact. The jax path in `gnome3d/mc/` batches many regions on the GPU for the genome scale run.
- The MC does stochastic search, not gradient descent. There are no gradients in the loop.

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
.venv/bin/pyright gnome3d/mc/numba/terms.py  # check one file
```

Run pyright before committing changes to `gnome3d/`. The tree currently reports errors that
predate the pipeline refactor; the rule for new work is **do not increase the count**. Check the
baseline with `git stash && .venv/bin/pyright | tail -1` before and after a change.

### Conventions

- **Numpy array aliases** live in [gnome3d/types.py](gnome3d/types.py): `F32Array`, `F64Array`, `I32Array`, `I64Array`, `BoolArray`. Use these instead of bare `np.ndarray` so dtype intent is visible.
- **Semantic int aliases** also in `types.py`: `ClusterIndex`, `LocalArcIndex`, `GenomicPos`. They are `int` at runtime — they document the intent of an `int` parameter (a cluster index into `Solver.clusters` is not the same thing as a chr-relative arc index, even though both are `int`).
- **Collection aliases**: `AnchorMap`, `ArcMap`, `RawArcMap`, `BreakpointMap`, `ChrRootMap`, `ChrFirstClusterMap`, `ChrLevel`, `SingletonContact`, `BeadOut`. Prefer these over inline `dict[str, list[…]]` spellings.
- **Output tuples**: `BeadOut = tuple[int, int, float, float, float]` = `(genomic_start_bp, genomic_end_bp, x, y, z)`. First field is start so `sorted(beads, key=lambda b: b[0])` still gives left-to-right genomic order.
- **Dataclass defaults**: use a named factory function (e.g. `_empty_cluster_index_list`) instead of `field(default_factory=list)` — bare `list` leaks `list[Unknown]` through pyright in strict mode.

### Numba interop

`@njit` from numba has no type stubs, which would otherwise force every JIT-compiled function to be typed as `Any`. [gnome3d/mc/numba/terms.py](gnome3d/mc/numba/terms.py) wraps it with a typed identity decorator:

```python
def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))
    return decorator
```

Decorated functions keep their original signatures under pyright while still being JIT-compiled at runtime. **Do not use `typing.cast` inside `@njit` kernels** — numba's nopython frontend doesn't recognise it. For random-index lines like `int(np.random.randint(0, n))` where numpy stubs return `Any`, use a per-line `# pyright: ignore[reportUnknownArgumentType]` comment instead.

### Formatting and linting

The project uses **`ruff`** for both formatting (Black-compatible) and linting. Config lives in `pyproject.toml` under `[tool.ruff]`. Installed in the project venv.

```bash
.venv/bin/ruff format gnome3d/ harness/         # apply formatting
.venv/bin/ruff format --check gnome3d/ harness/  # CI-style dry-run
.venv/bin/ruff check  gnome3d/ harness/         # lint
.venv/bin/ruff check  gnome3d/ harness/ --fix   # apply safe auto-fixes
```

Current state: **0 lint issues, all files formatted**. Lint groups enabled: `E F I UP B C4` (pycodestyle, pyflakes, isort, pyupgrade, bugbear-lite, comprehensions). Run format + lint before committing changes to `gnome3d/` or `harness/`.

Per-file ignores in `pyproject.toml`:
- `gnome3d/{solver,io,data,hierarchy,mc}.py`: F403/F405 — these modules re-export `gnome3d.types` via `from .types import *`.
- `gnome3d/data.py`: also B023 — `mark_arcs` uses an intentional closure pattern over loop locals.

### Adding new code

- Annotate every function signature (params + return). Strict mode requires it.
- Avoid bare `list`, `dict`, `tuple` in annotations — always parameterise.
- For numpy arrays in non-kernel code, use the aliases (`F32Array`, etc.).
- Inside `@njit` kernels, parameter and local annotations are advisory only (numba ignores them) but still required for pyright. Keep them realistic — they document the contract.

### Documentation and comments

Write doc strings and comments as short plain sentences. Where a dash or colon would open a clause, start a new sentence instead. Do not use dashes, colons or semicolons as sentence punctuation. Do not emphasise words by writing them in capitals. Keep backtick quotes for identifiers that need them, plain names read fine on their own. Avoid parenthetical asides. Turn an aside into its own sentence or drop it. State plainly what the code does. Do not reach for metaphors.

A doc describes what the thing is now. It does not record what it used to be, what it is not, or where related code moved. Migration history and cross references to relocated code belong in the commit message or in project memory, not in the source. Memory style [[links]] to related notes are fine.

Keep docs short and non redundant. When one fact holds for every function in a file, state it once at the module or class level and let each function doc stay brief. Keep the parameter and return descriptions. When a doc is too long, tighten the sentences rather than deleting the contract.

Do not restate what the line below already shows. A comment earns its place by carrying a constraint or a reason the code cannot express. A comment that paraphrases a visible type or repeats the called function's own doc is noise and should go.

A style pass is also a correctness pass. While rewriting a doc, confirm the claim still holds and fix drift such as a description of removed behaviour or a stale module name. See [[feedback_doc_comment_style]].

---

## Parity gate

`harness/compare.py` and `harness/scorer.cpp` were **removed**.  They compiled a scorer against
the real 3dnome sources and compared it against `gnome3d/energy.py` and `gnome3d/solver.py`, both
of which the pipeline refactor deleted.  Every one of the 51 checks had been silently skipping on
`ImportError` since then, so the harness reported `0 passed, 0 failed, 51 skipped` while looking
green.  A gate that cannot fail is worse than no gate.  The reference binary itself is untouched
and `harness/integration.py` still uses it.

**The parity gate for flag-off changes is a byte-exactness diff of a real reconstruction.**
Reconstruct one region at the previous commit and again on the working tree, then compare bead
coordinates exactly:

```bash
git worktree add --detach /tmp/g3d_head HEAD
ln -s "$PWD/data" /tmp/g3d_head/data
# run the same region in both trees, then compare the coordinates
.venv/bin/python <driver> /tmp/g3d_head  head.npz
.venv/bin/python <driver> "$PWD"         wip.npz
```

where `<driver>` calls `gnome3d.simulate.run_region` and dumps `(start, end, x, y, z)` per bead.
Any new opt-in feature must leave this diff byte-identical with its flag off.  Both runs must use
the same executor settings, since batch grouping selects which chain gets which RNG stream.

For the *behaviour* of a new energy term, unit-test it directly:

```bash
python harness/test_terms.py
```

That file covers the epigenome terms and is the template for the next one. Three properties per
term. A hand-built configuration whose energy is computable in closed form. A check that the
per-bead local scores sum to the full score, which is the contract the incremental MC update
depends on. And a check that the term stays non-negative over random configurations, which the
Metropolis rule requires because it divides by the running score. Plus whatever behavioural claim
the term makes that a closed form cannot express.

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
# gnome3d/simulate.py
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

The test auto-skips the Python comparison if `gnome3d/simulate.py` is missing or raises `NotImplementedError`. It never fails just because the Python side is absent.

---

## Epigenomic tracks

The compartment and accessibility energy terms read plain-text tracks derived from data the
repo already fetches.  Build them once per cell line:

```bash
python -m validation fetch  --manifest validation/manifests/<CELL>_hic.json --out data/_hic
python -m validation fetch  --manifest validation/manifests/<CELL>_accessibility.json \
                            --out data/_epigenome
python -m validation tracks --cell <CELL>
```

That writes `data/<CELL>/<CELL>_compartments.bedGraph` and `<CELL>_atac.bedGraph` plus a
lockfile recording resolution, source file and a per-chromosome quality number.  It is
idempotent, so re-running is free.

Then ablate the terms against that cell line's own Hi-C:

```bash
python -m validation epigenome --cell <CELL> --hic data/_hic/<CELL>/<file>.mcool \
                               --region chr1:20000000-40000000
```

Accessibility assay differs by cell line.  ENCODE has ATAC-seq for GM12878 only; H1 and HFFc6
use DNase-seq, which is what HiP-HoP itself used, so it is the intended input rather than a
substitute.  The manifests record which is which.

Three traps, each of which produces a silently wrong answer.  Pick the deepest contact file when
a cell line has several, since a shallow one yields a compartment eigenvector that is pure noise;
`validation/studies/tracks.py::_find_one` takes the largest file for this reason, and a lag-1
autocorrelation below 0.3 warns that the result is noise.  Pass `clip_percentile=99.9` when
calling `cooltools`' `cis_eig` directly, because it is not that function's default and without it
outlier pixels dominate the decomposition.  And the eigenvector sign is arbitrary and differs
between machines for the same input, so a derived track is not reproducible up to sign; anything
consuming it must either phase it or be invariant to a global flip.

---

## Python divergences from reference

Tracked list of intentional deviations from `3dnome/MC/`. Each entry: what diverges, why, and how to toggle/restore parity. Keep this list current — new entries must be added when introducing diverging behavior, and removed when behavior is brought back into parity.

### Settings hygiene

- **`noise_arcs` removed.** The reference declares `noiseCoefficientLevelAnchor` (read as `noise_arcs` in `[main]`) and multiplies it into a local `noise_size` variable in `LooperSolver.cpp:2085`, but the arc-MC call site on line 2136 passes a hardcoded `noise_size_small=0.005` instead — making the setting effectively dead in the reference. Python uses the same 0.005 hardcoded constant in [pipeline/ib/arcs.py](gnome3d/pipeline/ib/arcs.py); the setting was dropped to avoid implying configurability.
- **`random_walk` ported.** Previously loaded-but-unused; now drives [pipeline/coarse/build.py](gnome3d/pipeline/coarse/build.py), mirroring `LooperSolver.cpp:80-98` (chained 50.0-step walk per chromosome instead of segment-level heatmap MC). Honors `use_2d`.
- **`long_pet_*` ported.** Long-range arcs (gap > `max_pet_length`) are no longer discarded by [io.load_arcs](gnome3d/io.py) — they are carried on `ContactData.long_arcs` and folded into the segment heatmap by [pipeline/coarse/build.py](gnome3d/pipeline/coarse/build.py) as `long_pet_scale * arc.score ** long_pet_power`. Mirrors `LooperSolver.cpp:1069-1104`, including the asymmetric `h[st][end] += val` pattern (the downstream symmetrize step in `_normalize_heatmap` averages it to `val` on each side).
- **Chromosome-level MC ported.** Previously `steps_lvl1` / `noise_lvl1` were inert because Python had no chr-level reconstruction. Now [pipeline/coarse/build.py](gnome3d/pipeline/coarse/build.py) builds an n_chr × n_chr inter-chromosomal singleton heatmap, normalizes its first non-zero diagonal to 1.0, converts to expected distances, and runs `mc_heatmap` with `steps_lvl1` runs at `noise_lvl1 × avg_dist` step size. Mirrors `LooperSolver.cpp` lines 119-160 and 265-322. Triggered only when `len(chrs) > 1`; single-chr runs are unaffected. If the singleton input has no inter-chr contacts the chr roots are scattered randomly instead.
- **`normalizeHeatmapInter` ported.** [pipeline/coarse/heatmap.py](gnome3d/pipeline/coarse/heatmap.py) was previously a no-op stub; now it multiplies the segment heatmap by `heatmap_inter_scaling` and divides intra-chromosome blocks back, matching `LooperSolver.cpp:1422-1459`. Net effect: intra-chr unchanged, inter-chr × scale. Active only on multi-chr runs (length-1 `current_level` short-circuits).
- **Multi-chromosome CLI / API surface.** [cli.py](gnome3d/cli.py) and [simulate.py::run_genome](gnome3d/simulate.py) accept the same syntax as the reference's `-c` flag: single chromosome, comma-separated list, `chrN-chrM` numeric range, or single sub-chromosomal region. Default (empty string) matches the reference: `chr1-chr22,chrX`. Parsing is centralized in [io.py::parse_chrs_arg](gnome3d/io.py). The CLI now writes one CIF per chromosome when more than one chr was requested (using a `_<chr>_` suffix in the filename); single-chr/region runs keep the old single-CIF behavior. `run_region` keeps its original single-region semantics for backward compatibility; `run_genome` returns `list[dict[str, list[BeadOut]]]` so per-chromosome bead lists are preserved.
- **`data_singletons_inter` loaded.** [data.py::ContactData.from_files](gnome3d/data.py) reads the optional second singletons file (config key `[data] singletons_inter`) and appends to `singletons` whenever `len(chrs) > 1`. Mirrors `LooperSolver.cpp:970` which adds inter-chromosomal singleton files for multi-chr runs. Single-chr runs skip this file (per-chr contacts in the main file are sufficient and the inter file is often a sparse stub).

### Refactors (no behavior change at parity settings)

- **Unified smooth-MC kernel** ([gnome3d/mc/numba/terms.py](gnome3d/mc/numba/terms.py))
  The reference has separate `MonteCarloArcsSmooth` branches per feature combo. Python collapses the four prior specialized kernels (`_batch_smooth_nb`, `_batch_smooth_heat_nb`, `_batch_smooth_orientation_nb`, `_batch_smooth_orientation_heat_nb`) into one `batch_mc_nb` driven by `use_heat`/`use_orn` flags. Energy terms (struct, heat, orn) are tracked as independent components (`score_struct`, `score_orn`, `score_heat`) and combined per step.
  Side benefit: heat/orn paths previously did a full O(N) structure recompute every MC step — now incremental like the pure-smooth path. Verified drift stays at float-noise (~1e-9) under the codebase's reciprocal-neighbor invariant.

### Algorithm divergences

- **Interaction block boundaries: `[data] ib_split_source = arcs | tads`, default `arcs`.**
  The reference splits blocks at `hierarchy.py::find_gaps`, where ChIA-PET arc coverage falls to
  zero. That is partly a property of the library's depth rather than of the chromatin: where the
  assay is shallow no arc spans, a boundary appears, and the pipeline folds and places the two
  sides as independent objects. `tads` takes boundaries from a contact-map call supplied by
  `[data] ib_split` instead. A missing or empty boundary file raises rather
  than falling back to arc gaps, since a silent fallback would look like a `tads` run and behave
  like the baseline. `python -m validation tracks` writes the boundary file via
  `cooltools.insulation` at 10 kb with a 200 kb window.

  **`tads` is not in use. Prefer `arcs`, the default.** A TAD boundary comes from an insulation
  call that knows nothing about the arcs, so it cuts through them: measured on GM12878 chr1,
  5437 of 12474 arcs, 43.6 percent, have their two anchors in different blocks under `tads`
  against exactly 0 under `arcs`. An arc whose anchors sit in different blocks constrains
  nothing at anchor resolution, because the per-block arc MC only sees arcs internal to its own
  block and the blocks are held together by the chain bond between their centroids alone. So a
  `tads` run discards nearly half its arc data. Arc-gap boundaries cannot do this by
  construction, since they are placed where arc coverage is already zero. Reproduce with
  `playground/blockscope_cost.py`.

  The earlier case for `tads` was within-block over between-block contact enrichment on four
  GM12878 regions, 2.60, 11.16, 28160 and unbounded falling to 1.86, 0.68, 4.26 and 1.90, with
  the mean absolute compartment-saddle gap falling 1.664 to 0.373. Those numbers are real but
  do not carry: enrichment is strongly span and bin-size dependent, reading 44.2 on a 20 Mb
  region against 5.75 for the same pipeline on full chr1, so it ranks configurations only
  within one fixed span. Bead count also falls about 27 percent under `tads`, because more
  boundaries mean fewer inter-anchor gaps receive subanchors, so model resolution moves
  alongside block definition. See [[project_trio_expansion_artifact]].

- **IB placement scope: `[simulation_ib] refine_scope = segment | chromosome`, default `segment`.**
  `segment` is the prior behaviour: each segment's blocks are refined as a separate chain and any
  segment holding one block or fewer is skipped, so segment grouping decides which blocks get
  placed at all. `chromosome` refines every block on the chromosome as one chain and removes that
  dependency.

  `chromosome` is not the default despite being the more principled scoping. It also puts every
  block pair inside the excluded-volume term and the structures inflate: over four GM12878 regions
  mean simulated contact density fell 0.087 to 0.035 and within-block over between-block
  enrichment worsened about threefold. The compartment saddle improves, but sparsity alone drags
  that statistic toward 1.0, so the gain is not separable from the inflation. Adopting this scope
  means re-tuning `exclusion_*_ib` and `confinement_*_ib` first.

  This also decides whether `[data] ib_split` can share a file with `[data] segment_split`. Under
  `chromosome` they are interchangeable, measured as a wash. Under the default `segment` they are
  not: one shared file gives 16 segments holding 17 blocks, so nearly every segment holds a single
  block and placement skips it.

- **Orientation MC: weighted local scorer**
  Python uses a weighted local delta in `_local_score_orientation_nb` ([gnome3d/mc/numba/terms.py](gnome3d/mc/numba/terms.py)) so the incremental update is exact w.r.t. `_score_orientation_full_nb`. The reference uses an unweighted local scorer that drifts over many steps. See `[[project-orientation-mc-fix]]`. The reference scorer stayed unweighted for the old unit harness, which has since been removed.

- **Singleton chr filter** (data loading)
  The reference bins inter-chromosomal singletons by position; Python correctly filters by chromosome. Smooth-MC heat scores diverge 3-5× as a result, but final structures still match. See `[[project-singleton-chr-filter-divergence]]`.

- **`BeadOut` is a NamedTuple with a `kind` field** ([gnome3d/types.py](gnome3d/types.py))
  Output beads were widened from a 5-tuple `(start, end, x, y, z)` to a 6-field NamedTuple `(start, end, x, y, z, kind)` where `kind: Literal["anchor", "subanchor"]`. Iteration / positional unpacking still works (NamedTuple subclasses tuple) but consumers must expect length 6, not 5. Named access also available: `b.start`, `b.kind`, plus convenience properties `b.midpoint` and `b.span`. The CIF writer emits `kind` via a non-standard `_atom_site.gnome_bead_kind` column and additionally distinguishes anchors (`label_comp_id = ALA`) from subanchors (`label_comp_id = GLY`) so default mmCIF viewers color-code them automatically.

- **Subanchor densification: centered slots instead of single points** ([pipeline/ib/densify.py::densify](gnome3d/pipeline/ib/densify.py))
  The reference places each subanchor at a single genomic point `ca.end + (j+1) * gap_bp / (ld+1)`, so each subanchor bead has zero genomic width and adjacent anchors collapse all ld subanchors into one point when overlapping. Python keeps the same midpoint positions but gives each subanchor j a slot of width `d_bp = span // (ld+1)` centered on its midpoint. Properties:
    - Subanchor midpoints, 3D position interpolation (`t = (j+1)/(ld+1)`), and `dtn` are unchanged versus the reference's point scheme.
    - Each subanchor has a non-degenerate genomic range whenever the in-between region is at least `ld+1` bp wide.
    - Subanchor starts/ends never coincide with adjacent anchor boundaries (`ca.end` or `cb.start`), so sorting beads by start gives a unique total order — this fixes the bead-collision issue where overlapping anchors produced subanchors sharing a `start` value with the next anchor.

  Behavior on overlapping anchors is controlled by two independent settings:

    - **`overlap_anchor_strict`** (default `False`) — span computation.
      - `False`: `span = abs(cb.start - ca.end)`; non-overlap tiles the gap `[ca.end, cb.start]`, overlap tiles the overlap `[cb.start, ca.end]` with non-degenerate ranges. Python divergence.
      - `True`: `span = max(cb.start - ca.end, 0)` matches `LooperSolver.cpp:1829-1831` — overlap clamps to 0 so MC-chain subanchors between overlapping anchors collapse to a single boundary point.

    - **`drop_zero_length_subanchors`** (default `False`) — output filter, independent of the span setting.
      - `False`: every densified subanchor appears in the `BeadOut` output, even when `start == end`.
      - `True`: subanchor entries with `start == end` are filtered out of the externally visible bead list. The MC chain still contains them (needed for chain smoothness); only the output is cleaned. Useful in combination with `overlap_anchor_strict = True` to suppress the collapsed-overlap zero-length entries the reference would otherwise emit.

  The invariants are that every subanchor falls inside its gap and that bead starts are unique.

- **Dynamic loop density** — `settings.use_dynamic_loop_density = True` to enable. ([pipeline/ib/densify.py::_subanchor_counts](gnome3d/pipeline/ib/densify.py))
  The reference (and Python's default) inserts a fixed `loop_density` subanchor count between every adjacent anchor pair, regardless of arc span. With arcs spanning anything from a few hundred bp to hundreds of kb in real ChIA-PET data, this gives **>100× imbalance** in genomic-bp-per-bead between short and long arcs. Dynamic mode picks the count per arc so the *chain segments connecting beads* are roughly `target_bp_per_subanchor` long:

    `n_subanchors = round(span / target) − 1`  (clamped to `[min, max]`)

  An arc with `n` subanchors has `n + 1` consecutive chain segments. Short arcs (`span < 1.5 × target`) round to 1 segment → 0 subanchors → the two anchors are linked directly.

    - `target_bp_per_subanchor` (int, default `5000`) — target genomic distance per chain segment.
    - `min_subanchors_per_arc` (int, default `0`) — set to `1` to force at least one subanchor between every pair regardless of span.
    - `max_subanchors_per_arc` (int, default `50`) — cap to avoid runaway counts on huge gaps (a 444 kb arc at 5 kb target would otherwise insert 88 subanchors).

  Both [densify](gnome3d/pipeline/ib/densify.py) and [build_contact_heatmaps](gnome3d/pipeline/coarse/build.py) consume the same subanchor-count output, so the densified bead chain and the subanchor contact heatmap stay in sync — `use_subanchor_heatmap` remains compatible with dynamic mode. For arcs with `count == 0` the contact heatmap merges that gap's half into each flanking anchor's bin (midpoint break); the smooth-MC chain just links the two anchors directly.

  Default behavior (`use_dynamic_loop_density = False`) is unchanged.

### New features (opt-in via settings, default-off)

- **Epigenome energy terms** — A/B compartments and chromatin accessibility.
  The compartment family is ported from MultiMM (`add_compartment_blocks`,
  `add_Blamina_interaction`, `add_central_force`, `add_chromosomal_blocks`); accessibility from
  HiP-HoP (Buckle et al., Mol Cell 72(4):786-797, 2018, doi:10.1016/j.molcel.2018.09.016), which
  embeds ATAC-seq at 1 kbp beads, the same scale as our subanchors.  MultiMM has no ATAC energy
  term at all, so the two families come from different papers.  Six flags, all default off:
  `use_compartments`, `use_bridging`, `use_fibre_compaction`, `use_lamina`,
  `use_central_force`, `use_chromosomal_blocks`, under `[compartments]`, `[accessibility]` and
  `[nucleus]`. New `[data]` keys `compartments`, `accessibility`, `phasing_track`;
  `ContactData.from_dataframes` gains the matching frames.

  Divergences worth knowing:
  - **Attractive terms are written shifted and non-negative.** MultiMM and HiP-HoP write them
    as negative energies, which 3dgnome cannot use: the Metropolis rule divides by the running
    score and is guarded on `score > 0`, so a negative-definite term would silently disable the
    temperature branch. The shift changes an additive constant, not the minimum or the gradient.
  - **The pairwise affinity is divided by `N - 1`.** Not in MultiMM. Without it the term's
    strength grows with region size, because it sums over all partners while springs act per
    bond, and a weight tuned on a small region collapses a large one.
  - **ATAC drives both HiP-HoP mechanisms.** HiP-HoP uses H3K27ac for fibre compaction and ATAC
    for bridging; we drive both from accessibility because the pipeline loads one track.
    Compaction scales the existing `dtn` instead of adding i,i+2 springs.
  - **Accessibility normalisation is selectable: `[accessibility] mode = log | binary`,
    default `log`.** `log` is log-then-minmax. `binary` is HiP-HoP's own open/closed state,
    open at or above `[accessibility] percentile` (default 80) of the loaded values.
    The default is kept at `log` so existing configs are unchanged, but `binary` is the
    faithful one and `log` is close to inert on a track binned to several kb. Binning a
    narrow ATAC peak into a 5 kb bin already removes the upper tail the log exists to
    compress, so the log stretches the remaining background over most of `[0, 1]`: measured
    on GM12878 the median bead reads 0.85 open, leaving 0.224 of `1 - a` for fibre
    compaction, which applies 3.0% mean compaction and correlates with accessibility at
    r = -0.011. Under `binary` at the 80th percentile the same region gets 22.9% mean
    compaction at r = +0.668, and realised bond lengths fall 22.8%. `ContactData.from_files`
    reads both keys from settings; `from_dataframes` takes them as parameters.
  - **Bridging is an effective pairwise attraction.** HiP-HoP's explicit diffusing bridge
    particles are integrated out rather than simulated.
  - **Lamina, central and chromosomal blocks run at segment-level heatmap MC only.** They need a
    nuclear frame shared across the whole active region, and that is the one MC call spanning it.
  - **Compartment input is source-agnostic.** CALDER2 reads Juicer `.hic` only, and this repo's
    Hi-C is 4DN mcool, so a CALDER BED is one supported format rather than the required source.
    A signed eigenvector track is the recommended input. Eigenvector sign is arbitrary, so
    phasing is mandatory and explicit; an unphaseable chromosome is left unassigned rather than
    segregated backwards.
  - **The JAX smooth kernel carries the affinity terms; the checkerboard kernel does not.**
    `mc_smooth_jax` and `mc_smooth_jax_batch` implement compartment and bridging, agreeing with
    numba on initial energy to 1.2e-07 relative. `use_aff` is a static cache-key entry rather
    than a weight gate, because the term costs an extra O(N) pass per step and making it
    structural keeps that off runs that do not use it. The scores ride the excluded-volume
    accumulator: all three terms are pairwise, double counted and share the factor-2 delta, so
    the sum is exact for both the Metropolis ratio and the final score, and only the per-term
    breakdown is lost. The opt-in checkerboard kernel
    ([smooth_checker.py](gnome3d/mc/jax/smooth_checker.py)) does **not** implement them and
    raises rather than dropping them; unlike orientation they are not constant during smooth,
    so omitting them would change the structures.
  - **The estimate-dist dry pass excludes them.** They are attractive, so including them would
    shrink the estimated distances that become the heat target and the real smooth pass would
    then compact against an already-compacted target.

  These terms are purely attractive and need excluded volume or confinement enabled alongside
  them, both of which also default off. See the doc.


- **Excluded volume** — `settings.use_excluded_volume = true` to enable.
  Soft harmonic repulsion preventing bead overlap. For pairs `(i, j)` with `|i - j| > exclusion_skip_neighbors` and `d_ij < r₀`, contributes `exclusion_weight * ((r0 - d)/r0)²` to the score. Implemented as an independent score component with the same `2 * (local_curr - local_prev)` incremental pattern as the heat term.

  **Per-level radius + auto-factor.** EV runs at four different MC levels (arcs / smooth / heatmap / IB) whose natural bead-bead distance scales differ by orders of magnitude — anchor beads sit at unit scale, IB centroids at hundreds of units. Each level has its own radius knob and its own auto-factor knob:

    | Level | Radius field | Factor field | Auto formula (when radius = 0) |
    |---|---|---|---|
    | arcs   | `exclusion_radius_arcs`    | `exclusion_auto_factor_arcs`    | `factor × mean(positive arc expected distance)` |
    | smooth | `exclusion_radius_smooth`  | `exclusion_auto_factor_smooth`  | `factor × mean(dtn)` (chain bond distances) |
    | heatmap| `exclusion_radius_heatmap` | `exclusion_auto_factor_heatmap` | `factor × mean(active heatmap target distance)` |
    | ib     | `exclusion_radius_ib`      | `exclusion_auto_factor_ib`      | `factor × mean(IB chain-bond dtn)` |

    All radii default to `0` (auto). All auto factors default to `0.5` ("EV kicks in once beads get closer than half the typical bond distance at that level"). Setting a radius to a positive value disables auto-derive for that level.

  **Gates** (which MC passes consult EV at all):
    - `use_excluded_volume` (master)
    - `exclusion_apply_to_arcs` (default false)
    - `exclusion_apply_to_smooth` (default true)
    - `exclusion_apply_to_heatmap` (default false)
    - `exclusion_apply_to_ib` (default true, only kicks in when `use_ib_mc` is on)

  **Shared knobs:**
    - `exclusion_weight` (k, comparable to spring constants; `Settings` default 0.5, and the
      tuned config in `validation/core/config.py` sets 0.1)
    - `exclusion_skip_neighbors` (skip pairs with `|i-j|` ≤ this; default 1, i.e. skip bonded)

  Ini key names under `[excluded_volume]`: `use_excluded_volume`, `weight`, `apply_to_arcs`, `apply_to_smooth`, `apply_to_heatmap`, `apply_to_ib`, `skip_neighbors`, `radius_arcs`, `radius_smooth`, `radius_heatmap`, `radius_ib`, `auto_factor_arcs`, `auto_factor_smooth`, `auto_factor_heatmap`, `auto_factor_ib`.

  Why not in the reference: 3dnome uses an inverse-distance repulsion only on pairs where the input arc matrix is marked negative — a sparse, conditional repulsion. This adds a global polymer-physics excluded-volume term independent of input data. Touches arc-MC, smooth-MC and IB-MC through the shared `batch_mc_nb`, and heatmap-MC through its own `_batch_heatmap_nb`.

- **IB-level MC pass** — `settings.use_ib_mc = true` to enable. ([pipeline/coarse/build.py::ib_mc_refine](gnome3d/pipeline/coarse/build.py))
  Address the "central blob" pathology in full-chromosome runs (very visible with `use_dynamic_loop_density = true` + small `target_bp_per_subanchor`): the reference / Python default places IB centroids by random walk or linear interpolation with no MC, so when each IB's smooth-MC fills a fat sphere of beads the spheres overlap into a tangle around the origin.

  This pass runs `mc_ib` over each segment's child IB centroids with chain-bond targets from `genomic_length_to_distance` between consecutive IB midpoints, plus optional excluded volume so IBs push each other apart and optional confinement so the chain doesn't stretch out. `mc_ib` is a **peer stage** to `mc_smooth` — *not* a sub-mode of it. It owns its own MC schedule, chain spring constants, step noise, and EV/confinement knobs. Both stages share only the unified `_batch_mc_nb` kernel; neither extends the other.

  Ini layout (key names follow the same conventions as `[simulation_arcs_smooth]`, `[springs]`, and `[main]`; defaults mirror the smooth stage so existing configs keep working):
  ```ini
  [main]
  noise_ib = 0.5                                    # step_size = noise_ib * mean(dtn)

  [springs]
  stretch_constant_ib = 0.1
  squeeze_constant_ib = 0.1

  [simulation_ib]
  use_ib_mc                              = yes
  max_temp                               = 20.0
  delta_temp                             = 0.99995
  jump_temp_scale                        = 50.0
  jump_temp_coef                         = 20.0
  stop_condition_steps                   = 10000
  stop_condition_improvement_threshold   = 0.995
  stop_condition_successes_threshold     = 5
  dist_weight                            = 1.0

  [excluded_volume]
  apply_to_ib    = yes
  radius_ib      = 0                                # 0 = auto from IB chain-bond scale
  auto_factor_ib = 0.5                              # used when radius_ib = 0

  [confinement]
  use_confinement   = yes
  apply_to_ib       = yes
  radius_ib         = 0                             # 0 = auto = packing_factor * mean(dtn) * N^(1/3)
  packing_factor_ib = 0.75
  ```

  Why a peer (not a child of smooth): smooth-MC operates on per-IB sub-anchor chains with CTCF orientations, heat maps, and angle springs; IB-MC operates on inter-IB centroids — no orientations, no heat target, no angle term, different spatial scale and tolerance. Conflating their settings makes both stages harder to tune independently.

- **Spherical confinement** — `settings.use_confinement = true` to enable.
  Soft envelope mimicking a nuclear boundary. For each bead at distance `r` from the per-MC-call centroid, contributes `confinement_weight * ((r - R)/R)²` if `r > R`, else 0. Per-bead (single-counted) — delta is `(local_curr - local_prev)` with no factor of 2. Each MC level has its own radius and packing factor (the spatial scale differs across levels):

  | Level | Radius | Packing factor | Auto formula |
  |---|---|---|---|
  | arcs   | `confinement_radius_arcs`   | `confinement_packing_factor_arcs`   (default 1.5)  | `pf × mean(positive arc expected) × N^(1/3)` |
  | smooth | `confinement_radius_smooth` | `confinement_packing_factor_smooth` (default 1.5)  | `pf × mean(dtn) × N^(1/3)` |
  | ib     | `confinement_radius_ib`     | `confinement_packing_factor_ib`     (default 0.75) | `pf × mean(IB chain dtn) × N^(1/3)` |

  Shared knobs: `use_confinement` (master toggle), `confinement_weight` (k; comparable to spring constants; default 0.5). Apply flags per level: `confinement_apply_to_arcs`, `confinement_apply_to_smooth`, `confinement_apply_to_ib` (all default true). Center is always the centroid of starting positions at that MC call.

  Motivation: at the IB level, EV pushes IBs apart but only nearest-neighbor chain bonds pull them back, so the segment stretches out into a long sausage. The IB tether (small packing factor → tight sphere around the segment centroid) softly holds the chain together while EV still keeps IB spheres from overlapping. At arc / smooth levels, confinement instead acts as a nuclear-like envelope for under-constrained small IBs.

- **Small-IB spring boost** — described below but **not currently implemented**: `use_small_ib_boost`, `small_ib_threshold` and `small_ib_spring_multiplier` are not fields of `Settings`, and `solver.py` no longer exists. Kept as a design note.
  When an IB has fewer anchors than `small_ib_threshold`, multiplies `spring_stretch_arcs`, `spring_squeeze_arcs`, `spring_stretch`, `spring_squeeze`, `spring_angular` by `small_ib_spring_multiplier` for that IB only. No kernel changes — implemented in `solver.py::_settings_for_ib()` by passing a `copy.copy(self.s)` clone with boosted values to `_reconstruct_cluster_arcs` / `_reconstruct_cluster_smooth` via an `s_override` parameter. Thread-safe (never mutates `self.s`). Settings:
    - `use_small_ib_boost`
    - `small_ib_threshold` (anchor count below which an IB is "small"; default 10)
    - `small_ib_spring_multiplier` (default 5.0)

  Why not in the reference: complements confinement to prevent under-constrained small IBs from stretching out. The boost tightens chain and bond springs so the chain compresses against any repulsive/heatmap forces. Targeted: only affects small IBs, doesn't change behavior of large well-constrained IBs.

- **JAX/CUDA backend** — selected per stage via `mc_executor_<stage> = batch`. ([gnome3d/mc/jax/](gnome3d/mc/jax/))

  Routes selected MC levels to a JAX/CUDA kernel instead of the default numba CPU implementation. Measured ~2× total speedup on chr22 dryrun (21 min → 10:26), peak ~6× per-kernel on the largest smooth-MC call (N=10116: 570s numba → 100s JAX). The win compounds with chromosome size since smooth-MC is ~90% of total wall time.

  Architecture: `gnome3d/mc/numba/` holds the production numba implementations (`mc_heatmap_numba`, `mc_arcs_numba`, `mc_smooth_numba`, `mc_ib_numba`) and `gnome3d/mc/jax/` the JAX peers (`mc_smooth_jax`, `mc_smooth_jax_batch`, `mc_arcs_jax`, `mc_arcs_jax_batch`). There is no dispatcher module: the backend follows from the executor strategy resolved in [reconstruct.py::pick_executor](gnome3d/reconstruct.py), so `batch` means JAX and `serial`/`threaded` mean numba.

  Each MC level has its own JAX-routing flag, defaulting to the regime where JAX is measured to win:

  | level   | JAX kernel terms                                     | default flag       | rationale                                                    |
  |---------|------------------------------------------------------|--------------------|--------------------------------------------------------------|
  | smooth  | chain + EV + heat + orient + confinement (full set)  | **on**             | 5–6× win at N≥1024 with full energy terms                    |
  | arcs    | arc springs + EV + confinement                        | off                | arc energy is sparse; numba's per-pair early-continue wins at N<5000 |
  | heatmap | heatmap distance + EV                                 | off                | chr-level fires once at N=3-23, below JAX overhead floor; enable for multi-chr |
  | ib      | not ported                                            | n/a                | <1% of typical wall                                          |

  Settings (under `[simulation_backend]`): the per-stage `mc_executor_<stage>` values
  (`serial` | `threaded` | `batch` | `auto`). `batch` selects JAX. The old
  `mc_backend_apply_to_*` flags no longer exist.

  Install: `pip install gnome3d-ng[jax] "jax[cuda12]"` (NVIDIA) or `"jax[rocm6]"` (AMD). Without JAX installed, the dispatcher raises a clear error if `mc_backend="jax"` is set; otherwise it never imports JAX. The `[jax]` extras dep in `pyproject.toml` is **optional** — base install is numba-only.

  Implementation notes:
  - **Per-shape XLA compile cost** (~1–60s per (N, K, n_anchors) combo) is paid once per machine via the persistent compile cache at `~/.cache/gnome3d/jax` (override with `GNOME3D_JAX_CACHE`).
  - **Convergence loop runs on-device** via `lax.while_loop` — one JAX call drives the full annealing, no Python sync between batches.
  - **Float32 throughout the JAX path** — bench showed f64 is 2× slower on consumer GPUs (1/32 throughput) with no quality benefit at production run lengths.
  - **`cli.py` auto-forces `ib_workers=1` when `mc_backend=jax`** — multiple Python threads contending for a single GPU is net-negative; restarts go inside JAX via `mc_smooth_chains` (vmap), not via thread pools.
  - **Lazy import + thread-safe init** — `mc_jax` module loads without importing JAX; the first call to a JAX-backed entry triggers a one-time banner on stderr (`[mc_jax] JAX backend ready: backend=gpu devices=[...]`).

  **Kernel seeding is process-stable.** Every JAX kernel takes its PRNG offset from
  `util.stable_seed_offset(log.current(), problems[0]["seed"])`, a blake2b digest of the
  active scope path mixed with the seed the DAG node carries. The scope path separates
  concurrent kernels and the node seed ties the draw to `Seeded.seed`, matching what the
  numba path does. `PYTHONHASHSEED` no longer affects results, so comparing two runs needs
  no environment override.

  Chains inside one batch are separated by the kernel's own per-index fold rather than by
  their own seeds, so changing how IBs group into batches still shifts which chain gets
  which stream. Grouping is deterministic for a given config, so a run reproduces; runs
  compared across different executor settings do not.

  Why not in the reference: 3dgnome is CPU-only. The JAX port is a Python-side acceleration of the same algorithm; numerical results agree with numba within float32 RNG-trajectory noise. Harness/parity tests run with default settings (`mc_backend=numba`), so the new backend doesn't affect the parity baseline.

- **Multi-GPU sharding unit: `[simulation_backend] multigpu_mode = groups | within | off`,
  default `groups`.**
  The batch strategy splits a dispatch into groups by batch key, then runs the groups. `groups`
  makes the group the unit of parallelism, keeping each one whole on one device and running
  different groups side by side. `within` is the older behaviour, splitting one group's IBs
  across devices. `off` pins to a single device.

  `within` scales only as far as groups are wide, and they are not. A chr1 smooth dispatch is
  252 groups, 58% of its time in groups holding a single IB, and the estimate-dist dispatch is
  252 groups with 69% of its time single-IB, because the shape ladder puts differently sized
  IBs in different buckets. Measured on the chr1 profile, `within` tops out at 1.26x however
  many GPUs it is given, while whole-group sharding projects 7.96x on 8 and 15.84x on 16.

  `groups` is also the byte-exact one. The kernels key each IB's per-step RNG on its position
  within the launch, so an unsplit group draws what it would have drawn on one device and the
  output is identical whatever the device count. `within` changes the draw. Verified two ways
  on a chr1:1-8000000 TAD-block run: 4 devices against 1 agree on all 3390 beads, and a
  1-device run on this code is byte-identical to the same run at the previous commit.

  Results are applied in group order on the calling thread however the groups interleaved, so
  DAG growth stays deterministic.

  Why not in the reference: 3dgnome is CPU-only and single-process.

---

## Correctness Rules

These rules apply to the **parity baseline** (all new feature flags off). New feature work has its own rules in [Status: post-parity, feature-extension phase](#status-post-parity-feature-extension-phase) above.

1. When working on or near parity code, verify algorithmic choices against the reference source. Do not invent behavior on the parity path.
2. If behavior in the reference source is ambiguous or surprising, document it explicitly rather than working around it.
3. **Run the byte-exactness parity gate after touching parity-baseline scoring code.** See [Parity gate](#parity-gate). A flag-off change is not done until the diff is byte-identical.
4. **Run `python harness/integration.py` after touching parity-baseline MC code.** Bead-position distributions of reference and Python ensembles must remain statistically compatible.
5. Parity-baseline scoring functions must produce numerically equivalent results to the reference on the same inputs (within 1e-6 absolute tolerance).
6. New features are allowed and encouraged to diverge from the reference. They must be opt-in via `gnome3d/settings.py`, documented in the divergences section above, and must not change behavior when their flag is off.
