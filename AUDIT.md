# cudaMMC parity audit — Phase 1 (heatmap scoring + settings)

This document enumerates every place where `src/` diverges from the cudaMMC
reference for **heatmap scoring (`scores.py`, `heatmap.py`) and settings
(`settings.py`, `mc.py:monte_carlo_heatmap`, `solver.py:reconstruct_clusters_heatmap`)**.
Each deficiency is marked inline in the Python source with a
`# BUG (cudaMMC-mismatch): …` comment that cites the exact upstream lines.

A second audit pass for arc/smooth phases will follow.

---

## A. Architectural / pipeline-level (highest-impact)

### A1. Heatmap MC runs on the wrong tree level
- **cudaMMC**: `reconstructClustersHeatmap` (`LooperSolver.cpp:85-294`) runs
  `MonteCarloHeatmap` only on **LVL_CHROMOSOME (whole-chrom beads)** and
  **LVL_SEGMENT (~2 Mb beads)** via `reconstructClustersHeatmapSingleLevel(0)`
  / `(1)`. A heatmap with ~hundreds of beads.
- **Python**: `LooperSolver.reconstruct_clusters_heatmap` runs MC on the
  **anchor level** directly (one bead per CTCF anchor, N≈26 k). Wrong scale,
  wrong number of beads, wrong heatmap binning. This alone makes the structure
  meaningless.

### A2. The CPU `MonteCarloHeatmap` is the wrong entry point
- **cudaMMC**: `reconstructClustersHeatmapSingleLevel` calls
  `ParallelMonteCarloHeatmap(avg_dist)` (`LooperSolver.cpp:390`,
  `ParallelMonteCarloHeatmap.cu:241-244`). The CPU sequential
  `MonteCarloHeatmap` (`LooperSolver.cpp:421-518`) is **dead code** in the
  modern build (kept only as a fallback).
- **Python**: mirrors the dead-code CPU loop and ignores the actual GPU
  algorithm used in production.

### A3. Initial placement of beads is invented
- **cudaMMC**: each level starts from `initial_structure[i] +
  random_vector(avg_dist, use2D)` where `avg_dist = heatmap_dist.getAvg() *
  noiseCoefficientLevelChr/Segment` (`LooperSolver.cpp:301-312, 357-363`).
- **Python**: `solver.py:226-231` places beads uniformly in a sphere whose
  radius is the *median expected distance*. Pure invention; not in cudaMMC.

### A4. Multi-restart loop is missing for heatmap phase
- **cudaMMC** `LooperSolver.cpp:352-357`: runs MC `steps =
  simulationStepsLevelChr` (default 2) or `Segment` (default 2) times and
  keeps the best-scored structure.
- **Python**: heatmap MC runs **once** per chromosome.

### A5. `step_size_heatmap` is not derived from the data
- **cudaMMC**: `avg_dist = heatmap_dist.getAvg() * noiseCoefficient...`
  (`LooperSolver.cpp:307,312`), passed as `step_size` to
  `MonteCarloHeatmap`/`ParallelMonteCarloHeatmap`.
- **Python**: uses hard-coded `Settings.step_size_heatmap = 1.5` (× 0.5 in
  `mc.py:95`). Will not match any real heatmap’s scale.

---

## B. Heatmap normalisation (`heatmap.py`)

cudaMMC normalisation pipeline (`LooperSolver.cpp:148-149, 256-258`):

1. `normalizeHeatmap`  (`LooperSolver.cpp:1709-1751`):
   * compute `expected_sum = mean(row_sum_i)`;
   * for each row, multiply by `expected_sum / row_sum_i`;
   * symmetrise: `h[i][j] = h[j][i] = (h[i][j] + h[j][i]) / 2`.
2. `normalizeHeatmapDiagonalTotal(h, 1.0)` (`LooperSolver.cpp:1857-1876`):
   * compute mean of first off-diagonal at offset = `getDiagonalSize()`;
   * scale the **whole matrix** by `1.0 / avg` (i.e. so that off-diagonal
     mean = 1.0).
3. `normalizeHeatmapInter` (`LooperSolver.cpp:1817-1855`) — multi-chrom only.

Then `createDistanceHeatmap` (`LooperSolver.cpp:1753-1796`):
4. For each `(i,j)` with `val < 1e-6` → `heatmap_dist[i][j] = 0`;
5. Else, if `|i-j| < diagonal_size` → `heatmap_dist[i][j] = -1` (sentinel
   = repulsion in arc score; **must be preserved** through fp16 round-trip);
6. Else `heatmap_dist[i][j] = freqToDistanceHeatmap(val) = scale * val^power`.
7. Clip all values at `avg(heatmap_dist) *
   Settings::heatmapDistanceHeatmapStretching` (=2.0 by default).

Python `heatmap.normalize_heatmap` / `heatmap_to_expected_distances` mismatches:

| # | Step | cudaMMC | Python                                        |
|---|---|---|---|
| B1 | bin-length norm  | not present                  | divides by `length_mb[i] * length_mb[j]` (extra step) |
| B2 | row norm         | `expected_sum / row_sum_i`   | naive divide by `row_sum_i` (different normalising constant; not preserved across rows) |
| B3 | symmetrisation   | yes (cpp:1744-1748)          | **missing**                                    |
| B4 | diagonal-avg scale | offset = `getDiagonalSize()`, scales whole matrix to mean = 1.0 | only scales by mean of `diagonal_size`-offset (OK in principle but `getDiagonalSize()` is computed from data, not the hard-coded `Settings.diagonal_size=3`) |
| B5 | inter-chrom scale | `normalizeHeatmapInter`      | **missing**                                    |
| B6 | repulsion sentinel | inside-diagonal → -1 in `heatmap_dist` (becomes 1/d repulsion in score) | **missing** — Python zeroes those entries → no repulsion |
| B7 | max_dist clip     | `avg * heatmapDistanceHeatmapStretching` | **missing**                |
| B8 | dtype             | `float` throughout            | cast to `float16` (overflow / underflow risk for sentinel -1 → -1.0 OK but values >65504 become Inf) |

### Repulsion sentinel cross-impact
The heatmap score (`calcScoreHeatmapActiveRegion`, cpp:2216) treats only
`val < 1e-6` as “skip”, **not** negatives. With the sentinel preserved,
negative-valued diagonal-band pairs are evaluated as `(d - (-1))/(-1) - 1
= -(d+1) - 1`, contributing `(d+2)²` — i.e. a strong repulsion driving
near-diagonal beads apart. Python loses this entirely.

---

## C. Heatmap score functions (`scores.py`)

### C1. Hard-coded `diagonal_size=3`
- **cudaMMC**: `heatmap_dist.diagonal_size` is loaded from the heatmap file
  (`LooperSolver.cpp:1714,1756`, `Heatmap.cpp:430 diagonal_size =
  getDiagonalSize()`). It is per-heatmap and depends on the singleton
  resolution; for ChIA-PET data at anchor resolution it is typically 1 or 2,
  for segment-level it is larger.
- **Python**: `Settings.diagonal_size = 3` is a static default, used
  uniformly for all heatmaps. Must be plumbed from the loaded heatmap.

### C2. Sentinel handling in `score_heatmap` family
Per B6, when expected is -1 cudaMMC computes `(d-(-1))/(-1) - 1 = -d - 2`,
adding `(d+2)²` (repulsion). Python’s `valid = expected > 1e-3` masks
both 0 and negative entries out — so the repulsion term inside the
diagonal band is silently dropped.

### C3. Single-bead score returns sum over **chromosome range**, not full row
- **cudaMMC** `calcScoreHeatmapActiveRegion(moved)`, cpp:2208-2211:
  `getChromosomeHeatmapBoundary(moved, st, end)` and iterates `i ∈ [st, end]`.
- **Python**: when `same_chr_mask` is `None` (the common case in
  `reconstruct_clusters_heatmap`), iterates over **all N** beads — including
  beads of other chromosomes in a multi-chrom run. (Currently the Python
  pipeline runs per-chrom so this latent bug is hidden, but it is wired
  wrong.)

### C4. Full-vs-single ratio drift in `mc.monte_carlo_heatmap`
- **cudaMMC**’s `MonteCarloHeatmap` recomputes the **full** double-counted
  score at every iteration (cpp:468 `score_curr =
  calcScoreHeatmapActiveRegion()`).
- **Python** tracks `total_score` incrementally:
  `score_curr = total_score + (local_curr - local_prev)`.
  Per-pair the double-counted score changes by `2 × (local_curr -
  local_prev)`, **not** 1×. So `total_score` drifts further from the truth
  with every accepted move. The “milestone resync” at `mc.py:166` is a
  workaround for this bug; the correct fix is `delta = 2*(local_curr -
  local_prev)` (or track the single-counted score and use the same
  initialisation).

---

## D. Monte Carlo control loop (`mc.py:monte_carlo_heatmap`)

### D1. `_with_chance_ratio` greedy fallback is invented
`mc.py:58-59`:
```python
if score_prev <= 0.0 or score_curr <= 0.0:
    return score_curr <= score_prev
```
cudaMMC has no such guard (`LooperSolver.cpp:471-475`); the only branch is
`tp = scale * exp(-coef * (s_curr/s_prev) / T); ok = withChance(tp)`. The
guard only triggers because of bug C4 (drifted negative `total_score`).
Remove the guard; fix C4 instead.

### D2. Invented “cold-phase” stop condition
`mc.py:180-184`:
```python
cold = T < s.max_temp_heatmap * 0.005
... or (cold and ratio > 0.9999)
```
cudaMMC stop condition (`cpp:501-506`) is exactly:
```
(score_curr > improvement*milestone && milestone_success < min_successes)
|| score_curr < 1e-6
```
No cold check, no ratio>0.9999 guard. Violates the prime directive.

### D3. Stop-condition threshold uses the wrong setting
`mc.py:181` uses `s.milestone_improvement_ratio` (loaded from
`[simulation_arcs]`). cudaMMC uses
`Settings::MCstopConditionImprovementHeatmap` (`[simulation_heatmap]`,
`Settings.cpp:220`). Numerically identical by default (0.995) but a user
override would silently apply to the wrong phase.

### D4. `score_curr < 1e-6` constant
`mc.py:183` uses `total_score < 2e-6` (because of the doubled tracking).
With C4 fixed and matching cudaMMC double-counted total, the threshold
should be `1e-6` to mirror cpp:504 exactly.

### D5. Per-iteration `score_prev` update
`mc.py:191` does `score_prev = score_curr` at end of loop, but
`score_curr` has been overwritten by the `if accepted ... else` branch.
cudaMMC pattern is: after accept/reject, `score_curr` is either the new
total (accept) or `score_prev` (reject) — `cpp:513 score_prev = score_curr;`
keeps `score_prev` synced. The Python sequence works only because the
local variable `score_curr` is recomputed at the top of every iteration;
the explicit assignment is redundant and confusing.

---

## E. Settings (`settings.py`)

### E1. Missing INI sections in `from_ini`
The cudaMMC `Settings::loadFromINI` (`Settings.cpp:370-616`) reads:
- `[main]` keys: `output_level`, `random_walk`, `cache_input`, `use_2D`,
  `loop_density`, `max_pet_length`, `long_pet_power`, `long_pet_scale`,
  `steps_lvl1`, `steps_lvl2`, `steps_arcs`, `steps_smooth`,
  `noise_lvl1`, `noise_lvl2`, `noise_arcs`, `noise_smooth`;
- `[motif_orientation]`: `use_motif_orientation`, `symmetric_motifs`, `weight`;
- `[subanchor_heatmap]` / `[anchor_heatmap]` blocks;
- `[springs]`: `stretch_constant`, `squeeze_constant`, `angular_constant`,
  `stretch_constant_arcs`, `squeeze_constant_arcs`;
- `[simulation_heatmap]`: `max_temp_heatmap`, `delta_temp_heatmap`,
  `jump_temp_coef_heatmap`, `jump_temp_scale_heatmap`,
  `stop_condition_steps_heatmap`, `stop_condition_improvement_threshold_heatmap`,
  `stop_condition_successes_threshold_heatmap`;
- `[simulation_arcs_smooth]` jump_temp_scale/coef (Python only reads min_successes & milestone_steps);
- `[heatmaps]`: `inter_scaling`, `distance_heatmap_stretching`.

Python `from_ini` reads `[distance]`, parts of `[simulation_arcs]`, parts of
`[simulation_arcs_smooth]`, `[misc]`. **None** of the heatmap-phase MC
parameters can be overridden from an INI; **none** of the spring constants
can be tuned; multi-restart counts cannot be tuned.

### E2. Hard-coded spring symmetry
`Settings` collapses `springConstantStretch` and `springConstantSqueeze`
into a single `k_chain`, and `springConstantStretchArcs` /
`springConstantSqueezeArcs` into a single `k_spring`. cudaMMC keeps them
distinct and the score functions select stretch-vs-squeeze per-pair based
on sign of `diff = (d - exp)/exp` (`LooperSolver.cpp:1944-1946`,
`2041-2043`, `2085-2087`). At default values both are `1.0`/`0.1` so the
collapse is currently harmless, but any user override of
`squeeze_constant_arcs` (etc.) would be silently lost.

### E3. Missing `weightDistSmooth` / `weightAngleSmooth`
Python smooth-phase score adds chain + 1·angle. cudaMMC full score returns
`sca*weightDistSmooth + scb*weightAngleSmooth` (`LooperSolver.cpp:2062`).
At defaults `=1.0` both this is harmless, but the keys exist in
`[simulation_arcs_smooth]` and ought to be wired up. Also note the
upstream **swap bug** at `Settings.cpp:594-597`:
```
weightAngleSmooth = reader.GetReal("simulation_arcs_smooth", "dist_weight", ...);
weightDistSmooth  = reader.GetReal("simulation_arcs_smooth", "angle_weight", ...);
```
If we wire these up we must mirror the swap to remain bit-identical.

### E4. Missing motif-orientation parameters
`useCTCFMotifOrientation`, `motifsSymmetric`, `motifOrientationWeight`
(`Settings.cpp:394-399`) — none plumbed. The Python orientation score is
also a static-label approximation (see `scores.py:374-389`), but the
parameters should still be exposed.

### E5. Missing noise coefficients per level
`noiseCoefficientLevelChr/Segment/Anchor/Subanchor` are hard-coded as
literal `0.5` in `solver.py:330-331`. They should live in `Settings` and be
INI-overridable.

### E6. Missing simulation-step counts
`simulationStepsLevelChr` / `LevelSegment` (used for heatmap multi-restart,
defaults 2) are not in `Settings`. `simulationStepsLevelAnchor` /
`Subanchor` are present (good) but not overridable from `[main]` INI keys
(`steps_arcs`, `steps_smooth`).

### E7. Missing `heatmapDistanceHeatmapStretching`, `heatmapInterScaling`
Needed for the heatmap normalisation pipeline (B5, B7).

---

The list above is **not** a fix plan. It is the audit of deficiencies
limited to the user’s stated scope (heatmap scoring + settings). Markers
in source are tagged `# BUG (cudaMMC-mismatch): …` for grep-ability.

---

## F. Monte Carlo control loops — Phases 2 & 3 (`mc.py`)

Phase 1 (heatmap MC) is covered above in §§ A–D. This section adds Phase 2
(`monte_carlo_arcs` ↔ `MonteCarloArcs`, cpp:3058-3159) and Phase 3
(`monte_carlo_arcs_smooth` ↔ `MonteCarloArcsSmooth`, cpp:3161-3390).

### F1. Heatmap MC: cudaMMC recomputes the FULL score every iteration — Python tracks deltas
- **cudaMMC** `LooperSolver.cpp:468`:
  ```cpp
  clusters[ind].pos += displacement;
  score_curr = calcScoreHeatmapActiveRegion();   // O(N²) every iter — no delta
  ```
  i.e. Phase 1 deliberately does **not** use the delta-tracking trick that
  Phases 2/3 use. There is no possibility of drift, no resync needed.
- **Python** `mc.py:152, 182-187`: caches `total_score`, updates by
  `local_curr - local_prev`, resyncs at milestones (mc.py:221). Aside from
  the well-known 2× factor bug (§ C4), the whole pattern is inappropriate
  for Phase 1 — cudaMMC chose O(N²) per step because the heatmap-MC bead
  count is small (~hundreds at LVL_CHROMOSOME / LVL_SEGMENT, see § A1). At
  the (wrong) anchor-level scale Python runs it at (N≈26 k), an honest
  port would be unusable, which is why the author switched to deltas —
  papering over the A1 bug.

### F2. `monte_carlo_arcs_smooth` is missing the `T > 0` Metropolis guard
- **cudaMMC** `LooperSolver.cpp:3331`:
  ```cpp
  if (!ok && T > 0.0) { tp = …; ok = withChance(tp); }
  ```
- **Python** `mc.py:473`:
  ```python
  elif _with_chance_ratio(s.temp_jump_scale_smooth, …, T, rng):
  ```
  no `T > 0` check. With `dt_temp_smooth = 0.99995` and float64,
  `T` only asymptotes to 0, so this is *usually* benign, but it lets the
  Metropolis branch fire at temperatures cudaMMC would refuse to evaluate
  at — possibly accepting noise moves in the cold tail. The arcs phase
  (cpp:3113) does **not** have the `T>0` guard, so `monte_carlo_arcs` is
  correct in omitting it.

### F3. Arcs MC stop condition: `> 0.9999` direction mangled
- **cudaMMC** `LooperSolver.cpp:3146`:
  ```cpp
  || score_curr / milestone_score > 0.9999
  ```
  This stops on **any** near-equal milestone — including no-improvement
  *or slight worsening*. Direction-agnostic.
- **Python** `mc.py:374-375`:
  ```python
  or (ratio > 0.9999 and total_score <= milestone_score)
  or (cold and ratio > 0.999)
  ```
  Constrains the first clause to improvement-only (`<= milestone_score`),
  then adds an invented cold-phase second clause. Replace with the single
  unconditional `ratio > 0.9999` test from cpp:3146; delete the cold-only
  branch.

### F4. Smooth MC stop condition: extra `max_steps` budget cap
- **cudaMMC** `LooperSolver.cpp:3372-3376`: only the standard
  `(improvement && < min_successes) || score < 1e-6` test.
- **Python** `mc.py:502-505`: adds `or individual_steps >= max_steps`
  (computed at mc.py:415-416 from cooling-step heuristic). cudaMMC has no
  step budget; this is a Python-invented termination that will trigger on
  hard-to-converge IBs and short-circuit the optimisation.
- Same invented cap appears in `monte_carlo_arcs` at mc.py:293-294, 376.

### F5. Smooth MC: orientation delta is applied to every move, not gated on anchor/neighbour
- **cudaMMC** `LooperSolver.cpp:3275-3296, 3306-3314`: the orientation
  contribution is computed and added to `curr_score_orientation` **only**
  when `cluster_type[p] > 0` (moved bead is anchor or anchor-neighbour)
  and `Settings::useCTCFMotifOrientation` is on; the
  `2 * (local_curr - local_prev)` update at cpp:3311-3313 is inside the
  `if (orn_index != -1)` block (cpp:3306).
- **Python** `mc.py:444-446, 466`: calls `score_orientation_single` for
  **every** moved bead and unconditionally adds `2*(ori_a - ori_b)` to
  the score delta. For non-anchor beads the delta should be ≈0 if the
  single-bead score is well-defined, but:
  1. there is no `useCTCFMotifOrientation` gate — orientation is on by
     default while cudaMMC defaults `useCTCFMotifOrientation = false`
     (`Settings.cpp:148`);
  2. `score_orientation_single` semantics differ from
     `calcScoreOrientation(..., orn_index)`; see § G (scores audit).

### F6. Smooth MC: `cluster_type` neighbour-tracking machinery absent
- **cudaMMC** `LooperSolver.cpp:3214-3240` builds a `cluster_type` vector
  encoding for every bead whether it is an anchor (`3+idx`), a left
  neighbour (`1`), a right neighbour (`2`), or stores a coded negative
  pointer to the adjacent anchor. `calcScoreOrientation(orn_index)` uses
  this to locate the four beads (left neighbour, anchor, right
  neighbour) that define an anchor's orientation vector.
- **Python**: no equivalent. `score_orientation_single` operates on
  static "anchor orientation" string labels (see scores.py:374-389),
  which is a static approximation independent of the moved-bead
  geometry. The two scores are not numerically comparable.

### F7. Smooth MC: subanchor-heatmap branch missing
- **cudaMMC** `LooperSolver.cpp:3246-3247, 3299-3300, 3316-3320`:
  conditionally on `use_subanchor_heatmap && Settings::useSubanchorHeatmap`,
  adds `calcScoreSubanchorHeatmap` (full + per-bead) into the smooth
  score with the same `2 *` delta pattern as orientation.
- **Python**: no subanchor-heatmap code path. Defaults
  `useSubanchorHeatmap = false` (`Settings.cpp:147`) so this is dormant,
  but if a config turns it on, Python silently runs a different
  algorithm.

### F8. `_with_chance_ratio` greedy fallback corrupts arcs/smooth too
- Already noted for heatmap (§ D1). The same `mc.py:66-67` guard fires
  for arcs and smooth whenever Phase-2/3 score drifts non-positive (rare
  but possible because of delta accumulation; both phases resync at
  milestones, but within a milestone window the running `total_score` /
  `ts` can underflow). cudaMMC has no such guard at any phase
  (`cpp:3113, 3331`). Fix is to remove the guard once the delta math is
  proven not to drift negative.

### F9. Reject branch: cudaMMC restores `score_curr = score_prev`; Python skips it (arcs/smooth)
- **cudaMMC** arcs `LooperSolver.cpp:3127`, smooth `cpp:3349`: on reject
  sets `score_curr = score_prev`. This matters because cpp:3153 (arcs)
  / cpp:3342, 3349 (smooth) then carry `score_prev = score_curr`
  forward — i.e. the next iteration sees the **unchanged** score.
- **Python** arcs `mc.py:344-347`: on reject only un-applies `pos -= disp`
  and leaves `total_score` untouched (since it was never updated). This
  is *functionally* equivalent because `total_score` was only updated in
  the accept branches (lines 338, 342). Confirmed equivalent — flag as
  "harmless but easy to break next refactor". Annotate accordingly.

### F10. Heatmap MC: arcs-phase Setting reused for stop threshold
- See § D3. `mc.py:244` reads `s.milestone_improvement_ratio` (loaded from
  `[simulation_arcs]`); cudaMMC uses
  `Settings::MCstopConditionImprovementHeatmap` (`[simulation_heatmap]`,
  `Settings.cpp:220`). Same default value (0.995), but an INI that
  overrides arcs would silently re-tune heatmap. Tracked under § E1 also.

### F11. `step_size` decay is missing from arcs (correct), missing from smooth (correct), but present nowhere — yet `Settings.step_size_decay_arcs` / `_smooth` exist
- **cudaMMC**: neither `MonteCarloArcs` nor `MonteCarloArcsSmooth` decays
  the step size. Only `ParallelMonteCarloHeatmap` does (`.cu:325 step_size *= 0.95`).
- **Python `mc.py`**: arcs and smooth correctly do **not** decay step
  size. Good — but `Settings.step_size_decay_arcs` /
  `Settings.step_size_decay_smooth` exist as dead, misleading fields
  (see § E). Either remove or document as unused.

### F12. Arcs/smooth MC: random bead selector skips `is_fixed` via `continue` — counter still increments
- **cudaMMC** arcs `cpp:3099-3100`: `is_fixed` triggers `error(…)` (fatal)
  rather than retry — the arcs phase asserts no fixed beads exist in
  `active_region` at IB resolution.
- **cudaMMC** smooth `cpp:3269-3270`: `is_fixed` → `continue`, **no
  counter increment for that iteration** (the `i++` happens at the
  bottom of the loop *after* every path, so `continue` does still
  advance `i`). Wait — re-read: `i++` is at cpp:3384, *outside* and
  after the `continue`, so `continue` skips it. Each fixed-bead pick
  burns RNG draws but does NOT advance `i`, the cooling step, or the
  stop counter.
- **Python** `mc.py:312-313` (arcs) and `mc.py:437-438` (smooth): both
  use `continue` which jumps to the `while True:` top **before**
  `T *= dt` and `individual_steps += 1`. This matches the smooth
  behaviour. The arcs path differs from cudaMMC, which would `error()`
  out — Python silently skips, which is a defensive choice but masks
  the data-structure invariant that arcs-MC has no fixed beads.

---

## G. LooperSolver pipeline (`solver.py`)

This section audits the orchestrator against cudaMMC `runLooper()`
(main.cpp:459-565), `reconstructClustersHeatmap` (cpp:85-294),
`reconstructClustersHeatmapSingleLevel` (cpp:297-419),
`reconstructClustersArcsDistances` (cpp:2579-2702),
`positionInteractionBlocks` (cpp:2709-2725),
`reconstructClusterArcsDistances` (cpp:2735-2875), and
`calcAnchorExpectedDistancesHeatmap` (cpp:3837-3916).

### G1. **Phase-2 expected-distance matrix is sparse — cudaMMC's is DENSE with -1 repulsion default** ⚠️ CRITICAL
- **cudaMMC** `calcAnchorExpectedDistancesHeatmap` (cpp:3842-3844):
  ```cpp
  heatmap_exp_dist_anchor.init(active_region.size());
  heatmap_exp_dist_anchor.add(-1.0f);   // ALL pairs default to -1 (repulsion)
  heatmap_exp_dist_anchor.clearDiagonal(1);
  // … then only arc-connected pairs (cpp:3857-3882) overwrite with freqToDistance(freq).
  ```
  The Phase-2 score `calcScoreDistancesActiveRegion` (cpp:1932-1947) then
  evaluates **every (i,j) pair**:
  ```cpp
  if (heatmap_exp_dist_anchor.v[i][j] < 0.0f) { sc += 1.0f / v.length(); continue; }  // repulsion
  if (heatmap_exp_dist_anchor.v[i][j] < 1e-6) continue;                                // skip
  diff = (d - exp)/exp;
  sc += diff*diff * (diff >= 0 ? stretch : squeeze);                                   // spring
  ```
  i.e. **every non-arc pair contributes a `1/d` repulsion** that prevents
  bead collapse.
- **Python** `solver.py:130-176` `_arc_tensors_ib` builds a *list of arc
  pairs* — ONLY pairs that appear in `arcs_by_chr`. Non-arc pairs are
  absent from `arc_starts/arc_ends/arc_expected` → the score skips them
  entirely → **no global repulsion**. With typical IBs of ~30 anchors and
  only ~10-50 arcs, ≥ 80 % of the pairwise repulsion energy is missing.
- This single defect explains "structures collapse to a tiny ball": there
  is no force keeping non-interacting anchors apart.
- The single-bead score also can't recover from this: even if we add the
  `-1` entries to the sparse list, cudaMMC's per-bead score iterates
  *all* j ∈ active_region (cpp:1960), not just arc-connected j.

### G2. `_arc_tensors_ib` skips IB-boundary arcs, but cudaMMC's anchor-level matrix is built per-IB from `clusters[ai].arcs`
- **cudaMMC** cpp:3855-3882: iterates `clusters[ai].arcs` — the arc indices
  stored on each anchor cluster. These are filtered to "current IB"
  implicitly because `active_region` is the IB's children. Arcs whose
  other end lies outside the IB are silently skipped via
  `cluster_to_active_index[other_end]` not existing (would crash; the
  pre-condition is that `mark_arcs` placed arcs only between anchors in
  the same IB).
- **Python** `solver.py:151-167`: iterates **all arcs on the chromosome**
  for every IB, then filters via `idx_map.get(ai, -1)`. Functionally
  similar, but the iteration order doesn't match cudaMMC and the
  set of arcs depends on what `mark_arcs` and the IB partitioning
  produce — see audit of `tree.py` (`find_all_gaps` / `find_segments`).

### G3. cudaMMC reads `freqToDistance(freq, true)` for arc expected distances; Python reads `count_to_distance` with different defaults
- **cudaMMC** cpp:3875: `exp_dist = freqToDistance(freq, true)` —
  signature `freqToDistance(int freq, bool memo)` at cpp:2561-2578. This
  uses `Settings::countToDistA`, `countToDistScale`, `countToDistShift`,
  `countToDistBaseLevel` and memoises per-`freq`. Same family of
  parameters as Python's `count_to_distance`, but cudaMMC reads `int`
  PET-count whereas Python passes `arc.score` (which `load_pet_clusters`
  parses as `int` from BEDPE col 7 — OK).
- Confirm Python uses identical formula and defaults — already audited in
  prior pass; the **defaults** in `distances.py` are dead-code (settings
  override at call site), so behaviour matches at default config.

### G4. Missing `positionInteractionBlocks` — IB centroids never derived from segment chain
- **cudaMMC** `reconstructClustersArcsDistances` cpp:2599:
  ```cpp
  positionInteractionBlocks(current_level[chr]);
  ```
  which (cpp:2709-2725) is `interpolateChildrenPositionSpline(segments, true)`
  for multi-segment chroms, or a random walk with
  `Settings::ibRandomWalkJumps` for single-segment regions. **This places
  IB beads ALONG the spline of segment beads BEFORE any per-IB MC runs.**
- **Python** `solver.py:322`: just calls `tree._propagate_positions_up()`.
  That averages anchor positions UP to IB and segment beads — completely
  the opposite direction. Without segment-level MC having ever run (see
  § A1), the segment beads are all at the random-sphere centre, so this
  "propagate up" produces near-degenerate IB positions, and every IB's
  anchors start at essentially the same point regardless of genomic
  distance.

### G5. Missing `densifyActiveRegion` between arc and smooth phases
- **cudaMMC** cpp:2645: `densifyActiveRegion(current_level[chr][i], true)`
  inserts subanchor beads between consecutive anchors via spline
  interpolation. The smooth-phase MC then operates on this **denser**
  bead set (anchors + subanchors), refining the curved chromatin path.
- **Python** `solver.py:401-427`: smooth phase operates on the **same
  anchor beads** as arc phase, with no subanchor insertion. There is
  nothing to "smooth" — the smooth phase is just a second arc-style MC
  with chain + angular springs instead of arc springs.

### G6. Missing subanchor-heatmap construction
- **cudaMMC** cpp:2654-2685: if `useSubanchorHeatmap`, runs
  `subanchorEstimateDistancesReplicates` quick MCs, averages the
  resulting bead-bead distance matrices, and builds
  `heatmap_dist_subanchor` from them. This matrix feeds the smooth-phase
  score via `calcScoreSubanchorHeatmap` (cpp:2415).
- **Python**: not implemented (default `useSubanchorHeatmap = false` so
  dormant; § F7).

### G7. Smooth-phase noise magnitude wrong
- **cudaMMC** cpp:2844-2849: per-restart re-noising uses
  ```cpp
  random_vector(smooth ? noise_size : noise_size_small, use2D);
  ```
  i.e. arc-phase noise = `noise_size_small = 0.05f` (literal at cpp:2765);
  **smooth-phase noise = `noise_size` (= avg chain-length × noiseCoefSubanchor ≈ 5-100 units)**.
- **Python** `solver.py:409`:
  ```python
  pos_in = pos + s.noise_size_small * torch.randn_like(pos)
  ```
  uses `noise_size_small = 0.05` for *both* phases. Smooth phase is
  under-noised by 2-3 orders of magnitude → optimisation never escapes
  the arc-phase basin.
- Compounded with §F (smooth-phase MC `random_vector` → Gaussian `randn`,
  not uniform).

### G8. Per-restart re-noising uses Gaussian, cudaMMC uses uniform
- **cudaMMC** cpp:2847: `random_vector(noise, use2D)` →
  `common.cpp:14-25` → `(2u-1)*range` per axis (uniform).
- **Python** `solver.py:384, 409`: `torch.randn_like(pos)` → Gaussian.
  Already flagged generally for `_random_displacement`, but the
  per-restart noise call sites are independent and should also use
  the uniform helper.

### G9. Heatmap-phase `reconstructClustersHeatmap` bypasses the entire two-level cascade
- See § A. cudaMMC cpp:122-294 runs the cascade
  `LVL_CHROMOSOME (if multi-chr) → LVL_SEGMENT`, each with its own
  heatmap (chromosome / segment singleton heatmap), its own
  `createDistanceHeatmap`, its own `noiseCoefficient*` and its own
  `simulationStepsLevel*` multi-restart. Python collapses all of this
  into a single anchor-level MC.

### G10. `setLevel(LVL_SEGMENT)` / `setLevel(LVL_INTERACTION_BLOCK)` state transitions absent
- **cudaMMC** maintains a global `current_level[chr]` that
  `setLevel(L)` rewrites at every phase boundary. Most score functions
  read this state implicitly. This is a giant global-state machine.
- **Python** `ChromosomeTree` keeps all levels persistent in
  `clusters[]` and switches by passing different `cidx` lists into MC.
  Functionally equivalent, but be aware: any cudaMMC code that uses
  the implicit "active_region" cannot be ported by index translation
  alone — the active region's contents and length change across phases.

### G11. `runLooper` random-walk fast path missing
- **cudaMMC** cpp:89-109: if `Settings::randomWalk`, places segment beads
  by `displace(pos, 50.0f, use2D)` step-by-step then spline-interpolates
  anchors. Skips heatmap MC entirely. Used for degraded-input runs.
- **Python**: not implemented. Low priority (not the default), but the
  `Settings.randomWalk` flag silently does nothing.

### G12. `useDensity`, `useTelomerePositions`, `useCTCFMotifOrientation`, `useAnchorHeatmap`, `useInputCache`, `templateSegment` all silently ignored
- These cudaMMC features (densityCoord constraints, telomere fixing,
  motif orientation gating, anchor-heatmap modulation of expected
  distances, INI caching of intermediate heatmaps, template-segment
  initial structure) have **no Python wiring**. Most default to off so
  most runs are unaffected, but any INI that enables them will produce
  silently-divergent output.

### G13. `mark_arcs` vs cudaMMC arc-binding ordering
- **cudaMMC** `setContactData` (cpp:643-717): loads anchors, loads arcs,
  binds each arc to its two anchors (`clusters[anchor].arcs.push_back(arc_idx)`).
  An arc whose endpoints don't both hit anchors is dropped.
- **Python** `data_loading.mark_arcs`: similar in spirit, but a separate
  audit is required to confirm coordinate-rounding / midpoint matching is
  identical (BEDPE coords vs anchor BED coords). This pass does not
  resolve that; flag for follow-up under `data_loading.py` audit.

### G14. `_propagate_positions_up` uses mean-of-children; cudaMMC initialises children FROM the parent
- **cudaMMC** never "propagates up". The hierarchy is filled top-down:
  parent position is set first (via heatmap MC at LVL_SEGMENT), then
  children are placed at parent + noise (cpp:2624-2625, cpp:357-363).
- **Python** `solver.py:284, 322, 430`: `tree._propagate_positions_up()`
  sets parent = mean(children). For a tree that has only ever had
  anchor-level positions set (because § A1-A5 skipped LVL_SEGMENT MC),
  this is the only way to get IB/segment positions at all — but it is
  the **opposite direction** of cudaMMC's data flow.

---


## H. Chromatin tree (`tree.py`)

This section audits `ChromosomeTree` against cudaMMC `createTreeChromosome`
(cpp:1021-1158), `findGaps` (cpp:856-895), `findSplit` (cpp:900-994), and
`interpolateChildrenPositionSpline` (cpp:2939-3056).

### H1. ⚠️ `find_segments` is a Python-invented heuristic — cudaMMC has TWO modes, neither of which Python implements
- **cudaMMC** `findSplit` (cpp:900-994) has two branches:
  - **Branch A — `segments_predefined.regions.size() > 0`** (cpp:911-962):
    splits are taken from a user-provided BED of breakpoints (loaded at
    cpp:42 from `Settings::dataSegmentsSplit`). In the shipped configs
    this points at `ccds_all_hg38_breakpoints.bed` /
    `ccds_all_hg38_merged100k_*.breakpoints.bed`. Each predefined
    region's `start` coordinate that falls inside a `gaps[i]` span
    promotes that gap to a segment boundary.
  - **Branch B — no predefined segments** (cpp:964-994): builds `L` and
    `S` diagnostic arrays then `return gaps;` unchanged. **i.e. every
    gap becomes a segment boundary; each segment contains exactly ONE
    IB.** `Settings::segmentSize` is passed as `exp_size` but **never
    actually used** in this branch — the parameter is dead code.
- **Python** `find_segments` (tree.py:87-110): uses a `segment_size =
  2_000_000` distance heuristic to subsample gaps. This matches
  **neither** branch and is a third, invented partitioning algorithm.
- **Impact**: the segment hierarchy is structurally wrong. With the
  shipped GM12878 data + the breakpoints BED, cudaMMC's segments
  correspond to gene-coding regions / contiguous-density bins; Python's
  ≥ 2 Mb chunks ignore biology entirely. Combined with the absence of
  segment-level MC (§ A1), the segment tier exists in Python only as
  passive metadata.

### H2. `data_loading` does not read the breakpoints BED at all
- **cudaMMC** cpp:41-44: loads `Settings::dataSegmentsSplit` into
  `segments_predefined` before tree creation.
- **Python**: no field on `Settings` for this path; `data_loading.py`
  has no reader. The two `ccds_*.breakpoints.bed` files in
  `data/GM12878/` (and analogous in `H1ESC/`, `HFFC6/`) are vendored but
  unused.

### H3. Spline `interpolate_children_spline` uses ONE ghost reflection per end; cudaMMC uses TWO
- **cudaMMC** cpp:2946-2954:
  ```cpp
  pts[1] = mirrorPoint(P[0], P[1]);            // first reflection
  pts[0] = mirrorPoint(pts[1], P[0]);          // second reflection
  end_pt  = mirrorPoint(P[n-1], P[n-2]);
  end_pt2 = mirrorPoint(end_pt, P[n-1]);
  ```
  Four control points wrap the chain with two reflections at each end —
  this gives a sharper initial tangent and a non-zero curvature at the
  end-points.
- **Python** tree.py:185-188: one ghost at each end (`pts =
  [ghost_start] + parents + [ghost_end]`), and the Catmull-Rom sampler
  uses 4 consecutive control points starting from index 0. End-points
  collapse to straight-line continuation. Already noted in tree.py
  docstring — flag as `# BUG (cudaMMC-mismatch)`.

### H4. Spline parameterisation: cudaMMC uses sliding [0.5, 1.5) wrap with control-point switch; Python uses linear `u ∈ [0, n_segments]`
- **cudaMMC** cpp:3022-3032 (equidistant) and cpp:3007-3020 (genomic):
  knots for parent `i`'s children lie in [0.5, 1.0) ∪ [0.0, 0.5); when
  a child crosses 1.0 the four control points slide forward by one
  (cpp:3042-3050). Effect: each child interpolates between its parent's
  segment **and the neighbour's**, so the spline is continuous across
  parent boundaries.
- **Python** tree.py:191-203: `u = k/(N-1) * n_segments`, `seg = int(u)`,
  no control-point sliding. Each chunk uses the same 4 control points.
  At parent boundaries the curve direction can jump.
- Already noted in docstring; flag as `# BUG (cudaMMC-mismatch)`.

### H5. `interpolateChildrenPositionSpline(regions, use_genomic_dist=true)` mode is unimplemented
- **cudaMMC** cpp:2940 accepts `bool use_genomic_dist`. When `true`
  (cpp:2969-3020) knots are derived from the child's `genomic_pos`
  inside the parent's `[start, end]` (with flanking-aware weighting,
  cpp:2998-3003). When `false` (cpp:3021-3032) knots are equidistant.
- `positionInteractionBlocks` (cpp:2712) calls this with `true`, i.e.
  IB positioning along segment splines is **genomic-distance-weighted**.
- **Python** `interpolate_children_spline` has no `use_genomic_dist`
  parameter and always does equidistant sampling. Even after § G4 is
  fixed by wiring `positionInteractionBlocks`, IB knot positions will
  still be wrong unless the genomic-mode branch is added.

### H6. `ChromosomeTree` creates the root cluster FIRST (idx 0); cudaMMC creates it LAST
- **cudaMMC** cpp:1051, 1141-1156: `Cluster rootc;` is built locally
  throughout `createTreeChromosome` and pushed at the *end* of the
  cluster array (cpp:1156 `clusters.push_back(rootc)`). Anchors occupy
  the FIRST `arcs.anchors_cnt[chr]` indices for that chromosome
  (cpp:1030-1037).
- **Python** tree.py:258: `root_idx = self._new_cluster(1, ...)` is
  inserted at index 0, anchors land at indices 1..N. All downstream
  Python code uses `tree.anchors_idx` for translation so this is
  internally consistent, but it means cudaMMC's invariant
  `cluster_index == anchor_index for the first N clusters` is broken.
- **Impact**: any direct reuse of cudaMMC indices (e.g. reading a
  cudaMMC `.hcm` and matching beads) will be off-by-one for every
  chromosome. Plus: cudaMMC at cpp:1043 shifts `arcs.arcs[chr][i].start
  += cluster_start;` to make `clusters[start].arcs.push_back(i)` work
  on direct cluster indices. Python sidesteps this by going through
  `anchors_idx`, but the asymmetry creates fragility.

### H7. `clusters[].arcs` wiring guard is asymmetric
- **cudaMMC** cpp:1047-1048: unconditionally `push_back` arc index on
  both endpoints. Assumes both endpoints are valid anchors (already
  validated by `markArcs`).
- **Python** tree.py:307-311:
  ```python
  if arc.start < len(self.anchors_idx):
      self.clusters[self.anchors_idx[arc.start]].arcs.append(arc_i)
  if arc.end < len(self.anchors_idx):
      self.clusters[self.anchors_idx[arc.end]].arcs.append(arc_i)
  ```
  Each side is gated independently — an arc with `start < N` but
  `end >= N` would attach to the start anchor only, producing a
  half-bound dangling arc. cudaMMC would crash; Python silently
  corrupts.

### H8. `_propagate_positions_up` (already audited § G14): wrong data-flow direction
- See § G14. Used in `solver.py:284, 322, 430`. cudaMMC initialises
  positions **top-down** (segment from heatmap MC → IB via spline →
  anchor + noise); Python averages **bottom-up**.

### H9. `init_positions_linear` and `init_positions_random` are invented
- cudaMMC has no equivalent — initial anchor positions are always
  produced by an MC pass at the parent level. Python's
  `init_positions_linear` (tree.py:328-333) writes `(i, 0, 0)` to every
  anchor, which is then overwritten immediately by
  `reconstruct_clusters_heatmap`'s random-sphere init (§ A3). Dead
  scaffolding.

### H10. Anchor `orientation` stored as a string ('R'/'L'/'N'); cudaMMC stores a 3D vector computed from neighbours
- **Python** `data_loading.py:23-29`: BED strand `+` → `'R'`, `-` →
  `'L'`, else `'N'`. `tree.py:301` copies this label onto the cluster.
- **cudaMMC** `Cluster.h` declares `vector3 orientation;` and
  `calcOrientation(int cind)` (cpp:3437-3455) computes the orientation
  at run-time from the **3D positions** of the anchor and its left /
  right subanchor neighbours (taking the half-difference vector).
  The strand sign only enters via `Settings::motifsSymmetric` and the
  Anchor input parser.
- **Impact**: see § F6. Python's smooth-phase orientation score is a
  static-label approximation; cudaMMC's is a geometric quantity that
  changes whenever the anchor or its neighbours move. The two are not
  numerically comparable and the Python score has *no derivative*
  w.r.t. bead position — it cannot guide MC.

### H11. `find_all_gaps` arc-sweep correctly mirrors cudaMMC ✅
- tree.py:41-84 vs cpp:856-894: same loop structure, same
  `arcs_cnt += starts_at[i] - ends_at[i]` order, same first/last-pos
  insert. Confirmed identical. *(Positive finding — flag the file
  header so future agents don't disturb it.)*

### H12. IB construction loop matches cudaMMC inclusive-bounds correctly ✅
- tree.py:283-304 vs cpp:1084-1111: same `prev_gap = (i==1 ? gaps[0] :
  gaps[i-1]+1)` rule (tree.py:285), same `for k in [prev_gap, curr_gap]`
  inclusive iteration (tree.py:296). Confirmed identical. *(Positive
  finding.)*

---

## I. Data loading (`data_loading.py`)

Per the task scope — "loosely, Python data loading should be different
from C/CUDA loading" — this section only flags semantic divergences that
affect the downstream algorithm. Pure I/O / file-format differences are
acceptable.

### I1. `mark_arcs` filter: cudaMMC drops anchors with `length() <= 1`; Python doesn't
- **cudaMMC** `InteractionArcs.cpp:65`: `if (anchors[chr][j].length() > 1)`
  before the `contains()` check. Single-base anchors are silently
  skipped during arc matching.
- **Python** `data_loading.py:147-159`: `_find_anchor` checks `start <=
  pos <= end` without a length filter. A 1-bp anchor (start == end) is
  matchable.
- **Impact**: if the input BED contains zero/one-base anchors (rare but
  possible — e.g. a CTCF motif midpoint), Python will accept arcs that
  cudaMMC rejects. Different arc set → different IB partitioning (via
  `find_all_gaps`) → different downstream MC.

### I2. `mark_arcs` matching algorithm — linear scan vs binary search
- **cudaMMC** cpp:64-71 does an **O(A·N) linear scan** for every arc
  endpoint over every anchor on the chromosome.
- **Python** `_find_anchor` uses **O(log N) binary search** assuming
  anchors are sorted by `start` (data_loading.py:33) and non-overlapping.
- Result is identical *iff* anchors are sorted and disjoint. Both
  invariants hold for cudaMMC-style ChIA-PET anchor BED files, so this
  is a perf-only difference. ✅ acceptable per the "loose" scope.

### I3. BEDPE score-column heuristic
- **Python** `_parse_bedpe_line` (data_loading.py:38-54): tries
  `parts[7]` first, falls back to `parts[6]`. Comment says some BEDPE
  put score in col 7, others in col 6.
- **cudaMMC** `Cluster.cpp` / `Anchor.cpp` (verify separately):
  hard-codes the column expected by its preferred format. If the file
  has the alternative layout cudaMMC reads garbage; Python is more
  forgiving.
- **Impact**: PET counts could differ if column placement is
  ambiguous → different `freqToDistance` outputs → different expected
  distances for arc springs. Recommend: read the column index from
  `Settings`, mirroring cudaMMC's expectation rather than auto-detecting.

### I4. `_flush_tmp` factor merging matches cudaMMC ✅
- Python data_loading.py:167-197 vs cudaMMC `InteractionArcs.cpp:88-141`:
  same sort-by-factor, same `multiple_factors` flag, same
  `score = factor_score`, `eff_score = 0 or factor_score`, and same
  trailing summary arc with `score=0, eff_score=total`. Confirmed
  semantically identical.

### I5. Inter-chromosomal arc handling — Python aggregates under a synthetic key, cudaMMC keeps per-chrom maps
- **cudaMMC**: `arcs[chr]` is keyed by single chromosome; inter-chrom
  arcs live in separate plumbing (`dataSingletonsInter` etc.).
- **Python** data_loading.py:99-101: inter-chrom arcs stashed under
  `_inter_key(c1, c2) = "chrA:chrB"`. Downstream `solver.py` then
  iterates `arcs_by_chr.get(chrom, [])` which skips these synthetic
  keys, so inter-chrom arcs are effectively **dropped**.
- **Impact**: if any BEDPE has cross-chromosome rows they vanish
  silently. cudaMMC has a real inter-chrom code path
  (`normalizeHeatmapInter`, § B5); Python doesn't, so this drop is
  consistent with the rest of the codebase but should be explicit.

### I6. Anchor strand → orientation label loses information
- See § H10. Python stores 'R'/'L'/'N'; cudaMMC stores `vector3` and
  *recomputes* the orientation from neighbours during MC. The strand
  sign in the input BED is only an initial hint in cudaMMC, not the
  authoritative orientation.

### I7. `mark_arcs` `ignore_missing` parameter is a no-op
- data_loading.py:118-121: when an endpoint doesn't hit an anchor,
  Python `continue`s regardless of `ignore_missing`. cudaMMC prints
  `! error: non-matching arc` and continues (cpp:73-77). Python is
  always silent. Cosmetic but slightly misleading API.

### I8. Score / eff_score types
- cudaMMC `InteractionArc::score` is `int`. Python `InteractionArc.score`
  type — need to confirm in `data_structures.py` (this audit does not
  open it). If Python uses `int` everywhere, ✅.

---

`# BUG (cudaMMC-mismatch, AUDIT §H…)` markers are added inline at the
worst offenders (H1, H3-H4, H6-H7, H10) and the two positive findings
H11/H12 are tagged so future refactors don't break them.

---

## J. Mega-IB diagnosis

This section audits the mega-IB diagnosis pipeline (`solver.py`):
`reconstruct_clusters_heatmap` (cpp:85-294), `reconstruct_clusters_arcs` (cpp:2579-2702),
`position_interaction_blocks` (cpp:2709-2725), and `reconstruct_cluster_arcs_distances` (cpp:2735-2875).

### J1. Missing `reconstruct_clusters_arcs` call
- **cudaMMC**: `reconstructClustersHeatmap` (cpp:85-294) calls
  `reconstructClustersArcsDistances` (cpp:2579-2702) directly.
- **Python**: `reconstruct_clusters_heatmap` skips the arcs step entirely.
  The subsequent `position_interaction_blocks` (cpp:2709-2725) and
  `reconstruct_cluster_arcs_distances` (cpp:2735-2875) calls are then
  moot, as there are no arcs to process.
- **Impact**: the entire mega-IB diagnosis is bypassed. Structures are
  produced with no regard for long-range arc connections.

### J2. `position_interaction_blocks` uses the wrong source of segment beads
- **cudaMMC**: `positionInteractionBlocks(current_level[chr])` (cpp:2599)
  derives IB centroids from the segment beads of the current level.
- **Python**: `tree._propagate_positions_up()` (solver.py:322) averages
  the positions of child anchors **up** to the IB. The IB beads then
  inherit this averaged position.
- **Impact**: unless the segment-level MC has run (it hasn't, see A1),
  the segment beads are all at the random-sphere centre. The IBs are
  therefore all positioned at essentially the same point, leading to
  degenerate mega-IBs that don't reflect the actual chromatin
  architecture.

### J3. Missing `densifyActiveRegion` call for mega-IBs
- **cudaMMC**: after `reconstructClustersArcsDistances`, the active region
  is densified with `densifyActiveRegion(current_level[chr][i], true)`
  (cpp:2645). This adds subanchor beads between anchors, refining the
  chromatin path.
- **Python**: no equivalent. The active region remains sparse, with
  only the original anchor beads present.
- **Impact**: the mega-IBs lack the fine structure provided by the
  subanchors. The diagnosis will be based on an incomplete
  representation of the chromatin.

### J4. Arc score handling in mega-IBs
- **cudaMMC**: in `calcScoreDistancesActiveRegion` (cpp:1932-1947), every
  pair `(i,j)` in the active region is evaluated. If the expected distance
  is negative (non-arc pair), a repulsion term `1.0f / v.length()` is
  added to the score. This is crucial for maintaining the structure of
  non-interacting regions.
- **Python**: with the active region constructed from arcs only, the
  score function effectively becomes a no-op. There are no negative
  expected distances to trigger the repulsion term.
- **Impact**: the absence of the repulsion term means that non-interacting
  regions can collapse into each other, drastically altering the
  chromatin structure.

### J5. Missing output of the mega-IB diagnosis
- **cudaMMC**: after computing the scores, the results are written out
  for inspection.
- **Python**: no such output is generated. The user has no way to
  inspect the diagnosed mega-IBs.
- **Impact**: prevents validation of the mega-IB diagnosis. Users cannot
  verify if the mega-IBs have been correctly identified and scored.

---

The items above are tagged with `# BUG (cudaMMC-mismatch)` in the
relevant source sections. They represent critical divergences that
affect the functionality and output of the mega-IB diagnosis.

---

## K. 2026-05-14 diagnosis: "tangled ball" is a regime mismatch, not a port bug

User report: cudaMMC reference image shows clean loop rosettes; Python output
is a diffuse tangle. Suspicion: loop extrusion not enforced. Diagnosed with
`--debug-dump-stages` on `chr14:20000000:21500000`, GM12878 default config,
30 s/phase wall-cap.

### Stage-by-stage spatial extent (per-bead std deviation)

| Stage          | beads | std/axis | bbox diag |
|---|---|---|---|
| post-arc-MC    | 390   | **0.04** | ~0.07     |
| post-densify   | 2335  | **0.04** | ~0.07     |
| post-smooth-MC | 2335  | 4.56     | ~38       |

Arc-MC moved 390 stacked anchors by ≈ nothing (initial 0.05 cluster → 0.04
cluster after 3684 iters). Densify linearly interpolates between stacked
anchors → all subanchors stacked. Smooth-MC then re-noises subanchors by
`±noise_size_smooth = 6.83` (5.7× the median chain link target of 1.20) and
the Metropolis acceptance is so lax in the "hot" phase
(`tp = 50·exp(-20·(s_curr/s_prev)/T) ≈ 0.91` at T = 5) that the chain
random-walks rather than relaxes. Hard evidence: smooth-MC score **increased**
9457 → 13093 over 5872 iterations.

### Why no spread during arc-MC

- step = `avg_chain × noiseCoefficientLevelAnchor = 1.78 × 0.01 = 0.018`
- Initial cluster radius = `noise_size_small = 0.05` (cpp:2765 literal)
- 390 anchors are mutually repelled by the `-1` sentinel of the dense
  expected matrix (cpp:1932-1934 `sc += 1/d`). Repulsion energy scales as
  `O(N²) ≈ 76 k` pairs.
- Each MC step displaces ONE bead by ≤ 0.018. Expected per-bead std after
  `k` accepted moves: `0.018·√k`. To reach the equilibrium radius (~5 units
  where repulsion balances arc-spring at 0.5), each of 390 beads needs
  `k ≈ 80 000` ⇒ total iterations ≈ `3·10⁷`. `mc_stop_steps_arcs = 50 000`
  is 600× too small.
- cudaMMC's `MonteCarloArcs` (`cpp:3058-3159`) is a CPU sequential loop;
  there is no parallel arc-MC kernel in `ParallelMarkArcs.cu` (that file
  parallelises `markArcs`, not MC). So cudaMMC ALSO produces a near-point
  anchor cluster in this regime.

### Why no loops during smooth-MC

- step = `avg_chain_smooth × noiseCoefficientLevelSubanchor = 1.19 × 5.0
  = 6.83`
- Median chain link target = 1.20; max = 8.11 (heterogeneous across the IB)
- step/link ratio = **5.72** → every move several link-lengths → chain
  springs (`k = 0.1`, weak) cannot restore → score grows
- Lax Metropolis (`tp ≈ 0.91` while `T > ~1`) accepts ~91 % of bad moves.
  Cooling to `T < ~0.04` (where acceptance → 0) needs
  `0.9999^k · 5 < 0.04 ⇒ k > 49 000` iters. With 30 s cap (5872 iters) we
  never reach the cold refinement phase; the structure freezes wherever the
  random walk leaves it.

### Underlying input-scale problem

- `chr14:20-22.5 Mb` has **0 breakpoints** inside the test region in
  `ccds_all_hg38_merged100k_GM12878.breakpoints.bed`.
- `findSplit` Branch A (cpp:911-962) therefore yields **1 segment** for the
  region.
- `findGaps` (cpp:856-894) finds gaps where `arcs_cnt` hits zero. With ~170
  arcs over 396 anchors and dense overlap, the sweep yields just **7
  splits** — and one IB swallows **390 / 396 = 98 %** of all anchors.
- Result: arc-MC and smooth-MC are asked to position **one IB of ~400
  anchors** — exactly the regime where the algorithm scales `O(N²·iters)`
  and runs out of budget.

### Conclusion

- The Python port is **algorithmically faithful** to cudaMMC for this run.
- The user's reference image must come from a different region/dataset where
  IBs have **~10–50 anchors** (the regime where 50 k MC iterations suffice).
- Fix is **upstream of MC**: supply a denser `segment_split` BED so
  `findSplit` Branch A breaks the chromosome into smaller IBs, or pick a
  region that already has multiple breakpoints inside its bounds (≥ 1
  breakpoint per ~50 anchors). For GM12878, `segment_split` should average
  ≤ 500 kb per predefined segment to keep IB anchor counts below 100.

### Action taken on this branch

- `solver._reconstruct_single_ib` now emits a
  `WARNING: IB has N anchors — algorithm scales poorly beyond ~50` log
  line when `n_ib >= 100`, pointing at the `segment_split` configuration
  knob.
- New CLI flags in `main.py`:
  - `--debug-dump-stages DIR` — write
    `<chr>_ib<N>_(01_post_arc|02_post_densify|03_post_smooth).cif` plus
    `<chr>_00_ib_centers.cif` for visual inspection of each pipeline
    stage;
  - `--debug-max-ibs N` — stop after `N` IBs (any chromosome) for fast
    iteration on the mega-IB hotspot;
  - `--debug-max-mc-seconds SEC` — hard wall-cap per MC phase per restart
    so the user can complete the pipeline in a known budget.
- `tree.position_interaction_blocks`: single-segment branch now seeds the
  random walk at `(0, 0, 0)` to match cpp:2718
  `rw_pos.set(0.0f, 0.0f, 0.0f);` (previously seeded from the segment
  centroid which collapsed IBs on top of each other; verified
  `max IB pairwise = 20.88` after the fix vs near-zero before, on the
  same `chr14:20000000:21500000` test region).
