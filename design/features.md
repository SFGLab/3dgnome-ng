# Features and their record

Each entry states what a feature does, its settings, the place of its code and what was measured. The entries of the
last section are dated records and are not rewritten when production moves on. `SETTINGS.md` and
`gnome3d/production.py` state what holds now.

## Inputs and command line

- **Regions.** The command line and `run_genome` take one region such as `chr14:18288319-20307135`, one chromosome,
  a list `chr1,chr3,chrX` or a range `chr1-chr22,chrX`. An empty value is chr1 to chr22 and chrX.
  [io.parse_chrs_arg](../gnome3d/io.py) parses the value. A run of several chromosomes writes one CIF per chromosome
  with the chromosome in the file name. `run_region` returns one bead list per structure. `run_genome` returns one
  dict of chromosome to bead list per structure.

- **Singletons between chromosomes.** `[data] singletons_inter` is a second singletons file.
  [data.ContactData.from_files](../gnome3d/data.py) appends it to the singletons in a run of several chromosomes and
  skips it in a run of one.

- **Chromosome level.** A run of several chromosomes places the chromosome roots first, in
  [pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py). The singleton scores between each pair of
  chromosomes are summed into a heatmap. The law converts it to distances at the mean chromosome span. The heatmap MC
  runs `steps_lvl1` times with a step of `noise_lvl1` times the mean distance and the best run is kept. Without
  singletons between chromosomes the roots are scattered at random.

- **Scaling between chromosomes.** `[heatmaps] inter_scaling` multiplies the cells of the segment heatmap between
  two chromosomes and leaves the cells within a chromosome unchanged, in
  [pipeline/coarse/heatmap.py](../gnome3d/pipeline/coarse/heatmap.py). It acts in a run of several chromosomes.

- **Long loops.** A loop longer than `max_pet_length` gets no spring. [io.load_arcs](../gnome3d/io.py) returns these
  loops apart from the others and `ContactData.long_arcs` holds them. `add_long_pet_to_segment_heatmap` in
  [pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py) adds `long_pet_scale * score ^ long_pet_power` for
  each of them to the cell of its two segments.

- **Random walk.** `random_walk = yes` places the segment beads of each chromosome on a chained random walk in place
  of the segment heatmap MC. The walk takes one step of the distance of 1 Mb from the origin and then one step of 50
  units per segment. It honours `use_2D`.

- **Noise of the arcs stage.** The noise step of the arcs stage is fixed at 0.005, `_ARCS_NOISE` in
  [skeleton.py](../gnome3d/skeleton.py). No setting changes it.

## Kernel structure

- **One smooth kernel.** `batch_mc_nb` in [mc/numba/terms.py](../gnome3d/mc/numba/terms.py) runs the smooth stage for
  every combination of terms. Flags turn on the heat, orientation, excluded volume, confinement and compartment
  terms. Each term keeps its own running score and updates it by local deltas. The score of the chain is their sum.

## Model

- **Loops of more than one factor.** `[data] clusters` takes several files and `[data] factors` gives each a label,
  CTCF first. Every loop carries the index of its file. Each factor has its own strength fit: the PET counts of a
  second library are read against the typical count of that library at each span. The target matrix converts
  each loop with the fit of its factor. The orientation term reads the loops of the first factor alone. A loop that
  another factor holds has no motif orientation. A run with one file keeps the order of its loops and is byte
  identical by the parity gate. Built 2026-09-19 for RNAPII ChIA-PET as a second factor, `design/rnapii-loops.md`.
  The code is [io.load_arcs](../gnome3d/io.py), [data.fit_arc_strengths](../gnome3d/data.py),
  [polymer.PolymerLaw.arc_distance](../gnome3d/polymer.py) and [skeleton.py](../gnome3d/skeleton.py). Unit checks in
  `tests/unit/test_factors.py`.

  For a pair that two factors join, the reader adds a summary loop with score 0 and factor -1. The arcs stage leaves
  it out and the pair takes the closer of the targets of its factors. Found 2026-10-27 by the trio study: as a loop
  the summary gave the pair the background as its target. On a trio chr1 406 pairs, strong ones with a median of 14
  CTCF and 6 RNAPII PETs, had been held at the background in every run without loop dropout, which drops the summary
  loop by its score. Unit checks in `tests/unit/test_arc_matrix.py`.

- **Scope of block placement.** `[simulation_ib] refine_scope` is `segment` or `chromosome`, default `segment`.
  `segment` places the blocks of each segment as a separate chain and skips a segment with one block or none. The
  grouping into segments then decides the set of blocks that are placed. `chromosome` places every block of the
  chromosome as one chain.

  `chromosome` is not the default. It puts every pair of blocks inside the excluded volume term and the structures
  inflate. Over four GM12878 regions the mean simulated contact density fell from 0.087 to 0.035 and the enrichment
  within blocks over between blocks worsened about threefold. The compartment saddle improves, but sparsity alone
  moves that statistic toward 1.0 and the gain cannot be separated from the inflation. This scope needs the excluded
  volume and the confinement of block placement tuned again first.

- **Orientation term: weighted local score.** `_local_score_orientation_nb` in
  [mc/numba/terms.py](../gnome3d/mc/numba/terms.py) weights each neighbour of an anchor with the weight of their loop,
  as `score_orientation_full_nb` does. The update `score_orn += 2 * (curr - prev)` is then exact against the full
  score.

- **Singletons are filtered by chromosome.** [io.load_singletons](../gnome3d/io.py) keeps a contact when both of its
  chromosomes are in the run. In a run of one region both ends must also be inside the region.

- **`BeadOut` has a `kind` field.** An output bead is the NamedTuple `(start, end, x, y, z, kind)` of
  [types.py](../gnome3d/types.py), with `kind` either `anchor` or `subanchor` and the properties `midpoint` and
  `span`. The CIF writer stores the kind in the column `_atom_site.gnome_bead_kind`. Anchors have the residue name
  ALA and subanchors GLY, which a viewer colours apart.

- **Subanchor slots.** [pipeline/ib/densify.py](../gnome3d/pipeline/ib/densify.py) gives each of the `ld` subanchors
  between two anchors a slot of the width `d = span // (ld + 1)`. The midpoint of slot `j` is at `(j + 1) * d` from
  the start of the span and its position is interpolated between the two anchors at `t = (j + 1) / (ld + 1)`. Every
  subanchor is inside its gap. The starts of the beads are unique whenever the span is at least `ld + 1` bp.

  Two settings act on a pair of overlapping anchors.

    - `overlap_anchor_strict`, default no. Off, the span is `abs(start_b - end_a)` and the slots cover the overlap.
      On, the span is `max(start_b - end_a, 0)` and the subanchors between two overlapping anchors are at one point.
    - `drop_zero_length_subanchors`, default no. On, a subanchor with `start == end` is left out of the output. The
      chain keeps it.

- **Dynamic loop density.** `use_dynamic_loop_density = yes` chooses the subanchor count of each gap by its span,
  `n = round(span / target_bp_per_subanchor) - 1`, held between `min_subanchors_per_arc` and
  `max_subanchors_per_arc`. A gap with `n` subanchors has `n + 1` chain segments of about the target length. A gap
  shorter than 1.5 targets gets no subanchor and its two anchors are bonded directly. Off, every gap gets
  `loop_density` subanchors and the genomic length per bead differs by more than a hundred times between a short gap
  and a long one. The code is `_subanchor_counts` in [pipeline/ib/densify.py](../gnome3d/pipeline/ib/densify.py).

  The contact heatmap of the subanchors uses the same counts, in
  [pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py). A gap without subanchors adds one half of its
  contacts to the bin of each of its two anchors.

## Features and their measurements

- **Epigenome energy term** — A/B compartments.
  Ported from MultiMM (`add_compartment_blocks`).  One flag, default off: `use_compartments`,
  under `[compartments]`. New `[data]` keys `compartments` and `phasing_track`;
  `ContactData.from_dataframes` gains the matching frames.

  Divergences worth knowing:
  - **The attractive term is written shifted and non-negative.** MultiMM writes it as a
    negative energy, which 3dgnome cannot use: the Metropolis rule divides by the running
    score and is guarded on `score > 0`, so a negative-definite term would silently disable the
    temperature branch. The shift changes an additive constant, not the minimum or the gradient.
  - **The pairwise affinity is divided by `N - 1`.** Not in MultiMM. Without it the term's
    strength grows with region size, because it sums over all partners while springs act per
    bond, and a weight tuned on a small region collapses a large one.
  - **Compartment input is source-agnostic.** CALDER2 reads Juicer `.hic` only, and this repo's
    Hi-C is 4DN mcool, so a CALDER BED is one supported format rather than the required source.
    A signed eigenvector track is the recommended input. Eigenvector sign is arbitrary, so
    phasing is mandatory and explicit; an unphaseable chromosome is left unassigned rather than
    segregated backwards.
  - **The JAX smooth kernel carries the affinity term.**
    `mc_smooth_jax` and `mc_smooth_jax_batch` implement the compartment term, agreeing with
    numba on initial energy to 1.2e-07 relative. `use_aff` is a static cache-key entry rather
    than a weight gate, because the term costs an extra O(N) pass per step and making it
    structural keeps that off runs that do not use it. The score rides the excluded-volume
    accumulator: both terms are pairwise, double counted and share the factor-2 delta, so
    the sum is exact for both the Metropolis ratio and the final score, and only the per-term
    breakdown is lost.
  - **The estimate-dist dry pass excludes it.** It is attractive, so including it would
    shrink the estimated distances that become the heat target and the real smooth pass would
    then compact against an already-compacted target.

  The term is purely attractive and needs excluded volume or confinement enabled alongside
  it, both of which also default off. See `design/ab-compartments.md`.


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
      tuned config in `gnome3d/production.py` sets 0.1)
    - `exclusion_skip_neighbors` (skip pairs with `|i-j|` ≤ this; default 1, i.e. skip bonded)

  Ini key names under `[excluded_volume]`: `use_excluded_volume`, `weight`, `apply_to_arcs`, `apply_to_smooth`, `apply_to_heatmap`, `apply_to_ib`, `skip_neighbors`, `radius_arcs`, `radius_smooth`, `radius_heatmap`, `radius_ib`, `auto_factor_arcs`, `auto_factor_smooth`, `auto_factor_heatmap`, `auto_factor_ib`.

- **IB-level MC pass** — `settings.use_ib_mc = true` to enable. ([pipeline/coarse/build.py::ib_mc_refine](../gnome3d/pipeline/coarse/build.py))
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

  The packing factor has a physical floor. The sphere diameter is `2 × pf × N^(1/3)` chain bonds, so below `pf ≈ 0.58` a segment of fewer than `(1/(2·pf))³` blocks is asked to fold into a sphere narrower than one of its own bonds, and the layout is then set by where EV and confinement jam rather than by genomic separation. Measured on GM12878 chr1 the block-layout distance exponent is 0.021 at 0.15, 0.214 at 0.75 and 0.297 at 1.0 against 0.285 from the cell line's own Hi-C contact-probability curve (`playground/ps_curve.py`, `playground/ib_confine_ablate.py`).

- **JAX/CUDA backend** — selected per stage via `mc_executor_<stage> = batch`. ([gnome3d/mc/jax/](../gnome3d/mc/jax/))

  Routes selected MC levels to a JAX/CUDA kernel instead of the default numba CPU implementation. Measured ~2× total speedup on chr22 dryrun (21 min → 10:26), peak ~6× per-kernel on the largest smooth-MC call (N=10116: 570s numba → 100s JAX). The win compounds with chromosome size since smooth-MC is ~90% of total wall time.

  Architecture: `gnome3d/mc/numba/` holds the production numba implementations (`mc_heatmap_numba`, `mc_arcs_numba`, `mc_smooth_numba`, `mc_ib_numba`) and `gnome3d/mc/jax/` the JAX peers (`mc_smooth_jax`, `mc_smooth_jax_batch`, `mc_arcs_jax`, `mc_arcs_jax_batch`). There is no dispatcher module: the backend follows from the executor strategy resolved in [reconstruct.py::pick_executor](../gnome3d/reconstruct.py), so `batch` means JAX and `serial`/`threaded` mean numba.

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

  Install: `pip install gnome3d[jax] "jax[cuda12]"` (NVIDIA) or `"jax[rocm6]"` (AMD). Without JAX installed, the dispatcher raises a clear error if `mc_backend="jax"` is set; otherwise it never imports JAX. The `[jax]` extras dep in `pyproject.toml` is **optional** — base install is numba-only.

  Implementation notes:
  - **Per-shape XLA compile cost** (~1–60s per (N, K, n_anchors) combo) is paid once per machine via the persistent compile cache at `~/.cache/gnome3d/jax` (override with `GNOME3D_JAX_CACHE`).
  - **Convergence loop runs on-device** via `lax.while_loop` — one JAX call drives the full annealing, no Python sync between batches.
  - **Float32 throughout the JAX path** — bench showed f64 is 2× slower on consumer GPUs (1/32 throughput) with no quality benefit at production run lengths.
  - **The smooth kernel accepts a move on the summed local delta, not on `score_new < score`
    between two running totals.** On a whole chromosome block with the orientation term at its
    production weight the float32 totals are large enough that one unit in the last place
    exceeds the gain of resolving a shallow overlap, so such moves read as no change and were
    refused, and the stage plateaued early: with the coil start and the wall the JAX executor
    left 254 overlapping pairs per thousand beads on GM12878 chr1:1-60 Mb against 103 on numba,
    the same profile scaled by 2.4, while the two kernels agreed on any block whose totals
    stayed small. Deciding on the delta, 2026-09-12, brings JAX to 110 with numba's Hi-C. The
    totals are still carried for the Metropolis ratio and the plateau test. The JAX parity dump
    changed with it, by design.
  - **The smooth kernel prefetches: `[simulation_arcs_smooth] prefetch`, default 1, production
    32 since 2026-09-19.** Measured on GM12878 chr1:1-8 Mb, 93 percent of the stage's rounds
    accept under one percent of their proposals, and a step on the GPU costs its latency
    whatever the arithmetic. Above 1 a step evaluates that many proposals against the current
    state in one vectorised pass, applies the first accepted in draw order, counts only the
    proposals up to it as consumed and draws the rest afresh, so a round still delivers its
    serial step count and every applied move is a Metropolis move from the state it was drawn
    on; the chain has the serial law and the temperature is carried per chain. At 1 the body
    and its draws are the serial ones, byte identical to before. Smooth wall on the RTX 4060 Ti:
    chr1:1-8 Mb 57 s to 6 s at 32, chr1:1-60 Mb 420 s to 125 s at 32 and slower at 64, since
    the kernel's full excluded volume scan is paid once per proposal. Three structures on 60 Mb
    at 32 against 1: every battery number level, MultiMM 0.665 against 0.669. Numba ignores the
    key. Unit checks in `tests/unit/test_smooth_levers.py`, sweep in `slurm/ensemble/prefetch_sweep.sh`,
    reading and numbers in `design/parallel-mc-and-nn-reading.md`.
  - **The excluded volume and the wall run on a cell grid in the JAX kernel:
    `[simulation_arcs_smooth] jax_grid`, default no, production yes since 2026-09-19, on
    launches of `jax_grid_min_beads` (4096) beads or more.** Cells at least `r0` wide on up to
    48 per axis, capacity twice the fullest starting cell plus four, a table rebuilt every
    period of about 2,048 proposals and never written between rebuilds: a bead an accepted
    move carries away is flagged and listed, and a query takes listed beads from the list at
    their current position and skips them in the cells, so the sum is exact. Bit identical to
    the full scan on the GPU. Smooth wall on GM12878 chr1:1-60 Mb at prefetch 32: 116 s to 47
    to 85 s, three structures level with the full scan on every battery number. Two designs
    lost on the way and are recorded in `design/parallel-mc-and-nn-reading.md`: a per move
    relink is a batched scatter under the chain vmap that XLA does out of place and copied the
    table every step, an hour where the scan took two minutes; a sorted order with a binary
    search per cell was slower and a finer axis was slower, since the rebuild writes the table.
    Past 32 proposals a batch gets slower, so the floor is now the number of kernel launches in
    one step. The check needs a GPU, `tests/unit/test_jax_grid.py`.
  - **Two host and device costs the profile of 2026-09-20 removed, both byte identical by the
    parity gate on GM12878 chr1:1-60 Mb.** The orientation term scored a trial move by writing
    the anchor's new vector into a copy of the whole orientation array, and under the proposal
    vmap XLA materialised that as a broadcast of the array to one copy per proposal, 82 MB a
    batch on a chromosome and half of every batch's time; the trial score now takes the vector
    as an argument and the array is never copied. And the launch's initial energies were
    computed one chain at a time, each a scan of the chain's beads with a host sync between
    chains, a minute a structure on a trio chromosome; they are one vectorised call per term
    now. Kernel on chr1:1-60 Mb, 11 chains, 49.0 s to 35.5 s and the preparation before it 6 s
    to 1 s on the RTX 4060 Ti. Numbers and the per batch composition in
    `design/kernel-performance.md`.
  - **`cli.py` auto-forces `ib_workers=1` when `mc_backend=jax`** — multiple Python threads contending for a single GPU is net-negative; restarts go inside JAX via `mc_smooth_chains` (vmap), not via thread pools.
  - **Several processes on one card: `XLA_PYTHON_CLIENT_MEM_FRACTION`, JAX's own variable, no
    setting.** ([mc/jax/share.py](../gnome3d/mc/jax/share.py)) JAX preallocates 75 percent of the
    card when its backend starts. Measured 2026-11-12 on the RTX 4060 Ti with JAX 0.11: a second
    process on a held card gets a smaller pool than it asked for, `memory_stats()` still reports
    the full 11.7 GiB, the kernels size their launches from that and XLA aborts the process with
    a failed check in the allocator. Before the backend starts, `configure_gpu_pool` reads the
    free memory with nvidia-smi. On a free card it changes nothing. On a held card it sets the
    pool to the free memory less 0.5 GiB for the CUDA context and logs it, and refuses under
    1 GiB. A fraction set by hand that does not fit the free memory is refused with the numbers.
    Several instances that share a card on purpose set the variable in each, adding up to less
    than 1. A process with a smaller pool packs its launches differently, so its structures are
    not bit identical to a run on a free card; at a fixed pool it reproduces. Unit checks in
    `tests/unit/test_gpu_share.py`, two instances on one card in `playground/gpu_share_run.py`.
  - **The smooth stage is threaded numba in production since 2026-11-16, and `auto` picks it
    from four workers.** Measured on 36 cores against an RTX 3090 on GM12878 chr1: the stage of
    52 blocks takes 53 to 80 s threaded, 92 s on 8 workers, and 122 to 146 s on the device, and
    a 10 Mb region of two structures 35 s against 60 s. The relaxation pass follows it onto
    numba. The arcs solve stays on the device, 20 s against 106 s. Matched seeds are level on
    the pooled and battery numbers. The L-BFGS-B calls hold BLAS at up to eight threads,
    `util.blas_threads`, since OpenBLAS on every core spent a 5 Mb region's fit waking them.
    `design/kernel-performance.md`, section 14. Unit checks in `tests/unit/test_executor_auto.py`.
  - **Lazy import + thread-safe init** — `mc_jax` module loads without importing JAX; the first call to a JAX-backed entry triggers a one-time banner on stderr (`[mc_jax] JAX backend ready: backend=gpu devices=[...]`).

  **Kernel seeding is process-stable.** Every JAX kernel takes its PRNG offset from
  `util.stable_seed_offset(log.current(), ...)`, a blake2b digest of the active scope path.
  The scope path separates concurrent kernels. `PYTHONHASHSEED` no longer affects results,
  so comparing two runs needs no environment override.

  **The batched smooth kernel seeds each chain from its own seed.** `mc_smooth_jax_batch`
  takes its launch key from the scope alone and folds each problem's `seed` per chain, rather
  than taking the key from `problems[0]` and splitting it by slot. Every caller supplies one:
  the smooth stage from the node, and the estimate stage from the parent block and the
  replicate number, in `estimate_dist._expand_replicates`. A replicate carried no seed before
  and would have fallen back to its slot. A chain's stream therefore
  does not depend on how many chains share the launch, on where it sits among them, or on
  which sub-batch it lands in. That is what lets the grouping change without changing the
  algorithm. The arcs kernel and the multi-chain restart kernel still split by slot, which is
  correct for them: their widths come from settings rather than from grouping.

  What remains is arithmetic, not algorithm. XLA vectorises across the chain axis, so a
  reduction does not associate the same way at every launch width, and the first difference is
  one unit in the last place of the float32 score. Monte Carlo amplifies that into a different
  structure, so a run reproduces at a fixed configuration but structures cannot be compared bit
  for bit across launch widths. `tests/unit/test_batch_seeding.py` compares at equal width for
  exactly this reason, and measures the width effect separately.

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

- **Merged smooth launches: `[simulation_backend] merge_smooth_launches = yes`, default yes.**
  ([pipeline/ib/smooth.py](../gnome3d/pipeline/ib/smooth.py) `batch_key`,
  [mc/jax/smooth.py](../gnome3d/mc/jax/smooth.py) `_chunk_plan`)
  `SmoothStage.batch_key` was `(heat, orn, comp, bead bucket)`, so a set of interaction
  blocks agreeing on every energy term was still split into one launch per bead bucket. Cost per
  step in the batched kernel is flat in the launch width, so that split bought nothing. Measured
  on a real chr1 GM12878 run, 55.8 percent of smooth time sat in launches of four blocks or
  fewer at 4.8 to 22 us per step per block, against 0.18 to 0.69 in the sixty four wide
  `estimate_dist` launches in the same log.

  With the flag the bucket leaves the key and `mc_smooth_jax_batch` packs the group into as few
  launches as device memory allows, largest first so a launch's shape is fixed by its first
  member. Memory is the only reason to split: the heat target is one `(B, B)` float32 per chain,
  and merging every heat carrying group of one real dispatch would need 25.8 GB on a 16 GB card.
  Modelled on two real runs the merge is 3.7x on GM12878 chr1 and 3.5x on H1ESC chr1.

  Two changes make the merge safe, both unconditional because neither is a feature. Each chain
  seeds from its own seed rather than its slot, so regrouping does not change the draw. And a
  converged chain holds its state while the rest of the launch runs on, so a small block ends
  where it would have ended alone instead of annealing for as long as the largest block needs.

  Both change results once, on every JAX smooth run, and no flag restores the old streams.
  `merge_smooth_launches = no` restores the old grouping only.

  A merged dispatch is one or two groups where it used to be five to nine, so the multi GPU
  `groups` mode has fewer groups to shard. The wide groups it produces are what `within` needs,
  so the two changes point at different modes. Nothing here re-tunes that choice.

- **Boundary stitch: `[boundary_stitch] use_boundary_stitch = yes`, default no, production no since 2026-09-19.** Off at chromosome scope, kept for block scope, see the relaxation entry.
  ([gnome3d/pipeline/stitch.py](../gnome3d/pipeline/stitch.py))
  The per block chains place anchors only through their own block, so the last anchor of one
  block and the first anchor of the next have no term coupling them. Measured on GM12878
  chr1:1-60 Mb at matched separation of 562 kb to 1 Mb, an interior pair sits at 0.092 of
  `genomic_length_to_distance` and a boundary pair at 5.42, a factor of 59. This pass runs after
  every chain of a chromosome is done, in `reconstruct.py::_assemble`, and moves each block as a
  rigid body so every boundary pair sits at the distance the structure's own interior pairs
  realise at that separation. Rigid means the arcs and smooth results inside a block are
  untouched.

  Energy is a two sided spring per boundary plus a soft excluded volume between block
  centroids with a radius per pair of the two blocks' radii of gyration added, or one constant
  radius for every pair when `exclusion_radius_ib` is positive, minimised with L-BFGS-B over one
  rotation and one translation per block. The radius came from `genomic_length_to_distance` of
  the centroid gap at first, 153 units against block radii near 8 on chr1:1-60 Mb, which made
  the term a compaction penalty instead of an overlap guard. There is no chain bond and no confinement. Keys: `spring_weight` (1.0), `ev_weight`
  (1.0), `max_iter` (2000, which on a trio chr1 of 1,494 blocks costs 85 seconds and reaches a
  worst boundary of 1.32 times the curve, against 6.0 at 500 and nothing further at 5000). The pass uses no RNG and runs on the calling thread, so flag off is
  byte exact and flag on reproduces. Unit checks in `tests/unit/test_stitch.py`.

  **The energy carries its own gradient, and it has to.** Six variables per block puts a real
  chromosome in the thousands of dimensions, 8,964 for the 1,494 blocks of a trio chr1, where a
  finite difference gradient costs one evaluation per variable and scipy's default budget of
  15,000 evaluations buys about one step. The pass ran and moved almost nothing, which showed up
  as boundaries realised at up to 80 times what the structure's own interior realises at that
  separation, the visible strands between blocks. Validation had used an 11 block region, 66
  variables, where the default budget was 220 gradients and the pass converged. The gradient is
  Gallego and Yezzi's for the rotation derivative, chained through the boundary springs and the
  centroid excluded volume, and `maxfun` is set from `max_iter` so the budget cannot bind first.
  Replayed on a finished trio chr1 the worst boundary falls from 79.6 to 1.32 times the curve
  and the longest from 932 to 13 model units. `playground/restitch_model.py` runs both end of
  run passes on a finished cif, recovering block membership from the densification rule, so a
  change to either can be measured without paying for a reconstruction.

- **The distance law.** ([gnome3d/polymer.py](../gnome3d/polymer.py), `Settings.polymer_law`,
  [pipeline/coarse/heatmap.py](../gnome3d/pipeline/coarse/heatmap.py), [data.py](../gnome3d/data.py))
  One law sets every distance, in bead units, one bead being the distance at
  `target_bp_per_subanchor`. A pair with no contact sits at `max(1, (s / s0) ^ nu)`. A loop of
  strength `q` pulls its pair to `1 + (background - 1) / (1 + q / q_half)`, so a saturated loop
  sits at one bead and never inside; `q` is the PET count over the typical count at that span
  fitted on the run's own arcs. A heatmap cell sits at the background times observed over
  expected to the minus third, the expectation taken within the heatmap at that separation,
  and at the chromosome level, where pairs have no separation, at the mean chromosome span.
  `nu` is measured at load from the singletons every run reads, on the whole chromosome set
  before the region filter. The fit refuses what is not a polymer decay and says so in a
  STATUS line with the reason, then uses 0.285 as a named fallback; every ChIA-PET singletons
  file is refused, since it is enrichment filtered around CTCF, and every Hi-C file is
  accepted. A `Settings` with no law attached builds one from `polymer_exponent` or the
  fallback and warns once, so no distance call ever answers from a constant silently. Two keys,
  `polymer_exponent` (0 measures, a positive value pins) and `contact_half_saturation` (1.0).

  It replaced, on 2026-09-06, the reference's chain law, PET law and heatmap frequency law and
  two later patches on the PET law: fifteen constants copied from one config with no derivation
  and byte identical across cell lines, three absolute scales for one quantity in a unit with no
  meaning, and a derived exponent that was a mean over three of our own Hi-C files typed in as
  a default. Fitted on the inputs we run that exponent is 0.333, 0.299 and 0.298 for GM12878, H1ESC
  and HFFC6 on the 25 kb files, and 0.343 on the deep GM12878 map. Earlier figures of 0.275,
  0.299, 0.192 and 0.072 came from a fit that counted rows per separation rather than their
  counts, fixed 2026-09-08; the fit is contact probability per grid separation now. Measured against the old laws on GM12878, H1ESC and
  HFFC6, chr1:1-60 Mb, five structures each, raw contact maps: Pearson, Spearman and SCC up on
  every cell, MultiMM level, anchor overlaps down 96 to 98 percent, subanchor overlaps halved,
  cross block overlaps down four to ten times. The realised exponent runs 1.3 to 1.5 times the
  measured input on every cell, which one correction on the background exponent would absorb;
  not built. The old keys are not read and warn if present. Unit checks in
  `tests/unit/test_polymer.py` and `tests/unit/test_arc_matrix.py`; `playground/ps_from_singletons.py`
  fits the exponent on a file by hand. See `design/anchor-placement.md`, option H.

  **Tried and rejected, 2026-09-06: every arcless pair on the background.** The realised
  exponent overshoot is a kink, not a slope. By band the polymer arms were 0.09 to 0.20 under
  100 kb and 0.34 to 0.48 above, against inputs of 0.20 to 0.30, since loops pull pairs to
  touching and nothing holds the arcless pairs between them. The obvious fix, a weak spring
  holding every arcless pair at the background in place of the `1/d` repulsion, won a sweep on
  eight real blocks (short band onto the input, overlaps 101 to 27 per thousand) and then lost
  the three cell battery on every Hi-C statistic: GM12878 Pearson 0.462 to 0.329 and SCC 0.191
  to 0.130, H1ESC 0.258 to 0.119, HFFC6 0.267 to 0.181, with Rg halved and overlaps up. The
  kink went, but everything came out flatter than the input. The reason is embedding: an all
  pairs spring network asking every pair for `s^nu` with `nu` below a third cannot be realised
  in three dimensions, so the least squares compromise is a mean field blob. The `1/d` never
  constrained an arcless pair, which is why it did not hit this. The sweep could not see it
  because every isolated block came out at Rg 3.8 whatever the weight. Reverted the same day;
  the sweep is kept as `playground/background_weight_sweep.py` for the record and the battery
  reports the two band exponents from here on. Any future fix for the kink has to be short
  range only, holding pairs under about 100 kb and leaving the long range free.

  **Built as that, opt in, under measurement: `[springs] background_weight` (0, off) and
  `background_range_bp` (100000).** An arcless anchor pair inside the range carries minus the
  background in the arcs matrix and the kernels score it as the log symmetric spring; every
  other arcless pair carries `-0.5` and keeps the truncated repulsion. A background is never
  under one bead, so sign and magnitude tell the two kinds apart without a second matrix. The
  weight sits in every kernel beside the cutoff rather than replacing it, so weight zero is
  byte exact against the previous commit. JAX matches numba on a matrix carrying both arcless
  kinds, and the batched driver runs with the background on, both in
  `tests/unit/test_arc_matrix.py`. Swept in `playground/short_range_sweep.py`. Production since
  2026-09-07 at weight 0.1: with it the anchors sit on the input curve on all three cells,
  within blocks and across them. Weights 0.3 and 1.0 pull the closest pairs closer and cost
  Hi-C at every step.

- **Loop stiffness by strength: `[springs] arc_weight_exponent`, default 0, measured
  2026-09-22 and left at 0.** ([pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py)
  `arc_weight_matrix`, [mc/numba/arcs_solver.py](../gnome3d/mc/numba/arcs_solver.py),
  [mc/jax/arcs_energy.py](../gnome3d/mc/jax/arcs_energy.py), the arc terms of
  [mc/numba/terms.py](../gnome3d/mc/numba/terms.py))
  Every loop pulls with the same spring and only its target differs, so once two targets sit
  near one bead the stronger loop has nothing to win a competition with. Measured on the trio
  chromosomes, the strongest third of loops is realised worst, at 0.35 to 0.49 against 0.7 to
  0.8 for the rest, and the loss follows how many loops share an anchor. With the exponent
  above zero each arc pair carries its strength to that power as a weight on its spring
  constant, in the solver's energy on both backends and in the numba annealer's arc term; the
  targets, the background springs and the repulsion never read it. The weight matrix rides
  beside the target matrix as float32 and is None at zero, so the kernels take the path they
  took. The JAX annealer does not carry it and refuses a weighted problem rather than ignore
  the weights. Unit checks in `tests/unit/test_arcs_solver.py`.

  Measured at exponent 1 on the nine trio chr1 ensembles on the HGSVC map, ten conformations
  each, against the same arm at zero: target against realised over all loops 0.55 to 0.67 from
  0.54 to 0.66, within a span band up 0.01 to 0.06, the median realised over target 1.26 to 1.63
  from 1.29 to 1.65, and the strongest third level at 0.33 to 0.49 from 0.35 to 0.49 while the
  weak and middle thirds lose 0.005 to 0.02; every expression statistic level. A strong loop's
  competition is with the background springs and the repulsion among its anchor's partners,
  which the weight does not touch, so a stronger spring does not let a crowded loop close.
  Numbers in the triosformer project's tracker, `design/expression-from-structure.md` there,
  idea 37.

- **Loop dropout per conformation: `[springs] loop_dropout`, default no, under measurement
  since 2026-09-23.** ([pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py)
  `dropout_keep`, `_active_arcs`)
  A PET count is a frequency across cells, and every conformation of an ensemble solves the
  same targets from a different start, so a loop's count reaches the ensemble only through
  its target distance, which saturates. With the flag each conformation keeps a loop with
  probability `q / (q + loop_dropout_scale)`, `q` the law's strength, and a dropped loop
  leaves its pair arcless for that conformation, so the ensemble mean carries the loop's
  frequency and a person's count of a shared loop reaches the mean structure. The draw comes
  from a generator seeded by the conformation, the joint solve's seed or the block's, apart
  from every other stream, so a run reproduces and the gate holds with the flag off. Kept
  loops keep the law's target; the anchor set, the densification and the orientation term
  never read the draw. Measured before it was built, triosformer's tracker idea 43: its 2D form puts the shared loops'
  strength at +0.03 to +0.06 of the loops' +0.10 person specific part, which bounds what it
  can carry. Unit checks in `tests/unit/test_arcs_solver.py`.

  **Kept loops at contact: `[springs] loop_contact`, default no, under measurement since
  2026-10-27.** With it a loop a conformation keeps is held at one bead, touching, instead of
  the law's part way target, so with the dropout a loop's strength sets how often its anchors
  are in contact rather than how far they are pulled. Measured by triosformer's loop bench
  before it was built: a CTCF loop of 3 to 4 PETs asked for two thirds of the background in
  every conformation sits closer than a control pair on the mean and meets its anchors 0.60
  times as often within 1.5 beads, since the spring holds it at four to five beads and the
  control reaches contact by its spread. Off is byte exact. Unit checks in
  `tests/unit/test_arc_matrix.py`.

- **Transcription factories: `[factories] weight`, default 0, with `radius` and
  `[data] anchor_activity`, under measurement since 2026-09-23.**
  ([mc/numba/arcs_solver.py](../gnome3d/mc/numba/arcs_solver.py) `factory_energy_grad`,
  [mc/jax/arcs_energy.py](../gnome3d/mc/jax/arcs_energy.py) `DeviceFactoryEnergy`,
  [pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py) `calc_anchor_activity`)
  RNAPOL2 loops enter as pairwise springs like CTCF's, with no many body form. With the
  weight above zero each anchor carries an activity, the largest value of the track's
  intervals overlapping it, a broadPeak's signal or a bedGraph's value, and an active anchor
  is pulled toward the active anchors near it through `w a_i (log(1 + A) - log(1 + S_i))`
  with `S_i = sum_j a_j exp(-d_ij / r)` and `A` the total activity: a bead gains from joining
  one group and little from a second, and the term is never below zero, which the Metropolis
  rule needs. It is in the solver's energy on both backends, the gradient built in two passes
  since an anchor's move changes every group it is in; the annealers refuse it rather than
  ignore it, since the term needs a group sum kept per anchor that the local scorer does not
  have. Weight zero is byte exact. Unit checks in `tests/unit/test_arcs_solver.py`.

- **Arcs confinement radius from the law: `[confinement] packing_factor_arcs = 0`, default 1.5.**
  ([pipeline/ib/arcs.py](../gnome3d/pipeline/ib/arcs.py) `settings_for_block`,
  [polymer.py](../gnome3d/polymer.py) `radius_of_gyration`, `confinement_radius`)
  Above 100 kb nothing holds an arcless anchor pair but the block's confinement sphere, and its
  radius was the packing factor times the mean arc target times the cube root of the anchor
  count, a copied constant on a formula with no derivation. Measured on three cells, blocks
  under 200 kb came out the size the law says and blocks over 2 Mb two to three times it, and
  every pair beyond 100 kb ran away in proportion, 2.3 times the law at 1.5 Mb.

  At factor zero the radius is derived: a chain of span S whose pairs follow the law has a
  radius of gyration of `S^nu / sqrt(2 (2 nu + 1)(nu + 1))`, the Gaussian chain's N over 6 at
  nu one half, and a uniform sphere with that radius of gyration has radius root five thirds of
  it. The one assumption is that a confined block fills its sphere evenly. The value reaches the
  annealer, the solver and the JAX kernel as an explicit radius on a per block copy of the
  settings. The batch executor runs a launch on one settings and refuses rather than run every
  block at the first block's radius. Unit checks in `tests/unit/test_confinement_radius.py` and
  `tests/unit/test_polymer.py`.

  Production since 2026-09-07 as the principled form. Alone it moved nothing, the old formula
  already landed near it. What inflates a block is the arcless repulsion's reach; cutting that
  puts the long range on the law and the blocks in a pile, Hi-C down and cross block overlaps
  up nine times, which was the thin input: with the deep maps and the contact background the reach is
  1.5 in production, see that entry.

- **The smooth step and the subanchor heat term, production 2026-09-07.** `noise_smooth` is one
  bond per proposal, not the reference's five. At five the bonds never settle and come out 1.2
  to 1.4 times their target, the chain folds and the distance curve is flat under 100 kb; at
  one they sit at 1.05 and the curve follows the law from 10 to 100 kb, on H1ESC Hi-C Pearson
  0.244 to 0.338. `use_subanchor_heatmap` is off: inert on shallow Hi-C, where the pass is
  skipped for lack of active pairs, and on deep Hi-C it pulls bonds to 0.83 of target once
  the step lets them settle, GM12878 Pearson 0.476 without it against 0.435 with, at a fifth of
  the wall. Three cell battery: Pearson and SCC up on every cell, Spearman level, MultiMM down
  8 to 14 percent. See `design/anchor-placement.md`.

- **Contact data for arcless pairs beyond the short range: `[springs] use_contact_background`,
  default no.** ([pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py)
  `add_contact_background`)
  On the tree's own interaction blocks, which run until a position no arc covers and are 9 on
  H1ESC chr1:1-60 Mb at a median of 2.65 Mb, pairs up to 2 Mb apart are within one block 93
  percent of the time, so the range above 100 kb belongs to the arcs stage. An arcless anchor
  pair there had nothing behind its placement but the repulsion's reach and the sphere, and
  compacting those pairs without data scrambled the block's own contact map. With the flag the
  anchor level contact map, the run's singletons binned by anchor, is converted with the law as
  every heatmap is, and an arcless pair beyond the range whose cell puts it under the background
  carries minus that distance, which the kernels hold with the background spring. A pair at or
  below its expected contact keeps the repulsion, so the held set stays sparse; holding every
  pair at a power law could not be embedded and was rejected. No kernel change. Unit checks in
  `tests/unit/test_arc_matrix.py`.

  Production since 2026-09-08 together with `arcs_repulsion_cutoff_factor` 1.5, on the deep
  4DN maps. The files the project had used held 7.8 million, 104 million and 193 million
  contacts genome wide, so the term held under 1 percent of the far pairs and did nothing;
  `legacy/validation/manifests/<CELL>_hic.json` now fetch the 2.5 to 4.0 billion contact maps and
  the shallow ones are kept as `<CELL>_hic_shallow.json` for the record.
  With those as input, the term on and the reach at 1.5, the structures follow the law from
  20 kb to 2 Mb within a quarter on three cells, where before every pair beyond 100 kb ran
  away by 1.3 to 1.8 times; Hi-C against the deep maps is up on GM12878 and HFFC6 and a shade
  down on H1ESC. Reach 1.0 compacts past the law and doubles cross block overlaps. On a thin
  map the term holds next to nothing and that is allowed: no input has to be deep, and the
  reach at 1.5 then behaves as it did on the thin input. Anchors are about 13 kb wide against
  25 kb pixels, so even a dense map reaches only about 4 percent of the far pairs through
  the singletons; a denser anchor level map is a data path change and is not built.

  **Tried and dropped, 2026-09-19: the map itself as the background's source.** The run's
  singletons are thinned 230 fold and binned to anchors that hold few bin centres, so they hold
  no pair beyond 100 kb, while the raw 25 kb map has 70 million contacts in GM12878 chr1:1-60 Mb
  and a quarter of the 0.5 to 2 Mb anchor pairs and a tenth of the 2 to 10 Mb pairs three
  Poisson standard deviations over the expectation at their separation. Reading each pair's
  pixel from the mcool and holding the significant ones at the law's contact distance was
  measured over fourteen arms, thresholds 2 to 12 sigma, pooled and not, symmetric with pairs
  under expected held out, at a tenth to full pull, and across blocks only: SCC 0.374 to 0.39
  or 0.40 and Pearson 0.312 to up to 0.346, the first compartment eigenvector correlation of
  ours above 0.1, and on every arm Rg down 15 to 20 percent, within block overlaps two to four
  times and cross block four to eight times production, the saddle down. A held pair can only
  pull in, since beyond a few megabases the expectation is under a count and nothing is ever
  significantly below it; the within block pairs carry the correlation and the across block
  pairs the shrinkage, and weakening the pull loses both together. Removed the same day.
  Tables in `design/validation-2026-09.md`; `playground/far_pairs_from_map.py` measures what a
  map holds.

- **A contact term in the block layout: `[simulation_ib] heatmap_weight`, default 0.**
  ([pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py) `block_heatmap_distances`,
  [mc/numba/ib.py](../gnome3d/mc/numba/ib.py))
  The blocks of a segment were laid out by chain bonds, excluded volume and a sphere, with no
  contact data at all. The run's contacts are binned by block, the bins meeting halfway between
  neighbours, normalised and converted with the law the way the segment heatmap is, and the
  block kernel scores the matrix as its heat term. Weight zero is byte exact. Unit checks in
  `tests/unit/test_ib_heatmap.py`, which also record that at the defaults' `max_temp_ib` of 20 the
  layout never cools and ends with bonds nine times their target, while production's 5 settles.

- **Cross block relaxation: `[relax] use_cross_block_relax = yes`, default no, production yes since 2026-11-16.**
  Production with the arcless repulsion off, which tripled cross block overlaps on a chromosome;
  the pass with `local_window` 1 takes GM12878 chr1 from 62 to 14 per thousand. The paragraph
  below is the measurement that kept it off from 2026-09-19 with the repulsion on. With
  `within_blocks` and `ev_radius` 1.0, production since 2026-11-16, overlapping pairs inside a block
  move too: on CTCF only inputs the coils of two gaps whose anchors sit close met 240 to 300 times
  per thousand beads, and GM12878 chr1 goes from 246 to 37 within blocks at level balanced SCC.
  Both end passes were measured at chromosome scope on GM12878 chr1:1-60 Mb, three structures
  per arm: Hi-C level on all four arms, the joint solve already holding boundary pairs at a
  median of 1.03 times the curve and none over 2.34, the stitch adding a tenth of Rg through its
  centroid excluded volume and 17 cross block overlaps per thousand that the relaxation then
  removed. Both are kept, off, because a chromosome too large for one joint solve is solved by
  block, and there they are what places one block against the next. `design/validation-2026-09.md`.
  ([gnome3d/pipeline/relax.py](../gnome3d/pipeline/relax.py))
  The smooth stage's excluded volume acts within one block and the stitch guards block
  centroids only, so once blocks are stitched together nothing acts between their beads and two
  coils pass through each other. On GM12878 chr1:1-60 Mb the arm that agrees best with Hi-C
  had 11.9 percent of its beads within one bond of a bead from another block, four times the
  parity value. This pass runs the smooth kernel once over the whole chromosome, numba or JAX
  by `mc_executor_smooth`, with excluded volume on every pair and every anchor held fixed, so
  the arcs and the stitch are kept and only the subanchor coils re route. Bond targets are the
  bonds as they are.

  Three things it needs to work, each learned on a toy. The excluded volume acts at 1.5 bonds by
  default so that at equilibrium nothing is left under one bond, where contacts are counted. The
  chain springs carry their own weight, `bond_weight` (10), since at the smooth stage's 0.1 the
  excluded volume tears the coil instead of re routing it. And a little temperature, `temp`
  (0.1 of `max_temp_smooth`), because untangling two coils needs a bead to cross a neighbour's
  shell and a greedy pass stalls with contacts left. Keys: `ev_weight` (10), `ev_radius` (0 for
  1.5 bonds), `noise` (0.5 bonds). Runs after the stitch in `reconstruct.py::_assemble`. The
  gate is `cross_block_contacts`. Unit checks in `tests/unit/test_relax.py`.

  **`[relax] local_window` (default -1, off) is the better fix.** The round count is proportional
  to how many beads may move, measured on a real 129,457 bead chromosome at 19 rounds for 159
  movable, 104 for 1,177 and 850 for 11,766, which extrapolates to the 6,201 a trio run took with
  all 117,660 subanchors movable. A window of `k` keeps the beads touching another block and `k`
  chain neighbours either side, so the chain can take up the slack, and freezes everything else.
  With 53 beads touching, a window of 1 leaves 159 movable and about 19 rounds instead of 6,201.
  A negative value keeps every subanchor movable, which is what the pass did before.

  It runs on the wrong kernel as well. It picks from `mc_executor_smooth`, which production sets
  to `batch`, so a pass that is one single chain runs on the JAX kernel where one chain is its
  worst case. Measured against the trio run's own numbers, 22.6 us a step there against 6.6 on
  numba with the cell grid.

- **Solving the arcs stage: `[simulation_arcs] solver = mc | lbfgs`, default `mc`, production
  `lbfgs`.** ([gnome3d/mc/numba/arcs_solver.py](../gnome3d/mc/numba/arcs_solver.py))
  The arcs landscape is a funnel. Ten Monte Carlo starts from perturbed seeds land within one
  percent of the same energy at a coefficient of variation of a third of a percent, and
  temperature makes no difference to where a run converges, so there are no basins for a
  stochastic search to escape. A quasi Newton descent on the same energy therefore reaches the
  same minimum.

  `arcs_energy_grad` is the energy `mc_arcs_numba` scores, its arc term plus its truncated
  repulsion, an excluded volume and confinement, with the cutoff and the confinement centre and
  radius derived the way that driver derives them, and its gradient. The genomic floor is not
  implemented and is refused rather than dropped. Restarts, the per anchor start noise and best
  of are unchanged, so an ensemble still comes from the perturbed starts. Keys: `solver`,
  `solver_iters` (200).

  Measured over five structures on chr1:1-60Mb the two arms agree on every quality number, Hi-C
  Pearson 0.403 against 0.405, Spearman 0.318 against 0.321, SCC 0.110 against 0.109, MultiMM
  0.331 against 0.331, distance exponent 0.240 against 0.249, and the within block anchor
  overlap rate 89.2 against 89.1 per thousand beads. The stage's two calls fell from 492s to 6s
  and from 500s to 23s, and the whole five structure run from 1h57m to 1h13m.

  It is a speed change and nothing more. The anchor overlap rate is identical to three figures,
  so a faster solver of the same energy does not touch the within block compaction that
  `design/anchor-placement.md` exists for.

  Two ways it could be asked for and silently not run, both refused. An unrecognised name is a
  `ValueError` rather than a fall through to the annealer. And the batch executor is a JAX
  annealer with no solver in it, so it raises rather than annealing; the solver needs
  `mc_executor_arcs = serial` or `threaded`, which is what `CANONICAL` sets. Unit checks in
  `tests/unit/test_arcs_solver.py`.

  **The solver's energy on the device: `mc_executor_arcs = batch` with the solver, production
  since 2026-09-20 with `solver_iters` 800.** ([gnome3d/mc/jax/arcs_energy.py](../gnome3d/mc/jax/arcs_energy.py))
  The executor setting decides where every stage runs and `batch` means JAX, so the solver
  follows it: under `batch` each block is solved in turn with its energy on the device, on its
  own settings and from its own start, and under `serial` or `threaded` on the CPU. At
  chromosome scope the solve was most of a conformation's wall and ran on the CPU while the
  GPU idled: every pair of the chromosome visited against a dense target, up to 450 ms an
  evaluation on a trio chromosome, and the 200 iteration cap binding on every solve, which
  the solver's log line now reports. The device evaluates the same energy term for term in row
  chunks in float32, the rows summed in float64 on the host, so an evaluation costs one read
  of the target matrix; L-BFGS-B stays on the host. The energy and gradient agree with the
  numba kernel to about 1e-6 relative on a block carrying every kind of pair, and a solve on
  either side reaches the same energy; the two are not byte identical, since the summation
  order differs. On the RTX 4060 Ti, GM12878 chr1, 9,195 anchors: 200 iterations 31.6 s on
  the CPU to 4.8 s, 800 iterations 104.8 s to 13.4 s at energy 6274 against 6266, and a whole
  conformation 419 s to 144 s with the cap at 800. Three structures on chr1:1-60 Mb, CPU at
  200 against device at 200 and at 800: Pearson 0.312, 0.311, 0.312, SCC 0.374, 0.366,
  0.366, MultiMM 0.666, 0.663, 0.665, Rg 23.4, 23.3, 23.2, anchor overlaps 2.8, 2.7, 2.5 per
  thousand; the saddle 1.31, 1.38, 1.00 on three structures, which is noise at that count.
  Float64 on the device would need JAX's 64 bit mode, which is process wide and changes the
  smooth kernel's integer draws and its loop carries, so it is not used. Unit checks in
  `tests/unit/test_arcs_solver.py`.

  **The solve stops on the energy: `[simulation_arcs] solver_tol`, default 0, production 1e-6
  with `solver_iters` 5000 as the safety, since 2026-09-22.** The cap bound on every trio
  chromosome at 800 and at 3,000. Traced to 8,000 iterations on a chromosome of 23,080 anchors,
  the relative improvement per iteration falls to 1e-5 by 830 and to 1e-6 by 2,400 and then
  sits near 1e-6 with no plateau; the energy at 800 is 1.7 percent above the value at 8,000, at
  2,400 under one percent. The tolerance is L-BFGS-B's own `ftol`, so at zero the options are
  unchanged and the parity gate holds. `GNOME3D_ARCS_TRACE` names a file that receives every
  evaluation's energy, which is how the value was chosen, `playground/trace_analysis.py`.

  The annealer is kept by decision, 2026-09-06, not as a leftover. The solver's justification
  is a funnel landscape measured on a few real blocks. A dataset or an energy change that
  breaks that assumption has the annealer to fall back on, and the annealer is the reference's
  own method, so it is the comparison arm for replicating results. Do not propose removing
  `solver = mc`, its schedule keys or the JAX arcs kernel.

- **Chromosome scope for the arcs stage: `[simulation_arcs] scope = block | chromosome`,
  default block, with `start = centroid | walk` and `[confinement] weight_arcs`.**
  ([gnome3d/skeleton.py](../gnome3d/skeleton.py) `joint_arcs_solve`,
  [gnome3d/pipeline/ib/arcs.py](../gnome3d/pipeline/ib/arcs.py) `walk_start`)
  A compartment is a pattern over many blocks, and at block scope nothing places one block's
  anchors relative to another's but the block layout and the stitch. At chromosome scope every
  anchor of a chromosome is solved as one problem, each block's anchors starting at the
  centroid the block layout gave it, on a walk at the law's distance per gap under `walk`, with
  the target matrix built over the whole chromosome so the loops, the contact background and
  the compartment term act across blocks, and the law's sphere for the chromosome's span held
  at `weight_arcs`. The per block arcs stage then passes its anchors through and the chains,
  the stitch and the relaxation run as at block scope.

  Measured 2026-09-10 on GM12878 chr1:1-60 Mb with the chromosome as one block, before this
  scope existed: every anchor at one point collapses the solve to Rg 14 against production's
  22, the walk start alone leaves it at 44, and the walk with the sphere at weight 10 lands at
  22.9 with Pearson 0.303 against 0.270, SCC level, MultiMM 0.572 against 0.628, cross block
  overlaps level and the compartment eigenvector correlation 0.22 against 0.05 with the term
  off. The start decides the long range structure because the arcs energy has no term between
  far pairs. Block scope, centroid start and weight zero are byte exact by the parity gate.
  Unit checks in `tests/unit/test_arcs_scope.py` and `tests/unit/test_arcs_start.py`.

  Production since 2026-09-11 at chromosome scope with no arcs sphere, the start the Hilbert
  curve until 2026-11-16 and the maxent draw since, see the long range entry below.
  The joint solve holds no arc across blocks and the contact background 80 of 3.5 million
  cross block pairs on chr1:1-60 Mb, so the start is the long range arrangement, and a
  Hilbert curve scaled to the law's bond gives it the cube root size growth the maps show
  beyond a megabase with no weight. Like for like on three cells against the walk with the
  sphere at 10: Pearson 0.291, 0.318 and 0.304 to 0.295, 0.328 and 0.310, SCC level, MultiMM
  0.674, 0.667 and 0.673 to 0.631, 0.663 and 0.661, cross block overlaps 176, 173 and 143 to
  163, 126 and 100, Rg 28, 27 and 30 to 24, 25 and 26. The walk with the sphere was
  production for one day, 2026-09-10, on the gate that follows. Three cell
  gate on chr1:1-60 Mb against the deep maps: Pearson 0.271, 0.282 and 0.301 to 0.291, 0.318
  and 0.304, MultiMM 0.607, 0.568 and 0.652 to 0.674, 0.667 and 0.673, SCC level within 0.01,
  cross block overlaps 320, 416 and 170 to 176, 173 and 143 per thousand. With the compartment
  term at 0.5 on top the saddle rises on H1ESC 1.02 to 2.15 and HFFC6 0.71 to 1.09, not on
  GM12878, and SCC and MultiMM fall 0.07 to 0.10 on every cell, so the term stays opt in.

  **Tried and dropped, 2026-09-19: the loops beyond `max_pet_length` in the joint solve.**
  The loader sets those loops aside for the segment heatmap, a block scope rule, and at
  chromosome scope they could join the target matrix as arcs between the anchors holding
  their ends, 984 loops over 1 Mb on GM12878 chr1. Measured on chr1:1-60 Mb, three
  structures: Pearson 0.312 to 0.305, SCC 0.374 to 0.360, MultiMM 0.666 to 0.630, Rg 23.4 to
  19.4, cross block overlaps 20 to 47 per thousand. The arc law sets a loop's target by its
  PET count alone, so a 3 to 5 PET loop spanning megabases is asked to close to a bead or two
  and compacts the chromosome. Removed the same day; a span aware target would be option D
  of `design/anchor-placement.md` again.

- **Three smooth stage levers against within block overlaps: `[simulation_arcs_smooth]
  hard_wall` (default no, production yes), `anchor_cap` (0) and `start` (default line,
  production coil). The wall and the coil start are production since 2026-09-12.**
  ([gnome3d/mc/numba/terms.py](../gnome3d/mc/numba/terms.py) `batch_mc_nb`,
  [gnome3d/pipeline/ib/start.py](../gnome3d/pipeline/ib/start.py))
  The smooth stage ended with about 1,200 non neighbour pairs per thousand beads under 0.7 of
  a bond, twice an ideal chain with no excluded volume, on every cell and every gate. Traced
  on 2026-09-11: 94 percent are subanchor pairs 4 to 100 beads apart and 86 percent sit in the
  outer quarter of the radius, where the soft term `0.1 ((r0 - d) / r0)^2` has no force; the
  stage starts every coil on a straight line, 14,000 pairs per thousand, and pushes out
  against that term; and the anchors are pinned at 0.4 of an ideal coil's end to end. Widening
  the soft radius to one bond clears the count but flattens the distance curve and costs Hi-C
  on three cells, because it pushes every pair under a bond outward, contacts included.

  The wall rejects a move that adds a non neighbour pair under the excluded volume radius or
  deepens one that is there, so the count only falls and is a hard core once zero. It costs
  the same neighbour query the soft term makes, on the cell grid. The cap lets anchors move
  within that many mean bonds of where the arcs put them. The coil start places each gap's
  subanchors on a compact random bridge between its anchors at the bond targets, drawn from a
  generator seeded by the problem. Neither rule draws a random number, so with all three off
  the kernel's stream and the parity gate are untouched. One block, from the stage's own
  start: soft term alone 766 per thousand; wall 92; compact coil 487; compact coil with the
  wall 43 at the same Rg and bonds as production; a first clear coil with the wall reaches 0
  but swells the block by a third, which is why the compact rule is the one built. Both
  kernels carry the wall and the cap, as static flags on the JAX side since each is a pass per
  step; the relaxation pass does not. Three cells on the numba kernel, 2026-09-11: coil with
  the wall against the plain start raises Pearson 0.011 to 0.022 and MultiMM's own metric
  0.033 to 0.043, cuts overlaps five to seven times within blocks and three to four across at
  the same Rg, and flattens the short band exponent from 0.33 to 0.21; the cap adds a few
  percent fewer overlaps and nothing else. Unit checks in `tests/unit/test_smooth_levers.py`;
  three cell arms in `slurm/ensemble/overlap_levers.sh`.

  **Tried and dropped, 2026-11-13: a law spring on bead pairs in the smooth stage.** Traced on a
  web job, GM12878 chr1:20-25 Mb with CTCF and RNAPII loops, the squares in the contact map are
  anchorless gaps: anchor pairs sit at 1.0 to 1.2 of the law, subanchor pairs inside a gap at
  0.76 to 0.87 of it at 50 to 200 kb. Holding every bead pair under a range at the law through
  the subanchor heat term fixed the gaps, 1.02 at a 200 kb range, and cut overlaps tenfold, and
  lost the battery on chr1:20-25 and 10-15 Mb at five structures: Pearson 0.504 to 0.330 and
  0.366 to 0.243, MultiMM 0.671 to 0.630 and 0.613 to 0.545, the 20 to 100 kb contact exponent
  0.31 to 0.47. A spring at the mean distance removes the fluctuations contacts come from. A
  wider range was worse and 2 Mb compressed the block, as the all pairs springs did. The smooth
  wall tripled. Removed the same day; `playground/trace_square_blocks.py`,
  `playground/law_spring_readout.py` and `slurm/ensemble/law_spring_trial.sh` are the record.

  **Tried and dropped, 2026-11-13: a gap start that follows the law.** Under `start = law` each
  subanchor took the clear candidate whose distances to the beads 2 to 128 places behind it best
  matched the law in log, so a gap began at the law's size instead of packed into a ball. No
  energy term. On chr1:20-25, 10-15 and 35-40 Mb at five structures and chr1 at three, scored
  on 1-60 Mb, against the coil: Pearson down 0.03 to 0.07 on all four, the distance exponent
  1.06 to 1.17 times Hi-C down to 0.96 to 1.05, overlaps within blocks down two to twenty
  times, across blocks on chr1 11.7 to 20.5 per thousand. Gaps ended at about 1.2 of the law,
  since the smooth stage holds bonds and nothing holds a gap's interior, so its size is the
  start's. Without the hard wall overlaps rose tenfold, and with the subanchor heat term SCC
  went negative. Removed the same day; `slurm/ensemble/law_start_trial.sh`,
  `slurm/ensemble/gap_start_overnight.sh` and `playground/square_blocks/gapnight/` are the record.

  **A maximum entropy gap start: `start = maxent` with `[data] contact_map`, measured 2026-11-14
  and left opt in.** ([pipeline/ib/maxent.py](../gnome3d/pipeline/ib/maxent.py)) Both dropped
  trials gave every conformation the mean, and a contact map comes from the spread between
  conformations. Each gap's subanchors are drawn from the Gaussian chain whose mean pair distances
  are the law's, conditioned on the two anchors, after HIPPS and DIMES (Shi and Thirumalai). With
  a contact map, a cooler URI at 1 kb, each target is scaled by the count between the 10 kb
  windows of the pair against a power law fitted on the block's windows, observed over expected
  to the minus third. One set of springs cannot hold every pair at the law below an exponent of a
  third, and a distribution can, since the law's mean squared distance is a fractional Brownian
  motion variogram. The draw comes from the problem's seed. `line` and `coil` are unchanged.
  On three cells, three 5 Mb regions and chr1 it lost Pearson by 0.01 to 0.05 against the coil
  with SCC level and the distance exponent nearer Hi-C. The smooth stage widens the drawn gaps to
  1.2 of the law, where the coil ends under it. `slurm/ensemble/gap_maxent_trial.sh` and
  `playground/square_blocks/gapme/` are the record.
  Unit checks in `tests/unit/test_maxent_start.py`, options and trial in `design/gap-fill.md`.

- **Long range in the arcs stage: `[excluded_volume] arcs_repulsion = no`, `[springs]
  gyration_weight` 0.1, the arcs excluded volume at `auto_factor_arcs` 1.0 and `[simulation_arcs]
  start = maxent`, production since 2026-11-16.** ([pipeline/coarse/build.py](../gnome3d/pipeline/coarse/build.py)
  `repulsion_range`, [mc/numba/arcs_solver.py](../gnome3d/mc/numba/arcs_solver.py) `gyration_windows`,
  [pipeline/ib/maxent.py](../gnome3d/pipeline/ib/maxent.py) `anchor_draw`)
  Blocks held two to ten times too few contacts from 100 kb to 1 Mb against Hi-C at the right
  mean size. The arcless repulsion held every far anchor pair of every conformation at one
  distance, the 1 percent quantile beyond 100 kb at 0.40 to 0.53 of the law, so far pairs never
  met; an ablation of the arcs terms, one per arm, moved that floor only with the repulsion. Off,
  the blocks compact a third, so the gyration term holds the radius of gyration of every genomic
  window of anchors, 100 kb doubling up and tiled twice, at the law's in the solver's energy,
  which sets the size without holding a pair. The maxent start draws the anchors per
  conformation from the Gaussian whose pair variances best meet every pair's target, fitted by
  L-BFGS on a root of the covariance, and the solve descends from it. The cross block
  relaxation is on with it. Pooled ensemble scores on chr1 1 to 60 Mb, three cells: SCC up 0.017
  to 0.044, Spearman 0.018 to 0.028, MultiMM 0.06 to 0.10, the distance exponent 1.02 to 1.07
  of Hi-C's against 1.15 to 1.18, Rg at the law's where production sat 15 percent over it;
  balanced Pearson down 0.03 to 0.04, which follows the first diagonals. Arms are judged by the
  pooled map, `playground/ensemble_score.py`, since a per structure score rewards ensembles of one
  conformation. Unit checks in `tests/unit/test_gyration.py`, `tests/unit/test_maxent_start.py` and
  `tests/unit/test_arc_matrix.py`.

  Hole beads between blocks, `[holes] use_hole_beads` with `flank_beads`, stay opt in: on this
  arm they raise Spearman and lower pooled SCC 0.01 to 0.04 on three cells, since a correlation
  scores an empty hole's zeros against any filled level, while their contacts follow Hi-C's from
  pair to pair. `design/gap-fill.md` has every arm.

- **Uniform chain with loops on contact beads: `[main] uniform_chain = yes`, default no, under
  measurement since 2026-11-19.** ([pipeline/ib/spans.py](../gnome3d/pipeline/ib/spans.py) `densify_uniform`,
  [pipeline/ib/nodes.py](../gnome3d/pipeline/ib/nodes.py), [skeleton.py](../gnome3d/skeleton.py) `solve_nodes`,
  [mc/numba/arcs_solver.py](../gnome3d/mc/numba/arcs_solver.py) `reach_energy_grad`)
  `design/uniform-chain.md` has the rules of the model, the decisions and the stages of the build.
  The default chain has one bead per anchor of any width and one bead per kilobase between
  anchors. In the GM12878 anchor file 88 percent of consecutive anchors overlap and 42 percent are
  nested. The median width is 1.4 kb and 1.2 percent are over 10 kb. With `overlap_anchor_strict`
  off the densify stage fills each overlap with subanchors. On chr1 108.5 to 111 Mb the beads of
  the default chain cover 2,823 kb over 2,431 kb of sequence. The output is sorted by start and 62
  rows of that region leave the chain order. A viewer then draws bonds of 3 to 7 beads between
  beads of different runs near a nested anchor of 27 kb.

  The uniform chain divides the span of a block, from the start of its first anchor to the end of
  its last, into beads of equal width, one per `target_bp_per_subanchor`. Every base is in one
  bead and the sorted output is the chain. The span of an anchor is the run of beads whose
  midpoints are in its range. Each end of a kept loop is at one bead of the span of its anchor,
  drawn per conformation with equal chance. The distinct contact beads of a chromosome are the
  nodes of the joint arcs solve. The strengths of the loops on one pair of nodes add and go
  through the law once. Consecutive nodes of a block are held within the chain between them by
  the reach term, and the excluded volume of the solve acts at the radius of a bead. The nodes
  are the fixed beads of the smooth stage and the sites of the orientation term, with the
  orientation of the anchor of the CTCF loop with the most PETs. The relaxation holds the nodes.
  An output bead under a span has the kind anchor. The numba and the JAX smooth kernels run the
  chain unchanged. A run with the setting off is byte identical to 298685c, by
  `playground/parity_dump.py`.

  With `use_contact_background` and `[data] contact_map` a pair of nodes without a loop reads the
  Hi-C cell that holds its two beads at `[springs] contact_resolution`,
  [pipeline/ib/cells.py](../gnome3d/pipeline/ib/cells.py), under the Poisson test of
  `contact_background_z`. The anchor contact map and `use_anchor_heatmap` do not reach the nodes.

  The smooth stage keeps the centre of mass of a node, the bead before it and the bead after it,
  by the decision of 2026-11-20. Its setting `anchor_centre` and the fixed node of the uniform
  chain were removed. The rule: the node moves with its two neighbours taking minus half of its move each, or
  a neighbour moves with the other neighbour taking minus its move, `step_delta_nb` in
  [mc/numba/terms.py](../gnome3d/mc/numba/terms.py). The three single moves sum to the change of
  the whole move and the running score equals the full energy, with the orientation term. The
  JAX kernel does not have the move.

  `[springs] contact_resolution_max` reads a pair at the finest resolution whose cell can be
  tested, and `[springs] gyration_min_bp` starts the gyration windows under 100 kb.
  `design/uniform-chain.md` has every trial of the night of 2026-11-19 and
  `design/figures/uniform-2026-11-19/summary.md` one table of every arm.

  A third arm of the orientation term, `[motif_orientation] bond_limit`, refused a move of a bead
  beside an orientation site when it left a bond of that bead longer than the limit. In the trial
  of the rules it gained no score and raised the close pairs by a fifth. It was removed on
  2026-11-20.

  Stage 1 on GM12878 chr1 108.5 to 111 Mb, one structure, 154 nodes from 379 loops of 267 anchors,
  bonds before the relaxation:

  | arm | bonds over 2 beads | bonds over 3 beads | longest bond | pairs under 0.7 beads |
  |---|---|---|---|---|
  | as built | 212 | 89 | 8.90 | 183 |
  | orientation term off | 23 | 0 | 2.99 | 141 |
  | hard wall off | 60 | 6 | 3.89 | 1,740 |

  The distance of two consecutive nodes over the chain between them has a median of 0.93 and a
  95th percentile of 1.15. Of the 76 pairs one bond apart 55 are over 1.02. After the relaxation
  the as built arm has 301 bonds over 2 beads and 30 pairs under 0.7 beads, and the arm without
  the orientation term 79 bonds over 2 beads.

  The first build of 2026-11-19, commits 9d8b59f and a898a0e, held the centroid of each span at the
  arcs position of its anchor with a soft term of the smooth stage and is replaced.
  `design/uniform-chain.md` has its trial. The measurements below are of the default chain and of
  that first build.

  First measurement, one structure of GM12878 chr1 108.5 to 111 Mb at production settings, bonds
  in chain order after the relaxation, before the nodes were merged:

  | chain | beads | sequence in beads | bonds over 2 beads | bonds over 3 beads | 95th percentile bond | radius of gyration |
  |---|---|---|---|---|---|---|
  | default | 2,369 | 2,823 kb | 295 | 131 | 3.20 | 7.4 |
  | uniform | 2,483 | 2,483 kb | 159 | 44 | 2.16 | 7.3 |

  Trace of the long bonds, 2026-11-19, the same region and one structure per arm, bonds before
  the relaxation:

  | arm | bonds over 2 beads | bonds over 3 beads | bond beside an anchor, median | pairs under 0.7 beads |
  |---|---|---|---|---|
  | default chain | 243 | 117 | 2.48 | 66 |
  | default chain, orientation term off | 52 | 3 | 1.11 | 61 |
  | default chain, hard wall off | 103 | 25 | 1.77 | 970 |
  | uniform chain, nodes merged, span weight 1.0 | 167 | 58 | | 67 |
  | uniform chain, nodes merged, span weight 0.1 | 121 | 38 | | 47 |
  | the same, orientation term off | 46 | 0 | | 67 |
  | the same, hard wall off | 13 | 2 | | 1,593 |

  - The orientation term stretches the bonds beside an oriented anchor on both chains. Its weight
    is 50 and a bond has 0.1.
  - The hard wall keeps a stretched bond in place. Off, the bonds recover and the overlaps
    return.
  - The relaxation adds 40 to 60 bonds over 2 beads on either chain. It moves free beads and
    holds the anchor beads.
  - Before the merge 256 pairs of anchors with one knot had homes a median of 1.16 beads apart,
    held by the background spring at the floor of the law, one bead. Merged, they share a home.
  - Nodes one bead apart on the chain have homes a median of 1.5 to 1.7 beads apart. No term of
    the arcs stage holds chain neighbours within one bond. The arcs excluded volume radius at
    0.7 and 1.0 beads does not change it. Open.
  - The span weight was not tuned. At 1.0 the offset of a centroid from its home has a median of
    0.71 beads, at 0.1 of 1.12 and with the orientation term off of 0.57.

  Unit checks in `tests/unit/test_uniform_chain.py`, `tests/unit/test_contact_nodes.py` and
  `tests/integration/test_uniform_pipeline.py`.

- **Cell grid for excluded volume** ([gnome3d/mc/numba/cells.py](../gnome3d/mc/numba/cells.py),
  `[simulation_backend] neighbour_grid`, default yes)
  The excluded volume term sums over pairs closer than `r0` and was implemented as a scan over
  every bead. Only about fifty beads are ever that close whatever the structure's size, since
  that is a local density, so on a chromosome the scan did hundreds of times more work than it
  needed. Profiled on a finished 60 Mb region: the term is 99 percent of an MC step at 42,480
  beads, growing at 1.29 microseconds per thousand beads, while the useful neighbour count
  stayed at 57. The initial score was worse, a full pair scan quadratic in the structure.

  Beads are binned into a linked list per cell at cell size `r0`, so the twenty seven cells
  around a bead hold everything within `r0`. An accepted move unlinks the bead from its old cell
  and links it into the new one, which keeps the cells exactly `r0` wide with no margin for
  drift and the grid exact after every move. Both the step term and the initial score go
  through it.

  Results are identical bit for bit, not merely close, so no trajectory changes and the parity
  gate is unaffected. Two things make that true: only pairs inside the radius contribute
  anything, and the sum is built in ascending bead index, the order the full scan uses. A query
  that finds more neighbours than its buffer holds returns a sentinel and the caller falls back
  to the full scan, so correctness never depends on a capacity guess.

  Measured on the real structure, same seed, positions and scores identical: 5.3 times faster
  at 8,000 beads, 11.2 at 20,000, 22.8 at 42,480. It applies to every reconstruction, since the
  per block smooth stage runs the same term on blocks of up to 16,384 beads, and smooth is most
  of the pipeline's wall time. The grid is used from 128 beads, where it starts to win: on a
  block of 1,831 beads with the hard wall the stage takes 4.7 s against 45.9 s on the full
  scan. Unit checks in `tests/unit/test_cells.py`.

  A query clamps its cell index as the insertion does, since 2026-11-16. The box is the
  extent the structure had when the grid was built, a bead beyond it is filed in an edge cell,
  and a query from more than a cell beyond it searched no cell at all, so two beads out there
  did not see each other, neither the soft term nor the wall. A coil that swells past its
  starting extent ran without excluded volume in the swollen part. Found on a block the grid
  and the full scan solved differently. With the clamp the two agree bit for bit at every
  chain length from 32 to 1,831 beads, wall on and off. The JAX grid clamps both sides and
  was not affected.
