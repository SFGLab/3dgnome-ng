# Settings reference

A config is an ini file with the sections below. A key that appears in the file but is not one
of these produces a warning rather than silently doing nothing, so a misspelling is caught.
Booleans accept `yes`, `no`, `true`, `false`, `1` and `0`. Filenames in the data section are
taken relative to `data_dir` unless they are absolute.

## [main]

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `output_level` | int | 0 | 1 | Log verbosity. 0 prints the run banner and warnings, 1 adds milestones and per stage headers, 2 adds per batch MC step lines. |
| `log_file` | str |  |  | Path of a full detail structured log written alongside the console output. Empty writes none. |
| `random_walk` | bool | no | no | Place segment beads by a chained random walk of fixed 50.0 steps per chromosome instead of the segment heatmap MC. Honours `use_2D`. |
| `use_2D` | bool | no | no | Restrict every move to the plane. |
| `loop_density` | int | 5 | 5 | Subanchors inserted between every consecutive anchor pair when dynamic density is off. |
| `max_pet_length` | int | 1000000 | 1000000 | An arc spanning more than this many bp is a long arc. It is not a spring and is folded into the segment heatmap instead. |
| `long_pet_power` | float | 2.0 | 2.0 | A long arc adds `long_pet_scale * score ^ long_pet_power` to the segment heatmap. |
| `long_pet_scale` | float | 10.0 | 1.0 | See above. |
| `steps_lvl1` | int | 2 | 1 | Chromosome level heatmap MC runs, best kept. Multi chromosome runs only. |
| `steps_lvl2` | int | 2 | 1 | Segment level heatmap MC runs, best kept. |
| `steps_arcs` | int | 5 | 1 | Restarts of the arcs stage per block, each from a freshly noised start, best kept. Applies to the solver as well as the annealer. |
| `steps_smooth` | int | 5 | 1 | Restarts of the smooth stage per block, best kept. |
| `noise_lvl1` | float | 1.0 | 0.5 | Chromosome level step size, as a multiple of the mean target distance. |
| `noise_lvl2` | float | 0.1 | 0.5 | Segment level step size, as a multiple of the mean target distance. |
| `noise_smooth` | float | 0.5 | 5.0 | Smooth stage step size, as a multiple of the mean chain bond target. |
| `noise_ib` | float | 0.5 | 0.5 | Block placement step size, as a multiple of the mean block chain bond. |
| `overlap_anchor_strict` | bool | no | no | The reference's span rule for overlapping anchors, which collapses the subanchors between them to one point. Off tiles the overlap with non degenerate ranges. |
| `drop_zero_length_subanchors` | bool | no | yes | Leave zero width subanchors out of the written structure. The chain still carries them. |
| `use_dynamic_loop_density` | bool | no | yes | Pick the subanchor count per anchor gap so that each chain segment is about `target_bp_per_subanchor` long, instead of a fixed `loop_density`. |
| `target_bp_per_subanchor` | int | 5000 | 1000 | Target genomic length of one chain segment under dynamic density. |
| `min_subanchors_per_arc` | int | 0 | 0 | Lower bound on the count per gap under dynamic density. |
| `max_subanchors_per_arc` | int | 50 | 100 | Upper bound on the count per gap under dynamic density. |

## [data]

Filenames are relative to `data_dir` unless absolute. The region string is `chr:start-end`.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `data_dir` | str |  | Directory the other filenames resolve against. The CLI's `--data-dir` overrides it. |
| `anchors` | str |  | BED of loop anchors, `chr start end orientation`. |
| `clusters` | str |  | BEDPE of PET clusters, the arcs, `chr1 s1 e1 chr2 s2 e2 score`. |
| `singletons` | str |  | BEDPE of singleton contacts for the segment level heatmap. A Hi-C bin pair file works here too. |
| `singletons_inter` | str |  | A second singletons file appended for multi chromosome runs only. |
| `centromeres` | str |  | BED of centromere positions. |
| `segment_split` | str |  | BED of segment boundary breakpoints. |
| `compartments` | str |  | bedGraph of a signed compartment eigenvector or a CALDER BED, for `[compartments]`. |
| `accessibility` | str |  | bedGraph of ATAC or DNase signal, for `[accessibility]`. |
| `phasing_track` | str |  | Track used to fix the eigenvector's arbitrary sign. Required with `compartments`. |

## [distance]

One law sets every distance, in bead units, and it reads two keys. Its exponent is measured
from the run's own singletons at load, so the section is usually empty.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `polymer_exponent` | float | 0.0 | 0.0 | Zero measures the exponent from the singletons at load. A positive value pins it, and its presence in a config is the record that someone chose to. |
| `contact_half_saturation` | float | 1.0 | 1.0 | The loop strength, in multiples of a typical loop at that span, at which a contact pulls its pair halfway from the background to touching. |

## [heatmaps]

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `inter_scaling` | float | 1.0 | 1.0 | Multiply the inter chromosome blocks of the segment heatmap. Intra blocks are unchanged. Multi chromosome runs only. |
| `distance_heatmap_stretching` | float | 2.0 | 2.5 | Clip segment heatmap distances above the mean positive distance times this. |

## [springs]

Spring energy is `k * ((d - target) / target) ^ 2` with `k` the stretch constant when the pair
is too far and the squeeze constant when too close.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `stretch_constant` | float | 0.1 | 0.1 | Smooth stage chain bond, too far. |
| `squeeze_constant` | float | 0.1 | 0.1 | Smooth stage chain bond, too close. |
| `angular_constant` | float | 0.1 | 0.1 | Smooth stage bend penalty, the cube of the angle between consecutive bonds. |
| `stretch_constant_arcs` | float | 1.0 | 1.0 | Arcs stage, every target in the matrix, arcs and chain bonds alike. |
| `squeeze_constant_arcs` | float | 1.0 | 1.0 | Arcs stage. |
| `background_weight` | float | 0.3 | 0.3 | Weight of the spring holding an arcless anchor pair at the background for its separation in the arcs stage. Weak against the arc springs, since the background is an expectation and not a measured contact. |
| `use_arcs_chain_bonds` | bool | no | no | Give every consecutive anchor pair with no arc a full weight spring at `arcs_chain_bond_scale` times the background. Off in production: with every arcless pair already held at the background it is redundant, and at 1.5 it shoved neighbours into each other. |
| `arcs_chain_bond_scale` | float | 1.0 | 1.0 | Multiplier on that bond's target. |
| `stretch_constant_ib` | float | 0.1 | 0.1 | Block placement chain bond. |
| `squeeze_constant_ib` | float | 0.1 | 0.1 | Block placement chain bond. |

## [motif_orientation]

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_motif_orientation` | bool | no | yes | Score the angle between the CTCF motif vectors of anchors an arc joins, in the smooth stage. |
| `weight` | float | 1.0 | 50.0 | Weight of that term. |
| `symmetric_motifs` | bool | yes | yes | Yes scores two motif vectors best when they point the same way. No flips the partner first, so opposed vectors score best. |

## [anchor_heatmap]

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_anchor_heatmap` | bool | no | yes | Scale each arc target down in proportion to its pair's singleton contact count relative to the region's maximum. |
| `heatmap_influence` | float | 0.5 | 0.1 | The most a target can be scaled down, as a fraction. |

## [subanchor_heatmap]

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_subanchor_heatmap` | bool | no | yes | Add a contact distance target between subanchor pairs in the smooth stage, estimated from dry smooth passes and scaled by contact. |
| `heatmap_influence` | float | 0.5 | 0.1 | The most a pair's estimated distance is scaled down by its contact. |
| `heatmap_dist_weight` | float | 1.0 | 0.01 | Weight of that term against the chain and angle terms. |
| `estimate_distances_steps` | int | 2 | 4 | Dry smooth runs per replicate when estimating the mean pairwise distances. |
| `estimate_distances_replicates` | int | 5 | 4 | Replicate estimates averaged into the target. |
| `heat_min_reduction` | float | 0.0 | 0.0 | Skip the estimate for a block whose fraction of contact carrying pairs, an upper bound on the reduction the term can achieve, is below this. |
| `heatmap_workers` | int or auto | 1 |  | Threads used to build the per block contact heatmaps. `auto` uses every usable core. |

## The MC schedule sections

`[simulation_heatmap]`, `[simulation_arcs]`, `[simulation_arcs_smooth]` and `[simulation_ib]`
each parametrise one stage's annealing with the same keys. In `[simulation_heatmap]` every key
carries a `_heatmap` suffix. The meaning is shared.

An uphill move is accepted when `rand < jump_temp_scale * exp(-jump_temp_coef * (E_new / E_old)
/ T)`, and `T` is multiplied by `delta_temp` after every step from `max_temp`. Every
`stop_condition_steps` steps is one round. The run stops when a round neither lowered the score
below `stop_condition_improvement_threshold` times the previous round's score nor accepted at
least `stop_condition_successes_threshold` moves.

| key | type | default | production | what it does |
|---|---|---|---|---|
| `max_temp` | float | 20.0 | 5.0 | Starting temperature. |
| `delta_temp` | float | 0.99995 | 0.9999 | Per step cooling factor. |
| `jump_temp_scale` | float | 50.0 | 50.0 | Acceptance prefactor. |
| `jump_temp_coef` | float | 20.0 | 20.0 | Acceptance coefficient. |
| `stop_condition_steps` | int | 10000 | 50000 | Steps per round. |
| `stop_condition_improvement_threshold` | float | 0.995 | 0.999 | A round improved when the score fell below this times the previous round's. |
| `stop_condition_successes_threshold` | int | 5 | arcs and ib 100, smooth 50, heatmap 10 | Accepted moves in a round below which a non improving round ends the run. |

### [simulation_arcs] only

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `stop_condition_ratio` | float | 0.9999 | 0.9999 | Also stop when the score over the previous round's is at or above this, a plateau guard. |
| `solver` | str | mc | lbfgs | `mc` anneals, `lbfgs` minimises the same energy with L-BFGS-B. Same minimum, same overlaps, the stage's calls fell from minutes to seconds. Needs `mc_executor_arcs` of `serial` or `threaded`. The batch executor has no solver and refuses. |
| `solver_iters` | int | 200 | 200 | Iterations for the solver. |

### [simulation_arcs_smooth] only

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `dist_weight` | float | 1.0 | 1.0 | Weight of the chain bond term. |
| `angle_weight` | float | 1.0 | 1.0 | Weight of the bend term. |

### [simulation_ib] only

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `refine_scope` | str | segment | segment | `segment` places each segment's blocks as one chain and skips a segment with one block or fewer. `chromosome` places every block on the chromosome as one chain. It inflates structures and needs the excluded volume and confinement retuned. |
| `use_ib_mc` | bool | no | yes | Anneal block centroids with chain bonds, excluded volume and confinement instead of placing them by interpolation. |
| `dist_weight` | float | 1.0 | 1.0 | Weight of the chain bond term. |

## [simulation_backend]

Each stage runs on an executor. `serial` and `threaded` are the numba kernels, one chain per
thread. `batch` is the JAX kernels, which pack many blocks into one device launch. `auto`
resolves from the older backend keys.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `ib_workers` | int or auto | 1 | auto | Threads for the threaded executor. `auto` uses every usable core. |
| `heatmap_chains` | int | 1 | 1 | Independent heatmap MC chains run at once, best kept. |
| `smooth_chains` | int | 1 | 1 | Independent smooth chains run at once per block, best kept. |
| `mc_executor_arcs` | str | auto | threaded | Executor for the arcs stage. Threaded on the CPU because a vmapped launch cannot retire a converged block and one straggler holds every other block in the launch. |
| `mc_executor_densify` | str | auto | threaded | Executor for densification. |
| `mc_executor_estimate_dist` | str | auto | batch | Executor for the subanchor distance estimate. |
| `mc_executor_smooth` | str | auto | batch | Executor for the smooth stage. The cross block relaxation also picks its kernel from this, and one chain on the batch kernel is that kernel's worst case. |
| `mc_executor_jax_bucket_shapes` | bool | no | yes | Pad each block to a shape ladder so a stage is a few wide launches rather than one compile per size. |
| `merge_smooth_launches` | bool | yes | yes | Pack every smooth block that agrees on its energy terms into as few launches as device memory allows, regardless of size. |
| `mc_executor_jax_batch_width_smooth` | int or auto | auto | auto | Blocks per smooth launch. `auto` solves the largest count that fits the device. |
| `mc_executor_jax_batch_width_arcs` | int or auto | auto | auto | The same for the JAX arcs kernel. |
| `multigpu_mode` | str | groups | groups | `groups` runs whole batch groups on different devices and is byte exact in the device count. `within` splits one group across devices. `off` uses one device. |
| `neighbour_grid` | bool | yes | yes | Cell grid for the excluded volume term in the numba kernels. Identical results, faster above 2048 beads. |

## [excluded_volume]

Soft repulsion `weight * ((r0 - d) / r0) ^ 2` for pairs closer than `r0` and more than
`skip_neighbors` apart along the chain. Each stage has its own radius. A radius of zero derives
one as the stage's auto factor times the stage's mean bond scale.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_excluded_volume` | bool | no | yes | Master switch. |
| `weight` | float | 0.5 | 0.1 | Weight. The term saturates at this value for a full overlap. |
| `apply_to_arcs` | bool | no | yes | In the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | In the smooth stage. |
| `apply_to_heatmap` | bool | no | yes | In the heatmap stages. |
| `apply_to_ib` | bool | yes | yes | In block placement. |
| `skip_neighbors` | int | 1 | 1 | Pairs this close along the chain are exempt. |
| `radius_arcs` | float | 0.0 | 0.0 | Arcs stage radius, 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | Smooth stage radius. |
| `radius_heatmap` | float | 0.0 | 0.0 | Heatmap stage radius. |
| `radius_ib` | float | 0.0 | 0.0 | Block placement radius. Also the constant centroid radius of the boundary stitch when positive. |
| `auto_factor_arcs` | float | 0.5 | 0.5 | Times the mean positive arc target. |
| `auto_factor_smooth` | float | 0.5 | 0.7 | Times the mean chain bond target of the block. |
| `auto_factor_heatmap` | float | 0.5 | 0.5 | Times the mean active heatmap target. |
| `auto_factor_ib` | float | 0.5 | 0.5 | Times the mean block chain bond. |

## [confinement]

Soft envelope `weight * ((r - R) / R) ^ 2` for a bead further than `R` from the centroid of the
stage's starting positions. A radius of zero derives one as the packing factor times the
stage's mean bond scale times the cube root of the bead count.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_confinement` | bool | no | yes | Master switch. |
| `weight` | float | 0.5 | 0.1 | Weight. |
| `apply_to_arcs` | bool | yes | yes | In the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | In the smooth stage. |
| `apply_to_ib` | bool | yes | yes | In block placement. |
| `radius_arcs` | float | 0.0 | 0.0 | Arcs stage radius, 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | Smooth stage radius. |
| `radius_ib` | float | 0.0 | 0.0 | Block placement radius. |
| `packing_factor_arcs` | float | 1.5 | 1.5 | Arcs stage packing factor. |
| `packing_factor_smooth` | float | 1.5 | 1.5 | Smooth stage packing factor. |
| `packing_factor_ib` | float | 0.75 | 0.75 | Block placement packing factor. Below about 0.58 a small segment is asked to fold tighter than one of its own bonds, and 0.15 crushed the cross block distance scaling. |

## [boundary_stitch]

Runs after every chain of a chromosome is done. Moves each block as a rigid body so the last
anchor of one block and the first of the next sit at the distance the structure's own interior
pairs realise at that separation, with a soft excluded volume between block centroids. No RNG.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_boundary_stitch` | bool | no | yes | Master switch. |
| `spring_weight` | float | 1.0 | 1.0 | Weight of the boundary springs. |
| `ev_weight` | float | 1.0 | 1.0 | Weight of the centroid excluded volume. |
| `max_iter` | int | 2000 | 2000 | L-BFGS-B iterations. The energy carries its own gradient, so an iteration is one evaluation. 500 leaves a chromosome unconverged. 2000 converges a 1,494 block chromosome in 85 seconds. |

## [relax]

Runs after the stitch. The smooth kernel once over the whole chromosome with excluded volume on
every pair and every anchor held fixed, so coils from different blocks stop passing through each
other while the arcs and the stitch are kept.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_cross_block_relax` | bool | no | yes | Master switch. |
| `ev_weight` | float | 10.0 | 10.0 | Excluded volume weight for the pass. |
| `ev_radius` | float | 0.0 | 0.0 | Excluded volume radius. 0 uses 1.5 chain bonds, so nothing is left under one bond where contacts are counted. |
| `temp` | float | 0.1 | 0.1 | Starting temperature as a fraction of the smooth stage's `max_temp`. Untangling needs a bead to cross a neighbour's shell, and a greedy pass stalls. |
| `noise` | float | 0.5 | 0.5 | Step size in chain bonds. |
| `bond_weight` | float | 10.0 | 10.0 | Chain spring weight for the pass. At the smooth stage's 0.1 the excluded volume tears the coil. |
| `min_contact_fraction` | float | 0.0 | 0.0 | Decline the pass when cross block contacts are fewer than this fraction of the chromosome's beads. 0 always runs. |
| `local_window` | int | -1 | -1 | Let only the beads touching another block move, plus this many chain neighbours either side. -1 lets every subanchor move, which on a chromosome is hours. A window of 1 is minutes. |

## [compartments]

A/B compartment terms ported from MultiMM [7], written shifted and non negative so the Metropolis
ratio stays defined, and divided by `N - 1` so a weight tuned on a small region holds on a large
one. Needs `[data] compartments` and `phasing_track`, and excluded volume or confinement
alongside since the terms are attractive.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `use_compartments` | bool | no | Master switch. |
| `weight` | float | 1.0 | Weight of the pairwise affinity. |
| `energy_a` | float | 1.0 | Affinity between two A beads. |
| `energy_b` | float | 2.0 | Affinity between two B beads. |
| `apply_to_heatmap` | bool | yes | In the heatmap stages. |
| `apply_to_ib` | bool | yes | In block placement. |
| `apply_to_smooth` | bool | yes | In the smooth stage. |
| `radius_heatmap` | float | 0.0 | Interaction radius, 0 derives it. |
| `radius_ib` | float | 0.0 | Interaction radius. |
| `radius_smooth` | float | 0.0 | Interaction radius. |
| `auto_factor_heatmap` | float | 1.5 | Times the stage's mean bond scale. |
| `auto_factor_ib` | float | 1.5 | Times the stage's mean bond scale. |
| `auto_factor_smooth` | float | 1.5 | Times the stage's mean bond scale. |

## [accessibility]

The HiP-HoP mechanisms [6] driven from one accessibility track. Bridging is an effective pairwise
attraction between open beads. Fibre compaction shortens the chain bond where the bead is
closed.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `mode` | str | log | `log` is log then min max normalisation. `binary` is HiP-HoP's own open or closed state and is the faithful one. `log` is close to inert on a track binned to several kb. |
| `percentile` | float | 80.0 | Under `binary`, a bead is open at or above this percentile of the loaded values. |
| `use_bridging` | bool | no | Master switch for bridging. |
| `bridging_weight` | float | 1.0 | Weight of the bridging affinity. |
| `apply_to_heatmap` | bool | no | In the heatmap stages. |
| `apply_to_ib` | bool | no | In block placement. |
| `apply_to_smooth` | bool | yes | In the smooth stage. |
| `radius_heatmap` | float | 0.0 | Interaction radius, 0 derives it. |
| `radius_ib` | float | 0.0 | Interaction radius. |
| `radius_smooth` | float | 0.0 | Interaction radius. |
| `auto_factor_heatmap` | float | 1.5 | Times the stage's mean bond scale. |
| `auto_factor_ib` | float | 1.5 | Times the stage's mean bond scale. |
| `auto_factor_smooth` | float | 1.5 | Times the stage's mean bond scale. |
| `use_fibre_compaction` | bool | no | Master switch for compaction. |
| `fibre_compaction` | float | 0.3 | A bead's chain bond target is scaled by `1 - fibre_compaction * (1 - accessibility)`. |

## [nucleus]

Whole nucleus terms from MultiMM [7]. They run in the segment level heatmap MC only, since that is
the one call that spans the whole active region.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `use_lamina` | bool | no | Pull B beads toward the nuclear envelope. |
| `lamina_weight` | float | 400.0 | Weight. |
| `use_central_force` | bool | no | Pull A beads toward the centre. |
| `central_weight` | float | 20.0 | Weight. |
| `use_chromosomal_blocks` | bool | no | Keep each chromosome in its own territory. Multi chromosome runs. |
| `chrom_block_kc` | float | 0.3 | Territory stiffness. |
| `chrom_block_weight` | float | 0.0001 | Weight. |
| `radius` | float | 0.0 | Outer nuclear radius, 0 derives it. |
| `packing_factor` | float | 1.0 | The derived outer radius is this times the mean bond scale times the cube root of the bead count. |
| `inner_fraction` | float | 0.2 | The inner radius is the outer one times the cube root of this. |

## How distances are set

One law turns everything the pipeline knows about a pair of beads into a distance, and every
stage reads it: the bond between consecutive beads in the smooth stage, the bond between
block centroids in block placement, the target for a pair of anchors an arc joins, the bond a
consecutive arcless anchor pair gets, and the target for a cell of the segment heatmap.

In plain terms. The unit is the bead, the distance two beads hold at the resolution the run
declared, so nothing is measured in a number that means nothing. Two loci with nothing between
them sit at the crumpled chromatin distance for how much DNA separates them. Chromatin in a
nucleus is crumpled, not stretched out, so distance grows only slowly with the DNA in between:
ten times more DNA puts two loci only about twice as far apart. That exponent is read off the
run's own contacts rather than typed in. A loop draws its two ends in from there, more for a
loop stronger than is typical at its span, and never closer than touching. A pair of big
segments on the coarse map sits closer the more contacts they share.

Precisely, with `s0` being `target_bp_per_subanchor` and `nu` the measured exponent.

- No contact: `d = max(1, (s / s0) ^ nu)` for separation `s`. Mean spatial distance grows
  with genomic separation as a power law [4, 5]. In the arcs stage every arcless anchor pair
  is held there by a weak spring, `background_weight`, so no pair is free to collapse onto
  another.
- A loop of strength `q`: `d = 1 + (background - 1) / (1 + q / q_half)`. `q` is the loop's PET
  count over the typical count at its span, fitted on the run's own arcs, which is observed
  over expected [8, 9]. A saturated loop sits at one bead, touching, as polymer loop models
  hold loop anchors [6, 11], and never inside.
- A heatmap cell: the background times observed over expected contact to the minus third,
  the fractal globule relation [4, 5, 10], the expectation being the mean of the cells at that
  separation within the heatmap. At the chromosome level, where pairs have no separation,
  every cell shares one expectation and the background is taken at the mean chromosome span.
- `nu` is the slope of log contact count against log separation on the run's singletons,
  over minus three [4, 5], fitted at load on the whole chromosome set before any region
  filter. Contact probability falls close to the inverse of separation in Hi-C [4], and if
  distance grows as `s ^ nu` contact falls as `s ^ (-3 nu)` [5]. When the singletons cannot
  supply one, which a ChIA-PET singletons file cannot since it is enrichment filtered around
  CTCF and does not decay, the run says so in an always visible line and uses 0.285, the
  fractal globule value, as a named fallback.

Measured on this project's own inputs the exponent is 0.275 for GM12878, 0.299 for H1ESC,
0.192 for HFFC6 and 0.072 for a trio sample, which is why it is measured and not a constant.

This law replaced, on 2026-09-06, the three laws 3D-GNOME set these distances with [1]: a chain
law from genomic separation, a PET law from loop count, and a heatmap frequency law, plus two
later patches on the PET law. Together they carried fifteen constants copied from one config
with no derivation, set one quantity on three unrelated scales, and asked loop ends to sit
inside each other. Measured against them on three cell lines the polymer law is better on
Hi-C correlation by every statistic, cuts anchor overlaps by 97 percent and halves subanchor
overlaps. The old constants are not accepted in a config and warn if present.

## References

1. Szałaj P, Tang Z, Michalski P, Pietal MJ, Luo OJ, Sadowski M, Li X, Radew K, Ruan Y,
   Plewczynski D. An integrated 3-Dimensional Genome Modeling Engine for data-driven simulation
   of spatial genome organization. Genome Research 26, 1697 to 1709 (2016).
   doi:10.1101/gr.205062.116
2. Szałaj P, Michalski PJ, Wróblewski P, Tang Z, Kadlof M, Mazzocco G, Ruan Y, Plewczynski D.
   3D-GNOME: an integrated web service for structural modeling of the 3D genome. Nucleic Acids
   Research 44, W288 to W293 (2016). doi:10.1093/nar/gkw437
3. Wlasnowolski M, Kadlof M, Sengupta K, Plewczynski D. 3D-GNOME 3.0: a three-dimensional
   genome modelling engine for analysing changes of promoter-enhancer contacts in the human
   genome. Nucleic Acids Research 51, W5 to W10 (2023). doi:10.1093/nar/gkad354
4. Lieberman-Aiden E, van Berkum NL, Williams L, Imakaev M, Ragoczy T, Telling A, Amit I, Lajoie
   BR, Sabo PJ, Dorschner MO, Sandstrom R, Bernstein B, Bender MA, Groudine M, Gnirke A,
   Stamatoyannopoulos J, Mirny LA, Lander ES, Dekker J. Comprehensive mapping of long-range
   interactions reveals folding principles of the human genome. Science 326, 289 to 293 (2009).
   doi:10.1126/science.1181369
5. Mirny LA. The fractal globule as a model of chromatin architecture in the cell. Chromosome
   Research 19, 37 to 51 (2011). doi:10.1007/s10577-010-9177-0
6. Buckle A, Brackley CA, Boyle S, Marenduzzo D, Gilbert N. Polymer Simulations of Heteromorphic
   Chromatin Predict the 3D Folding of Complex Genomic Loci. Molecular Cell 72, 786 to 797
   (2018). doi:10.1016/j.molcel.2018.09.016
7. Korsak S, Banecki K, Plewczynski D. Multiscale molecular modeling of chromatin with MultiMM:
   From nucleosomes to the whole genome. Computational and Structural Biotechnology Journal 23,
   3537 to 3548 (2024). doi:10.1016/j.csbj.2024.09.025
8. Rao SSP, Huntley MH, Durand NC, Stamenova EK, Bochkov ID, Robinson JT, Sanborn AL, Machol I,
   Omer AD, Lander ES, Aiden EL. A 3D map of the human genome at kilobase resolution reveals
   principles of chromatin looping. Cell 159, 1665 to 1680 (2014). doi:10.1016/j.cell.2014.11.021
9. Serra F, Baù D, Goodstadt M, Castillo D, Filion GJ, Marti-Renom MA. Automatic analysis and
   3D-modelling of Hi-C data using TADbit reveals structural features of the fly chromatin
   colors. PLoS Computational Biology 13, e1005665 (2017). doi:10.1371/journal.pcbi.1005665
10. Varoquaux N, Ay F, Noble WS, Vert JP. A statistical approach for inferring the 3D structure
    of the genome. Bioinformatics 30, i26 to i33 (2014). doi:10.1093/bioinformatics/btu268
11. Fudenberg G, Imakaev M, Lu C, Goloborodko A, Abdennur N, Mirny LA. Formation of chromosomal
    domains by loop extrusion. Cell Reports 15, 2038 to 2049 (2016).
    doi:10.1016/j.celrep.2016.04.085
