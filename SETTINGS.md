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
| `step_decay_floor` | float | 0.1 | 0.1 | Lower bound on the step size under the per stage `step_decay` keys, as a fraction of the starting step. |

## [data]

Filenames are relative to `data_dir` unless absolute. The region string is `chr:start-end`.

| key | type | default | what it does |
| --- | --- | --- | --- |
| `data_dir` | str |  | Directory the other filenames resolve against. The CLI's `--data-dir` overrides it. |
| `anchors` | str |  | BED of loop anchors, `chr start end orientation`. |
| `clusters` | str |  | BEDPE of PET clusters, the arcs, `chr1 s1 e1 chr2 s2 e2 score`. |
| `singletons` | str |  | BEDPE of singleton contacts for the segment level heatmap. A Hi-C bin pair file works here too. |
| `singletons_inter` | str |  | A second singletons file appended for multi chromosome runs only. |
| `factors` | str |  | Accepted and never used. The reference implementation's protein factor filter. |
| `split_singleton_files_by_chr` | bool | no | Accepted and never used. |
| `centromeres` | str |  | BED of centromere positions. |
| `segment_split` | str |  | BED of segment boundary breakpoints. |
| `segment_heatmap` | str |  | Accepted and never used. |
| `compartments` | str |  | bedGraph of a signed compartment eigenvector or a CALDER BED, for `[compartments]`. |
| `accessibility` | str |  | bedGraph of ATAC or DNase signal, for `[accessibility]`. |
| `phasing_track` | str |  | Track used to fix the eigenvector's arbitrary sign. Required with `compartments`. |

## [distance]

Three keys are the polymer law. Every other key in this section is one of the parity era laws
and is not read while `use_polymer_law` is on.

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `use_polymer_law` | bool | no | no | One law for every distance, in bead units, with the exponent measured from the run's own contacts. See the section above. Measured, not yet default. |
| `polymer_exponent` | float | 0.0 | 0.0 | Zero measures the exponent from the singletons at load. A positive value pins it, and its presence in a config is the record that someone chose to. |
| `contact_half_saturation` | float | 1.0 | 1.0 | The loop strength, in multiples of a typical loop at that span, at which a contact pulls its pair halfway from the background to touching. |
| `genomic_dist_base` | float | 0.0 | 1.0 | Chain law, `base + scale * (bp / 1000) ^ power`. |
| `genomic_dist_scale` | float | 1.0 | 0.5 | Chain law. |
| `genomic_dist_power` | float | 0.5 | 0.75 | Chain law. |
| `freq_dist_scale` | float | 100.0 | 25.0 | Segment heatmap frequency to distance, `scale * freq ^ power`. |
| `freq_dist_power` | float | -0.333 | -0.6 | Segment heatmap frequency to distance. |
| `freq_dist_scale_inter` | float | 100.0 | 120.0 | The same for the chromosome level heatmap. |
| `freq_dist_power_inter` | float | -1.0 | -1.0 | The same for the chromosome level heatmap. |
| `count_dist_a` | float | 0.5 | 0.2 | PET law, `base_level + scale / exp(a * (PET + shift))`. |
| `count_dist_scale` | float | 20.0 | 1.8 | PET law. |
| `count_dist_shift` | float | 1.0 | 8.0 | PET law. |
| `count_dist_base_level` | float | 0.01 | 0.2 | PET law. The distance a saturated arc asks for, and the lower of the two limits the unified law normalises against. |
| `use_separation_arc_target` | bool | no | yes | Multiply the PET law by `max(1, s_kb / arc_target_pivot_kb) ^ arc_target_exponent`. Superseded by the unified law when both are on. |
| `arc_target_exponent` | float | 0.285 | 0.285 | Separation aware exponent. |
| `arc_target_pivot_kb` | float | 10.0 | 10.0 | Span below which the separation aware law leaves the PET law alone. |
| `use_unified_arc_target` | bool | no | no | One background for arc targets and chain bonds, with the PET count setting only the pull. See the section above. Measured a strict improvement on GM12878 and H1ESC at a background exponent of 0.15. |
| `arc_target_pull` | float | 0.45 | 0.45 | The fraction of the background a saturated arc pulls its pair to. A zero PET arc sits on the background. |
| `arc_target_background_exponent` | float | 0.0 | 0.0 | Zero rides the chain law itself, whose exponent is far steeper than the contact probability curves give. A positive value continues the chain law from the reference separation at this slope. 0.15 measured best on two cell lines. The optimum differs by dataset. |
| `arc_target_background_ref_bp` | int | 1000 | 1000 | The separation at which the background agrees with the chain law when an exponent is set. |

## [template]

All four keys are accepted and never used. They are the reference implementation's structural
template and MDS distance heatmap inputs, which are not ported. The production config sets
`template_scale` and `dist_heatmap_scale` and neither has any effect.

| key | type | default |
| --- | --- | --- |
| `template_segment` | str |  |
| `template_scale` | float | 1.0 |
| `dist_heatmap` | str |  |
| `dist_heatmap_scale` | float | 1.0 |

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
| `use_arcs_chain_bonds` | bool | no | yes | Give every consecutive anchor pair with no arc a spring at the chain law of its gap, entered after the anchor heatmap scaling. |
| `arcs_chain_bond_scale` | float | 1.0 | 1.5 | Multiplier on that bond's target. |
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
| `batch_trials` | bool | no |  | Accepted and never used. |
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
least `stop_condition_successes_threshold` moves. `step_decay` multiplies the step size by that
factor each round, down to `step_decay_floor` of the start, and only the numba kernels honour
it.

| key | type | default | production | what it does |
|---|---|---|---|---|
| `max_temp` | float | 20.0 | 5.0 | Starting temperature. |
| `delta_temp` | float | 0.99995 | 0.9999 | Per step cooling factor. |
| `jump_temp_scale` | float | 50.0 | 50.0 | Acceptance prefactor. |
| `jump_temp_coef` | float | 20.0 | 20.0 | Acceptance coefficient. |
| `stop_condition_steps` | int | 10000 | 50000 | Steps per round. |
| `stop_condition_improvement_threshold` | float | 0.995 | 0.999 | A round improved when the score fell below this times the previous round's. |
| `stop_condition_successes_threshold` | int | 5 | arcs and ib 100, smooth 50, heatmap 10 | Accepted moves in a round below which a non improving round ends the run. |
| `step_decay` | float | 1.0 | 1.0 | Per round step size multiplier. 1.0 is off. Not in `[simulation_heatmap]`. |

### [simulation_arcs] only

| key | type | default | production | what it does |
| --- | --- | --- | --- | --- |
| `stop_condition_ratio` | float | 0.9999 | 0.9999 | Also stop when the score over the previous round's is at or above this, a plateau guard. |
| `force_bias` | float | 0.0 | 0.0 | Steer each proposal along the local gradient by this fraction of the step. 0 is a plain random step. |
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
| `use_ib_arcs` | bool | no | no | Add attraction only targets between block centroids from the arcs crossing their boundary, normalised so adjacent blocks sit at unit frequency and mapped through the heatmap frequency law. |
| `arcs_weight` | float | 1.0 | 1.0 | Weight of those targets. |
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
| `arcs_repulsion_cutoff_factor` | float | 0.0 | 3.0 | Truncate the arcless pair repulsion `1 / d` beyond this times the mean arc target. 0 leaves it unbounded, which is the reference's behaviour and lets a sparse block explode. |
| `use_genomic_floor` | bool | no | no | Give every arcless anchor pair an excluded volume radius `scale * (separation / 1000) ^ exponent` in the arcs stage, and retire the `1 / d` for them. Sets block size rather than shape. The arcs solver does not implement it. |
| `genomic_floor_factor` | float | 0.44 | 0.44 | The scale is this times the median consecutive anchor distance of a first pass. |
| `genomic_floor_exponent` | float | 0.285 | 0.285 | Separation exponent of the floor. |
| `genomic_floor_scale` | float | 0.0 | 0.0 | An explicit scale in model units, overriding the calibration. |
| `genomic_floor_polish_temp` | float | 0.0 | 0.0 | Starting temperature of the second pass, as a fraction of `max_temp`. |
| `genomic_floor_weight` | float | 1.0 | 1.0 | Weight of the floor term, separate from `weight`. |

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

## How the distance targets fit together

The pipeline has several laws that turn something about the data into a distance in model
units, and they meet in the arcs stage's target matrix. This is the part of the configuration
where a wrong reading costs the most, so it is laid out before the tables. Each law is given
first in plain terms with a worked number, then precisely with its source. The chain law, the
PET law, the heatmap frequency law and the anchor heatmap scaling are 3D-GNOME's own, from the
modelling engine paper [1] and carried through its later releases [2, 3]. The separation aware
and unified laws are this project's, and they add no physics of their own. Each applies two
established results to the arc targets. Mean spatial distance grows with genomic separation as
a power law [4, 5], and a contact's meaning depends on how much more often the pair meets than
two loci that far apart meet anyway, which is the observed over expected normalisation that
loop calling [8] and restraint based modelling [9, 10] both rest on. What is this project's
alone is the parameterisation, and that is said where it applies.

One number to hold onto. Chromatin in a nucleus is crumpled, not stretched out, so distance
grows only slowly with the amount of DNA in between. Ten times more DNA between two loci puts
them only about twice as far apart. Written as a power law that is an exponent near 0.285, and
it is the target every distance law here is measured against. It comes from Hi-C. Contact
probability falls close to the inverse of separation [4], and if distance grows as separation
to the power `nu`, contact falls as separation to the power `-3 nu` [5]. This project's three
cell lines measure a contact slope near `-0.86`, which gives `nu` near 0.285.

**The polymer law**, `[distance] use_polymer_law`. The one to use.

In plain terms. Everything below collapses into one rule, in one unit. The unit is the bead,
the distance two beads hold at the resolution the run declared, so nothing is measured in a
number that means nothing. Two loci with nothing between them sit at the crumpled chromatin
distance for how much DNA separates them, and that exponent is read off the run's own
contacts rather than typed in. A loop draws its two ends in from there, more for a loop
stronger than is typical at its span, and never closer than touching. Turning it on retires
every other law in this section and every constant they carry. It reads three keys, the
exponent, which is measured unless pinned, and one shape number for how hard a loop pulls.

Precisely. `d = max(1, (s / s0) ^ nu)` for a pair at separation `s` with no contact, `s0`
being `target_bp_per_subanchor`. With a loop of strength `q`, `d = 1 + (background - 1) /
(1 + q / q_half)`, where `q` is the loop's PET count over the typical count at its span fitted
on the run's own arcs, which is observed over expected [8, 9]. A heatmap cell sits at the
background times observed over expected to the minus third, the fractal globule relation
[4, 5], the expectation being the mean of the cells at that separation within the heatmap. The
exponent `nu` is the slope of log contact count against log separation on the run's singletons,
over minus three [4, 5], measured at load. When the singletons cannot supply one, which a
ChIA-PET singletons file cannot since it is enrichment filtered around CTCF and does not decay,
the run says so in an always visible line and uses 0.285, the fractal globule value.

Measured on this project's own inputs the exponent is 0.275 for GM12878, 0.299 for H1ESC, 0.192
for HFFC6 and 0.072 for a trio sample, which is why it is measured and not a constant.

**The chain law**, `[distance] genomic_dist_*`. Parity law, not read when the polymer law is on.

In plain terms. The chain is a rope. Two beads next to each other on it are held a set
distance apart, and that distance is larger the more DNA lies between them. Under the
production values two beads 1 kb apart are held 1.5 units apart, 20 kb apart 5.7 units, and
1 Mb apart 90 units. That last number is the catch. Ninety is sixty times 1.5, but crumpled
chromatin would put loci 1 Mb apart only about seven times further than loci 1 kb apart. The
rope is right for neighbours and far too stiff over long stretches.

Precisely. `base + scale * (bp / 1000) ^ power` [1]. It sets the bond between consecutive
beads in the smooth stage, the bond between consecutive block centroids in block placement, the
bond a consecutive arcless anchor pair gets in the arcs stage when `use_arcs_chain_bonds` is
on, and the background of the unified arc target. It is calibrated for consecutive beads a kb
or so apart. Its exponent under the production values is 0.75 against the 0.285 above, which
matters wherever it is evaluated at large separations.

**The PET law**, `[distance] count_dist_*`. Parity law, not read when the polymer law is on.

In plain terms. An arc is a loop the experiment saw, and the PET count is how many times it
saw it. The more often a loop was seen, the closer its two ends are pulled. Seen twice, the
ends are asked to sit 0.44 units apart. Seen fifty times, 0.20. Two catches. Those distances
are a fraction of one bead's size, which is about 1.5 units, so the law asks the two ends to
sit inside each other. And the law says the same distance whether the loop spans 5 kb or
1 Mb, because it never looks at the span.

Precisely. `base_level + scale / exp(a * (PET + shift))` [1]. It runs from its zero PET value
down to `base_level` as the count grows, which under the production values is 0.56 down to
0.20.

**The separation aware law**, `[distance] use_separation_arc_target`. Not read when the polymer law is on.

In plain terms. The PET law with one correction. A longer loop is allowed to be longer. A loop
spanning 1 Mb is asked for about 3.7 times the distance of one spanning 10 kb with the same
PET count, which is the crumpled chromatin scaling above. Loops under 10 kb are left as the
PET law had them. This fixes the second catch and not the first. A 1 Mb loop seen four times is
now asked for 1.35 units instead of 0.36, still less than one bead.

Precisely. The PET law times `max(1, s_kb / pivot) ^ exponent`. The factor is the polymer
scaling of mean distance with separation [4, 5], applied to the arc target so that two arcs of
equal PET count but different span are no longer asked for the same distance. It gives the
target the polymer slope and leaves its scale where the PET law put it. The pivot of 10 kb is
a choice of this project.

**The unified law**, `[distance] use_unified_arc_target`. Superseded by the polymer law, which is the same idea with the scale and the exponent taken from the data instead of from constants.

In plain terms. Start from where two loci would sit anyway, given how much DNA lies between
them and nothing holding them together. Call that the background. Then, if a loop joins them,
pull them in from the background, more for a loop seen more often, but never closer than
touching. Under the production values two loci 50 kb apart sit at about 10 units with no loop
between them. With the strongest possible loop they are pulled to 0.45 of that, about 4.7
units, a few beads apart and well clear of sitting inside each other. A loop seen a few times
lands in between. The rope sets how far apart two points
can be, a loop is a clip that draws them together, and two beads cannot pass through each
other. The same background sets the bond between neighbouring anchors with no loop, so
everything in the block is measured on one scale.

Precisely. This is the form restraint based genome modelling generally takes. A pair sits at
the expected distance for its separation [4, 5], and its contact only says how far in from that
background to pull. That is observed over expected [8] written as a distance: TADbit sets its
restraints from a pair's contact relative to the expectation at its separation [9], and
distance inference methods map contact to distance through the same power law [10]. A contact
pulls a pair toward touching and no further, since two beads cannot be closer than their own
size, which is how polymer loop models hold loop anchors [6, 11], so the pull is bounded below
by `arc_target_pull` of the background. The PET law supplies the pull factor rather than the
distance, normalised by its own two limits, which reuses 3D-GNOME's calibrated shape of PET
count against strength [1]. The background is the chain law, or with
`arc_target_background_exponent` set, the chain law's value at `arc_target_background_ref_bp`
continued at that exponent, which is the power law of [4, 5] anchored where the chain law is
calibrated. When it is on, the chain bond a consecutive arcless anchor pair gets rides the same
background, so the two families agree in scale and in slope. It supersedes the separation aware
law. The pull of 0.45 and the exponent are this project's, chosen on measurement against Hi-C
rather than derived.

**The anchor heatmap**, `[anchor_heatmap]`.

In plain terms. If Hi-C also sees a pair of anchors touching often, shrink their target a
little more, by up to a tenth under the production values. It is a nudge on top of whichever
arc law is on, not a law of its own.

Precisely. Scales an arc target down by up to `heatmap_influence` in proportion to the pair's
singleton contact count relative to the region's maximum [1]. It runs after the arc targets are
set and before the chain bonds are added, so contact between neighbours does not shrink a
bond.

**The arcs stage matrix**, what every anchor pair ends up with.

In plain terms. Every pair of anchors in a block is one of three things. Joined by a loop, and
pulled to the loop distance. Next to each other on the chain with no loop, and held at the rope
distance. Neither, and given no target at all, only a push apart if they come very close.

Precisely. A pair an arc joins carries the arc target, from whichever law is on. A consecutive
pair with no arc carries `arcs_chain_bond_scale` times the chain law, or times the unified
background, when `use_arcs_chain_bonds` is on. Every other pair carries no target and feels
only the repulsion [1], `max(0, 1 / d - 1 / (arcs_repulsion_cutoff_factor * mean arc
target))`, unbounded when the factor is zero, plus the genomic floor when `use_genomic_floor`
is on, an excluded volume radius per pair that grows with separation.

**The heatmap frequency law**, `[distance] freq_dist_*`. Parity law, not read when the polymer law is on.

In plain terms. This one is for the coarse map, not for loops. The genome is first laid out as
big segments, and two segments that share more contacts are placed closer. Contact drops off
as the cube of distance, so distance is taken as contacts to the power of minus one third.
Twice the contacts means about a fifth closer.

Precisely. `scale * freq ^ power` for the segment level heatmap, and the `_inter` pair does
the same for the chromosome level heatmap [1]. The default power of minus one third is the
fractal globule relation, contact falling as the cube of distance [4, 5], and the same
exponent distance inference methods use [10]. It is not used for arcs.

**The smooth stage**, where the anchors stop moving.

In plain terms. Once the arcs stage has placed the anchors, they are nailed down. The smooth
stage only threads the beads between them along the rope, bending as little as it can, and
packs the rope tighter where the chromatin is closed. Nothing after the arcs stage moves an
anchor, so whatever the arcs stage asked for is what the structure has.

Precisely. Consecutive beads are held at the chain law of their gap, compacted by
`fibre_compaction` where accessibility is low following HiP-HoP [6], and every anchor is held
fixed.

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
