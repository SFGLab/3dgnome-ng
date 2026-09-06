# Settings reference

Every key the pipeline reads from a config, what it does, and what it is set to. Two values are
given where they differ. The default is what `Settings()` holds with no config at all. The
production value is what `validation/core/config.py::CANONICAL` sets, which is where every
ensemble config under `slurm/ensemble/` and every validation run is generated from. Change
`CANONICAL` and regenerate rather than editing a generated ini.

A config is an ini file. `Settings().load_ini(path)` reads it and `Settings.from_dict(mapping)`
builds one from a nested `{section: {key: value}}` mapping. A key present in the file that no
reader consults is reported by `_warn_unknown_keys`, so a misspelled key is a warning rather
than a silent no-op. Booleans accept `yes`, `no`, `true`, `false`, `1`, `0`. The python
attribute column is the field on `Settings` that carries the value, which is what code and the
validation helpers use. It is often not the same word as the ini key.

Some keys are read and never consulted. They are kept because the reference declares them and
a config carrying them should not warn, and each is marked as such below rather than given a
meaning it does not have.

## How the distance targets fit together

The pipeline has several laws that turn something about the data into a distance in model
units, and they meet in the arcs stage's target matrix. This is the part of the configuration
where a wrong reading costs the most, so it is laid out before the tables.

**The chain law**, `[distance] genomic_dist_*`, gives the distance between two beads from their
genomic separation, `base + scale * (bp / 1000) ^ power`. It sets the bond between consecutive
beads in the smooth stage, the bond between consecutive block centroids in block placement, the
bond a consecutive arcless anchor pair gets in the arcs stage when `use_arcs_chain_bonds` is
on, and the background of the unified arc target. It is calibrated for consecutive beads a kb or
so apart. Its exponent under the production values is 0.75, and the distance it gives grows far
faster with separation than the 0.285 the cell lines' contact probability curves give, which
matters wherever it is evaluated at large separations.

**The PET law**, `[distance] count_dist_*`, gives an arc's target from its PET count alone,
`base_level + scale / exp(a * (PET + shift))`. It runs from `freq_to_distance(0)` down to
`base_level` as the count grows, which under the production values is 0.56 down to 0.20. That
scale is a fraction of one chain bond, so on its own it asks two anchors an arc joins to sit
inside each other.

**The separation aware law**, `[distance] use_separation_arc_target`, multiplies the PET law by
`max(1, s_kb / pivot) ^ exponent`. It gives an arc's target the right slope in separation and
leaves its scale where the PET law put it.

**The unified law**, `[distance] use_unified_arc_target`, replaces both. A pair sits on a
background distance for its separation, and its PET count only says how far in to pull from
there, between `arc_target_pull` and 1. The PET law supplies that factor rather than the
distance, normalised by its own limits. The background is the chain law, or with
`arc_target_background_exponent` set, the chain law's value at `arc_target_background_ref_bp`
continued at that exponent. When it is on the chain bond a consecutive arcless anchor pair gets
rides the same background, so the two families agree in scale and in slope. It supersedes the
separation aware law.

**The anchor heatmap**, `[anchor_heatmap]`, then scales an arc target down by up to
`heatmap_influence` in proportion to the pair's Hi-C contact. It runs after the arc targets are
set and before the chain bonds are added, so contact between neighbours does not shrink a bond.

**The arcs stage matrix**, per anchor pair, is therefore one of three things. A pair an arc
joins carries the arc target, from whichever law is on. A consecutive pair with no arc carries
`arcs_chain_bond_scale` times the chain law, or times the unified background, when
`use_arcs_chain_bonds` is on. Every other pair carries no target and feels only the repulsion,
`max(0, 1 / d - 1 / (arcs_repulsion_cutoff_factor * mean arc target))`, unbounded when the
factor is zero, plus the genomic floor when `use_genomic_floor` is on, an excluded volume radius
per pair that grows with separation.

**The heatmap frequency law**, `[distance] freq_dist_*`, is a different thing. It turns an
aggregate singleton contact count into a distance, `scale * freq ^ power`, for the segment level
heatmap, and the `_inter` pair does the same for the chromosome level heatmap. It is not used
for arcs.

**The smooth stage** holds consecutive beads at the chain law of their gap, compacted by
`fibre_compaction` where accessibility is low, and holds every anchor fixed. So an anchor's
position is set in the arcs stage and nothing after it moves an anchor. What the arcs stage
matrix asks for is what the structure gets.

## [main]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `output_level` | int | 0 | 1 | `output_level` | Log verbosity. 0 prints the run banner and warnings, 1 adds milestones and per stage headers, 2 adds per batch MC step lines. |
| `log_file` | str | | | `log_file` | Path of a full detail structured log written alongside the console output. Empty writes none. |
| `random_walk` | bool | no | no | `random_walk` | Place segment beads by a chained random walk of fixed 50.0 steps per chromosome instead of the segment heatmap MC. Honours `use_2D`. |
| `use_2D` | bool | no | no | `use_2d` | Restrict every move to the plane. |
| `loop_density` | int | 5 | 5 | `loop_density` | Subanchors inserted between every consecutive anchor pair when dynamic density is off. |
| `max_pet_length` | int | 1000000 | 1000000 | `max_pet_length` | An arc spanning more than this many bp is a long arc. It is not a spring and is folded into the segment heatmap instead. |
| `long_pet_power` | float | 2.0 | 2.0 | `long_pet_power` | A long arc adds `long_pet_scale * score ^ long_pet_power` to the segment heatmap. |
| `long_pet_scale` | float | 10.0 | 1.0 | `long_pet_scale` | See above. |
| `steps_lvl1` | int | 2 | 1 | `steps_lvl1` | Chromosome level heatmap MC runs, best kept. Multi chromosome runs only. |
| `steps_lvl2` | int | 2 | 1 | `steps_lvl2` | Segment level heatmap MC runs, best kept. |
| `steps_arcs` | int | 5 | 1 | `steps_arcs` | Restarts of the arcs stage per block, each from a freshly noised start, best kept. Applies to the solver as well as the annealer. |
| `steps_smooth` | int | 5 | 1 | `steps_smooth` | Restarts of the smooth stage per block, best kept. |
| `noise_lvl1` | float | 1.0 | 0.5 | `noise_lvl1` | Chromosome level step size, as a multiple of the mean target distance. |
| `noise_lvl2` | float | 0.1 | 0.5 | `noise_lvl2` | Segment level step size, as a multiple of the mean target distance. |
| `noise_smooth` | float | 0.5 | 5.0 | `noise_smooth` | Smooth stage step size, as a multiple of the mean chain bond target. |
| `noise_ib` | float | 0.5 | 0.5 | `noise_ib` | Block placement step size, as a multiple of the mean block chain bond. |
| `overlap_anchor_strict` | bool | no | no | `overlap_anchor_strict` | The reference's span rule for overlapping anchors, which collapses the subanchors between them to one point. Off tiles the overlap with non degenerate ranges. |
| `drop_zero_length_subanchors` | bool | no | yes | `drop_zero_length_subanchors` | Leave zero width subanchors out of the written structure. The chain still carries them. |
| `use_dynamic_loop_density` | bool | no | yes | `use_dynamic_loop_density` | Pick the subanchor count per anchor gap so that each chain segment is about `target_bp_per_subanchor` long, instead of a fixed `loop_density`. |
| `target_bp_per_subanchor` | int | 5000 | 1000 | `target_bp_per_subanchor` | Target genomic length of one chain segment under dynamic density. |
| `min_subanchors_per_arc` | int | 0 | 0 | `min_subanchors_per_arc` | Lower bound on the count per gap under dynamic density. |
| `max_subanchors_per_arc` | int | 50 | 100 | `max_subanchors_per_arc` | Upper bound on the count per gap under dynamic density. |
| `step_decay_floor` | float | 0.1 | 0.1 | `mc_step_decay_floor` | Lower bound on the step size under the per stage `step_decay` keys, as a fraction of the starting step. |

## [data]

Filenames are relative to `data_dir` unless absolute. The region string is `chr:start-end`.

| key | type | default | attribute | what it does |
|---|---|---|---|---|
| `data_dir` | str | | `data_dir` | Directory the other filenames resolve against. The CLI's `--data-dir` overrides it. |
| `anchors` | str | | `data_anchors` | BED of loop anchors, `chr start end orientation`. |
| `clusters` | str | | `data_pet_clusters` | BEDPE of PET clusters, the arcs, `chr1 s1 e1 chr2 s2 e2 score`. |
| `singletons` | str | | `data_singletons` | BEDPE of singleton contacts for the segment level heatmap. A Hi-C bin pair file works here too. |
| `singletons_inter` | str | | `data_singletons_inter` | A second singletons file appended for multi chromosome runs only. |
| `factors` | str | | `data_factors` | Read and never consulted. The reference's protein factor filter. |
| `split_singleton_files_by_chr` | bool | no | `data_split_singletons_by_chr` | Read and never consulted. |
| `centromeres` | str | | `data_centromeres` | BED of centromere positions. |
| `segment_split` | str | | `data_segment_split` | BED of segment boundary breakpoints. |
| `segment_heatmap` | str | | `data_segment_heatmap` | Read and never consulted. |
| `compartments` | str | | `data_compartments` | bedGraph of a signed compartment eigenvector or a CALDER BED, for `[compartments]`. |
| `accessibility` | str | | `data_accessibility` | bedGraph of ATAC or DNase signal, for `[accessibility]`. |
| `phasing_track` | str | | `data_phasing_track` | Track used to fix the eigenvector's arbitrary sign. Required with `compartments`. |

## [distance]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `genomic_dist_base` | float | 0.0 | 1.0 | `genomic_dist_base` | Chain law, `base + scale * (bp / 1000) ^ power`. |
| `genomic_dist_scale` | float | 1.0 | 0.5 | `genomic_dist_scale` | Chain law. |
| `genomic_dist_power` | float | 0.5 | 0.75 | `genomic_dist_power` | Chain law. |
| `freq_dist_scale` | float | 100.0 | 25.0 | `freq_dist_scale` | Segment heatmap frequency to distance, `scale * freq ^ power`. |
| `freq_dist_power` | float | -0.333 | -0.6 | `freq_dist_power` | Segment heatmap frequency to distance. |
| `freq_dist_scale_inter` | float | 100.0 | 120.0 | `freq_dist_scale_inter` | The same for the chromosome level heatmap. |
| `freq_dist_power_inter` | float | -1.0 | -1.0 | `freq_dist_power_inter` | The same for the chromosome level heatmap. |
| `count_dist_a` | float | 0.5 | 0.2 | `count_dist_a` | PET law, `base_level + scale / exp(a * (PET + shift))`. |
| `count_dist_scale` | float | 20.0 | 1.8 | `count_dist_scale` | PET law. |
| `count_dist_shift` | float | 1.0 | 8.0 | `count_dist_shift` | PET law. |
| `count_dist_base_level` | float | 0.01 | 0.2 | `count_dist_base_level` | PET law. The distance a saturated arc asks for, and the lower limit the unified law normalises against. |
| `use_separation_arc_target` | bool | no | yes | `use_separation_arc_target` | Multiply the PET law by `max(1, s_kb / arc_target_pivot_kb) ^ arc_target_exponent`. Superseded by the unified law when both are on. |
| `arc_target_exponent` | float | 0.285 | 0.285 | `arc_target_exponent` | Separation aware exponent. |
| `arc_target_pivot_kb` | float | 10.0 | 10.0 | `arc_target_pivot_kb` | Span below which the separation aware law leaves the PET law alone. |
| `use_unified_arc_target` | bool | no | no | `use_unified_arc_target` | One background for arc targets and chain bonds, with the PET count setting only the pull. See the section above. Measured a strict improvement on GM12878 and H1ESC at a background exponent of 0.15. See `design/anchor-placement.md`, option G. |
| `arc_target_pull` | float | 0.45 | 0.45 | `arc_target_pull` | The fraction of the background a saturated arc pulls its pair to. A zero PET arc sits on the background. |
| `arc_target_background_exponent` | float | 0.0 | 0.0 | `arc_target_background_exponent` | Zero rides the chain law itself, whose exponent is far steeper than the contact probability curves give. A positive value continues the chain law from the reference separation at this slope. 0.15 measured best on two cell lines. The optimum differs by dataset. |
| `arc_target_background_ref_bp` | int | 1000 | 1000 | `arc_target_background_ref_bp` | The separation at which the background agrees with the chain law when an exponent is set. |

## [template]

All four keys are read and never consulted. They are the reference's structural template and
MDS distance heatmap inputs, which are not ported. `CANONICAL` sets `template_scale` and
`dist_heatmap_scale` and neither has any effect.

| key | type | default | attribute |
|---|---|---|---|
| `template_segment` | str | | `template_segment` |
| `template_scale` | float | 1.0 | `template_scale` |
| `dist_heatmap` | str | | `dist_heatmap` |
| `dist_heatmap_scale` | float | 1.0 | `dist_heatmap_scale` |

## [heatmaps]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `inter_scaling` | float | 1.0 | 1.0 | `heatmap_inter_scaling` | Multiply the inter chromosome blocks of the segment heatmap. Intra blocks are unchanged. Multi chromosome runs only. |
| `distance_heatmap_stretching` | float | 2.0 | 2.5 | `heatmap_distance_stretching` | Clip segment heatmap distances above the mean positive distance times this. |

## [springs]

Spring energy is `k * ((d - target) / target) ^ 2` with `k` the stretch constant when the pair
is too far and the squeeze constant when too close.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `stretch_constant` | float | 0.1 | 0.1 | `spring_stretch` | Smooth stage chain bond, too far. |
| `squeeze_constant` | float | 0.1 | 0.1 | `spring_squeeze` | Smooth stage chain bond, too close. |
| `angular_constant` | float | 0.1 | 0.1 | `spring_angular` | Smooth stage bend penalty, the cube of the angle between consecutive bonds. |
| `stretch_constant_arcs` | float | 1.0 | 1.0 | `spring_stretch_arcs` | Arcs stage, every target in the matrix, arcs and chain bonds alike. |
| `squeeze_constant_arcs` | float | 1.0 | 1.0 | `spring_squeeze_arcs` | Arcs stage. |
| `use_arcs_chain_bonds` | bool | no | yes | `use_arcs_chain_bonds` | Give every consecutive anchor pair with no arc a spring at the chain law of its gap, entered after the anchor heatmap scaling. |
| `arcs_chain_bond_scale` | float | 1.0 | 1.5 | `arcs_chain_bond_scale` | Multiplier on that bond's target. |
| `stretch_constant_ib` | float | 0.1 | 0.1 | `spring_stretch_ib` | Block placement chain bond. |
| `squeeze_constant_ib` | float | 0.1 | 0.1 | `spring_squeeze_ib` | Block placement chain bond. |

## [motif_orientation]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_motif_orientation` | bool | no | yes | `use_ctcf_motif` | Score the angle between the CTCF motif vectors of anchors an arc joins, in the smooth stage. |
| `weight` | float | 1.0 | 50.0 | `motif_weight` | Weight of that term. |
| `symmetric_motifs` | bool | yes | yes | `motifs_symmetric` | Yes scores two motif vectors best when they point the same way. No flips the partner first, so opposed vectors score best. |

## [anchor_heatmap]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_anchor_heatmap` | bool | no | yes | `use_anchor_heatmap` | Scale each arc target down in proportion to its pair's singleton contact count relative to the region's maximum. |
| `heatmap_influence` | float | 0.5 | 0.1 | `anchor_heatmap_influence` | The most a target can be scaled down, as a fraction. |

## [subanchor_heatmap]

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_subanchor_heatmap` | bool | no | yes | `use_subanchor_heatmap` | Add a contact distance target between subanchor pairs in the smooth stage, estimated from dry smooth passes and scaled by contact. |
| `heatmap_influence` | float | 0.5 | 0.1 | `subanchor_heatmap_influence` | The most a pair's estimated distance is scaled down by its contact. |
| `heatmap_dist_weight` | float | 1.0 | 0.01 | `subanchor_heatmap_dist_weight` | Weight of that term against the chain and angle terms. |
| `estimate_distances_steps` | int | 2 | 4 | `subanchor_estimate_steps` | Dry smooth runs per replicate when estimating the mean pairwise distances. |
| `estimate_distances_replicates` | int | 5 | 4 | `subanchor_estimate_replicates` | Replicate estimates averaged into the target. |
| `batch_trials` | bool | no | | `subanchor_batch_trials` | Read and never consulted. |
| `heat_min_reduction` | float | 0.0 | 0.0 | `subanchor_heat_min_reduction` | Skip the estimate for a block whose fraction of contact carrying pairs, an upper bound on the reduction the term can achieve, is below this. |
| `heatmap_workers` | int or auto | 1 | | `heatmap_workers` | Threads used to build the per block contact heatmaps. `auto` uses every usable core. |

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

The attributes are `max_temp`, `dt_temp`, `jump_scale`, `jump_coef`, `mc_stop_steps`,
`mc_stop_improvement`, `mc_stop_successes` for the arcs stage, and the same with `_smooth`,
`_ib` and `_heatmap` suffixes for the others. `step_decay` is `mc_step_decay_arcs`,
`mc_step_decay_smooth` and `mc_step_decay_ib`.

### [simulation_arcs] only

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `stop_condition_ratio` | float | 0.9999 | 0.9999 | `mc_stop_ratio_arcs` | Also stop when the score over the previous round's is at or above this, a plateau guard. |
| `force_bias` | float | 0.0 | 0.0 | `arcs_force_bias` | Steer each proposal along the local gradient by this fraction of the step. 0 is a plain random step. |
| `solver` | str | mc | lbfgs | `arcs_solver` | `mc` anneals, `lbfgs` minimises the same energy with L-BFGS-B. Same minimum, same overlaps, the stage's calls fell from minutes to seconds. Needs `mc_executor_arcs` of `serial` or `threaded`. The batch executor has no solver and refuses. |
| `solver_iters` | int | 200 | 200 | `arcs_solver_iters` | Iterations for the solver. |

### [simulation_arcs_smooth] only

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `dist_weight` | float | 1.0 | 1.0 | `smooth_dist_weight` | Weight of the chain bond term. |
| `angle_weight` | float | 1.0 | 1.0 | `smooth_angle_weight` | Weight of the bend term. |

### [simulation_ib] only

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `refine_scope` | str | segment | segment | `ib_refine_scope` | `segment` places each segment's blocks as one chain and skips a segment with one block or fewer. `chromosome` places every block on the chromosome as one chain. It inflates structures and needs the excluded volume and confinement retuned. |
| `use_ib_mc` | bool | no | yes | `use_ib_mc` | Anneal block centroids with chain bonds, excluded volume and confinement instead of placing them by interpolation. |
| `use_ib_arcs` | bool | no | no | `use_ib_arcs` | Add attraction only targets between block centroids from the arcs crossing their boundary, normalised so adjacent blocks sit at unit frequency and mapped through the heatmap frequency law. |
| `arcs_weight` | float | 1.0 | 1.0 | `ib_arcs_weight` | Weight of those targets. |
| `dist_weight` | float | 1.0 | 1.0 | `dist_weight_ib` | Weight of the chain bond term. |

## [simulation_backend]

Each stage runs on an executor. `serial` and `threaded` are the numba kernels, one chain per
thread. `batch` is the JAX kernels, which pack many blocks into one device launch. `auto`
resolves from the older backend keys.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `ib_workers` | int or auto | 1 | auto | `mc_executor_threaded_workers` | Threads for the threaded executor. `auto` uses every usable core. |
| `heatmap_chains` | int | 1 | 1 | `mc_heatmap_chains` | Independent heatmap MC chains run at once, best kept. |
| `smooth_chains` | int | 1 | 1 | `mc_smooth_chains` | Independent smooth chains run at once per block, best kept. |
| `mc_executor_arcs` | str | auto | threaded | `mc_executor_arcs` | Executor for the arcs stage. Threaded on the CPU because a vmapped launch cannot retire a converged block and one straggler holds every other block in the launch. |
| `mc_executor_densify` | str | auto | threaded | `mc_executor_densify` | Executor for densification. |
| `mc_executor_estimate_dist` | str | auto | batch | `mc_executor_estimate_dist` | Executor for the subanchor distance estimate. |
| `mc_executor_smooth` | str | auto | batch | `mc_executor_smooth` | Executor for the smooth stage. The cross block relaxation also picks its kernel from this, and one chain on the batch kernel is that kernel's worst case. |
| `mc_executor_jax_bucket_shapes` | bool | no | yes | `mc_executor_jax_bucket_shapes` | Pad each block to a shape ladder so a stage is a few wide launches rather than one compile per size. |
| `merge_smooth_launches` | bool | yes | yes | `merge_smooth_launches` | Pack every smooth block that agrees on its energy terms into as few launches as device memory allows, regardless of size. |
| `mc_executor_jax_batch_width_smooth` | int or auto | auto | auto | `mc_executor_jax_batch_width_smooth` | Blocks per smooth launch. `auto` solves the largest count that fits the device. |
| `mc_executor_jax_batch_width_arcs` | int or auto | auto | auto | `mc_executor_jax_batch_width_arcs` | The same for the JAX arcs kernel. |
| `multigpu_mode` | str | groups | groups | `mc_multigpu_mode` | `groups` runs whole batch groups on different devices and is byte exact in the device count. `within` splits one group across devices. `off` uses one device. |
| `neighbour_grid` | bool | yes | yes | `mc_neighbour_grid` | Cell grid for the excluded volume term in the numba kernels. Identical results, faster above 2048 beads. |

## [excluded_volume]

Soft repulsion `weight * ((r0 - d) / r0) ^ 2` for pairs closer than `r0` and more than
`skip_neighbors` apart along the chain. Each stage has its own radius. A radius of zero derives
one as the stage's auto factor times the stage's mean bond scale.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_excluded_volume` | bool | no | yes | `use_excluded_volume` | Master switch. |
| `weight` | float | 0.5 | 0.1 | `exclusion_weight` | Weight. The term saturates at this value for a full overlap. |
| `apply_to_arcs` | bool | no | yes | `exclusion_apply_to_arcs` | In the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | `exclusion_apply_to_smooth` | In the smooth stage. |
| `apply_to_heatmap` | bool | no | yes | `exclusion_apply_to_heatmap` | In the heatmap stages. |
| `apply_to_ib` | bool | yes | yes | `exclusion_apply_to_ib` | In block placement. |
| `skip_neighbors` | int | 1 | 1 | `exclusion_skip_neighbors` | Pairs this close along the chain are exempt. |
| `radius_arcs` | float | 0.0 | 0.0 | `exclusion_radius_arcs` | Arcs stage radius, 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | `exclusion_radius_smooth` | Smooth stage radius. |
| `radius_heatmap` | float | 0.0 | 0.0 | `exclusion_radius_heatmap` | Heatmap stage radius. |
| `radius_ib` | float | 0.0 | 0.0 | `exclusion_radius_ib` | Block placement radius. Also the constant centroid radius of the boundary stitch when positive. |
| `auto_factor_arcs` | float | 0.5 | 0.5 | `exclusion_auto_factor_arcs` | Times the mean positive arc target. |
| `auto_factor_smooth` | float | 0.5 | 0.7 | `exclusion_auto_factor_smooth` | Times the mean chain bond target of the block. |
| `auto_factor_heatmap` | float | 0.5 | 0.5 | `exclusion_auto_factor_heatmap` | Times the mean active heatmap target. |
| `auto_factor_ib` | float | 0.5 | 0.5 | `exclusion_auto_factor_ib` | Times the mean block chain bond. |
| `arcs_repulsion_cutoff_factor` | float | 0.0 | 3.0 | `arcs_repulsion_cutoff_factor` | Truncate the arcless pair repulsion `1 / d` beyond this times the mean arc target. 0 leaves it unbounded, which is the reference's behaviour and lets a sparse block explode. |
| `use_genomic_floor` | bool | no | no | `use_genomic_floor` | Give every arcless anchor pair an excluded volume radius `scale * (separation / 1000) ^ exponent` in the arcs stage, and retire the `1 / d` for them. Sets block size rather than shape. The arcs solver does not implement it. |
| `genomic_floor_factor` | float | 0.44 | 0.44 | `genomic_floor_factor` | The scale is this times the median consecutive anchor distance of a first pass. |
| `genomic_floor_exponent` | float | 0.285 | 0.285 | `genomic_floor_exponent` | Separation exponent of the floor. |
| `genomic_floor_scale` | float | 0.0 | 0.0 | `genomic_floor_scale` | An explicit scale in model units, overriding the calibration. |
| `genomic_floor_polish_temp` | float | 0.0 | 0.0 | `genomic_floor_polish_temp` | Starting temperature of the second pass, as a fraction of `max_temp`. |
| `genomic_floor_weight` | float | 1.0 | 1.0 | `genomic_floor_weight` | Weight of the floor term, separate from `weight`. |

## [confinement]

Soft envelope `weight * ((r - R) / R) ^ 2` for a bead further than `R` from the centroid of the
stage's starting positions. A radius of zero derives one as the packing factor times the
stage's mean bond scale times the cube root of the bead count.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_confinement` | bool | no | yes | `use_confinement` | Master switch. |
| `weight` | float | 0.5 | 0.1 | `confinement_weight` | Weight. |
| `apply_to_arcs` | bool | yes | yes | `confinement_apply_to_arcs` | In the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | `confinement_apply_to_smooth` | In the smooth stage. |
| `apply_to_ib` | bool | yes | yes | `confinement_apply_to_ib` | In block placement. |
| `radius_arcs` | float | 0.0 | 0.0 | `confinement_radius_arcs` | Arcs stage radius, 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | `confinement_radius_smooth` | Smooth stage radius. |
| `radius_ib` | float | 0.0 | 0.0 | `confinement_radius_ib` | Block placement radius. |
| `packing_factor_arcs` | float | 1.5 | 1.5 | `confinement_packing_factor_arcs` | Arcs stage packing factor. |
| `packing_factor_smooth` | float | 1.5 | 1.5 | `confinement_packing_factor_smooth` | Smooth stage packing factor. |
| `packing_factor_ib` | float | 0.75 | 0.75 | `confinement_packing_factor_ib` | Block placement packing factor. Below about 0.58 a small segment is asked to fold tighter than one of its own bonds, and 0.15 crushed the cross block distance scaling. |

## [boundary_stitch]

Runs after every chain of a chromosome is done. Moves each block as a rigid body so the last
anchor of one block and the first of the next sit at the distance the structure's own interior
pairs realise at that separation, with a soft excluded volume between block centroids. No RNG.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_boundary_stitch` | bool | no | yes | `use_boundary_stitch` | Master switch. |
| `spring_weight` | float | 1.0 | 1.0 | `boundary_stitch_spring_weight` | Weight of the boundary springs. |
| `ev_weight` | float | 1.0 | 1.0 | `boundary_stitch_ev_weight` | Weight of the centroid excluded volume. |
| `max_iter` | int | 2000 | 2000 | `boundary_stitch_max_iter` | L-BFGS-B iterations. The energy carries its own gradient, so an iteration is one evaluation. 500 leaves a chromosome unconverged. 2000 converges a 1,494 block chromosome in 85 seconds. |

## [relax]

Runs after the stitch. The smooth kernel once over the whole chromosome with excluded volume on
every pair and every anchor held fixed, so coils from different blocks stop passing through each
other while the arcs and the stitch are kept.

| key | type | default | production | attribute | what it does |
|---|---|---|---|---|---|
| `use_cross_block_relax` | bool | no | yes | `use_cross_block_relax` | Master switch. |
| `ev_weight` | float | 10.0 | 10.0 | `relax_ev_weight` | Excluded volume weight for the pass. |
| `ev_radius` | float | 0.0 | 0.0 | `relax_ev_radius` | Excluded volume radius. 0 uses 1.5 chain bonds, so nothing is left under one bond where contacts are counted. |
| `temp` | float | 0.1 | 0.1 | `relax_temp` | Starting temperature as a fraction of the smooth stage's `max_temp`. Untangling needs a bead to cross a neighbour's shell, and a greedy pass stalls. |
| `noise` | float | 0.5 | 0.5 | `relax_noise` | Step size in chain bonds. |
| `bond_weight` | float | 10.0 | 10.0 | `relax_bond_weight` | Chain spring weight for the pass. At the smooth stage's 0.1 the excluded volume tears the coil. |
| `min_contact_fraction` | float | 0.0 | 0.0 | `relax_min_contact_fraction` | Decline the pass when cross block contacts are fewer than this fraction of the chromosome's beads. 0 always runs. |
| `local_window` | int | -1 | -1 | `relax_local_window` | Let only the beads touching another block move, plus this many chain neighbours either side. -1 lets every subanchor move, which on a chromosome is hours. A window of 1 is minutes. |

## [compartments]

A/B compartment terms ported from MultiMM, written shifted and non negative so the Metropolis
ratio stays defined, and divided by `N - 1` so a weight tuned on a small region holds on a large
one. Needs `[data] compartments` and `phasing_track`, and excluded volume or confinement
alongside since the terms are attractive.

| key | type | default | attribute | what it does |
|---|---|---|---|---|
| `use_compartments` | bool | no | `use_compartments` | Master switch. |
| `weight` | float | 1.0 | `compartment_weight` | Weight of the pairwise affinity. |
| `energy_a` | float | 1.0 | `compartment_energy_a` | Affinity between two A beads. |
| `energy_b` | float | 2.0 | `compartment_energy_b` | Affinity between two B beads. |
| `apply_to_heatmap` | bool | yes | `compartment_apply_to_heatmap` | In the heatmap stages. |
| `apply_to_ib` | bool | yes | `compartment_apply_to_ib` | In block placement. |
| `apply_to_smooth` | bool | yes | `compartment_apply_to_smooth` | In the smooth stage. |
| `radius_heatmap` | float | 0.0 | `compartment_radius_heatmap` | Interaction radius, 0 derives it. |
| `radius_ib` | float | 0.0 | `compartment_radius_ib` | Interaction radius. |
| `radius_smooth` | float | 0.0 | `compartment_radius_smooth` | Interaction radius. |
| `auto_factor_heatmap` | float | 1.5 | `compartment_auto_factor_heatmap` | Times the stage's mean bond scale. |
| `auto_factor_ib` | float | 1.5 | `compartment_auto_factor_ib` | Times the stage's mean bond scale. |
| `auto_factor_smooth` | float | 1.5 | `compartment_auto_factor_smooth` | Times the stage's mean bond scale. |

## [accessibility]

The HiP-HoP mechanisms driven from one accessibility track. Bridging is an effective pairwise
attraction between open beads. Fibre compaction shortens the chain bond where the bead is
closed.

| key | type | default | attribute | what it does |
|---|---|---|---|---|
| `mode` | str | log | `accessibility_mode` | `log` is log then min max normalisation. `binary` is HiP-HoP's own open or closed state and is the faithful one. `log` is close to inert on a track binned to several kb. |
| `percentile` | float | 80.0 | `accessibility_percentile` | Under `binary`, a bead is open at or above this percentile of the loaded values. |
| `use_bridging` | bool | no | `use_bridging` | Master switch for bridging. |
| `bridging_weight` | float | 1.0 | `bridging_weight` | Weight of the bridging affinity. |
| `apply_to_heatmap` | bool | no | `bridging_apply_to_heatmap` | In the heatmap stages. |
| `apply_to_ib` | bool | no | `bridging_apply_to_ib` | In block placement. |
| `apply_to_smooth` | bool | yes | `bridging_apply_to_smooth` | In the smooth stage. |
| `radius_heatmap` | float | 0.0 | `bridging_radius_heatmap` | Interaction radius, 0 derives it. |
| `radius_ib` | float | 0.0 | `bridging_radius_ib` | Interaction radius. |
| `radius_smooth` | float | 0.0 | `bridging_radius_smooth` | Interaction radius. |
| `auto_factor_heatmap` | float | 1.5 | `bridging_auto_factor_heatmap` | Times the stage's mean bond scale. |
| `auto_factor_ib` | float | 1.5 | `bridging_auto_factor_ib` | Times the stage's mean bond scale. |
| `auto_factor_smooth` | float | 1.5 | `bridging_auto_factor_smooth` | Times the stage's mean bond scale. |
| `use_fibre_compaction` | bool | no | `use_fibre_compaction` | Master switch for compaction. |
| `fibre_compaction` | float | 0.3 | `fibre_compaction` | A bead's chain bond target is scaled by `1 - fibre_compaction * (1 - accessibility)`. |

## [nucleus]

Whole nucleus terms from MultiMM. They run in the segment level heatmap MC only, since that is
the one call that spans the whole active region.

| key | type | default | attribute | what it does |
|---|---|---|---|---|
| `use_lamina` | bool | no | `use_lamina` | Pull B beads toward the nuclear envelope. |
| `lamina_weight` | float | 400.0 | `lamina_weight` | Weight. |
| `use_central_force` | bool | no | `use_central_force` | Pull A beads toward the centre. |
| `central_weight` | float | 20.0 | `central_weight` | Weight. |
| `use_chromosomal_blocks` | bool | no | `use_chromosomal_blocks` | Keep each chromosome in its own territory. Multi chromosome runs. |
| `chrom_block_kc` | float | 0.3 | `chrom_block_kc` | Territory stiffness. |
| `chrom_block_weight` | float | 0.0001 | `chrom_block_weight` | Weight. |
| `radius` | float | 0.0 | `nucleus_radius` | Outer nuclear radius, 0 derives it. |
| `packing_factor` | float | 1.0 | `nucleus_packing_factor` | The derived outer radius is this times the mean bond scale times the cube root of the bead count. |
| `inner_fraction` | float | 0.2 | `nucleus_inner_fraction` | The inner radius is the outer one times the cube root of this. |

## Not settings

`[small_ib_boost]` appears in `CANONICAL` and in the generated configs with `use_small_ib_boost
= no`. No reader consults it and the feature is not implemented. It is a design note in the
divergences list. `[cuda]` in the reference's config is the reference binary's own and is
ignored.
