# Settings reference

A config is an ini file with the sections below. A key that is not in this list produces a warning. Booleans accept
`yes`, `no`, `true`, `false`, `1` and `0`. A file name of the data section is relative to `data_dir` unless it is
absolute.

The production column is `gnome3d.production.CANONICAL`. `slurm/ensemble/make_configs.py` generates the ensemble
configs from it with two changes, `mc_executor_jax_bucket_shapes = yes` and `heat_min_reduction = 0.001`.
`python tests/unit/test_settings_doc.py` checks every row against the loader, the defaults and the production config.

The measurements behind a production value are in `design/features.md`.

## [main]

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `output_level` | int | 0 | 1 | Log verbosity. 0 prints the run banner and warnings, 1 adds milestones and stage headers, 2 adds one line per batch of MC steps. |
| `log_file` | str |  |  | Path of a full detail log that is written beside the console output. Empty writes none. |
| `random_walk` | bool | no | no | Places the segment beads on a chained random walk with steps of 50 units, in place of the segment heatmap MC. Honours `use_2D`. |
| `use_2D` | bool | no | no | Restricts every move to the plane. |
| `loop_density` | int | 5 | 5 | Subanchors between two consecutive anchors with dynamic density off. |
| `max_pet_length` | int | 1000000 | 1000000 | A loop that spans more than this many bp is a long loop. It gets no spring and is added to the segment heatmap. |
| `long_pet_power` | float | 2.0 | 2.0 | A long loop adds `long_pet_scale * score ^ long_pet_power` to the segment heatmap. |
| `long_pet_scale` | float | 10.0 | 1.0 | Scale of that sum. |
| `steps_lvl1` | int | 2 | 1 | Runs of the chromosome level heatmap MC. The best is kept. Used with several chromosomes. |
| `steps_lvl2` | int | 2 | 1 | Runs of the segment level heatmap MC. The best is kept. |
| `steps_arcs` | int | 5 | 1 | Restarts of the arcs stage per block, each from a new noised start. The best is kept. Applies to the solver and to the annealer. |
| `steps_smooth` | int | 5 | 1 | Restarts of the smooth stage per block. The best is kept. |
| `noise_lvl1` | float | 1.0 | 0.5 | Step size of the chromosome level, as a multiple of the mean target distance. |
| `noise_lvl2` | float | 0.1 | 0.5 | Step size of the segment level, as a multiple of the mean target distance. |
| `noise_smooth` | float | 0.5 | 1.0 | Step size of the smooth stage, as a multiple of the mean chain bond target. |
| `noise_ib` | float | 0.5 | 0.5 | Step size of block placement, as a multiple of the mean bond between blocks. |
| `overlap_anchor_strict` | bool | no | no | Sets the span between two overlapping anchors to zero, which puts the subanchors between them at one point. Off spreads them over the overlap with ranges of nonzero width. |
| `drop_zero_length_subanchors` | bool | no | yes | Leaves subanchors of zero width out of the output. The chain keeps them. |
| `use_dynamic_loop_density` | bool | no | yes | Chooses the subanchor count of each anchor gap to make one chain segment about `target_bp_per_subanchor` long. Off uses `loop_density` for every gap. |
| `target_bp_per_subanchor` | int | 5000 | 1000 | Genomic length of one chain segment under dynamic density. It is also the bead unit of the distance law. |
| `min_subanchors_per_arc` | int | 0 | 0 | Lower bound of the count per gap under dynamic density. |
| `max_subanchors_per_arc` | int | 50 | 100 | Upper bound of the count per gap under dynamic density. |

## [data]

| key | type | default | meaning |
| --- | --- | --- | --- |
| `data_dir` | str |  | Directory that the other file names are relative to. `--data-dir` of the command line replaces it. |
| `anchors` | str |  | BED of loop anchors, `chr start end orientation`. |
| `clusters` | str |  | BEDPE of PET clusters, the loops, `chr1 s1 e1 chr2 s2 e2 score`. One path or several separated by commas, one per loop factor, with the CTCF set first. |
| `factors` | str |  | One label per cluster file, separated by commas, CTCF first. Each factor has its own strength fit and only the loops of the first factor enter the orientation term. |
| `singletons` | str |  | BEDPE of singleton contacts for the exponent fit and the segment heatmap. A file of Hi-C bin pairs is valid. |
| `singletons_inter` | str |  | A second singletons file, read only with several chromosomes. |
| `centromeres` | str |  | BED of centromere positions. |
| `segment_split` | str |  | BED of segment breakpoints. |
| `compartments` | str |  | Input of `[compartments]`. A bedGraph of a signed compartment eigenvector, or a BED with a label in the fourth column, `A`, `B` or a CALDER rank. |
| `phasing_track` | str |  | Track that fixes the sign of an eigenvector in `compartments`. Without it the side with more loop anchors is A. A file of labels needs none. |
| `anchor_activity` | str |  | BED with a signal column that gives each anchor an activity for `[factories]`. The signal is the seventh column of a line with seven or more, as in a broadPeak. It is the fourth column otherwise, as in a bedGraph. An anchor gets the largest value of the intervals that overlap it and zero without one. |
| `anchor_contacts` | str |  | Counts between anchor pairs from a full depth map, an npz from `slurm/ensemble/prep_anchor_contacts.py`. The anchor contact map is built from it in place of the singletons, which still give the exponent fit and the coarse heatmaps. A log line states the share of the anchors of the run that the file holds. |
| `contact_map` | str |  | A cooler URI at fine resolution, such as an mcool at 1 kb. `start = maxent` takes from it the raw counts between windows of about 10 kb inside each gap. Empty draws the gaps from the law alone. |

## [distance]

One law sets every distance in bead units, see the section on the distance law below. Its exponent is measured from
the singletons of the run at load and a config usually leaves this section empty.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `polymer_exponent` | float | 0.0 | 0.0 | Zero measures the exponent from the singletons at load. A positive value fixes it. |
| `contact_half_saturation` | float | 1.0 | 1.0 | The loop strength at which a loop pulls its pair halfway from the background to one bead. The strength is in multiples of a typical loop at the same span. |

## [heatmaps]

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `inter_scaling` | float | 1.0 | 1.0 | Multiplies the cells of the segment heatmap between two chromosomes. Cells within a chromosome are unchanged. Used with several chromosomes. |
| `distance_heatmap_stretching` | float | 2.0 | 2.5 | Clips the distances of the segment heatmap at this multiple of the mean positive distance. |

## [springs]

The spring energy is `k * ((d - target) / target) ^ 2`. `k` is the stretch constant for a pair that is too far and the
squeeze constant for a pair that is too close.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `stretch_constant` | float | 0.1 | 0.1 | Chain bond of the smooth stage, too far. |
| `squeeze_constant` | float | 0.1 | 0.1 | Chain bond of the smooth stage, too close. |
| `angular_constant` | float | 0.1 | 0.1 | Bend penalty of the smooth stage, on the cube of the angle between consecutive bonds. |
| `stretch_constant_arcs` | float | 1.0 | 1.0 | Arcs stage, every target of the matrix, loops and chain bonds, too far. |
| `squeeze_constant_arcs` | float | 1.0 | 1.0 | Arcs stage, too close. |
| `arc_weight_exponent` | float | 0.0 | 0.0 | Scales the spring constant of each loop by its strength to this power. The strength is the PET count over the typical count at the span of the loop. Zero gives every loop the same spring. A value of 1 changed neither the strong loops nor expression on the trio chromosomes. |
| `loop_dropout` | bool | no | no | Keeps a loop in a conformation with probability `q / (q + loop_dropout_scale)`, with `q` the strength of the loop. A loop that the draw drops gets no spring in that conformation. The draw is seeded by the conformation. |
| `loop_dropout_scale` | float | 1.0 | 1.0 | The strength at which a loop is kept in half of the conformations. A typical loop has strength 1. |
| `loop_contact` | bool | no | no | Holds a kept loop at one bead in place of the target of the law. With `loop_dropout` the strength of a loop then sets the share of conformations with its anchors in contact. |
| `background_weight` | float | 0.0 | 0.1 | Weight of a weak spring in the arcs stage that holds an anchor pair without a loop at the background distance of its separation. It acts on pairs closer than `background_range_bp`. Zero turns it off. |
| `background_range_bp` | int | 100000 | 100000 | The separation under which a pair without a loop is held at the background. A power law distance matrix can be embedded in three dimensions over a band of separations and not over every pair. |
| `gyration_weight` | float | 0.0 | 0.1 | Weight of a term of the arcs solver that holds the radius of gyration of every genomic window of anchors at the value of the law. The windows cover the span at every scale from `background_range_bp` upward by doubling, each scale twice with a shift of half a window. The annealer refuses it. |
| `factor_strength` | str |  |  | A multiplier on the loop strength per cluster file, separated by commas in file order. Empty is 1 for each. At 0 the loops of a factor are held at the background and its anchors stay as beads. |
| `use_contact_background` | bool | no | yes | Holds a pair without a loop beyond `background_range_bp` at the contact distance of the law, with the same spring, when its contact cell puts it closer than the background. A pair at or below its expected contact gets no spring. Needs `use_anchor_heatmap`. |
| `contact_background_z` | float | 0.0 | 0.0 | Above zero, the contact background holds a pair without a loop when its count is this many Poisson deviations above or below its expectation and the expectation is above this value squared. The target is `background * ((count + 0.5) / (expected + 0.5)) ^ (-1/3)`, clipped to one bead and to `distance_heatmap_stretching` backgrounds. Zero keeps the rule of `use_contact_background`. |
| `stretch_constant_ib` | float | 0.1 | 0.1 | Bond between blocks in block placement, too far. |
| `squeeze_constant_ib` | float | 0.1 | 0.1 | Bond between blocks in block placement, too close. |

## [factories]

The factory term of the arcs stage pulls an anchor with an activity toward the active anchors near it. The activity
comes from `[data] anchor_activity`. The energy is `w * a_i * (log(1 + A) - log(1 + S_i))` with
`S_i = sum_j a_j exp(-d_ij / r)` and `A` the total activity. It saturates: an anchor gains from one group and little
from a second. The term is in the energy of the solver on both backends. The annealers refuse it.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `weight` | float | 0.0 | 0.0 | Weight `w` of the term. Zero turns it off. |
| `radius` | float | 2.0 | 2.0 | Reach `r` of a group, in beads. |

## [motif_orientation]

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_motif_orientation` | bool | no | yes | Scores the angle between the CTCF motif vectors of two anchors that a loop joins, in the smooth stage. |
| `weight` | float | 1.0 | 50.0 | Weight of the term. |
| `symmetric_motifs` | bool | yes | yes | Yes scores two motif vectors best when they point the same way. No flips the partner first and opposed vectors score best. |

## [anchor_heatmap]

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_anchor_heatmap` | bool | no | yes | Scales the target of each loop down in proportion to the singleton count of its pair over the maximum of the region. |
| `heatmap_influence` | float | 0.5 | 0.1 | The largest fraction by which a target is scaled down. |

## [subanchor_heatmap]

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_subanchor_heatmap` | bool | no | no | Adds a contact distance target between subanchor pairs in the smooth stage. The target is estimated from dry smooth passes and scaled by contact. |
| `heatmap_influence` | float | 0.5 | 0.1 | The largest fraction by which the contact of a pair scales its estimated distance down. |
| `heatmap_dist_weight` | float | 1.0 | 0.01 | Weight of the term against the chain and angle terms. |
| `estimate_distances_steps` | int | 2 | 4 | Dry smooth runs per replicate of the distance estimate. |
| `estimate_distances_replicates` | int | 5 | 4 | Replicates averaged into the target. |
| `heat_min_reduction` | float | 0.0 | 0.0 | Skips the estimate for a block when its fraction of pairs with a contact is below this value. The fraction is an upper bound on the reduction that the term can reach. |
| `heatmap_workers` | int or auto | 1 |  | Threads that build the contact heatmaps of the blocks. `auto` uses every usable core. |

## The MC schedule sections

`[simulation_heatmap]`, `[simulation_arcs]`, `[simulation_arcs_smooth]` and `[simulation_ib]` each set the annealing
of one stage with the same keys. In `[simulation_heatmap]` every key has the suffix `_heatmap`.

An uphill move is accepted when `rand < jump_temp_scale * exp(-jump_temp_coef * (E_new / E_old) / T)`. `T` starts at
`max_temp` and is multiplied by `delta_temp` after every step. A round is `stop_condition_steps` steps. The run stops
after a round that neither lowered the score below `stop_condition_improvement_threshold` times the score of the round
before nor accepted `stop_condition_successes_threshold` moves.

| key | type | default | production | meaning |
|---|---|---|---|---|
| `max_temp` | float | 20.0 | 5.0 | Starting temperature. |
| `delta_temp` | float | 0.99995 | 0.9999 | Cooling factor per step. |
| `jump_temp_scale` | float | 50.0 | 50.0 | Prefactor of the acceptance. |
| `jump_temp_coef` | float | 20.0 | 20.0 | Coefficient of the acceptance. |
| `stop_condition_steps` | int | 10000 | 50000 | Steps per round. |
| `stop_condition_improvement_threshold` | float | 0.995 | 0.999 | A round improved when its score is below this times the score of the round before. |
| `stop_condition_successes_threshold` | int | 5 | arcs and ib 100, smooth 50, heatmap 10 | A round that did not improve ends the run when it accepted fewer moves than this. |

### [simulation_arcs] only

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `stop_condition_ratio` | float | 0.9999 | 0.9999 | Stops the run when the score over the score of the round before is at or above this. |
| `solver` | str | mc | lbfgs | `mc` anneals. `lbfgs` minimises the same energy with L-BFGS-B and reaches the same minimum and the same overlaps. `mc_executor_arcs` sets the place of the energy evaluation: `serial` and `threaded` use the numba kernel on the CPU, `batch` uses the JAX device in float32 and agrees to about 1e-6 relative. |
| `solver_iters` | int | 200 | 5000 | Iteration cap of the solver. Zero keeps the start as it is. With `solver_tol` at zero the cap is the only stop. |
| `solver_tol` | float | 0.0 | 1e-6 | Stops the solve when an iteration improves the energy by less than this fraction of it. Zero leaves the cap as the only stop. At 1e-6 a chromosome solve stops near 2,400 iterations, within one percent of the energy at 8,000. |
| `start` | str | centroid | maxent | Start of the anchors. `centroid` puts every anchor at the centroid of its block. `walk` places consecutive anchors at the distance of the law along random directions. `hilbert` places them along a 3D Hilbert curve scaled to the bond of the law. `maxent` draws them once per conformation from the Gaussian whose pair variances best meet the target of every pair. It needs `scope = chromosome` and the annealer of the batch executor refuses it. |
| `scope` | str | block | chromosome | `block` solves the anchors of each block alone. `chromosome` solves every anchor of a chromosome as one problem from the placed block centroids. Loops, the contact background and the compartment term then act across blocks. |

### [simulation_arcs_smooth] only

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `dist_weight` | float | 1.0 | 1.0 | Weight of the chain bond term. |
| `angle_weight` | float | 1.0 | 1.0 | Weight of the bend term. |
| `hard_wall` | bool | no | yes | Rejects a move that puts a pair of non neighbours under the excluded volume radius or brings such a pair closer. The count of these pairs can only fall. |
| `anchor_cap` | float | 0.0 | 0.0 | Lets the anchors move up to this many mean bonds from their position after the arcs stage. 0 keeps them fixed. |
| `prefetch` | int | 1 | 32 | Number of proposals that one step of the JAX kernel evaluates against the current state. The first accepted in draw order is kept. Numba ignores it. |
| `jax_grid` | bool | no | yes | Puts the excluded volume and the wall of the JAX kernel on a cell grid. A proposal then visits the 27 cells around it in place of every bead, with the same result. |
| `jax_grid_min_beads` | int | 4096 | 4096 | The grid is used on launches whose padded bead count reaches this value. |
| `start` | str | line | coil | Start of the subanchors of a gap. `line` is the straight line between the two anchors. `coil` is a compact random bridge at the bond targets. `maxent` is a draw from the maximum entropy Gaussian bridge whose mean pair distances follow the law, scaled by window contacts from `[data] contact_map` when one is given. `maxent` lowered Pearson by 0.01 to 0.05 on three cell lines. |

### [simulation_ib] only

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `refine_scope` | str | segment | segment | `segment` places the blocks of each segment as one chain and skips a segment with one block or none. `chromosome` places every block of the chromosome as one chain. It inflates the structures unless the excluded volume and the confinement of block placement are tuned again. |
| `use_ib_mc` | bool | no | yes | Anneals the block centroids with chain bonds, excluded volume and confinement. Off places them by interpolation. |
| `dist_weight` | float | 1.0 | 1.0 | Weight of the chain bond term. |
| `heatmap_weight` | float | 0.0 | 0.0 | Weight of a contact term between the blocks of one chain. The contacts of the run are binned by block and converted with the law, as for the segment heatmap. Zero turns it off. |

## [simulation_backend]

Each stage runs on an executor. `serial` and `threaded` run the numba kernels, one chain per thread. `batch` runs the
JAX kernels, which pack many blocks into one device launch. `auto` takes `batch` for the distance estimate and for
the smooth stage under 4 workers when JAX is installed. Otherwise it takes `threaded` with more than one worker and
`serial` with one.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `ib_workers` | int or auto | 1 | auto | Threads of the threaded executor. `auto` uses every usable core. |
| `heatmap_chains` | int | 1 | 1 | Independent chains of the heatmap MC that run at once. The best is kept. |
| `smooth_chains` | int | 1 | 1 | Independent smooth chains per block that run at once. The best is kept. |
| `mc_executor_arcs` | str | auto | batch | Executor of the arcs stage. With the solver, `batch` evaluates the energy on the JAX device and `serial` and `threaded` evaluate it on the CPU. With the annealer, `batch` is the vmapped JAX kernel, in which one slow block holds every other block of its launch. |
| `mc_executor_densify` | str | auto | threaded | Executor of densification. |
| `mc_executor_estimate_dist` | str | auto | batch | Executor of the subanchor distance estimate. |
| `mc_executor_smooth` | str | auto | threaded | Executor of the smooth stage and of the cross block relaxation. `auto` is `threaded` from 4 workers. Threaded numba from 8 workers is faster than an RTX 3090 on a chromosome and on a region of any size. |
| `mc_executor_jax_bucket_shapes` | bool | no | no | Pads each block to a ladder of shapes. A stage is then a few wide launches in place of one compile per size. |
| `merge_smooth_launches` | bool | yes | yes | Packs the smooth blocks with the same energy terms into as few launches as the device memory allows, at any size. |
| `mc_executor_jax_batch_width_smooth` | int or auto | auto | auto | Blocks per smooth launch. `auto` takes the largest count that fits the device. |
| `mc_executor_jax_batch_width_arcs` | int or auto | auto | auto | Blocks per launch of the JAX arcs kernel. `auto` takes the largest count that fits the device. |
| `multigpu_mode` | str | groups | groups | `groups` runs whole batch groups on different devices, with results that do not depend on the device count. `within` splits one group across devices. `off` uses one device. |
| `neighbour_grid` | bool | yes | yes | Cell grid for the excluded volume term of the numba kernels, used from 128 beads. The results are identical. |

## [excluded_volume]

A soft repulsion `weight * ((r0 - d) / r0) ^ 2` acts on pairs closer than `r0` and more than `skip_neighbors` apart
along the chain. Each stage has its own radius. A radius of zero is derived as the auto factor of the stage times its
mean bond scale.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_excluded_volume` | bool | no | yes | Turns the term on. |
| `weight` | float | 0.5 | 0.1 | Weight of the term, which is its value at a full overlap. |
| `apply_to_arcs` | bool | no | yes | Applies it in the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | Applies it in the smooth stage. |
| `apply_to_heatmap` | bool | no | yes | Applies it in the heatmap stages. |
| `apply_to_ib` | bool | yes | yes | Applies it in block placement. |
| `skip_neighbors` | int | 1 | 1 | Pairs this close along the chain are exempt. |
| `radius_arcs` | float | 0.0 | 0.0 | Radius of the arcs stage. 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | Radius of the smooth stage. 0 derives it. |
| `radius_heatmap` | float | 0.0 | 0.0 | Radius of the heatmap stages. 0 derives it. |
| `radius_ib` | float | 0.0 | 0.0 | Radius of block placement. 0 derives it. A positive value is also the constant centroid radius of the boundary stitch. |
| `auto_factor_arcs` | float | 0.5 | 1.0 | Multiplies the mean positive loop target. |
| `auto_factor_smooth` | float | 0.5 | 0.7 | Multiplies the mean chain bond target of the block. |
| `auto_factor_heatmap` | float | 0.5 | 0.5 | Multiplies the mean active heatmap target. |
| `auto_factor_ib` | float | 0.5 | 0.5 | Multiplies the mean bond between blocks. |
| `arcs_repulsion_cutoff_factor` | float | 0.0 | 1.5 | Cuts the repulsion `1 / d` between pairs without a loop beyond this multiple of the mean loop target. 0 leaves it unbounded and a sparse block then expands without limit. 1.5 with the contact background follows the law up to 2 Mb, 3 lets every pair beyond 100 kb drift apart and 1.0 compacts the structure. |
| `arcs_repulsion` | bool | yes | no | The repulsion between anchor pairs without a loop in the arcs stage. Off, such a pair beyond the short range background has only the excluded volume and `[springs] gyration_weight` holds the size. Off with the gyration term, the maxent start and the cross block relaxation raised the pooled SCC by 0.017 to 0.044 on three cell lines. |

## [confinement]

A soft envelope `weight * ((r - R) / R) ^ 2` acts on a bead further than `R` from the centroid of the starting
positions of the stage. A radius of zero is derived as the packing factor times the mean bond scale of the stage times
the cube root of the bead count.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_confinement` | bool | no | yes | Turns the term on. |
| `weight` | float | 0.5 | 0.1 | Weight of the term. |
| `weight_arcs` | float | 0.0 | 0.0 | Confinement weight of the arcs stage. 0 uses `weight`. |
| `apply_to_arcs` | bool | yes | yes | Applies it in the arcs stage. |
| `apply_to_smooth` | bool | yes | yes | Applies it in the smooth stage. |
| `apply_to_ib` | bool | yes | yes | Applies it in block placement. |
| `radius_arcs` | float | 0.0 | 0.0 | Radius of the arcs stage. 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | Radius of the smooth stage. 0 derives it. |
| `radius_ib` | float | 0.0 | 0.0 | Radius of block placement. 0 derives it. |
| `packing_factor_arcs` | float | 1.5 | 0 | Packing factor of the arcs stage. At 0 the radius of each block is derived from the law as the sphere that a chain of the genomic span of the block fills, root five thirds of its radius of gyration `S^nu / sqrt(2 (2 nu + 1)(nu + 1))`. The annealer of the batch executor refuses 0. |
| `packing_factor_smooth` | float | 1.5 | 1.5 | Packing factor of the smooth stage. |
| `packing_factor_ib` | float | 0.75 | 0.75 | Packing factor of block placement. Below about 0.58 a small segment is asked to fold tighter than one of its own bonds. |

## [boundary_stitch]

The pass runs after every chain of a chromosome is done. It moves each block as a rigid body until the last anchor of
one block and the first anchor of the next are at the distance that pairs inside the blocks have at the same
separation. A soft excluded volume acts between block centroids. The pass uses no random numbers.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_boundary_stitch` | bool | no | no | Turns the pass on. It is for `scope = block`. At chromosome scope the joint solve places the blocks and the pass inflates the radius of gyration by a tenth. |
| `spring_weight` | float | 1.0 | 1.0 | Weight of the boundary springs. |
| `ev_weight` | float | 1.0 | 1.0 | Weight of the centroid excluded volume. |
| `max_iter` | int | 2000 | 2000 | Iteration cap of L-BFGS-B. 500 leaves a chromosome unconverged and 2000 converges a chromosome of 1,494 blocks. |

## [relax]

The pass runs after the stitch. It runs the smooth kernel once over the whole chromosome with excluded volume on every
pair and every anchor fixed. Coils of different blocks stop passing through each other and the anchors keep their
positions.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_cross_block_relax` | bool | no | yes | Turns the pass on. At chromosome scope with `arcs_repulsion` off it lowered the overlaps across blocks from 62 to 14 per thousand on GM12878 chr1. |
| `ev_weight` | float | 10.0 | 10.0 | Excluded volume weight of the pass. |
| `ev_radius` | float | 0.0 | 1.0 | Excluded volume radius of the pass. 0 uses 1.5 chain bonds. 1.0 is about one bond. |
| `temp` | float | 0.1 | 0.1 | Starting temperature as a fraction of `max_temp` of the smooth stage. A bead must cross the shell of a neighbour to untangle and a pass at zero temperature stalls. |
| `noise` | float | 0.5 | 0.5 | Step size in chain bonds. |
| `bond_weight` | float | 10.0 | 10.0 | Chain spring weight of the pass. At the 0.1 of the smooth stage the excluded volume tears the coil. |
| `local_window` | int | -1 | 1 | Moves only the beads that touch another block and this many chain neighbours on each side. -1 moves every subanchor, which takes hours on a chromosome against minutes at 1. |
| `within_blocks` | bool | no | yes | With `local_window` set, also moves the pairs of non neighbours of one block that are closer than 0.7 of a bond. These are the coils of two gaps whose anchors the arcs stage placed close. |

## [holes]

The pass runs before the stitch. The stretch between two blocks has no beads. With the pass on it gets one bead per
`target_bp_per_subanchor`, with no cap, drawn from the maximum entropy bridge between the two anchors that bound it
and scaled by `[data] contact_map` when one is given. The stretches before the first block and after the last block
stay empty.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_hole_beads` | bool | no | no | Turns the pass on. |
| `hard_wall` | bool | yes | yes | Keeps the hard wall of the smooth stage in the chromosome pass over the hole beads. `no` leaves the soft excluded volume alone and a hole can rest against its blocks. |
| `flank_beads` | int | 0 | 0 | Block beads on each side of a hole, beside the two anchors, that the draw is conditioned on. Zero draws a bridge between the anchors alone. |

## [compartments]

The A/B compartment terms follow MultiMM [4]. Each is shifted to be non negative, which the acceptance rule requires,
and divided by `N - 1`, which keeps one weight valid from a small region to a large one. The terms need
`[data] compartments`, with `phasing_track` for a signed track. They are attractive and need excluded volume or
confinement beside them.

| key | type | default | production | meaning |
| --- | --- | --- | --- | --- |
| `use_compartments` | bool | no | no | Turns the terms on. |
| `weight` | float | 1.0 | 1.0 | Weight of the pairwise affinity. |
| `energy_a` | float | 1.0 | 1.0 | Affinity between two A beads. |
| `energy_b` | float | 2.0 | 2.0 | Affinity between two B beads. |
| `apply_to_heatmap` | bool | yes | yes | Applies them in the heatmap stages. |
| `apply_to_ib` | bool | yes | yes | Applies them in block placement. |
| `apply_to_smooth` | bool | yes | yes | Applies them in the smooth stage. |
| `radius_heatmap` | float | 0.0 | 0.0 | Interaction radius of the heatmap stages. 0 derives it. |
| `radius_ib` | float | 0.0 | 0.0 | Interaction radius of block placement. 0 derives it. |
| `radius_smooth` | float | 0.0 | 0.0 | Interaction radius of the smooth stage. 0 derives it. |
| `auto_factor_heatmap` | float | 1.5 | 1.5 | Multiplies the mean bond scale of the heatmap stages. |
| `auto_factor_ib` | float | 1.5 | 1.5 | Multiplies the mean bond scale of block placement. |
| `auto_factor_smooth` | float | 1.5 | 1.5 | Multiplies the mean bond scale of the smooth stage. |

## Distance law

One law gives the distance of every pair of beads. It sets the bond between consecutive beads in the smooth stage, the
bond between block centroids in block placement, the target of an anchor pair that a loop joins, the bond of a
consecutive anchor pair without a loop and the target of a cell of the segment heatmap.

The unit is the bead, the distance of two consecutive beads at the resolution of the run. Chromatin in a nucleus is
crumpled. Ten times more DNA between two loci puts them only about twice as far apart. A loop draws its two
anchors closer than that, more for a loop that is stronger than a typical loop at its span, down to one bead. Two
segments of the coarse map are closer the more contacts they share.

In the formulas `s0` is `target_bp_per_subanchor` and `nu` is the measured exponent.

- A pair without a contact is at `d = max(1, (s / s0) ^ nu)` for the separation `s`. The mean spatial distance grows
  with the genomic separation as a power law [1, 2]. In the arcs stage `background_weight` holds a pair without a loop
  at this distance inside `background_range_bp`.
- A loop of strength `q` is at `d = 1 + (background - 1) / (1 + q / q_half)`. `q` is the PET count of the loop over
  the typical count at its span, fitted on the loops of the run. It is an observed over expected ratio [5, 6]. A
  saturated loop is at one bead, as polymer loop models hold loop anchors [3, 8].
- A heatmap cell is at the background times observed over expected contact to the power of minus one third, the
  fractal globule relation [1, 2, 7]. The expectation is the mean of the cells at the same separation in the heatmap.
  Chromosome pairs have no separation. At the chromosome level every cell shares one expectation and the background is
  taken at the mean chromosome span.
- `nu` is minus one third of the slope of log contact count against log separation on the singletons of the run. It is
  fitted at load on the whole chromosome set, before any region filter. Contact probability falls close to the inverse
  of separation in Hi-C [1]. A distance that grows as `s ^ nu` gives a contact that falls as `s ^ (-3 nu)` [2]. A
  singletons file from ChIA-PET is enriched around CTCF and does not decay with separation. For such a file the run
  logs a line that is always visible and uses 0.285, the value that three 4DN Hi-C cell lines measure from 20 kb to
  1 Mb.

The measured exponent is 0.275 for GM12878, 0.299 for H1ESC, 0.192 for HFFC6 and 0.072 for a trio sample.

## References

1. Lieberman-Aiden E, van Berkum NL, Williams L, Imakaev M, Ragoczy T, Telling A, Amit I, Lajoie BR, Sabo PJ,
   Dorschner MO, Sandstrom R, Bernstein B, Bender MA, Groudine M, Gnirke A, Stamatoyannopoulos J, Mirny LA, Lander ES,
   Dekker J. Comprehensive mapping of long-range interactions reveals folding principles of the human genome. Science
   326, 289 to 293 (2009). doi:10.1126/science.1181369
2. Mirny LA. The fractal globule as a model of chromatin architecture in the cell. Chromosome Research 19, 37 to 51
   (2011). doi:10.1007/s10577-010-9177-0
3. Buckle A, Brackley CA, Boyle S, Marenduzzo D, Gilbert N. Polymer Simulations of Heteromorphic Chromatin Predict the
   3D Folding of Complex Genomic Loci. Molecular Cell 72, 786 to 797 (2018). doi:10.1016/j.molcel.2018.09.016
4. Korsak S, Banecki K, Plewczynski D. Multiscale molecular modeling of chromatin with MultiMM: From nucleosomes to
   the whole genome. Computational and Structural Biotechnology Journal 23, 3537 to 3548 (2024).
   doi:10.1016/j.csbj.2024.09.025
5. Rao SSP, Huntley MH, Durand NC, Stamenova EK, Bochkov ID, Robinson JT, Sanborn AL, Machol I, Omer AD, Lander ES,
   Aiden EL. A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. Cell 159,
   1665 to 1680 (2014). doi:10.1016/j.cell.2014.11.021
6. Serra F, Baù D, Goodstadt M, Castillo D, Filion GJ, Marti-Renom MA. Automatic analysis and 3D-modelling of Hi-C
   data using TADbit reveals structural features of the fly chromatin colors. PLoS Computational Biology 13, e1005665
   (2017). doi:10.1371/journal.pcbi.1005665
7. Varoquaux N, Ay F, Noble WS, Vert JP. A statistical approach for inferring the 3D structure of the genome.
   Bioinformatics 30, i26 to i33 (2014). doi:10.1093/bioinformatics/btu268
8. Fudenberg G, Imakaev M, Lu C, Goloborodko A, Abdennur N, Mirny LA. Formation of chromosomal domains by loop
   extrusion. Cell Reports 15, 2038 to 2049 (2016). doi:10.1016/j.celrep.2016.04.085
