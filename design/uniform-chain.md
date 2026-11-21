# Uniform chain: rules of the model

Status 2026-11-19: design, not built. The first build of `[main] uniform_chain` is in the code and is described in
its own section at the end. The rules below replace it in place. Its trial is the baseline.

## The object

A block is a chain of beads of equal width, one per `target_bp_per_subanchor`, from the start of its first anchor to
the end of its last. Every base of the block is in one bead. The span of an anchor is the run of beads whose
midpoints are in its range. Anchors that overlap share beads. In the GM12878 anchor file 88 percent of consecutive
anchors overlap, 42 percent are nested, the median width is 1.4 kb and 1.2 percent are over 10 kb.

Every rule below is stated on beads. No rule is stated on anchors as points.

## Decisions of 2026-11-19

Every row is the maintainer's decision.

| Question | Decision |
|---|---|
| The object a loop acts on | One bead of each of its two spans, the contact beads, drawn per conformation |
| The scope of the draw | Each loop end draws its own bead |
| PET counts of loops on one bead pair | The strengths add and the sum goes through the law once |
| The draw of a contact bead inside a span | Uniform over the beads of the span, measured against the bead at the midpoint |
| The smooth stage and the contact beads | Small and bounded adjustments are allowed. Three arms: fixed, bounded drift, centre of mass kept |
| The reach of the chain in the arcs stage | A one sided spring at the loop spring weight beyond k bonds |
| The orientation term | Three arms: as it is, off, and with the beads beside a site held within their bonds |
| Contact data of a pair of beads | The Hi-C cell that holds the two beads, at a resolution finer than 25 kb |
| The resolution of the cell | One setting, measured at 1, 5 and 10 kb |
| The cells that act | A cell that passes the Poisson test of `contact_background_z`, over or under its expectation |
| The pairs that a cell acts on | Every pair of nodes without a loop, near and far. A pair with a loop keeps its loop target |
| The counts and the expectation | Raw counts against the mean count at the same separation on the same chromosome and resolution |
| The order of the build | Three stages with an arms trial after each |
| The centre of mass that the smooth stage keeps | That of a node, the bead before it and the bead after it |
| The exceptions of that rule and the reach of the bond limit, both the assistant's | Kept as built |
| The resolution of the contact cells | Decided on the larger trial. 5 kb is the working base |
| The draw of the contact bead | Both draws stay as arms |
| The rule of the smooth stage for a node | Decided after the trial of the rules |
| Room for the free chain | Gyration windows under 100 kb first, blobs for the free chain when that does not work |
| The three bead move and the bond limit in the JAX kernel | After the rules are chosen |
| The larger trial | Three cells, six regions of 5 Mb across chr1, 20 structures per arm, for the arms that remain |
| The first build | Replaced in place. Comparisons use its trial of 2026-11-19 |

## Decisions of 2026-11-20

Every row is the maintainer's decision, after the trials of the night.

| Question | Decision |
|---|---|
| The rule of the smooth stage for a node | The centre of mass rule |
| The bond of the smooth stage against a squeeze | `squeeze_constant` of 1.0 on the uniform chain |
| The node rule in the settings | The centre of mass rule is the one node rule of the chain. Its setting and the fixed node are deleted, after step 3 of `uniform_night4.sh` |
| The squeeze in the settings | No setting implies another. There are two canonical configs, `CANONICAL` of the default chain and `CANONICAL_UNIFORM` of the uniform chain, in `gnome3d/production.py`. A value that works better on the uniform chain is stated in the second: `uniform_chain` and `squeeze_constant` 1.0 so far |
| The draw of the contact bead | Uniform. The draw at the midpoint is deleted |
| The orientation term | On as it is. `bond_limit` is deleted |
| The radius of the arcs excluded volume of a node | The key `[excluded_volume] radius_arcs` with its value 0, which derives the radius from the loop targets of the run, as the default. Decided on 2026-11-21 after the sweep, where the derived radius scored as 3.0 |
| The cap of the radius of a pair | The cap by the chain is deleted. The cap by the target is decided after the second round of the sweep |
| A radius of a node from its chain | Waits for the arm at a radius of 3.85 under the dropout, `uniform_sweep8c.sh` |
| The subanchor heatmap | Off, as in production |
| The resolution of the contact cells | Decided again after the radius rule, 5 kb against the ladder |
| `loop_dropout` | Decided after the radius rule. The rise of the overlap of the ends under it is traced first |
| `contact_background_z` | Open. 2 gains 0.03 of SCC at a radius of 3.0 and costs 0.035 of Spearman, `uniform_night8.sh` |
| The three bead move in the JAX kernel | Not yet |

The draw at the midpoint and `bond_limit` were removed on 2026-11-20, and `uniform_chain` implies the centre of
mass rule since the same day. The byte exactness gate on GM12878 chr1 18.29 to 20.31 Mb is identical for the
default chain and for the uniform chain with the centre of mass rule, `squeeze_constant` 1.0 and `loop_dropout`,
two structures, across the three commits.

A first run of the gate for the uniform chain differed, and commit 01503e1 states a cause that is wrong: the order
of the float operations under `fastmath`. The cause was the gate. A worktree has its own `data/` folder with the
files that git tracks, which lack the Hi-C singletons. The run in the worktree fitted the exponent on the ChIA-PET
singletons, 0.175, and the run in the checkout on the Hi-C singletons, 0.344. With one data folder for every tree
the dumps are identical. The gate of a config built by `settings_for_cell` passes the data folder as an absolute
path.

## Loops

A PET is one ligation between one place of anchor A and one place of anchor B. A conformation is one cell. In a
conformation each end of each kept loop is at one bead of the span of its anchor, the contact bead of that end.

- Each loop end draws its bead by itself. Two loops of one anchor can have two contact beads.
- The draw is uniform over the beads of the span. It comes from a generator seeded by the conformation, the seed that
  `loop_dropout` uses.
- A span of one bead has one choice. This is the case of most anchors.
- The comparison arm takes the bead at the midpoint of the anchor in every conformation.

Over an ensemble the contacts of a loop cover the bead pairs of its two spans evenly. The cluster file gives the two
intervals and a count and nothing on the places of the PET ends inside them.

### Strength

The strength of a loop is its PET count over the typical count at its span, `PolymerLaw.arc_strength`. It is linear
in the count.

- The strengths of the loops whose two contact beads are the same bead pair add.
- The pair is held at `contact_distance` of the law for the summed strength and the separation of the two beads.
- Loops of two factors on one bead pair add with the multiplier of each factor.
- A loop whose two contact beads are one bead is left out of the conformation.

Anchors that are copies of one site in several clusters then act as one stronger loop when their contact beads
agree.

## Arcs stage

The nodes of the arcs stage are the distinct contact beads of the conformation. A node is a bead of the chain and
its position is the position of that bead.

- Loop springs act between nodes at the target of the summed strength.
- The background spring of a pair without a loop is at the law for the separation of the two beads. Two nodes are
  different beads and the floor of one bead holds for them.
- Reach. Two nodes with k bonds of chain between them are at most k bonds apart. The term has no cost up to k
  bonds and is a spring at the weight of a loop spring beyond. It acts on consecutive nodes of a block. The triangle
  inequality carries it to every other pair.
- Excluded volume acts at the radius of the arcs stage, `auto_factor_arcs` times the mean loop target, 2.6 to
  3.1 beads, as on the default chain. A node stands for the chain around it and that chain needs the room. The
  radius of a bead of the first rule, 0.7 of a bond, let the nodes of two megabases pass through each other.
- The radius of a pair has an optional cap, `[excluded_volume] arcs_cap = target`. The excluded volume then does
  not ask a pair to be further apart than the distance its spring holds it at:

  | pair of nodes | limit |
  |---|---|
  | with a loop | the loop target |
  | without a loop, closer than `background_range_bp` | the background of the law at its separation |
  | held by its contact cell | the distance of the cell |
  | any other pair | none, the radius of the stage |

  The radius of a pair is the smaller of the radius of the stage and its limit. The nodes next to each other in
  the order of the chromosome stay out of the excluded volume, as before. A second limit, the chain between the
  two beads, changed no score on the sweep and was deleted on 2026-11-21.
- The gyration term is unchanged. It reads the genomic midpoints of the beads.

### Contact data

A pair of nodes reads the cell of the contact map that holds its two beads. The map is `[data] contact_map`, a
cooler, at a resolution finer than 25 kb. The value of a cell is its count over the expectation at its separation,
and the law puts a pair at the background times that ratio to the minus third.

The default chain reads the singletons at 25 kb. A singleton is given by the start of its bin and a contact is
counted for the bead that holds that start.

The resolution is `[springs] contact_resolution`. A cell acts on a pair of nodes without a loop when its expectation is
above z squared and its count is more than z Poisson deviations from it, with z of `contact_background_z`. The pair
is then held at the background times `((count + 0.5) / (expected + 0.5))` to the minus third, clipped to one bead and
to `distance_heatmap_stretching` backgrounds, with the background spring. The expectation of a cell is the mean count
at its offset over the stretch of the chromosome that holds the nodes, pooled in eight bins of offset a decade,
without the cells of a bin that has no count. Two nodes in one cell are left as they are.

The cells need a deep map. On the map of 7.8 million contacts in the laptop's data no cell of chr1 108.5 to 111 Mb
has an expectation of 9 at 1, 5 or 10 kb and no pair is held. The maps of the trials hold 2.5 to 4.0 billion
contacts.

The count of nodes changes between conformations with the draw. The joint solve is over the chromosome as now.

## Smooth stage

The chain starts with each node at its arcs position and the beads between two nodes on the bridge of
`smooth_start`. The beads before the first node and after the last continue the chain.

The arcs stage owns the positions of the nodes. Three arms state the freedom of the smooth stage over them.

| Arm | Rule |
|---|---|
| Fixed | A node does not move in the smooth stage |
| Bounded drift | A node moves inside a sphere of a stated radius around its arcs position and a move that leaves the sphere is refused. `smooth_anchor_cap` is this rule |
| Centre of mass kept | The centre of mass of a node, the bead before it and the bead after it stays as it is at the start of the stage. This is the rule of the uniform chain since 2026-11-20. Its setting was `anchor_centre` |

No arm has a penalty with a weight between the two stages. Every bead that is not a node moves freely, with the other
beads of a wide span among them.

The maintainer chose the centre of mass of the node and its two neighbours on 2026-11-19, in place of that of the
span: a loop acts on one bead, three spans of four are one bead, and the centre of mass of a span at the start of
the stage is set by the start chain. Under the rule the three beads have two moves. The node moves by the draw and
its two neighbours by minus half of it each. A neighbour moves by the draw and the other neighbour by minus it.
Neither neighbour moves alone. A node at an end of the chain, a node beside another node and a node closer than
three beads to the node of the rule before it stay fixed. Those three cases are the assistant's choice: the centres
of mass of two nodes that share a bead cannot both stay under moves of three beads. On chr1 108.5 to 111 Mb half of
the consecutive nodes are one bond apart.

The cross block relaxation follows the arm of the smooth stage. It reads the nodes from the pipeline and not from
the kind of the output bead.

## Orientation

A site of the orientation term is a contact bead of a CTCF loop. Its direction is from the bead before it to the
bead after it.

Measured on 2026-11-19: at weight 50 against bonds at 0.1 it stretches the bonds beside a site on the default
chain and on the first build. With the term off the pooled SCC of the trial moved by 0.011 or less in the mean over
six rows on either chain. The arms are the term as it is, the term off, and the term with a move of the two beads
beside a site refused when it stretches their bonds past a stated length.

The third arm was `[motif_orientation] bond_limit`, a hard rule of the numba kernel. On GM12878 chr1 108.5 to
111 Mb, one structure, with the term on, the bonds over 2 and over 3 beads before the relaxation were 212 and 89
without a limit, 78 and 0 at a limit of 1.5 and 74 and 4 at 1.25. In the trial of the rules the limit gained no
score and raised the close pairs by a fifth. The setting and its rule were removed on 2026-11-20. The term stays
on as it is.

## Output

The kind of an output bead is anchor when the bead is under a span, in every conformation. The nodes of a
conformation are not written to the output.

## Cases

| Case | Rule |
|---|---|
| Two anchors of one site, each with a loop to one partner | The contact beads agree when the spans are one bead. The strengths add |
| An anchor nested in a wide anchor | Each loop end draws inside its own span. The beads of the wide span that carry no loop end are free chain |
| A wide anchor with loops to several partners | Each loop end draws its own bead. The partners meet the anchor at several beads |
| Two contact beads next to each other on the chain | Two nodes one bond apart, by the reach term |
| A loop with both contact beads on one bead | Left out of the conformation |
| An anchor at the end of a block | Its span ends the chain. No bead is added beyond it |
| An anchor without a loop in the conformation | No node. Its beads are free chain |
| The beads before the first node of a block and after the last | A tail. Under `start = coil` it starts as a compact walk from its node at the bond targets. Under `start = line` it continues the line of the nodes |
| A block without a node in the conformation | Its middle bead is at the centroid of the block and the chain is two tails from it |

## Measurements that decide

Arms run by `slurm/ensemble/arms_trial.sh` on GM12878, H1ESC and HFFC6, five structures per arm, against production
and against the first build.

- Pooled SCC and Spearman against raw and balanced Hi-C, the distance exponent over the exponent of Hi-C.
- Overlapping pairs. The battery counts pairs by the kind of bead and the kinds differ between the chains. The count
  is checked against a count over all non neighbour pairs under 0.7 beads before two chains are compared.
- Bonds in chain order: the share over 2 beads and over 3 beads.
- Loops: realised distance over target, per conformation.
- For the draw: the contacts of the pairs of two wide anchors in the pooled map against Hi-C at 1 kb, uniform draw
  against midpoint.
- For the smooth arms: the three rules on the same arcs solution.

## Stages of the build

1. The contact beads, the summed strengths, the reach spring, the excluded volume at the radius of a bead and the
   contact data per pair of beads in the arcs stage. The nodes are fixed in the smooth stage. The first build is
   removed.
2. The bounded drift arm and the centre of mass arm of the smooth stage.
3. The arms of the orientation term.

State on 2026-11-19: stage 1 is built. The joint solve runs on the nodes with the reach spring, the excluded
volume at the radius of a bead and the contact cells of `gnome3d/pipeline/ib/cells.py`. The nodes are the fixed
beads of the smooth stage and of the relaxation, and the first build is removed. `design/features.md` has the first
numbers. Stage 2 is built: bounded drift is `anchor_cap`, which the uniform chain runs as it is, and the centre of
mass rule is a move of three beads of the numba kernel, `step_delta_nb` and `batch_mc_nb` in
`gnome3d/mc/numba/terms.py`. The JAX kernel does not have it. Stage 3 was the third arm of the orientation
term, `[motif_orientation] bond_limit`, measured and removed.

### Trial of stage 1 without the contact cells

`design/figures/uniform-2026-11-19/nodes/` has the figure and the table, `scores-nodes/` the score files. Commit
28374db, the cells, regions and structures of the trial of the first build. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | exponent over Hi-C |
|---|---|---|---|---|---|
| default chain | 0.327 | 0.307 | 0.448 | 0.716 | 1.062 |
| default chain, orientation off | 0.320 | 0.296 | 0.440 | 0.717 | 1.073 |
| default chain, contact terms off | 0.327 | 0.302 | 0.446 | 0.717 | 1.047 |
| contact beads, uniform draw | 0.319 | 0.329 | 0.444 | 0.671 | 1.042 |
| contact beads, uniform draw, orientation off | 0.314 | 0.323 | 0.437 | 0.670 | 1.050 |
| contact beads, midpoint | 0.332 | 0.332 | 0.453 | 0.684 | 1.053 |
| contact beads, midpoint, orientation off | 0.332 | 0.331 | 0.446 | 0.683 | 1.065 |

- The midpoint is ahead of the uniform draw on raw SCC, 0.332 against 0.319, and level on balanced SCC.
- Both draws gain balanced SCC over the default chain, 0.02 to 0.03, and lose balanced Pearson, 0.03 to 0.05.
- HFFC6 chr1 10 to 15 Mb falls from 0.477 to 0.32 to 0.34 of raw SCC on every contact bead arm, as on the first
  build. See the bins without a count below.
- The default chain without its two contact terms scores as the default chain. The missing contact cells do not
  explain a difference between the arms.
- The battery's overlap count is 9.8 per thousand beads on the default chain, 24.5 with the uniform draw and 15.2
  with the midpoint. The count is still not checked between the chains.

### Bins of Hi-C without a count

The pooled score counts a bin of Hi-C without a count as zero and a model bead there counts against the model. The
three rows of chr1 10 to 15 Mb have 12, 17 and 20 such bins of 200 and the rows of 20 to 25 Mb have none. The
default chain caps a gap at 100 beads and has fewer beads in those rows: 2,597, 2,823 and 3,301 against 3,025, 3,900
and 4,565 of the uniform chain, 16 to 38 percent more. The two rows where the contact beads lose raw SCC are the
two with 38 percent more beads.

Over the bins that Hi-C maps, mean of the six rows:

| arm | SCC raw | SCC balanced | balanced stratum Spearman |
|---|---|---|---|
| default chain | 0.295 | 0.266 | 0.113 |
| default chain, contact terms off | 0.294 | 0.258 | 0.129 |
| contact beads, uniform draw | 0.290 | 0.297 | 0.149 |
| contact beads, midpoint | 0.307 | 0.305 | 0.141 |

HFFC6 chr1 10 to 15 Mb over mapped bins: raw SCC 0.392 on the default chain against 0.311 and 0.289, balanced SCC
0.309 against 0.313 and 0.300. The loss of that row is in the raw score. The share of it that the beads of the long
gaps carry is not measured bead by bead.

### Trial of the contact cells

`design/figures/uniform-2026-11-19/cells/` has the figure and the table, `scores-cells/` the score files. Commit
00c67bd, the uniform draw unless stated, z of 3, the maps of 2.5 to 4.0 billion contacts. Mean over the six rows:

| arm | SCC raw | SCC balanced | SCC raw, mapped bins | SCC balanced, mapped bins | Spearman raw | Pearson balanced | exponent over Hi-C |
|---|---|---|---|---|---|---|---|
| default chain | 0.327 | 0.307 | 0.295 | 0.266 | 0.448 | 0.716 | 1.062 |
| contact beads, no cells | 0.319 | 0.329 | 0.290 | 0.297 | 0.444 | 0.671 | 1.042 |
| cells at 1 kb | 0.321 | 0.335 | 0.291 | 0.302 | 0.447 | 0.676 | 1.053 |
| cells at 5 kb | 0.374 | 0.364 | 0.351 | 0.341 | 0.437 | 0.678 | 1.038 |
| cells at 10 kb | 0.391 | 0.376 | 0.369 | 0.354 | 0.387 | 0.682 | 1.065 |
| midpoint, no cells | 0.332 | 0.332 | 0.307 | 0.305 | 0.453 | 0.684 | 1.053 |
| midpoint, cells at 5 kb | 0.374 | 0.361 | 0.352 | 0.336 | 0.411 | 0.667 | 1.013 |

Pairs of nodes that the cells hold in one structure of chr1 10 to 15 Mb, in and out, of all pairs:

| resolution | GM12878, of 29,646 | H1ESC, of 77,815 | HFFC6, of 66,066 |
|---|---|---|---|
| 1 kb | 168 and 52 | 54 and 47 | 76 and 33 |
| 5 kb | 3,761 and 224 | 2,879 and 297 | 2,774 and 398 |
| 10 kb | 7,141 and 830 | 7,131 and 1,827 | 6,943 and 2,216 |

- At 1 kb few cells pass the test and the scores are those of the arm without cells.
- At 5 kb raw SCC rises 0.055 and balanced SCC 0.035 over the arm without cells, and 0.047 and 0.057 over the default
  chain. Spearman falls 0.007.
- At 10 kb SCC rises a further 0.012 to 0.017 and Spearman falls 0.057 under the arm without cells.
- The midpoint with cells at 5 kb scores as the uniform draw on SCC and 0.026 lower on Spearman.
- Balanced Pearson stays 0.03 to 0.05 under the default chain on every contact bead arm.

### Trial of the bounded drift

`design/figures/uniform-2026-11-19/drift/` and `scores-drift/`. Commit 00c67bd, the uniform draw, cells at 5 kb, z of
3. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | overlaps per thousand beads |
|---|---|---|---|---|---|
| nodes fixed | 0.374 | 0.364 | 0.437 | 0.678 | 21.9 |
| drift within 0.3 beads | 0.373 | 0.365 | 0.436 | 0.678 | 21.5 |
| drift within 0.5 beads | 0.371 | 0.365 | 0.435 | 0.678 | 20.9 |
| drift within 1 bead | 0.372 | 0.362 | 0.434 | 0.678 | 18.3 |

The drift moves no pooled score by more than 0.003. The battery's overlap count falls from 21.9 to 18.3 at one
bead, a count that is not yet checked between the chains.

### Trial of the rules

`design/figures/uniform-2026-11-19/rules/` and `scores-rules/`. Commit e4577f1, the uniform draw, cells at 5 kb, z of
3. Mean over the six rows, with the close pairs of `playground/uniform_overlap_check.py` per thousand beads:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other |
|---|---|---|---|---|---|---|
| default chain | 0.327 | 0.307 | 0.448 | 0.716 | 10.2 | 2.6 |
| nodes fixed | 0.374 | 0.364 | 0.437 | 0.678 | 34.0 | 6.6 |
| orientation term off | 0.368 | 0.358 | 0.429 | 0.677 | 32.9 | 8.9 |
| centre of mass kept | 0.386 | 0.374 | 0.439 | 0.678 | 32.7 | 4.3 |
| bond limit 1.25 | 0.371 | 0.360 | 0.431 | 0.677 | 42.4 | 8.0 |
| bond limit 1.5 | 0.370 | 0.360 | 0.432 | 0.679 | 40.0 | 7.4 |
| centre of mass kept, bond limit 1.5 | 0.381 | 0.367 | 0.431 | 0.679 | 42.0 | 5.5 |

- The centre of mass rule gains 0.012 of raw SCC and 0.010 of balanced SCC over the fixed node and has the fewest
  close pairs that are not next to each other.
- The bond limit gains nothing and raises the close pairs by a fifth.
- Without the orientation term the scores fall by 0.006 to 0.008.

### Overlaps, checked without the bead kind

`playground/uniform_overlap_check.py` counts the pairs of beads closer than 0.7 model units per thousand beads, over
every pair and over the pairs that are not next to each other in the file sorted by start. Mean of the six rows,
five structures per arm, after the relaxation:

| arm | every pair | pairs not next to each other |
|---|---|---|
| default chain | 10.2 | 2.6 |
| contact beads, uniform draw | 37.6 | 10.3 |
| contact beads, midpoint | 30.5 | 5.9 |
| contact beads, cells at 5 kb | 34.0 | 6.6 |
| cells at 5 kb, drift within 1 bead | 24.9 | 3.1 |

- The uniform chain has three to four times the close pairs of the default chain, with and without the contact
  cells. The count of the battery was right about the direction.
- Most of them are two consecutive beads closer than 0.7: 7.6 per thousand on the default chain against 24 to 27.
- On GM12878 chr1 108.5 to 111 Mb, one structure before the relaxation, 83 of 2,482 bonds are under 0.7 beads: 1 of
  the 76 bonds between two nodes, 8 of the 155 between a node and a free bead and 74 of the 2,251 between two free
  beads. The bonds between two nodes have a median of 1.05.
- The short bonds are in the free chain between nodes. In the smooth stage a bond at 0.6 beads costs 0.016 and the
  wall does not act between consecutive beads. The relaxation takes the realised bond as its target. The reading
  of that evening, that the arcs stage gives the beads between two nodes no room, did not hold: see the trial of
  the gyration windows below.
- The drift within one bead lowers both counts by a third to a half.

## Room for the free chain

Status 2026-11-19: options, nothing built. The maintainer asked for this section before a choice.

A bound on the distance of two nodes by the beads between them is wrong. The two anchors of a loop touch whatever
the length of the chain between them, and that chain is beside them. The arcs stage is missing the volume of the
free beads, not a distance between two nodes.

Sizes. A trial region of 5 Mb has 244 to 395 nodes in a conformation and 3,000 to 4,900 beads. Chromosome 1 has
24,000 to 31,000 anchors in the three cell lines and about 230,000 beads. At the node density of the trial regions,
49 to 79 a megabase, it has 11,000 to 18,000 nodes. The arcs solve holds the target matrix of every pair of nodes,
1.0 to 2.6 GB in float64, and the contact cells read two more matrices of that size.

| Option | Rule | New in the solve | Cost on chromosome 1 |
|---|---|---|---|
| A. Blobs | A run of free beads between two nodes enters the solve as blobs of b beads. A blob excludes at the radius of gyration of the law for b beads. Consecutive nodes and blobs are within the chain between their centres | One unit per b free beads | b of 10 adds 23,000 units and the matrix grows to 9 to 13 GB. b of 50 adds 4,600 units, 2 to 4 GB |
| B. Short gyration windows | The gyration term holds every window of four nodes or more at the radius of gyration of the law. It starts at windows of `background_range_bp`, 100 kb. It would start at a smaller window | Windows, no units | A sparse matrix of windows by nodes. Small |
| C. Drift | A node moves within a sphere in the smooth stage, `anchor_cap` | Nothing | None. Measured: a third to a half fewer close pairs at one bead, the pooled scores to 0.003 |
| D. Bond floor | A hard least distance of two consecutive beads in the smooth stage | Nothing | None. Not chosen by the maintainer on 2026-11-19 |

Notes on each.

- A states the missing volume directly and makes the arcs stage solve a coarse chain. It needs three choices: the
  size b, the start of the free beads from the blob positions in the smooth stage, and the contact data of a blob.
  It is the largest change.
- B uses a term the model has. The radius of gyration of the law for a span counts every bead of the span, and
  nodes are a sample of them at the bases of loops. For a window of few nodes the sample is biased. The term holds
  a window at the radius of the law from both sides.
- C acts after the cause and does not need the arcs stage.
- A and B can both be measured against C on the trial rows by the close pairs of
  `playground/uniform_overlap_check.py` and the pooled scores.

The maintainer on 2026-11-19: B first, A when B does not work.

B is `[springs] gyration_min_bp`, the smallest window of the gyration term, 0 for `background_range_bp`. On GM12878
chr1 108.5 to 111 Mb, one structure without contact cells and without the orientation term, the bonds under 0.7
beads before the relaxation are 72 of 2,482 at 100 kb, 103 at 10 kb, 59 at 25 kb and 49 at 50 kb. One structure
decides nothing. The trial runs 10, 25 and 50 kb on the six rows.

### Trial of the gyration windows, and the cause of the close pairs

`design/figures/uniform-2026-11-19/gyr/` and `scores-gyr/`, step 1 of `slurm/ensemble/uniform_night.sh`, commit
5f98a91, cells at 5 kb. Mean over the six rows, close pairs per thousand beads:

| arm | SCC raw | SCC balanced | Spearman raw | close pairs | not next to each other |
|---|---|---|---|---|---|
| nodes fixed, windows from 100 kb | 0.374 | 0.364 | 0.437 | 34.0 | 6.6 |
| windows from 50 kb | 0.362 | 0.358 | 0.440 | 33.2 | 6.6 |
| windows from 25 kb | 0.373 | 0.363 | 0.430 | 33.4 | 6.8 |
| windows from 10 kb | 0.369 | 0.361 | 0.411 | 38.4 | 7.4 |
| midpoint, windows from 25 kb | 0.363 | 0.349 | 0.421 | 31.1 | 5.1 |
| centre of mass kept, windows from 100 kb | 0.386 | 0.374 | 0.439 | 32.7 | 4.3 |
| centre of mass kept, windows from 50 kb | 0.370 | 0.364 | 0.442 | 32.6 | 4.6 |
| centre of mass kept, windows from 25 kb | 0.378 | 0.370 | 0.436 | 32.3 | 4.3 |

B does not work. No window size lowers the close pairs and every one lowers SCC.

The close pairs are not a lack of room. Measured the same night:

- On the trace region the bonds under 0.7 beads are spread over the gaps of every length at 22 to 39 per thousand
  bonds, 55 of the 72 in gaps of more than 20 bonds whose two nodes are 0.04 to 0.09 of the chain apart. They are
  not at the crowded nodes.
- A bond is soft. At `squeeze_constant` 0.1 a bond at 0.7 beads costs 0.009. Over the 30 structures of the trial the
  bonds of the uniform chain have a first percentile of 0.59 and a fifth of 0.77, and 28 per thousand are under 0.7.
- The default chain has the same tail with the orientation term off, 19 per thousand under 0.7 and a first
  percentile of 0.63. With the term on it has 8, since the term stretches its bonds, median 1.06 and 95th percentile
  2.41. The default chain hides its short bonds behind stretched ones.
- With `squeeze_constant` at 1.0 the trace structure has 2 bonds under 0.7 in place of 72, median bond 1.10.
- The smooth confinement does not act: the sphere has a radius of 20 beads and 95 percent of the beads are within
  10.7.

The close pairs that are not next to each other are the overlaps of the chain: 2.6 per thousand on the default
chain, 6.6 with fixed nodes and 4.3 under the centre of mass rule.

Option A, the blobs, is not built. The cause it answers is not the one measured. The stiffness of a bond is step 4
of the queue of the night.

### Trial of the resolution and the test of the cells

`design/figures/uniform-2026-11-19/res/` and `scores-res/`, step 2 of the night, commit 5f98a91, the uniform draw, the
centre of mass rule. Mean over the six rows. The pairs held are those of one structure of GM12878 chr1 10 to 15 Mb,
in and out, of 29,646.

| cells | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs not next to each other | pairs held |
|---|---|---|---|---|---|---|
| none | 0.321 | 0.326 | 0.450 | 0.673 | 7.5 | 0 |
| 5 kb, z of 4 | 0.359 | 0.353 | 0.445 | 0.675 | 6.2 | 2,842 and 107 |
| 2 kb, z of 3 | 0.353 | 0.353 | 0.430 | 0.671 | 7.4 | 962 and 107 |
| 5 kb, z of 3 | 0.386 | 0.374 | 0.439 | 0.678 | 4.3 | 3,761 and 224 |
| 5 kb, z of 2 | 0.406 | 0.394 | 0.390 | 0.667 | 3.7 | 5,528 and 852 |
| 10 kb, z of 3 | 0.401 | 0.384 | 0.388 | 0.682 | 3.4 | 7,141 and 830 |
| 25 kb, z of 3 | 0.415 | 0.395 | 0.401 | 0.738 | 2.1 | 13,656 and 2,343 |

- SCC rises with the count of pairs that the cells hold, from 0.32 with none to 0.41 with half of all pairs.
- Spearman falls from 0.45 to 0.39 to 0.40 once more than a fifth of the pairs are held.
- At 25 kb balanced Pearson passes that of the default chain, 0.738 against 0.716, and the close pairs that are not
  next to each other fall under those of the default chain, 2.1 against 2.6.

The trial scores the pooled map at 25 kb against the map that the cells are read from. At 25 kb the cells are the
scored cells. A gain there is in part the input read back. The cells need a test on counts that the run did not
read: the map thinned into two halves, one for the cells and one for the score.

### Trial of the draw and the loops

`design/figures/uniform-2026-11-19/draw/` and `scores-draw/`, step 3 of the night, commit 5f98a91, cells at 5 kb, z of
3, the centre of mass rule. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other |
|---|---|---|---|---|---|---|
| uniform draw | 0.386 | 0.374 | 0.439 | 0.678 | 32.7 | 4.3 |
| midpoint | 0.377 | 0.367 | 0.417 | 0.668 | 33.7 | 3.9 |
| `loop_dropout` | 0.397 | 0.385 | 0.445 | 0.689 | 29.0 | 2.9 |
| `loop_dropout` with `loop_contact` | 0.418 | 0.391 | 0.574 | 0.626 | 73.8 | 21.6 |
| `arc_weight_exponent` 0.5 | 0.387 | 0.377 | 0.443 | 0.670 | 35.6 | 5.0 |
| orientation term off | 0.379 | 0.365 | 0.433 | 0.679 | 34.6 | 6.3 |
| orientation weight 10 | 0.382 | 0.372 | 0.435 | 0.680 | 34.9 | 4.7 |

- The loop dropout gains every column: 0.011 of SCC, 0.006 of Spearman, 0.011 of balanced Pearson and a third fewer
  close pairs that are not next to each other. A conformation then keeps a loop with a chance by its strength.
- With the kept loops at one bead Spearman rises to 0.574, from 0.439, and raw SCC to 0.418. Balanced Pearson falls
  to 0.626 and the close pairs rise five times. The gain and the overlaps come together.
  These dropout arms carry the stretched tails of the section on the tails below. Their scores are not a result.
- The midpoint is under the uniform draw on every score with the centre of mass rule.
- The loop weight by strength changes nothing. The orientation term at 10 or off loses 0.004 to 0.009 of SCC.

### Trial of the smooth stage and the relaxation

`design/figures/uniform-2026-11-19/smooth/` and `scores-smooth/`, step 4 of the night, commit 5f98a91, cells at 5 kb, z
of 3, the centre of mass rule. Mean over the six rows, with the bonds under 0.7 beads per thousand bonds:

| arm | SCC raw | SCC balanced | Spearman raw | close pairs | not next to each other | bonds under 0.7 | median bond |
|---|---|---|---|---|---|---|---|
| as it is | 0.386 | 0.374 | 0.439 | 32.7 | 4.3 | 28.6 | 1.03 |
| `squeeze_constant` 1.0 | 0.388 | 0.378 | 0.450 | 7.9 | 3.1 | 4.8 | 1.10 |
| `squeeze_constant` and `stretch_constant` 1.0 | 0.376 | 0.367 | 0.432 | 22.9 | 6.8 | 16.3 | 1.02 |
| `noise_smooth` 0.5 | 0.373 | 0.362 | 0.429 | 51.1 | 15.6 | 35.9 | 1.01 |
| no relaxation | 0.389 | 0.375 | 0.434 | 47.9 | 28.2 | 19.9 | 1.02 |
| no hard wall | 0.374 | 0.366 | 0.438 | 26.6 | 10.9 | 15.9 | 1.03 |
| excluded volume at one bead | 0.337 | 0.263 | 0.421 | 45.7 | 1.8 | 45.5 | 1.02 |

- A bond that resists a squeeze ten times more has 7.9 close pairs per thousand beads, under the 10.2 of the
  default chain, and gains 0.011 of Spearman and 0.002 to 0.004 of SCC. The median bond rises from 1.03 to 1.10.
- With the stretch as stiff as well the gain is lost.
- The relaxation removes most of the close pairs that are not next to each other, 28.2 without it against 4.3.
- The half step, no wall and the excluded volume at one bead all lose.

`squeeze_constant` at 1.0 joins the base of the second round.

### Trial of the arcs stage

`design/figures/uniform-2026-11-19/arcs/` and `scores-arcs/`, step 5 of the night, commit 5f98a91, cells at 5 kb, z of
3, the centre of mass rule. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other |
|---|---|---|---|---|---|---|
| as it is | 0.386 | 0.374 | 0.439 | 0.678 | 32.7 | 4.3 |
| `background_weight` 0.3 | 0.367 | 0.356 | 0.459 | 0.668 | 35.2 | 4.6 |
| `background_weight` 1.0 | 0.375 | 0.362 | 0.442 | 0.632 | 44.5 | 7.6 |
| `gyration_weight` 0.3 | 0.370 | 0.363 | 0.423 | 0.696 | 27.2 | 3.4 |
| `gyration_weight` 0 | 0.352 | 0.335 | 0.467 | 0.533 | 97.4 | 27.3 |
| arcs excluded volume at one bead | 0.376 | 0.367 | 0.434 | 0.671 | 33.7 | 3.3 |
| `background_range_bp` 50 kb | 0.384 | 0.371 | 0.437 | 0.685 | 33.2 | 4.9 |

No setting of the arcs stage beats the values of production on SCC. A stronger background spring trades 0.02 of SCC
for 0.02 of Spearman. Without the gyration term the chain collapses: balanced Pearson 0.533 and three times the
close pairs.

### Trial on counts the run did not read, with the singletons of the deep map

`design/figures/uniform-2026-11-19/heldw/` and `scores-heldw/`, step 1 of `slurm/ensemble/uniform_night2.sh`, commit
0b653a9. The earlier trials score a run against the map its contact cells read. Here
`playground/thin_cooler.py` splits every count of the deep map of a cell into two halves by a binomial draw. The
halves of a Poisson count are independent. A run reads half a and is scored against half b. The base of every
uniform arm is the centre of mass rule with `squeeze_constant` 1.0. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other |
|---|---|---|---|---|---|---|
| default chain | 0.305 | 0.294 | 0.374 | 0.745 | 3.3 | 1.5 |
| no cells | 0.280 | 0.314 | 0.397 | 0.717 | 1.6 | 1.0 |
| cells at 5 kb, z of 3 | 0.317 | 0.341 | 0.368 | 0.712 | 1.8 | 0.8 |
| cells at 5 kb, z of 2 | 0.326 | 0.340 | 0.343 | 0.720 | 1.4 | 0.2 |
| cells at 10 kb | 0.317 | 0.328 | 0.357 | 0.738 | 1.3 | 0.3 |
| cells at 25 kb | 0.358 | 0.370 | 0.349 | 0.764 | 1.3 | 0.2 |
| ladder from 5 to 25 kb | 0.334 | 0.349 | 0.377 | 0.761 | 1.3 | 0.2 |
| cells at 5 kb, `loop_dropout` | 0.362 | 0.372 | 0.407 | 0.745 | 0.7 | 0.3 |
| ladder, `loop_dropout` | 0.358 | 0.360 | 0.399 | 0.768 | 0.5 | 0.2 |

The gain of the contact cells holds on counts the run did not read: 0.04 of raw SCC at 5 kb and 0.08 at 25 kb over
the chain without cells, against 0.07 and 0.09 on the map the run read. `loop_dropout` adds 0.045 of raw SCC and
0.04 of Spearman at 5 kb.

The round has two faults of its setup. Both were found by a check of the instrument.

- The halves were balanced over the bins of the regions alone. A bin at the end of a region loses its contacts
  beyond the region and gets a weight 20 to 40 percent above its weight in the whole map. The first score of the
  round, `scores-held/`, has balanced SCC between 0.13 and 0.19 and is not valid. `thin_cooler.py` now gives a half
  the weights of the whole map. The table above is the same structures against the reweighted half,
  scored by `slurm/ensemble/rescore_trial.sh`.
- Every run read the Hi-C singletons of half a of the deep map. Production reads another file with 20,746 contacts
  in chr1 10 to 15 Mb of GM12878, where half a has 1.67 million. The exponent of the law is 0.386 with the deep
  singletons and 0.344 in production. The round is a model on other inputs. Its arms compare with each other and
  not with the earlier trials. The close pairs of the default chain fall from 10.2 to 3.3 per thousand beads with
  these inputs, which is a result of its own: deeper singletons and a larger exponent leave the chain more room.

The round was stopped in its second step. `slurm/ensemble/uniform_night3.sh` repeats it on the data of production,
with the contact cells from half a alone.

### Trial on counts the run did not read, with the inputs of production

`design/figures/uniform-2026-11-19/held3/` and `scores-held3/`, step 1 of `slurm/ensemble/uniform_night3.sh`, commit
1b62c1d. A run reads the data of production. Its contact cells alone come from half a of the deep map, and its
score is against half b with the weights of the whole map. The default chain and the chain without cells are the
structures of the earlier trials, scored on half the counts. Mean over the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other |
|---|---|---|---|---|---|---|
| default chain | 0.325 | 0.305 | 0.448 | 0.715 | 10.2 | 2.6 |
| no cells | 0.331 | 0.333 | 0.456 | 0.667 | 9.6 | 6.2 |
| cells at 5 kb | 0.366 | 0.361 | 0.459 | 0.683 | 7.8 | 4.0 |
| cells at 10 kb | 0.378 | 0.361 | 0.437 | 0.677 | 7.6 | 2.9 |
| cells at 25 kb | 0.423 | 0.400 | 0.403 | 0.710 | 6.8 | 1.4 |
| ladder from 5 to 25 kb | 0.380 | 0.368 | 0.432 | 0.706 | 6.8 | 2.4 |
| cells at 5 kb, `loop_dropout` | 0.395 | 0.380 | 0.445 | 0.677 | 5.1 | 2.4 |
| cells at 25 kb, `loop_dropout` | 0.423 | 0.398 | 0.415 | 0.705 | 3.5 | 1.3 |
| ladder, `loop_dropout` | 0.402 | 0.384 | 0.460 | 0.718 | 2.9 | 0.9 |

The gain of the cells holds on counts the run did not read. At 5 kb it is 0.035 of raw SCC and 0.03 of balanced SCC
over the chain without cells. On the map the run read it was 0.07 and 0.05. The cells of this trial read half the
counts, and a part of the difference is the depth. At 25 kb the gain is 0.09 and 0.07, as on the map the run read.

The same structures scored at 10 kb and at 5 kb, `scores-held3_10k/` and `scores-held3_5k/`, raw and balanced SCC:

| arm | at 25 kb | at 10 kb | at 5 kb | Spearman at 5 kb |
|---|---|---|---|---|
| default chain | 0.325, 0.305 | 0.246, 0.216 | 0.178, 0.139 | 0.292 |
| no cells | 0.331, 0.333 | 0.256, 0.245 | 0.191, 0.166 | 0.325 |
| cells at 5 kb | 0.366, 0.361 | 0.293, 0.267 | 0.221, 0.182 | 0.326 |
| cells at 10 kb | 0.378, 0.361 | 0.299, 0.264 | 0.214, 0.176 | 0.304 |
| cells at 25 kb | 0.423, 0.400 | 0.300, 0.264 | 0.215, 0.177 | 0.302 |
| ladder from 5 to 25 kb | 0.380, 0.368 | 0.293, 0.264 | 0.220, 0.176 | 0.319 |
| cells at 5 kb, `loop_dropout` | 0.395, 0.380 | 0.299, 0.267 | 0.218, 0.181 | 0.327 |
| cells at 25 kb, `loop_dropout` | 0.423, 0.398 | 0.300, 0.265 | 0.211, 0.176 | 0.313 |
| ladder, `loop_dropout` | 0.402, 0.384 | 0.304, 0.266 | 0.222, 0.178 | 0.335 |

The lead of the cells at 25 kb exists in the score at 25 kb alone. At 10 kb every resolution of the cells scores
within 0.007, and at 5 kb the cells at 5 kb lead by 0.006 with the higher Spearman. A cell of 25 kb fits the bin of
the score at 25 kb. It does not place the beads inside the bin better. The uniform chain is ahead of the default
chain at every resolution of the score, with and without cells. Five structures make a sparse pooled map at 5 kb.
The larger trial has twenty.

### Stretched tails, found under the loop dropout

The trials with `loop_dropout` have a radius of gyration of 18 to 65 on chr1 10 to 15 Mb, where the other arms have 8
to 12. `playground/uniform_stretch_check.py` counts the structures with a piece far from the rest. On HFFC6 chr1
10 to 15 Mb three structures of five have a bond of 210 to 280 beads across the gap of 355 kb at 13.24 Mb, with
480 to 1270 beads far from the rest. A run on a laptop without contact cells gives the same three structures.

The cause is the start of the chain in `spans.start_chain`. The beads beyond the last node of a block continued the
line of the nodes at one bond per bead, and a block without a node was one straight line. Without the dropout
every anchor with a loop has a node and a tail is short. With the dropout a block loses nodes. A tail of 480 beads
is then a line 480 bonds long, and a block of 1270 beads whose loops are all dropped is a line through its
centroid. The smooth stage does not coil a line of that length within its budget. The pooled scores do not show
it: a piece far from the rest makes no contact, and the bins of the gap have no count.

`start.coil_start` now starts a tail as a walk from its node at the bond targets. Each bead takes the clear
candidate of twenty draws nearest the node. A chain without a node is two tails from its middle bead. The draws
of the tails follow the draws of every gap. The default chain has an anchor at both ends and draws nothing new.
The byte exactness gate is identical for it.

| HFFC6 chr1 10 to 15 Mb, five structures, no cells | radius of gyration in bonds | bonds over ten per structure | structures with a stretched piece |
|---|---|---|---|
| no dropout | 7.2 | 0.8 | 0 of 5 |
| dropout, tails on a line | 54.3 | 11.2 | 4 of 5 |
| dropout, tails on a coil | 6.9 | 1.0 | 0 of 5 |

The check over the thirty structures of every arm of the earlier trials, six rows of five:

| arm | radius of gyration in bonds | bonds over ten per structure | structures with a stretched piece |
|---|---|---|---|
| default chain | 9.2 | 0.4 | 0 of 30 |
| no cells | 9.3 | 0.6 | 0 of 30 |
| cells at 5 kb | 9.5 | 0.6 | 0 of 30 |
| cells at 25 kb | 9.9 | 0.4 | 0 of 30 |
| ladder from 5 to 25 kb | 9.7 | 0.5 | 0 of 30 |
| cells at 5 kb, `loop_dropout` | 19.9 | 2.9 | 11 of 30 |
| cells at 25 kb, `loop_dropout` | 20.1 | 2.7 | 10 of 30 |
| ladder, `loop_dropout` | 20.3 | 2.8 | 11 of 30 |
| `loop_dropout` of the trial of the draw | 20.8 | 3.1 | 11 of 30 |
| `loop_dropout` with `loop_contact` of the trial of the draw | 19.6 | 2.9 | 13 of 30 |

No arm without the dropout has a stretched piece, in these trials and in the trials of the rules, the smooth stage
and the arcs stage. Their scores stand. A third of the structures of every arm with `loop_dropout` has one, and
the scores of those arms are not a result. The arms run again in `slurm/ensemble/uniform_night4.sh`, which also
keeps the stretch check of every step.

### Trial of the loop dropout with the tails on a coil

`design/figures/uniform-2026-11-19/tail/` and `scores-tail/`, step 1 of `slurm/ensemble/uniform_night4.sh`, commit
f5ec43a. The inputs are those of the trial above: the data of production, the contact cells from half a, the score
against half b. The default chain and the chain without cells are the structures of the earlier trials. Mean over
the six rows, thirty structures per arm:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | close pairs | not next to each other | radius of gyration in bonds | structures with a stretched piece |
|---|---|---|---|---|---|---|---|---|
| default chain | 0.325 | 0.305 | 0.448 | 0.715 | 10.2 | 2.6 | 9.2 | 0 of 30 |
| default chain, `loop_dropout` | 0.322 | 0.294 | 0.428 | 0.701 | 10.3 | 2.9 | 9.1 | 0 of 30 |
| no cells | 0.331 | 0.333 | 0.456 | 0.667 | 9.6 | 6.2 | 9.3 | 0 of 30 |
| no cells, `loop_dropout` | 0.363 | 0.365 | 0.439 | 0.674 | 5.3 | 2.7 | 9.2 | 0 of 30 |
| cells at 5 kb | 0.367 | 0.363 | 0.461 | 0.684 | 7.7 | 4.0 | 9.5 | 0 of 30 |
| cells at 5 kb, `loop_dropout` | 0.388 | 0.384 | 0.442 | 0.669 | 5.5 | 2.6 | 9.1 | 0 of 30 |
| ladder from 5 to 25 kb | 0.375 | 0.364 | 0.432 | 0.706 | 7.0 | 2.6 | 9.7 | 0 of 30 |
| ladder, `loop_dropout` | 0.402 | 0.389 | 0.460 | 0.712 | 3.1 | 1.0 | 9.6 | 0 of 30 |
| cells at 25 kb, `loop_dropout` | 0.424 | 0.405 | 0.419 | 0.701 | 3.4 | 1.3 | 9.6 | 0 of 30 |

No structure has a stretched piece. The radius of gyration of every arm with the dropout is that of its arm
without it.

In the score at 25 kb the gain of the dropout on the uniform chain stands with the tails on a coil. It is 0.02 to
0.03 of SCC with the cells and 0.03 without them. The scores of the stretched structures were within 0.007 of
these: a piece far from the rest makes no contact and takes no part in the score. The dropout does not gain on the
default chain. The cause of that difference is not traced.

The same structures scored at 10 kb and at 5 kb, `scores-tail_10k/` and `scores-tail_5k/`, raw and balanced SCC:

| arm | at 25 kb | at 10 kb | at 5 kb | Spearman at 5 kb |
|---|---|---|---|---|
| default chain | 0.325, 0.305 | 0.246, 0.216 | 0.178, 0.139 | 0.292 |
| default chain, `loop_dropout` | 0.322, 0.294 | 0.237, 0.202 | 0.171, 0.130 | 0.280 |
| no cells | 0.331, 0.333 | 0.256, 0.245 | 0.191, 0.166 | 0.325 |
| no cells, `loop_dropout` | 0.363, 0.365 | 0.272, 0.253 | 0.198, 0.171 | 0.327 |
| cells at 5 kb | 0.367, 0.363 | 0.294, 0.267 | 0.221, 0.183 | 0.326 |
| cells at 5 kb, `loop_dropout` | 0.388, 0.384 | 0.295, 0.271 | 0.216, 0.182 | 0.326 |
| ladder from 5 to 25 kb | 0.375, 0.364 | 0.291, 0.260 | 0.220, 0.176 | 0.319 |
| ladder, `loop_dropout` | 0.402, 0.389 | 0.304, 0.269 | 0.221, 0.179 | 0.335 |
| cells at 25 kb, `loop_dropout` | 0.424, 0.405 | 0.297, 0.266 | 0.208, 0.176 | 0.314 |

With the cells, the gain of the dropout exists in the score at 25 kb alone. At 10 kb it is 0.001 to 0.013 and at
5 kb the arms tie. Without the cells it is 0.016 at 10 kb and 0.007 at 5 kb. The close pairs are the measure the
dropout moves at every resolution: from 7.7 to 5.5 per thousand beads with the cells at 5 kb, from 7.0 to 3.1 with
the ladder and from 9.6 to 5.3 without cells. The default chain with the dropout is behind the default chain at
every resolution of the score. Five structures make a sparse pooled map at 5 kb. Step 2 has twenty.

### The larger trial on counts the run did not read

`design/figures/uniform-2026-11-19/large4/` and `scores-large4/`, with `scores-large4_10k/` and `scores-large4_5k/`,
step 2 of `slurm/ensemble/uniform_night4.sh`, commit f5ec43a. Three cells, six regions of 5 Mb of chr1, twenty
structures per arm. No structure of 360 per arm has a stretched piece. Mean over the eighteen rows, raw and
balanced SCC and raw Spearman:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | Pearson balanced at 25 kb | close pairs | not next to each other |
|---|---|---|---|---|---|---|---|
| default chain | 0.316, 0.277 | 0.250, 0.196 | 0.190, 0.128 | 0.532, 0.463, 0.401 | 0.705 | 11.5 | 3.5 |
| cells at 5 kb | 0.341, 0.349 | 0.299, 0.263 | 0.241, 0.186 | 0.487, 0.462, 0.429 | 0.638 | 10.7 | 5.9 |
| cells at 5 kb, `loop_dropout` | 0.386, 0.387 | 0.323, 0.285 | 0.249, 0.198 | 0.461, 0.455, 0.433 | 0.634 | 6.8 | 3.5 |
| ladder, `loop_dropout` | 0.395, 0.387 | 0.330, 0.282 | 0.253, 0.195 | 0.456, 0.451, 0.436 | 0.670 | 4.5 | 1.8 |

- Balanced SCC at 25 kb is above the default chain on 17 rows of 18 for the cells at 5 kb, raw SCC on 11.
- The dropout raises raw SCC at 25 kb on 18 rows of 18 and balanced SCC on 17. With twenty structures its gain
  also shows at 10 kb, 0.02, and at 5 kb, 0.01.
- Raw Spearman and balanced Pearson at 25 kb are higher for the default chain. Spearman is higher on 12 rows of 18
  against the cells at 5 kb. At 10 kb the Spearman of the arms ties and at 5 kb the uniform chain is ahead.

### Check of the pooled score

`playground/pooled_score_check.py`, on three rows of the larger trial, the default chain against the cells at 5 kb.
The pooled map counts the bead pairs within the contact radius and adds each to the bins of its two bead starts.

The beads of a bin. A bin of 25 kb of the uniform chain holds 25 beads. A bin of the default chain holds 0 to 64,
with a median of 23 to 25. The count of a cell grows with the product of the bead counts of its two bins. The bead
count of the default chain correlates with the raw coverage of Hi-C at 0.49, 0.09 and 0.36 on the three rows.

| row | arm | raw SCC | raw SCC per bead pair | raw SCC of the bead counts alone | Spearman | Spearman per bead pair | Spearman of the decay alone |
|---|---|---|---|---|---|---|---|
| GM12878 200 to 205 Mb | default chain | 0.437 | 0.208 | 0.455 | 0.595 | 0.592 | 0.849 |
| | cells at 5 kb | 0.382 | 0.382 | 0.171 | 0.450 | 0.449 | 0.699 |
| GM12878 20 to 25 Mb | default chain | 0.193 | 0.178 | 0.107 | 0.550 | 0.552 | 0.727 |
| | cells at 5 kb | 0.330 | 0.345 | -0.028 | 0.396 | 0.394 | 0.535 |
| HFFC6 200 to 205 Mb | default chain | 0.308 | 0.232 | 0.190 | 0.600 | 0.600 | 0.799 |
| | cells at 5 kb | 0.424 | 0.412 | 0.082 | 0.622 | 0.622 | 0.826 |

Per bead pair is the pooled map over the product of the bead counts. The bead counts alone is that product times
the mean of each diagonal, a map without a 3D position. The decay alone is the mean of each diagonal of the pooled
map.

- Raw SCC favours the default chain through its bead counts. On GM12878 200 to 205 Mb a map of the bead counts
  alone scores 0.455, above the 0.437 of the structures, and the score per bead pair is 0.208. The scores of the
  uniform chain do not move. The lead of the uniform chain in raw SCC is larger than the tables show.
- Spearman does not depend on the bead counts. It is set by the decay: the decay alone scores above the map of
  the structures on every row, and the difference of the two arms in Spearman, 0.145, 0.154 and -0.022, is the
  difference of their decays alone, 0.150, 0.192 and -0.027.
- The decay of the uniform chain is not monotone on the GM12878 rows. At 4 Mb the mean count is 0.0175 of the
  count at 25 kb, above the 0.0076 at 2 Mb. Hi-C has 0.0030 and the default chain 0.0005. The rise is also in
  the arms without cells.
- Per structure on GM12878 200 to 205 Mb, the bead pairs in contact over 3.5 Mb are 0 to 496 on the default
  chain and 0 to 6250 on the uniform chain, over 600 in 7 structures of 20. Those structures have a radius of
  gyration of 6.7 to 7.8 bonds against 8.0 to 8.7 of the default chain, and 1 Mb at each end of the region
  shares its space with the other. The cause is not traced.

The larger trial scored again with the SCC per bead pair, `scores-large4p/`, `scores-large4p_10k/` and
`scores-large4p_5k/`. `playground/ensemble_score.py` prints it in its last two columns. Mean over the eighteen
rows:

| arm | raw SCC at 25 kb | per bead pair | balanced SCC at 25 kb | per bead pair | raw SCC at 10 kb | per bead pair | raw SCC at 5 kb | per bead pair |
|---|---|---|---|---|---|---|---|---|
| default chain | 0.316 | 0.204 | 0.277 | 0.268 | 0.250 | 0.181 | 0.190 | 0.133 |
| cells at 5 kb | 0.341 | 0.337 | 0.349 | 0.345 | 0.299 | 0.298 | 0.241 | 0.240 |
| cells at 5 kb, `loop_dropout` | 0.386 | 0.381 | 0.387 | 0.382 | 0.323 | 0.322 | 0.249 | 0.249 |
| ladder, `loop_dropout` | 0.395 | 0.390 | 0.387 | 0.383 | 0.330 | 0.329 | 0.253 | 0.253 |

A third of the raw SCC of the default chain at 25 kb is the bead counts of its bins. The difference per row is
-0.07 to 0.29. Balanced SCC moves by 0.009: the balancing takes the coverage out of Hi-C. Per bead pair the cells
at 5 kb are above the default chain in raw SCC on 18 rows of 18 at every resolution and in balanced SCC on 15
or 16. Every raw SCC of the default chain in an earlier comparison, the comparison of the tools among them,
holds this share. Those are not scored again.

### The resolution of the contact cells on the larger trial

`design/figures/uniform-2026-11-19/res20p/` and `scores-res20p/`, with `scores-res20p_10k/` and
`scores-res20p_5k/`. Step 1 of `slurm/ensemble/uniform_night5.sh` adds the ladder alone and the cells at 25 kb to
the larger trial, commit f5ec43a. Eighteen rows of twenty structures, raw and balanced SCC per bead pair and raw
Spearman. The ends overlap in a structure with more than 300 bead pairs between its end megabases:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|
| default chain | 0.204, 0.268 | 0.181, 0.195 | 0.133, 0.127 | 0.532, 0.463, 0.401 | 11.5 | 3.5 | 12 of 360 |
| cells at 5 kb | 0.337, 0.345 | 0.298, 0.263 | 0.240, 0.185 | 0.487, 0.462, 0.429 | 10.7 | 5.9 | 90 of 360 |
| ladder from 5 to 25 kb | 0.343, 0.338 | 0.303, 0.252 | 0.242, 0.176 | 0.442, 0.428, 0.412 | 8.7 | 3.4 | 71 of 360 |
| cells at 25 kb | 0.385, 0.373 | 0.303, 0.258 | 0.230, 0.175 | 0.395, 0.382, 0.376 | 8.2 | 2.4 | 66 of 360 |
| cells at 5 kb, `loop_dropout` | 0.381, 0.382 | 0.322, 0.284 | 0.249, 0.197 | 0.461, 0.455, 0.433 | 6.8 | 3.5 | 117 of 360 |
| ladder, `loop_dropout` | 0.390, 0.383 | 0.329, 0.281 | 0.253, 0.195 | 0.456, 0.451, 0.436 | 4.5 | 1.8 | 77 of 360 |

- The cells at 25 kb lead in the score at 25 kb alone, on 18 rows of 18 over the cells at 5 kb. At 10 kb the three
  resolutions are within 0.005 of raw SCC, and at 5 kb the cells at 25 kb are behind on 14 rows of 18. Their
  Spearman is the lowest at every resolution.
- The ladder and the cells at 5 kb tie. The ladder is ahead in raw SCC on 9 to 11 rows of 18 and behind in
  balanced SCC and in Spearman. It has a fifth fewer close pairs.
- The dropout gains 0.01 to 0.04 of SCC at every resolution under either rule of the cells.
- The ends overlap in a fifth to a third of the structures of every arm of the uniform chain and in 3 percent of
  the default chain. Every arm of this table has the radius of a bead in the arcs excluded volume.

### Trial of the loops, the bonds and the node rule on the base

`design/figures/uniform-2026-11-19/combo4p/` and `scores-combo4p/`, with `scores-combo4p_10k/` and
`scores-combo4p_5k/`, step 3 of `slurm/ensemble/uniform_night4.sh`, commit f5ec43a. Every arm has the cells at
5 kb and `loop_dropout`, six rows of five structures. Raw and balanced SCC per bead pair, raw Spearman at 25 kb,
the close pairs, and the structures with more than 300 bead pairs between the end megabases,
`playground/uniform_ends_check.py`:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|
| default chain | 0.290, 0.304 | 0.223, 0.213 | 0.152, 0.138 | 0.448 | 10.2 | 2.6 | 1 of 30 |
| cells at 5 kb, no dropout | 0.362, 0.359 | 0.291, 0.264 | 0.220, 0.182 | 0.461 | 7.7 | 4.0 | 4 of 30 |
| the base with dropout | 0.382, 0.378 | 0.293, 0.268 | 0.215, 0.181 | 0.442 | 5.5 | 2.6 | 9 of 30 |
| nodes fixed | 0.377, 0.372 | 0.289, 0.263 | 0.213, 0.178 | 0.440 | 6.0 | 3.1 | 10 of 30 |
| `squeeze_constant` 0.3 | 0.385, 0.374 | 0.294, 0.265 | 0.213, 0.177 | 0.439 | 11.0 | 2.5 | 8 of 30 |
| `squeeze_constant` 3.0 | 0.385, 0.379 | 0.296, 0.271 | 0.217, 0.183 | 0.444 | 4.2 | 2.6 | 8 of 30 |
| `loop_dropout_scale` 0.5 | 0.379, 0.370 | 0.298, 0.269 | 0.219, 0.180 | 0.456 | 5.5 | 2.7 | 7 of 30 |
| `loop_dropout_scale` 2.0 | 0.378, 0.366 | 0.287, 0.257 | 0.206, 0.171 | 0.455 | 3.9 | 1.5 | 9 of 30 |
| dropout with `loop_contact` | 0.399, 0.379 | 0.300, 0.274 | 0.211, 0.180 | 0.582 | 35.1 | 26.9 | 2 of 30 |

- The centre of mass rule is ahead of the fixed node by 0.002 to 0.006 of SCC at every resolution, with fewer
  close pairs. The maintainer's choice of the base stands.
- `squeeze_constant` 0.3 doubles the close pairs at the same scores. 3.0 has a quarter fewer close pairs than 1.0
  and scores within 0.003.
- The scale of the dropout moves no score by more than 0.01. At 2.0 the close pairs fall and the scores at 10 and
  5 kb fall with them.
- `loop_contact` raises Spearman by 0.14 at 25 kb and has six times the close pairs. It has the ends overlap
  in 2 structures of 30.
- The dropout doubles the structures with overlapping ends, from 4 to 9 of 30. A dropped loop leaves fewer nodes
  for the excluded volume of the arcs stage. Its lower Spearman follows the overlap, as in the section below.

### The ends of a region share their space: the excluded volume radius of the arcs stage

A trace on a laptop with the config of the trial, `slurm/ensemble/gm12878_xc_R.ini`, without contact cells, one
seed per structure and the same seeds in every arm. The distance is between the centroids of the first and the last
megabase of the region, in bonds. The pairs are the bead pairs of the two megabases within the contact radius.

| GM12878 chr1 200 to 205 Mb, 16 structures | radius of the arcs excluded volume | mean distance after arcs | smallest distance in the output | most pairs | structures over 300 pairs | Rg of the output |
|---|---|---|---|---|---|---|
| default chain | 2.67 | 12.0 | 6.5 | 731 | 3 of 16 | 8.37 |
| uniform chain | 0.70 | 12.5 | 4.6 | 3021 | 6 of 16 | 7.73 |
| uniform chain, `radius_arcs` 3.0 | 3.00 | 14.3 | 9.0 | 251 | 0 of 16 | 8.34 |

On GM12878 chr1 10 to 15 Mb, eight structures, the smallest distance after arcs is 7.5 on the default chain, 5.5 on
the uniform chain and 7.1 at a radius of 3.0, with at most 216, 910 and 376 pairs.

- The arcs stage places the megabases. The distance after arcs and the distance in the output correlate at 0.92
  to 0.97 in every arm. The smooth stage and the relaxation do not move a megabase.
- A node of the uniform chain excludes at the radius of a bead of the smooth stage, 0.7 of a bond, by the
  decision of 2026-11-19. An anchor of the default chain excludes at `auto_factor_arcs` times the mean target
  distance, 2.6 to 3.1 bonds. With `arcs_repulsion` off the excluded volume and the gyration term are all that
  hold two nodes without a loop and beyond 100 kb apart. At the radius of a bead the nodes of two megabases pass
  through each other.
- At a radius of 3.0 no structure of 16 has the overlap and the radius of gyration of the output is that of the
  default chain. At the radius of a bead the output is 7 percent smaller than its nodes.
- The trial of the arcs stage had `radius_arcs` 1 as an arm and scored it by SCC, which does not read the largest
  separations.

A node stands for the chain around it, and that chain needs room that a radius of one bead does not keep. A
radius of 3.0 on every pair also asks two nodes with one bond of chain between them to be 3 bonds apart, against
the reach term. The rule of the radius is open. `slurm/ensemble/uniform_night6.sh` measures the radius of 3.0 on
the larger trial, with and without `loop_dropout`, and `playground/uniform_ends_check.py` counts the overlap of
the ends for every arm.

### Trial of the radius of the arcs excluded volume

`design/figures/uniform-2026-11-19/ev20q/` and `scores-ev20q/`, with `scores-ev20q_10k/` and `scores-ev20q_5k/`,
step 1 of `slurm/ensemble/uniform_night6.sh`, commit f5ec43a. The cells at 5 kb with `[excluded_volume]
radius_arcs` 3.0, with and without `loop_dropout`, beside the same arms at the radius of a bead. Eighteen rows of
twenty structures, raw and balanced SCC per bead pair:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | Pearson balanced at 25 kb | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|---|
| default chain | 0.204, 0.268 | 0.181, 0.195 | 0.133, 0.127 | 0.532, 0.463, 0.401 | 0.705 | 11.5 | 3.5 | 12 of 360 |
| cells at 5 kb | 0.337, 0.345 | 0.298, 0.263 | 0.240, 0.185 | 0.487, 0.462, 0.429 | 0.638 | 10.7 | 5.9 | 90 of 360 |
| cells at 5 kb, radius 3.0 | 0.313, 0.335 | 0.278, 0.255 | 0.226, 0.181 | 0.547, 0.485, 0.434 | 0.695 | 3.2 | 0.6 | 14 of 360 |
| cells at 5 kb, `loop_dropout` | 0.381, 0.382 | 0.322, 0.284 | 0.249, 0.197 | 0.461, 0.455, 0.433 | 0.634 | 6.8 | 3.5 | 117 of 360 |
| cells at 5 kb, `loop_dropout`, radius 3.0 | 0.353, 0.367 | 0.303, 0.274 | 0.236, 0.190 | 0.537, 0.494, 0.450 | 0.688 | 2.3 | 0.5 | 34 of 360 |

No structure of 360 per arm has a stretched piece, and the radius of gyration is 9.4 bonds at a radius of 3.0
against 9.1.

The battery per structure over the eight arms of the larger trial is in `all20p/` and `scores-all20p/`. The
exponent of the decay over that of Hi-C is 1.044 on the default chain, 1.004 with the cells at 5 kb, 1.018 with
the radius of 3.0, 1.04 with the dropout at either radius and 1.11 to 1.12 with the ladder and with the cells at
25 kb.

- The radius of 3.0 takes the overlap of the ends from 90 structures of 360 to 14, the count of the default
  chain. Spearman at 25 kb rises from 0.487 to 0.547, above the default chain, and it is above the default chain
  at every resolution. Balanced Pearson rises from 0.638 to 0.695.
- The close pairs fall from 10.7 to 3.2 per thousand beads and those not next to each other from 5.9 to 0.6, a
  sixth of the default chain.
- SCC falls by 0.02 of raw and 0.01 of balanced at 25 kb and by 0.02 and 0.01 at 10 kb. The arm at 3.0 is above
  the arm at the radius of a bead in raw SCC on 4 rows of 18. It stays above the default chain on 17.
- With the radius of 3.0 the dropout still gains 0.04 and 0.03 of SCC at 25 kb and has the ends overlap in 34
  structures of 360 against 14.

The radius of 3.0 on every pair is the test. It pushes apart two nodes with little chain between them, which is
the likely cost in SCC. The rule of the radius is the maintainer's decision.

### The loop dropout and the overlap of the ends

With `loop_dropout` the ends of a region overlap in more structures, 34 of 360 against 14 at a radius of 3.0 and
117 against 90 at the radius of a bead. Over the 360 structures of each arm on the larger trial the two end
megabases are 1.0 bond closer with the dropout, 12.61 against 13.65 at a radius of 3.0 with a standard error of
0.17, and the radius of gyration of a megabase is the same, 5.28 against 5.26.

Four causes were tested and are not the cause.

- Chain without a node. At a radius of 3.0 the beads of one end in contact with the other end are as far from
  their nearest node as every bead of the end megabases, a median of 13 to 15 kb.
- The megabases. Their radius of gyration does not change.
- The start. The maximum entropy draw realises its targets at a median of 1.000 without the dropout and 1.001
  with it, and the end megabases are 16.5 and 16.9 bonds apart at the start.
- The gyration term. With `gyration_weight` 0 the radius of gyration of the region moves by 0.02 without the
  dropout and by 0.09 with it.

The cause is the excluded volume of the arcs stage. A trace that stops after the joint solve, 120 solves per arm
on GM12878 chr1 20 to 25 Mb, without contact cells:

| arm | nodes | radius of the excluded volume | end megabases apart after the solve | radius of gyration of the region |
|---|---|---|---|---|
| no dropout | 465 | 3.0 | 13.40 ± 0.24 | 8.44 ± 0.02 |
| dropout | 320 | 3.0 | 12.93 ± 0.27 | 8.30 ± 0.02 |
| dropout | 320 | 3.4 | 13.19 ± 0.25 | 8.36 ± 0.02 |
| dropout | 320 | 3.8 | 13.66 ± 0.25 | 8.47 ± 0.02 |

The solve packs the region from a radius of gyration of 9.0 at the start to 8.4. The excluded volume pushes out
with every pair of nodes inside the radius, and the count of such pairs grows with the square of the number of
nodes. The dropout leaves 320 nodes of 465 and the region packs 1.7 percent tighter. The megabases keep their size
and are closer to each other. A radius larger by the ratio of the nodes to the power two thirds, 3.85, gives the
size without the dropout back.

The push of the excluded volume of the arcs stage depends on the number of nodes of the conformation. A node that
stands for more chain after the dropout excludes no more than before.

A larger weight of the term is the weaker lever. The same trace with the dropout at a radius of 3.0:

| weight of the excluded volume | end megabases apart after the solve | radius of gyration of the region |
|---|---|---|
| 0.1, production | 12.93 ± 0.27 | 8.30 ± 0.02 |
| 0.146, times the ratio of the nodes | 13.10 ± 0.27 | 8.33 ± 0.02 |
| 0.211, times the square of the ratio | 13.33 ± 0.27 | 8.37 ± 0.02 |

The square of the ratio of the nodes gives the count of the pairs without the dropout back and restores half of
the size, 8.37 of 8.44. The radius at the ratio to the power two thirds restores all of it.

#### Proposal: the radius of a node from its chain

Not built. It waits for the maintainer.

A node stands for the chain to the midpoints of its two neighbours, `c` beads. The volume a pair excludes grows
with the cube of its radius. The sum over the pairs keeps its value under a change of the node count when the
radius of a pair is `R * (c_i * c_j / c0^2) ^ (1/3)`, with `R` the radius of the arcs stage and `c0` the mean
chain of a node of the conformation before the dropout. Without the dropout the mean of the factor is near 1.
With it the 320 nodes of the trace get a mean radius of 3.85, the value that restored the size. A node beside
a long stretch without a loop excludes more than a node in a dense run of loops. The cap by the chain of
`arcs_cap` stays on top of it. The rule needs a radius per pair in the kernel of the solver and in its JAX
twin, where the radius is one number now.

### Trial of the threshold of the contact cells

`design/figures/uniform-2026-11-19/z20p/` and `scores-z20p/`, with `scores-z20p_10k/` and `scores-z20p_5k/`,
`slurm/ensemble/uniform_night8.sh`, commit f5ec43a. The cells at 5 kb with `contact_background_z` 2 beside the
same arms at 3, at the radius of a bead and at `[excluded_volume] radius_arcs` 3.0. Eighteen rows of twenty
structures, raw and balanced SCC per bead pair:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | Pearson balanced at 25 kb | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|---|
| default chain | 0.204, 0.268 | 0.181, 0.195 | 0.133, 0.126 | 0.532, 0.463, 0.401 | 0.705 | 11.5 | 3.5 | 12 of 360 |
| z 3 | 0.337, 0.345 | 0.298, 0.263 | 0.240, 0.185 | 0.487, 0.462, 0.429 | 0.638 | 10.7 | 5.9 | 90 of 360 |
| z 2 | 0.363, 0.363 | 0.315, 0.271 | 0.250, 0.189 | 0.429, 0.407, 0.381 | 0.606 | 10.1 | 5.0 | 83 of 360 |
| z 3, radius 3.0 | 0.313, 0.335 | 0.278, 0.255 | 0.226, 0.181 | 0.547, 0.485, 0.434 | 0.695 | 3.2 | 0.6 | 14 of 360 |
| z 2, radius 3.0 | 0.346, 0.362 | 0.302, 0.272 | 0.241, 0.188 | 0.512, 0.459, 0.416 | 0.689 | 3.4 | 0.5 | 16 of 360 |

No structure of 360 per arm has a stretched piece.

- At a radius of 3.0 the threshold of 2 gains 0.03 of raw and 0.03 of balanced SCC at 25 kb, 0.02 and 0.02 at
  10 kb and 0.015 and 0.01 at 5 kb. It is above the threshold of 3 in raw SCC on 17 rows of 18 at 25 kb, 18 at
  10 kb and 17 at 5 kb.
- The gain costs Spearman, 0.512 against 0.547 at 25 kb, 0.459 against 0.485 at 10 kb and 0.416 against 0.434
  at 5 kb. At 25 kb the arm is below the default chain, 0.532. Balanced Pearson is 0.689 against 0.695.
- The close pairs and the overlap of the ends do not change with the threshold.
- At the radius of a bead the threshold of 2 has the lowest Spearman of the trial, 0.429.

The threshold of 2 with the radius of 3.0 has the SCC of the threshold of 3 at the radius of a bead, with the
close pairs and the ends of the radius of 3.0. It is a comparison of the sweep of `uniform_sweep8.sh`.

### Sweep at eight structures per arm

`design/figures/uniform-2026-11-19/sweep8a/` and `scores-sweep8a/`, with `scores-sweep8a_10k/` and
`scores-sweep8a_5k/`, step 1 of `slurm/ensemble/uniform_sweep8.sh`, commit 5636a87. The eighteen rows of the larger
trial with eight structures per arm, the cells at 5 kb on counts the run does not read. The first five arms are
the first eight structures of the arms of the larger trial, built at commit f5ec43a. A pooled map of eight
structures scores lower than one of twenty, and the arms compare within this table alone. Raw and balanced SCC
per bead pair, with the rows of 18 above the cells at 5 kb with a radius of 3.0 in raw SCC at 25 kb:

| arm | SCC at 25 kb | rows above | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | Pearson balanced at 25 kb | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|---|---|
| default chain | 0.191, 0.247 | 0 | 0.161, 0.169 | 0.112, 0.105 | 0.502, 0.407, 0.335 | 0.723 | 11.2 | 3.5 | 6 of 144 |
| radius of a bead | 0.318, 0.324 | 15 | 0.269, 0.235 | 0.205, 0.157 | 0.501, 0.436, 0.374 | 0.671 | 10.1 | 5.5 | 34 of 144 |
| radius 3.0 | 0.289, 0.311 | | 0.247, 0.225 | 0.190, 0.151 | 0.517, 0.433, 0.366 | 0.715 | 3.2 | 0.6 | 9 of 144 |
| radius 3.0, `loop_dropout` | 0.326, 0.336 | 16 | 0.263, 0.235 | 0.191, 0.151 | 0.518, 0.447, 0.381 | 0.714 | 2.2 | 0.5 | 13 of 144 |
| radius 3.0, z 2 | 0.325, 0.341 | 16 | 0.270, 0.243 | 0.203, 0.157 | 0.502, 0.421, 0.359 | 0.717 | 3.4 | 0.5 | 4 of 144 |
| radius 3.0, cap by the target | 0.298, 0.315 | 12 | 0.257, 0.228 | 0.198, 0.151 | 0.516, 0.433, 0.367 | 0.715 | 4.5 | 0.6 | 9 of 144 |
| radius 3.0, cap by the chain | 0.292, 0.311 | 10 | 0.248, 0.227 | 0.190, 0.151 | 0.516, 0.433, 0.367 | 0.714 | 3.3 | 0.6 | 9 of 144 |
| derived radius | 0.295, 0.312 | 13 | 0.249, 0.226 | 0.191, 0.149 | 0.518, 0.436, 0.368 | 0.713 | 3.7 | 0.8 | 9 of 144 |
| derived radius, cap by the target | 0.303, 0.319 | 14 | 0.259, 0.229 | 0.200, 0.150 | 0.524, 0.438, 0.370 | 0.713 | 4.8 | 0.8 | 7 of 144 |
| radius 3.0, cap by the target, `loop_dropout` | 0.333, 0.344 | 18 | 0.272, 0.239 | 0.199, 0.155 | 0.517, 0.448, 0.383 | 0.716 | 2.6 | 0.6 | 14 of 144 |
| radius 2.0 | 0.316, 0.324 | 16 | 0.264, 0.232 | 0.201, 0.154 | 0.525, 0.447, 0.378 | 0.692 | 5.2 | 1.2 | 19 of 144 |
| radius 3.0, ladder to 25 kb | 0.326, 0.330 | 16 | 0.270, 0.232 | 0.203, 0.151 | 0.515, 0.442, 0.380 | 0.749 | 4.3 | 0.5 | 2 of 144 |
| radius 3.0, `background_weight` 0.3 | 0.298, 0.317 | 13 | 0.256, 0.228 | 0.198, 0.151 | 0.523, 0.439, 0.370 | 0.709 | 3.8 | 0.9 | 13 of 144 |

No structure has a stretched piece. The radius of gyration is 9.9 bonds with the ladder and 9.1 to 9.4 on the
other arms.

- The ladder at a radius of 3.0 gains 0.04 of raw and 0.02 of balanced SCC at 25 kb, 0.02 and 0.01 at 10 kb, with
  the same Spearman at 25 kb and 0.01 more at 10 and 5 kb. It has the highest balanced Pearson of the table, 0.749,
  and the ends overlap in 2 structures of 144. It has the SCC of the radius of a bead without its close pairs and
  its overlap of the ends.
- The cap by the chain does not change a score. The cap by the target gains 0.01 of raw SCC at each resolution, on
  12 rows of 18 at a radius of 3.0, 12 with the derived radius and 14 under the dropout, and adds 1.3 close pairs
  per thousand beads next to each other.
- The derived radius of the default chain scores as the radius of 3.0.
- The radius of 2.0 has the SCC of the radius of a bead, half of its close pairs and the highest Spearman at
  25 kb. The ends overlap in 19 structures of 144 against 9 at 3.0 and 34 at the radius of a bead.
- The dropout and the threshold of 2 each gain 0.04 of raw SCC at 25 kb, as on twenty structures.
- `background_weight` 0.3 gains 0.01 of raw SCC and has the ends overlap in 13 structures against 9.

Step 2 of the sweep, `scores-sweep8b/`, is the subanchor heatmap of the smooth stage at a radius of 3.0, on the
singletons of production at 25 kb, which are not held out. On 12 rows of 18 the term does not act and the
structures are those of the base. On the 6 rows where it acts raw SCC is above on 2 and below on 4, and three
of them have 18 to 26 close pairs per thousand beads next to each other against 1 to 3. Over the 18 rows it
scores 0.281 and 0.309 against 0.289 and 0.311, a difference of -0.008 ± 0.006. It stays off.

The differences of the table over the 18 rows, paired by row, from `playground/arm_stats.py`, with the p value
of the Wilcoxon signed rank test. The rows of one cell share its data and the p value is a lower bound.

| arm against the radius of 3.0 | raw SCC at 25 kb | balanced SCC at 25 kb | Spearman at 25 kb |
|---|---|---|---|
| ladder to 25 kb | +0.037 ± 0.009, p 0.003 | +0.019 ± 0.008, p 0.03 | -0.003 ± 0.015, p 0.7 |
| `loop_dropout` | +0.036 ± 0.006, p 0.0005 | +0.025 ± 0.006, p 0.001 | +0.001 ± 0.008, p 1.0 |
| z 2 | +0.035 ± 0.006, p 0.00005 | +0.029 ± 0.006, p 0.0007 | -0.016 ± 0.012, p 0.03 |
| radius 2.0 | +0.027 ± 0.006, p 0.0003 | +0.013 ± 0.005, p 0.02 | +0.008 ± 0.006, p 0.2 |
| cap by the target | +0.009 ± 0.004, p 0.06 | +0.004 ± 0.003, p 0.4 | -0.001 ± 0.005, p 1.0 |
| cap by the chain | +0.003 ± 0.003, p 0.2 | +0.000 ± 0.004, p 0.6 | -0.001 ± 0.003, p 0.5 |
| derived radius | +0.005 ± 0.003, p 0.09 | +0.001 ± 0.003, p 0.7 | +0.000 ± 0.005, p 1.0 |
| `background_weight` 0.3 | +0.009 ± 0.005, p 0.06 | +0.006 ± 0.005, p 0.06 | +0.006 ± 0.006, p 0.04 |

The cap by the target against the same arm without it: +0.007 ± 0.003 of raw SCC under the dropout, p 0.02, and
+0.008 ± 0.003 on the derived radius, p 0.02. On the larger trial of twenty structures the threshold of 2 against
3 at a radius of 3.0 is +0.033 ± 0.005 of raw SCC, p 0.0003, and -0.035 ± 0.013 of Spearman, p 0.01.

The second round, `slurm/ensemble/uniform_sweep8c.sh`, joins the ladder at a radius of 3.0 with the dropout, the
threshold of 2 and the cap by the target, and tries the ladder up to 50 kb, the ladder at a radius of 2.0 and the
dropout at a radius of 3.85.

### Second round of the sweep

`design/figures/uniform-2026-11-19/sweep8c/` and `scores-sweep8c/`, with `scores-sweep8c_10k/` and
`scores-sweep8c_5k/`, `slurm/ensemble/uniform_sweep8c.sh`, commit 43163bd on the code of 5636a87. The rows and
the inputs of the first round, eight structures per arm. Every new arm has the ladder to 25 kb. Raw and balanced
SCC per bead pair:

| arm | SCC at 25 kb | SCC at 10 kb | SCC at 5 kb | Spearman at 25, 10, 5 kb | Pearson balanced at 25 kb | close pairs | not next to each other | ends overlap |
|---|---|---|---|---|---|---|---|---|
| default chain | 0.191, 0.247 | 0.161, 0.169 | 0.112, 0.105 | 0.502, 0.407, 0.335 | 0.723 | 11.2 | 3.5 | 6 of 144 |
| radius 3.0, cells at 5 kb | 0.289, 0.311 | 0.247, 0.225 | 0.190, 0.151 | 0.517, 0.433, 0.366 | 0.715 | 3.2 | 0.6 | 9 of 144 |
| radius 3.0, ladder | 0.326, 0.330 | 0.270, 0.232 | 0.203, 0.151 | 0.515, 0.442, 0.380 | 0.749 | 4.3 | 0.5 | 2 of 144 |
| ladder, `loop_dropout` | 0.360, 0.360 | 0.283, 0.248 | 0.202, 0.158 | 0.518, 0.456, 0.393 | 0.740 | 2.2 | 0.4 | 8 of 144 |
| ladder, z 2 | 0.357, 0.361 | 0.288, 0.252 | 0.214, 0.162 | 0.520, 0.444, 0.382 | 0.759 | 4.8 | 0.6 | 7 of 144 |
| ladder, z 2, `loop_dropout` | 0.380, 0.379 | 0.296, 0.259 | 0.210, 0.165 | 0.549, 0.468, 0.399 | 0.753 | 2.3 | 0.4 | 5 of 144 |
| ladder, `loop_dropout`, cap by the target | 0.363, 0.360 | 0.288, 0.247 | 0.208, 0.158 | 0.525, 0.458, 0.394 | 0.742 | 2.8 | 0.5 | 8 of 144 |
| ladder, `loop_dropout`, radius 3.85 | 0.352, 0.358 | 0.277, 0.245 | 0.198, 0.156 | 0.526, 0.456, 0.389 | 0.748 | 1.9 | 0.3 | 6 of 144 |
| ladder, radius 2.0 | 0.328, 0.322 | 0.275, 0.229 | 0.206, 0.151 | 0.477, 0.425, 0.373 | 0.721 | 6.4 | 1.1 | 15 of 144 |

No structure has a stretched piece. The radius of gyration is 9.7 to 10.4 bonds on the arms with the ladder. The
arm with the ladder up to 50 kb did not run. The map of half a holds no resolution of 50 kb.

The differences at 25 kb over the 18 rows, paired by row, with the p value of the Wilcoxon signed rank test:

| comparison | raw SCC | balanced SCC | Spearman |
|---|---|---|---|
| the dropout on the ladder | +0.034 ± 0.006, p 0.0007 | +0.030 ± 0.008, p 0.001 | +0.004 ± 0.004, p 0.4 |
| z 2 on the ladder | +0.030 ± 0.006, p 0.0001 | +0.031 ± 0.006, p 0.0003 | +0.005 ± 0.009, p 0.5 |
| z 2 and the dropout on the ladder | +0.054 ± 0.008, p 0.0003 | +0.048 ± 0.010, p 0.00004 | +0.034 ± 0.007, p 0.0004 |
| z 2 on the ladder with the dropout | +0.020 ± 0.005, p 0.0009 | +0.018 ± 0.004, p 0.001 | +0.030 ± 0.007, p 0.0007 |
| the cap by the target on the ladder with the dropout | +0.003 ± 0.004, p 0.4 | -0.001 ± 0.004, p 0.8 | +0.007 ± 0.004, p 0.07 |
| the radius of 3.85 against 3.0 on the ladder with the dropout | -0.008 ± 0.004, p 0.08 | -0.003 ± 0.004, p 0.5 | +0.008 ± 0.005, p 0.2 |
| the radius of 2.0 against 3.0 on the ladder | +0.002 ± 0.003, p 0.5 | -0.008 ± 0.004, p 0.1 | -0.037 ± 0.010, p 0.002 |
| ladder, z 2 and the dropout against the default chain | +0.189 ± 0.018, p 0.000008 | +0.132 ± 0.019, p 0.0002 | +0.046 ± 0.012, p 0.002 |

- The ladder, the threshold of 2 and the dropout add up. The three together have a raw SCC of 0.380 against
  0.191 of the default chain, above it on 18 rows of 18, with a higher Spearman at every resolution, a higher
  balanced Pearson, a fifth of its close pairs and the same overlap of the ends.
- With the ladder the threshold of 2 does not cost Spearman. With the cells at 5 kb alone it cost 0.035 on the
  larger trial.
- The cap by the target does not change a score on the ladder with the dropout.
- The radius of 3.85 under the dropout loses 0.008 of raw SCC and has the ends overlap in 6 structures of 144
  against 8. The overlap of the ends under the dropout is small with the ladder, and a radius of a node from its
  chain has nothing to repair there.
- The radius of 2.0 on the ladder gains no SCC and loses 0.037 of Spearman.

## Settings

A setting is added when a run reads it.

- `[main] uniform_chain` selects the chain.
- The draw is uniform over the span. `[main] contact_bead` selected the draw at the midpoint, which lost the trial
  of the draw, and was removed on 2026-11-20.
- Bounded drift is `[simulation_arcs_smooth] anchor_cap`, which exists.
- The centre of mass rule is the rule of the uniform chain. `[simulation_arcs_smooth] anchor_centre` selected it
  against the fixed node, which lost the trial of the rules and step 3 of `uniform_night4.sh`, and was removed on
  2026-11-20. The batch executor refuses the smooth stage of the uniform chain until its kernel has the move.

## The first build

Commits 9d8b59f and a898a0e. It kept the arcs stage on points and added a hold to the smooth stage.

- Each anchor is one arcs node at its midpoint. Anchors with their midpoints in one bead are one node, and a pair of
  nodes takes the shortest loop target of its anchor pairs. PET counts do not add.
- The smooth stage moves every bead and adds `anchor_span_weight * |centroid - home|^2` per span. On the traced
  region the centroids end a median of 0.7 beads from their homes. No stage owns the position of an anchor.
- The arcs stage has no reach term and nodes one bead apart have homes a median of 1.5 beads apart.
- The numba and the JAX smooth kernels both have the span term.

`design/features.md` has its measurements and the trace of the long bonds.

### Trial of the first build

`design/figures/uniform-2026-11-19/` has the figure `arms.png`, the table `arms.md` and the score files. Six arms on
GM12878, H1ESC and HFFC6, chr1 10 to 15 Mb and 20 to 25 Mb, five structures per arm, pooled maps at 25 kb. Mean over
the six rows:

| arm | SCC raw | SCC balanced | Spearman raw | Pearson balanced | exponent over Hi-C |
|---|---|---|---|---|---|
| default chain | 0.327 | 0.307 | 0.448 | 0.716 | 1.062 |
| default chain, orientation off | 0.320 | 0.296 | 0.440 | 0.717 | 1.073 |
| first build | 0.332 | 0.334 | 0.452 | 0.708 | 1.038 |
| first build, orientation off | 0.334 | 0.335 | 0.451 | 0.708 | 1.042 |
| first build, span weight 0.1 | 0.319 | 0.323 | 0.452 | 0.708 | 1.038 |
| first build, span weight 0.1, orientation off | 0.317 | 0.320 | 0.449 | 0.709 | 1.048 |

- The first build gains raw SCC on five rows of six, 0.02 to 0.09, and loses 0.20 on HFFC6 chr1 10 to 15 Mb, 0.477 to
  0.278. That row is not traced.
- Balanced SCC is up 0.027 in the mean, Spearman level, balanced Pearson down 0.008.
- The battery counts 9.8 overlapping pairs per thousand beads on the default chain and 16.2 on the first build. It
  reads neighbours from the file order and splits pairs by the kind of bead. Both differ between the chains and the
  two counts are not yet comparable.
- Five structures per arm are few for a pooled map. The trial is a baseline and decides nothing.
