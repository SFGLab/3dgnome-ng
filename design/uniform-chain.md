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
- Excluded volume acts at the radius of a bead, the radius of the smooth stage. It does not act at the mean loop
  target, which was 3.1 beads on the traced region.
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
| Centre of mass kept | The centre of mass of a node, the bead before it and the bead after it stays as it is at the start of the stage. `anchor_centre` is this rule |

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

The third arm is `[motif_orientation] bond_limit`. A move of a bead beside a site is refused when it leaves a bond
of that bead longer than the limit, in bond targets, and longer than it was. A bond over the limit can still
shorten. The rule is a hard rule of the numba kernel, beside the wall and the cap, in the move of one bead and in
the move of three. On GM12878 chr1 108.5 to 111 Mb, one structure, with the term on, the bonds over 2 and over 3
beads before the relaxation are 212 and 89 without a limit, 78 and 0 at a limit of 1.5 and 74 and 4 at 1.25.

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
`gnome3d/mc/numba/terms.py`. The JAX kernel does not have it. Stage 3 is built: the third arm of the orientation
term is `[motif_orientation] bond_limit`, in the numba kernel alone.

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

## Settings

A setting is added when a run reads it.

- `[main] uniform_chain` selects the chain.
- `[main] contact_bead` is the draw, uniform or midpoint.
- Bounded drift is `[simulation_arcs_smooth] anchor_cap`, which exists.
- The centre of mass arm is `[simulation_arcs_smooth] anchor_centre`.

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
