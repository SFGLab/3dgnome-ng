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
- The gyration term and the contact background are unchanged. They read the genomic midpoints of the beads.

The count of nodes changes between conformations with the draw. The joint solve is over the chromosome as now.

## Smooth stage

The chain starts with each node at its arcs position and the beads between two nodes on the bridge of
`smooth_start`. The beads before the first node and after the last continue the chain.

The arcs stage owns the positions of the nodes. Three arms state the freedom of the smooth stage over them.

| Arm | Rule |
|---|---|
| Fixed | A node does not move in the smooth stage |
| Bounded drift | A node moves inside a sphere of a stated radius around its arcs position and a move that leaves the sphere is refused. `smooth_anchor_cap` is this rule |
| Centre of mass kept | The centre of mass of the beads of each span stays as it is at the start of the stage. A move of one bead of a span by a vector moves another bead of the same span by the opposite vector. A bead under several spans moves only with a partner under the same spans. A span of one bead does not move |

No arm has a penalty with a weight between the two stages. Every bead that is not a node moves freely, with the other
beads of a wide span among them.

The cross block relaxation follows the arm of the smooth stage. It reads the nodes from the pipeline and not from
the kind of the output bead.

## Orientation

A site of the orientation term is a contact bead of a CTCF loop. Its direction is from the bead before it to the
bead after it.

Measured on 2026-11-19: at weight 50 against bonds at 0.1 it stretches the bonds beside a site on the default
chain and on the first build. With the term off the pooled SCC of the trial moved by 0.011 or less in the mean over
six rows on either chain. The arms are the term as it is, the term off, and the term with a move of the two beads
beside a site refused when it stretches their bonds past a stated length.

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

## Settings

A setting is added when a run reads it.

- `[main] uniform_chain` selects the chain.
- The draw needs one key with the values uniform and midpoint.
- Bounded drift is `[simulation_arcs_smooth] anchor_cap`, which exists.
- The centre of mass arm needs one key.

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
