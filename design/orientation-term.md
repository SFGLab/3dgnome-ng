# The orientation term of the smooth stage

The term gives the chain at a CTCF anchor a direction that follows the motifs of its loops. This note has the
term as it is coded, the trace of 2026-11-22 of the damage it does to the chain and the open design.

## The term as coded

- A site is an anchor of the block with a CTCF loop to another anchor of the block. A loop of another factor
  gives no site.
- The vector of a site at bead `a` is `pos[a + 1] - pos[a - 1]`, scaled to length one. An anchor with the motif
  L has the vector reversed. An anchor without a motif has it as it is.
- A loop between the sites `i` and `j` has the energy `weight * w * ((1 - v_i . v_j) / 2)^2`, with `w` the square
  root of the PET count of the loop. The energy is zero for two equal vectors and largest for two opposite ones.
  A convergent pair then has the chain run in opposite directions through its two anchors.
- `[motif_orientation] weight` is 50 in production. A bond has `[springs] stretch_constant` 0.1, and
  `squeeze_constant` 0.1 on the default chain and 1.0 on the uniform chain.
- The kernels are `_local_score_orientation_nb` in `gnome3d/mc/numba/terms.py` and `_local_orientation_at` in
  `gnome3d/mc/jax/smooth.py`. The loops and their weights are built in `gnome3d/skeleton.py`.

The reference has the same form and the same three numbers: the vector over the two neighbours, the weight 50
and bonds at 0.1.

## Reports

The maintainer reported on 2026-11-21 two jobs of the site with long straight strands folded into sharp hairpins,
over about half of a model by eye, at chr3 134.39 Mb and at chr1 11.94 Mb.

The trace of 2026-11-19 in `design/features.md` on chr1 108.5 to 111 Mb counted 117 bonds over 3 beads on the
default chain against 3 with the term off.

## Trace of 2026-11-22

`playground/orientation_trace.py` records every call of the numba smooth kernel with its chain before and after
and reads the chain at the sites of the term. GM12878 chr1 11 to 13 Mb, 4 structures per arm, production configs,
before the cross block relaxation. The arm with the term off is read at the sites of the arm with it on.

| | default chain, on | default chain, off | uniform chain, on | uniform chain, off |
|---|---|---|---|---|
| sites | 368 | 368 | 228 | 228 |
| bonds of a site to its neighbours over 2 targets | 23.9 % | 1.5 % | 33.3 % | 5.3 % |
| bonds from those neighbours outwards over 2 targets | 22.7 % | 0.5 % | 30.3 % | 1.3 % |
| bonds three beads or more from a fixed bead over 2 targets | 0.6 % | 0.3 % | 0.9 % | 0.5 % |
| longest bond, in targets | 6.9 | 3.0 | 6.6 | 3.3 |
| bends sharper than 120 degrees within two beads of a site | 26.4 % | 4.6 % | 21.3 % | 5.7 % |
| bends sharper than 120 degrees three beads or more from a fixed bead | 0.4 % | 0.2 % | 1.1 % | 0.9 % |
| chord of a site over 1.5 | 16.6 % | 1.4 % | 36.4 % | 7.9 % |
| chord of a site under 0.5 | 13.3 % | 6.2 % | 7.5 % | 3.5 % |
| misfit of the term before the stage | 0.324 | 0.324 | 0.349 | 0.349 |
| misfit of the term after the stage | 0.076 | 0.338 | 0.007 | 0.319 |
| bond energy of the final chains | 146 | 19 | 167 | 35 |
| orientation energy of the final chains | 30,920 | 87,612 | 836 | 53,322 |
| beads within five beads of a site | 30.9 % | 30.9 % | 23.5 % | 23.5 % |

The chord of a site is the distance of its two neighbours over the sum of the two bond targets. The misfit is the
mean of `((1 - v_i . v_j) / 2)^2` over the loops. Two random vectors have 0.333.

- The damage is at the sites. A quarter to a third of the four bonds around a site are over twice their target
  with the term on. Away from the fixed beads the bonds are the same with the term on and off.
- The bonds of a site and the bonds one further out stretch by the same share. The term moves the two neighbours
  of a site one by one to the places that give the vector its direction, and both bonds of a moved neighbour
  stretch. The chord is both longer and shorter than without the term: a neighbour is pulled out along the
  direction or folded back beside the other one.
- The chain enters the stage with a random vector at every site: the misfit is that of two random vectors. No
  earlier stage gives the chain a direction at an anchor. The whole direction is made by two beads.
- The energy of the term is 200 times the energy of the bonds at the end of the stage on the default chain and
  5 times on the uniform chain. At the start of the stage it is 87,668 and 57,645. A move that turns the vector is
  taken at any cost in bonds.
- On the default chain the term ends at a misfit of 0.076 with 35 % of its energy left. On the uniform chain it
  ends at 0.007 with 1.4 %.
- A quarter to a third of the beads of a model are within five beads of a site.

## The model without the term, 2026-11-22

The trial `orn20` on the workstation: 18 rows of three cell lines and six regions of 5 Mb of chr1, twenty
structures an arm, the inputs of production with the contact cells of one half of the deep map and the score
against the other half. `o_on` is the locked uniform chain, `o_off` the same without the term, `d_off` the default
chain without the term, `prod` the default chain. Differences are over the rows with their standard error,
`playground/arm_stats.py`. Scores in `design/figures/uniform-2026-11-19/scores-orn20*`.

| without the term, against with it | uniform chain | rows above | p | default chain | rows above | p |
|---|---|---|---|---|---|---|
| SCC per bead pair at 25 kb, raw | -0.005 ± 0.002 | 5 of 18 | 0.032 | +0.002 ± 0.003 | 10 of 18 | 0.45 |
| SCC per bead pair at 25 kb, balanced | -0.007 ± 0.002 | 3 of 18 | 0.0007 | -0.003 ± 0.003 | 9 of 18 | 0.47 |
| Spearman at 25 kb | -0.009 ± 0.001 | 0 of 18 | 0.0002 | -0.005 ± 0.005 | 7 of 18 | 0.29 |
| balanced Pearson at 25 kb | +0.004 ± 0.001 | 17 of 18 | 0.0003 | +0.005 ± 0.001 | 15 of 18 | 0.002 |
| SCC balanced at 10 kb | -0.008 ± 0.002 | 1 of 18 | 0.001 | | | |
| SCC balanced at 5 kb | -0.005 ± 0.001 | 1 of 18 | 0.003 | | | |

- On the uniform chain the term is worth 0.005 to 0.008 of balanced SCC and 0.007 to 0.009 of Spearman at
  every resolution. The raw SCC is not consistent: lower without the term at 25 kb, equal at 10 kb and higher
  at 5 kb.
- On the default chain no score of SCC or Spearman moves without the term.
- The close pairs are 1.5 per thousand beads with the term and 1.2 without it on the uniform chain, and 11.5
  and 10.6 on the default chain. The radius of gyration is 1 to 2 % larger without the term.
- The locked uniform chain at the commit of this trial repeats the arm `L_node` of 1a1b93d on every score, the
  largest difference 0.001. The commits between them, with the terms of the arcs solve on the device, did not
  move the scores.

## Design of the rewrite, 2026-11-22

The maintainer chose to rewrite the term on 2026-11-22. This is the design on paper. Steps 1 to 3 of the build
are done.

### The direction of a site

The direction of a site at bead `a` is the unit vector from the centroid of the `k` beads before it to the
centroid of the `k` beads after it, reversed for the motif L. The anchor bead itself is in neither centroid.

- With `k` of 1 this is the vector of the term as it is.
- One bead moved by one bond turns the vector of the term as it is by half a radian. It turns the vector over
  `k` beads a side by `1 / (k (k + 1))` radians: 4.8 degrees at `k` 3. No single bead makes the direction. The
  stretch of `2 k + 1` beads turns as a whole, and the chain beside it follows by its bonds.
- `k` is 3, a documented choice: the shortest stretch in which a bead moved by one bond turns the vector by
  under 5 degrees. On the uniform chain it is 3 kb a side.

### The energy of a site

A loop between the sites `i` and `j` keeps the form `((1 - v_i . v_j) / 2)^2`. Its weight changes.

- The weight of a loop is the root of its PET count over the larger of the sums of those roots at its two
  sites. The weights of the loops of a site then sum to one at most, and the weight of a loop is the same seen
  from either site, which the update of the running score needs.
- The weight of the term is the energy of the `2 k` bonds of a stretch each one target too long,
  `2 k * stretch_constant * smooth_dist_weight`: 0.6 at `k` 3. A site at its worst misfit has the energy of
  its own stretch pulled out by one bond a bead. `[motif_orientation] weight` is not read by this form.
- The term stays non negative and its local scores sum to its full score.

### The cases

| case | rule |
|---|---|
| a site at the first or the last bead of a block | the centroid of the missing side is the anchor bead |
| a site under `k` beads from an end of the block | the centroid of that side is of the beads there are |
| two sites under `k` beads apart | each stretch holds the beads there are, the other anchor among them |
| a moved bead within `k` beads of several sites | every one of those sites is scored again, one after another |
| a node of the uniform chain moved with its two neighbours | the sites within `k` beads of any of the three are scored again |
| an anchor without a CTCF loop inside the block | no term, as now |
| an anchor without a motif on a CTCF loop | a site with its vector not reversed, as now |
| a loop of another factor | no term, as now |
| the cross block relaxation | no term, as now |

The term as it is scores one site for a moved bead, the later of two anchors when the bead is between two. The
running score of the orientation then leaves the full score when two anchors are two beads apart. The rewrite
scores every site that reads the moved bead.

### What each stage may move

No stage before the smooth stage reads the direction. The smooth stage turns the stretch of a site with the
moves it has, one bead or a node with its two neighbours. The arcs stage, the densify stage and the relaxation
are as they are.

### The build and the trial

1. The numba kernel first, the executor of production: the vector over a stretch, the table of the sites that
   a move reads, the weights. The term as it is stays beside it under `[motif_orientation] form`, `neighbours`
   or `stretch`, a key under trial that is deleted with the losing form.
2. Unit file after `tests/unit/test_terms.py`: the vector of a hand built chain in closed form, the local
   scores against the full score with sites under `k` beads apart, the term non negative, the weights of the
   loops of a site.
3. The trace of `playground/orientation_trace.py` on the three arms: the term as it is, the rewrite and no
   term. The rewrite is right when the bonds over twice their target at the sites are at the level of the arm
   without the term and the misfit falls.
4. The 18 held out rows at twenty structures: the rewrite against the term as it is and against no term, by
   paired differences.
5. With the rewrite adopted: the JAX kernel gets the same form, the term as it is and its key are deleted. The
   batch executor refuses `stretch` until then.

## Trace of the rewrite, 2026-11-22

The numba kernel has the rewrite as `[motif_orientation] form = stretch` since 450522e. The same trace as above,
GM12878 chr1 11 to 13 Mb, 4 structures an arm: the term as it is, the rewrite and no term. The misfit is read
with the vector over three beads a side for every arm.

| | default: as it is | rewrite | no term | uniform: as it is | rewrite | no term |
|---|---|---|---|---|---|---|
| bonds within two beads of a site over 2 targets | 25.2 % | 0.8 % | 1.1 % | 31.0 % | 4.3 % | 4.2 % |
| the same over 3 targets | 9.1 % | 0.0 % | 0.0 % | 16.3 % | 0.4 % | 0.3 % |
| longest bond, in targets | 6.9 | 2.7 | 3.0 | 6.6 | 3.4 | 3.3 |
| bends sharper than 120 degrees within two beads of a site | 26.4 % | 5.8 % | 4.6 % | 21.3 % | 7.0 % | 5.7 % |
| bond energy of the final chains | 146 | 21 | 19 | 167 | 38 | 35 |
| misfit over three beads a side, before the stage | 0.322 | 0.322 | 0.322 | 0.329 | 0.329 | 0.329 |
| misfit over three beads a side, after | 0.183 | 0.147 | 0.326 | 0.111 | 0.185 | 0.338 |
| misfit over the two neighbours, after | 0.076 | 0.259 | 0.338 | 0.007 | 0.265 | 0.319 |

- The rewrite leaves the bonds and the bends at the sites as the model without the term has them, on both chains.
- The chain turns at the sites: the misfit over three beads a side falls from 0.32 to 0.15 on the default chain
  and from 0.33 to 0.19 on the uniform chain. It does not reach zero at a weight of 0.6.
- On the default chain the rewrite gives the stretch more direction than the term as it is, 0.147 against 0.183.
  The term as it is turns the two neighbours and leaves the rest of the stretch. On the uniform chain the term
  as it is reaches 0.111.

## The rewrite on held out inputs, 2026-11-22

The trial `ornst20`: `o_st`, the locked uniform chain with the form `stretch`, on the 18 rows at twenty
structures, beside `o_on` and `o_off` of `orn20`. Scores in `design/figures/uniform-2026-11-19/scores-ornst20*`.

| at 25 kb | the rewrite against the term as it is | rows above | p | the rewrite against no term | rows above | p |
|---|---|---|---|---|---|---|
| SCC per bead pair, raw | -0.003 ± 0.001 | 4 of 18 | 0.052 | +0.001 ± 0.002 | 10 of 18 | 0.42 |
| SCC per bead pair, balanced | -0.007 ± 0.001 | 1 of 18 | 0.0005 | +0.001 ± 0.001 | 11 of 18 | 0.54 |
| Spearman | -0.009 ± 0.001 | 0 of 18 | 0.0002 | -0.000 ± 0.001 | 7 of 18 | 0.46 |
| balanced Pearson | +0.003 ± 0.001 | 16 of 18 | 0.0004 | -0.000 ± 0.000 | 5 of 18 | 0.33 |

- The rewrite scores as the model without the term on every score, at 25, 10 and 5 kb. It has 1.2 close pairs
  per thousand beads, as the model without the term, against 1.5 with the term as it is.
- The 0.007 of balanced SCC and the 0.009 of Spearman of the term as it is on the uniform chain are not kept.
  They come with what the term as it is does and the rewrite does not: the two neighbours of a site moved at a
  weight of 50, with the bonds that stretch for it, or a direction held harder than the rewrite holds it. The
  misfit over three beads a side is 0.111 with the term as it is and 0.185 with the rewrite. The trial does not
  separate the two.

## The rewrite at a larger weight, 2026-11-22

The maintainer chose one more arm, the rewrite at ten times its weight, `[motif_orientation] stretch_weight`, a
key of the trial. The trace on GM12878 chr1 11 to 13 Mb, 4 structures an arm:

| | default: 1 | 10 | 100 | uniform: 1 | 10 | 100 |
|---|---|---|---|---|---|---|
| bonds within two beads of a site over 2 targets | 0.8 % | 3.3 % | 12.5 % | 4.3 % | 6.7 % | 15.4 % |
| bends sharper than 120 degrees within two beads of a site | 5.8 % | 12.8 % | 19.6 % | 7.0 % | 11.8 % | 18.7 % |
| longest bond, in targets | 2.7 | 3.6 | 6.5 | 3.4 | 3.6 | 4.9 |
| bond energy of the final chains | 21 | 33 | 89 | 38 | 47 | 77 |
| misfit over three beads a side, after the stage | 0.147 | 0.035 | 0.010 | 0.185 | 0.045 | 0.005 |

- At 10 the direction is held harder than by the term as it is, 0.035 and 0.045 against 0.183 and 0.111, with
  2 points more of bonds over twice their target than the model without the term.
- At 100 an eighth of the bonds at the sites are over twice their target. The stretch of a site is pulled out
  along its direction.

The arm `o_st10` on the 18 held out rows is building on the workstation as `ornst10`.

## Open

The choice of the maintainer between the rewrite as it is, the model without the term and a trial of the rewrite
at a larger weight. With a form adopted, the same form in the JAX kernel and the removal of the other form and
of the key.
