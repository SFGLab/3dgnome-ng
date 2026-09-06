# Anchor placement

Reconstructions come out as compact balls of anchors scattered in space and joined by thin
strands. This records what causes that, what has been measured, what has been ruled out, and
which changes are worth making.

Status. Diagnosis complete. B, the boundary stitch, converges and removes the strands between
blocks. E, the cross block relaxation, removes the interpenetration. H, one distance law in
bead units with its exponent measured from the input, is the law as of 2026-09-06: it replaced
the three reference laws and their fifteen constants after beating them on three cell lines,
and A, D and G went with them. What remains open is the realised exponent running 1.3 to 1.5
times the measured input, which one correction on the background exponent would absorb.

## Symptom

A whole chromosome renders as roughly one ball per interaction block, each ball a tight tangle,
with long single strands between them. On chr1 that is about 52 balls for the arc gap block
partition and about 69 for the trio partition.

## The target

Polymer physics links contact probability to spatial distance. If distance grows as `s^nu` with
genomic separation `s`, two loci meet with probability near `s^(-3 nu)`, because the capture
volume is fixed and the coil volume grows as the cube of its radius. A contact probability slope
measured from Hi-C therefore fixes `nu` without appeal to any other implementation.

Measured with `playground/ps_curve.py` on the deepest 4DN mcool for each cell line, five
chromosomes, 5 kb bins, log binned ten per decade.

| cell line | 20 kb to 1 Mb | 1 to 10 Mb |
|---|---|---|
| GM12878 | 0.309 | 0.345 |
| H1ESC | 0.269 | 0.260 |
| HFFC6 | 0.277 | 0.261 |

`nu` is near 0.285 in every cell line and both ranges, the fractal globule value. Every
comparison below uses that number as the yardstick.

`genomic_length_to_distance` is `1 + 0.5 * (s / 1000)^0.75`, so its own exponent is 0.75, about
0.70 effective over 20 kb to 1 Mb once the base is included. That is 2.5 times steeper than the
data. Over 20 kb to 1 Mb it separates loci 5.2 times more than Hi-C implies. It is the right
chain bond target at the scale it was tuned for and the wrong shape for anything that has to hold
across separations.

## What was measured

All numbers are from this repository. Unless stated otherwise they come from GM12878 on the
arc gap block partition.

### Three defects, not one

The realised distance curve is not one flat line. It is two flat regimes joined by a jump, and
each piece has its own cause.

| regime | separation | realised distance | cause |
|---|---|---|---|
| within a block | 3 kb to 200 kb | flat near 3.4 | arc target has no separation, everything else is scale free repulsion |
| across blocks | 1 Mb to 60 Mb | flat near 75 | `confinement_packing_factor_ib = 0.15` |
| the jump | any boundary | 33 to 59 times at matched separation | nothing couples the two edges |

Pooling them produces an apparent overall exponent near 0.9 that is pure mixture, since short
pairs are all within block and long pairs are all across.

### Within a block, anchors collapse

The arc target does not depend on genomic separation. `freq_to_distance(freq)` is
`base_level + scale / exp(a * (freq + shift))`, a function of PET count alone, and at
`base_level = 0.2` any arc above about twenty reads returns approximately 0.2. Measured on H1ESC
chr11 with `playground/scale_clash.py`:

| genomic separation | arc target | chain target | realised |
|---|---|---|---|
| 5 to 20 kb | 0.200 | 5.14 | 0.359 |
| 20 to 50 kb | 0.202 | 8.24 | 0.369 |
| 50 to 100 kb | 0.213 | 13.64 | 0.404 |
| 100 to 300 kb | 0.249 | 25.51 | 0.450 |
| 300 kb to 1 Mb | 0.309 | 50.73 | 0.530 |

Almost no pair is constrained at all. `calc_anchor_expected_distances` gives a pair either an arc
target or -1, which the kernel scores as an unbounded `1/d` repulsion. On HG00512 chr1 over the
twelve largest blocks, 5.17M anchor pairs, `playground/hic_coverage.py`, between 95 and 100
percent of pairs at every separation have nothing but that repulsion, and Hi-C reaches only 0.1
percent of them. Arc springs pulling everything to 0.2 against a repulsion with no sense of scale
equilibrates at a uniform ball.

Measured on five finished models from three data sources with `playground/calibrate_beta.py`,
the realised exponent over 20 kb to 1 Mb is 0.143 to 0.212, always well under 0.285. The
simulated contact probability from the beads themselves decays as `s^-0.41` against `s^-0.86` in
the Hi-C, which is the same fact on the observable Hi-C measures directly.

### Across blocks, the layout was a setting

Block centroids are placed by the coarse stage with chain bonds between consecutive centroids,
excluded volume, and a confinement sphere of radius `pf * mean(dtn) * N^(1/3)` per segment.
`validation/core/config.py::CANONICAL` set `pf = 0.15`, five times tighter than the `Settings`
default of 0.75. It entered in commit c7dc086 and never went through the harness grid, which
had validated 0.75 to 1.0.

That sphere is `2 * pf * N^(1/3)` chain bonds wide. At 0.15 a segment needs 37 blocks before the
sphere can hold one bond. GM12878 chr1 segments hold 3 to 20, so every bond in every segment is
violated by construction. Excluded volume wants centroids at least `0.5 * dtn` apart, confinement
wants all of them inside a sphere narrower than one bond, both cannot hold, and whichever jams
first sets the layout instead of genomic separation.

Coarse stage alone on chr1, 52 blocks, `playground/ib_confine_ablate.py`:

| `packing_factor_ib` | block layout exponent | Rg |
|---|---|---|
| 0.15 | 0.021 | 99 |
| 0.29 | 0.068 | 146 |
| 0.50 | 0.146 | 190 |
| 0.75 | 0.214 | 261 |
| 1.00 | 0.297 | 320 |
| 1.50 | 0.280 | 320 |
| off | 0.280 | 320 |

Finished structures on chr1:1-60 Mb, 11 blocks in 5 segments, 3158 anchors, 3M sampled pairs,
`playground/boundary_wide.py`:

| `packing_factor_ib` | cross block exponent | within block exponent | Rg, all beads |
|---|---|---|---|
| 0.15 | 0.017 | 0.648 | 146 |
| 0.75 | 0.183 | 0.645 | 342 |
| 1.00 | 0.168 | 0.648 | 418 |
| off | 0.168 | 0.648 | 418 |

The finished structures inherit the coarse layout. Within block numbers are identical across all
four, so the setting is orthogonal to the within block collapse. At 1.0 the sphere never binds
on this region's 3 to 5 block segments and the run reproduces the unconfined trajectory exactly.
At 0.75 it binds during the trajectory and ends inside the sphere, realised over auto radius
0.75, so it still nudges the layout. The binding threshold scales with block count, so the
choice between 0.75 and 1.0 is decided by segments with many blocks, where the coarse sweep
shows 1.0 landing on 0.297 and 0.75 short at 0.214.

Changed to 0.75 in `CANONICAL` and all seventeen configs on 2026-09-02. Every production model
built before that carries 0.15, including the three cell line genome ensembles and the trio
array 1805610.

### The jump at a boundary survives that change

Relaxing confinement restored separation scaling across blocks. It did not close the gap
between a block's last anchor and the next block's first, because nothing couples them. Both
are placed only through their own block centroid. Measured at matched separation of 562 kb to
1 Mb on the finished chr1:1-60 Mb structures:

| `packing_factor_ib` | within block, over `gld` | across a boundary, over `gld` | ratio |
|---|---|---|---|
| 0.15 | 0.092 | 3.04 | 33 |
| 0.75 | 0.092 | 5.42 | 59 |
| off | 0.092 | 5.72 | 62 |

In model units two loci 750 kb apart sit at about 7 within a block and about 400 across a
boundary. The ratio grew when confinement was relaxed, because blocks moved apart and nothing
pulled the adjacent edges back together. This is a separate defect from the scaling and it is
what option B addresses. The earlier estimate of a flat factor of 27 came from 35 consecutive
pairs whose gaps barely overlapped the comparison group. On 3.7M pairs the penalty is 15.5 at
300 to 560 kb on whole chromosome models and falls to 3.8 above 5 Mb, so it is separation
dependent, and the short range end is where it bites.

Cross block contacts beyond 1 Mb at twice the bond length are 0.0008 of pairs in all three
structures. The models predict essentially no contact across a boundary, whatever the packing
factor.

### The subanchor chain balloons

Between anchors 35 kb apart the chain is handed about 8.7 units of contour for a gap of 0.37,
roughly 36 times more than it needs. Anchors are held fixed during smooth MC, so the excess
relaxes into a free coil. That is the fuzz on each ball rather than a clean path.

## What has been ruled out

- Excluded volume, confinement at the arcs and smooth levels, and the CTCF motif weight, for
  the within block collapse. Ablated one at a time on a 10 Mb region. The defect survives all
  three off. Confinement at those levels never binds. The IB level confinement is a different
  term and does bind, see above.
- Smooth MC failing to converge. It converges to a minimum that is not at the target distances.
- Ensemble depth. Five, ten and twenty conformations give the same answer to within 0.006 on a
  pinned gene set.
- Being worse than cudaMMC. Its leaf chain is equally flat. An earlier investigation compared
  our raw beads against cudaMMC's `.smooth.txt`, which is resampled onto a uniform 1 kb lattice
  and manufactures short range structure the model does not have.
- Hi-C supplying the missing within block targets. It reaches 0.1 percent of arcless pairs.
- Finer blocks. An arc gap boundary sits where no arc spans, so the arc gap partition is by
  construction the finest one that discards no arc. Any finer partition cuts arcs, and a cut arc
  constrains nothing at anchor resolution. TAD boundaries orphan 5437 of 12474 arcs on GM12878
  chr1 against zero under arc gaps. Block granularity trades directly against arc retention.

## Solutions

### A. Excluded volume floor that grows with genomic separation. Removed 2026-09-06

Implemented in `gnome3d/pipeline/ib/floor.py`, the arcs stage and both arcs kernels, gated on
`[excluded_volume] use_genomic_floor`. Unit checks in `harness/test_genomic_floor.py`.

First measurement, chr1:1-60 Mb on the workstation, same seed as the stitch runs, with the floor
riding the excluded volume term's weight of 0.1. It failed, and the reason is the weight.

| arm | within block exponent | block Rg median | Rg | jump at 562 kb to 1 Mb | overlapping block pairs |
|---|---|---|---|---|---|
| off | 0.208 | 7.7 | 342 | 58.9 | 1 |
| floor | 0.267 | 7.2 | 342 | 84.9 | 1 |
| floor and stitch | 0.266 | 7.2 | 36 | 0.86 | 13 |

The exponent moved toward 0.285 as designed, but every within block bin got smaller, 3.19 to
1.86 at 3 to 5 kb and 6.89 to 4.88 at 0.5 to 1 Mb, so the exponent rose only because short
range shrank faster than long range. Pairs sat at about half their floor. The floor retires the
`1/d` repulsion, which reaches 5 at a distance of 0.2 and was the term actually holding each
ball at 3.2, and replaced it with a quadratic at a tenth of the arcs' weight, which cannot hold
anything against springs saturated at 0.2. Stitching onto that shrunken interior curve then
packed the blocks into 13 overlapping pairs, worse than the stitch alone.

The floor now carries its own weight, `genomic_floor_weight`, default 1. Swept on the same
region, floor alone:

| weight | 3 to 5 kb | 100 to 177 kb | 562 kb to 1 Mb | within block exponent | block Rg median |
|---|---|---|---|---|---|
| off | 3.19 | 3.46 | 6.89 | 0.208 | 7.7 |
| 0.1 | 1.86 | 2.07 | 4.88 | 0.267 | 7.2 |
| 1 | 3.87 | 4.23 | 8.30 | 0.219 | 8.0 |
| 10 | 6.78 | 6.69 | 12.04 | 0.184 | 9.5 |

The floor sets the size of the ball and not its shape. Every bin moves together, collapse at
0.1, a fifth up at 1, double at 10, and the exponent stays between 0.18 and 0.27 at every
weight against a target of 0.285. The mid range pairs the floor was calibrated to lift do reach
their floors at weight 10, and drag every other pair with them.

Why. Inside a block there are two populations at every separation. Arc pairs sit at 0.55 to
0.94, arcless pairs at 2.8 to 6.7, and both carry an exponent near 0.12. The 4240 arcs of this
region target 0.218 at 10 to 30 kb and 0.363 at 300 kb to 1 Mb, a range of 0.20 to 0.40 in every
bin, because `freq_to_distance` maps PET count alone and 30 percent of arcs sit at its floor. A
1 Mb arc with 4 PETs targets 0.36 while the chain law says 90. Arcs are realised at 2.6 to 3.2
times their target in every bin, the equilibrium of a network of near equal links against a
repulsion. A floor on the pairs that are not links can inflate that network or let it
collapse, and cannot change its shape, because the shape is the network's. The first row of
the cause table, the arc target has no separation, is the one that sets the exponent, and A
never touched it.

What A is good for. It retires the unbounded `1/d` cleanly, it gives the block a size knob with
a physical meaning, and it is parity safe and tested in both kernels. It is not the fix for the
within block collapse.

Two caveats on the measurements. They ran the numba arcs kernel, since the measurement config
leaves `mc_executor_arcs` at `auto`, which resolves arcs to threaded numba, while production
sets `batch`; the JAX path is covered by the unit checks and initial energy agreement only, and
`out/floor_jax/run.sh` on the workstation runs it on this region at a given weight. And the
stitch on top of weight 0.1 gave 13 overlapping block pairs, so no floor weight has yet been
combined with the stitch.

For pairs with no arc, replace the constant `exclusion_radius_arcs` with a floor
`r0(s) = beta * (s / 1000)^nu`. Keep it a floor rather than a spring. Arcs must still be able to
pull distant loci together, since that is the biology the model exists to capture.

The exponent is fixed by the data. `nu = 0.285` from the contact probability curve, as a
setting so a cell line can carry its own value. Do not use `genomic_length_to_distance` as the
shape, since its exponent is 0.70 and would impose a chain 2.5 times more extended than the
data at every scale.

The prefactor is derived at runtime. `beta` is not a constant, it is 1.17 to 1.36 on the cell
line models and 0.70 to 0.76 on the trio models, but `beta` over the median consecutive anchor
distance is 0.447, 0.456, 0.447, 0.410 and 0.437 across those five models. So the rule is
`beta = factor * d_bond` with `factor = 0.44`, following the pattern the other excluded volume
levels already use. `d_bond` is the median consecutive anchor distance, which is measurable in
a pass that runs after the arcs MC has placed anchors and only approximately at arcs MC setup.

Calibrated that way the floor binds only between 44 kb and 780 kb, peaks at 1.69 times inflation
near 250 kb, and touches nothing beyond 1.4 Mb. It is a bounded correction, not an inflation.

Cost is free. The kernel already visits every pair for the repulsion. It also retires the
unbounded `1/d`, which is the known cause of the small interaction block blow ups.

### B. Stitch block edges together. Built, opt in

Implemented in `gnome3d/pipeline/stitch.py`, wired into `reconstruct.py::_assemble`, gated on
`[boundary_stitch] use_boundary_stitch`. Unit checks in `harness/test_stitch.py`.

Measured through the pipeline on the workstation, chr1:1-60 Mb at packing factor 0.75, flag
off and flag on from the same seed so the two structures differ only by the pass, then
`playground/boundary_wide.py` on each. The offline application of the pass to the laptop's
finished structure, `playground/stitch_offline.py`, gave the same answer to within 0.05 in every
bin.

| separation | jump, flag off | jump, flag on |
|---|---|---|
| 562 kb to 1 Mb | 58.9 | 0.98 |
| 1 to 1.78 Mb | 43.6 | 0.92 |
| 1.78 to 3.16 Mb | 34.5 | 0.86 |
| 3.16 to 5.62 Mb | 22.9 | 0.84 |
| 5.62 to 10 Mb | 16.3 | 1.06 |

Cross block exponent 0.183 to 0.750 against within block 0.645, and the whole structure's
exponent 1.39 to 0.71, so the curve is no longer bimodal. Cross block anchor pairs closer than
one bond are 1 in 100,000, so blocks do not interpenetrate. The flag off structure reproduces
the laptop's to 0.1 in Rg, 342.3 against 342.4, so the GPU and CPU paths agree on this region.

Two consequences to know. Rg of the region fell from 342 to 46. That is the pass doing its
job. It makes a boundary pair look like an interior pair, and interior pairs are collapsed by
the within block defect, so the collapse now spans the chromosome instead of stopping at each
block. Option A is what lifts it, and B without A produces a compact structure by design.

And the excluded volume regulariser was mis-scaled in the first build. Its radius derived
from `genomic_length_to_distance` of the centroid gaps, 153 units here, against a median block
Rg of 7.7, so 53 of 55 centroid pairs sat inside it and the term acted as a weak global
compaction penalty rather than an overlap guard. It did no harm on this structure but it did not
mean what it said. Replaced by a per pair radius `rg_k + rg_l` from each block's own radius of
gyration over all its beads, with `exclusion_radius_ib` kept as the explicit override.

With that radius the guard still leaves 8 of 55 block pairs inside touching on this structure,
5 of them chain neighbours. Raising `ev_weight` from 1 to 30 moves that to 7, with the worst
pair going from 0.18 to 0.46 of touching, while the jump stays between 1.0 and 1.24 and Rg
between 45 and 50. The overlaps are not a weight problem. A neighbouring pair is asked to put
its edge anchors about 7 units apart while each block is a ball of radius 7 to 25, so the
springs and the guard cannot both be satisfied until the blocks themselves are less collapsed,
which is option A. The two worst pairs both involve a 102 bead block whose radius of gyration
is inflated by a loose coil rather than by dense material. `ev_weight` stays at 1.

Add a bond between the last anchor of one block and the first anchor of the next, targeting the
distance a within block pair at that separation realises. About 68 boundaries on chr1, so one
spring each and no quadratic cost.

Those pairs are few but they set the global layout, and the jump they close is a factor of 33 to
62 at matched separation. This is the narrow version of what `use_segment_arcs` attempted and
lost on. That change widened the arcs MC to whole segments, which dragged every arcless pair into
the unbounded repulsion. Touching only the boundary pairs avoids that.

Placement needs a stage that sees both blocks. The natural shape is a short pass after anchors
are placed, adjusting each block rigidly to satisfy its edge bonds while leaving the intra block
arrangement the arcs MC produced untouched. That is the same pass A wants for measuring
`d_bond`, so the two share a stage.

### D. Separation aware arc targets. Superseded by H, removed 2026-09-06

Implemented as `util.arc_target_with_separation`, `Settings.arc_expected_distance` and
`build.arc_expected_matrix`, gated on `[distance] use_separation_arc_target`, multiplicative above
a 10 kb pivot. Unit checks in `harness/test_arc_target.py`.

Measured alone on chr1:1-60 Mb, same seed and executor as every other arm:

| arcs by span | 10 to 30 kb | 30 to 100 kb | 100 to 300 kb | 300 kb to 1 Mb |
|---|---|---|---|---|
| target, parity law | 0.22 | 0.23 | 0.27 | 0.36 |
| target, D | 0.27 | 0.39 | 0.65 | 1.00 |
| realised, parity | 0.67 | 0.68 | 0.82 | 1.00 |
| realised, D | 0.56 | 0.86 | 1.52 | 2.50 |
| realised over target, parity | 2.60 | 2.55 | 2.88 | 3.16 |
| realised over target, D | 1.85 | 1.99 | 2.35 | 2.62 |

The arc network gains the separation gradient it lacked. Realised arc distance runs 0.56 to
2.50 across 10 kb to 1 Mb where the parity law gave 0.67 to 1.00, arcs sit closer to their
targets in every bin, and the sub pivot arcs tighten rather than loosen. The all pairs within
block exponent moves 0.208 to 0.241 and the contact probability slope 0.41 to 0.50, less than
the network's own change, because the arcless majority still sits at the flat `1/d` equilibrium
and is inflated uniformly, 3.19 to 5.75 at 3 to 5 kb and 6.89 to 13.18 at 0.5 to 1 Mb.

With the stitch, D is the first arm that moves every gate the right way at once:

| arm | jump at 562 kb to 1 Mb | Rg | overlapping block pairs | within block exponent | P(s) slope |
|---|---|---|---|---|---|
| stitch alone | 0.98 | 46 | 8 | 0.207 | 0.41 |
| D and stitch | 0.95 | 56 | 5 | 0.240 | 0.50 |

Rg rises because blocks are larger, block Rg median 7.7 to 10.4, and fewer of them overlap when
stitched. The cost is a corona. Under D, 342 anchors sit more than 50 units from their block
centroid where the parity law leaves none, and the largest block's radius of gyration goes 24
to 41. Those anchors are not dangling, their arcs are intra block and realised at 1.3 times
target. They are whole sub networks. The two largest blocks hold 66 and 48 arc graph connected
components, and the components outside the largest sit about 40 units from the block centroid
under the parity law and about 67 under D. Nothing in the arcs MC joins one component to
another except the `1/d` repulsion, which is cut off beyond three times the mean arc target,
so their radius is not set by any term that knows the genome. The arcs step size is a constant,
so D does not change the anneal's reach. What sets that radius is the arcs level confinement,
measured on the largest block, 1227 anchors, annealed offline four ways from the same seed:

| law | arcs confinement | Rg | anchors beyond 50 | max radius |
|---|---|---|---|---|
| parity | on | 26.0 | 0 | 49 |
| D | on | 44.0 | 157 | 78 |
| parity | off | 32.7 | 139 | 63 |
| D | off | 52.5 | 410 | 93 |

The islands are held by nothing but the confinement leash. Its radius is
`1.5 * mean arc target * N^(1/3)`, 4.6 units under the parity law and 10.7 under D because D
lifts the mean target from 0.29 to 0.67, and at weight 0.1 it is a leash rather than a wall,
so the islands float where the leash balances the anneal's diffusion, 40 units out under the
parity law and 67 under D. Turn the leash off and the parity law grows the same corona. The
corona is not something D created, it is the island problem D made visible by scaling it.

That corona is where option C, chain bonds between consecutive anchors in the arcs MC, acts: it
is exactly the term that ties a component to its genomic neighbours.

A on top of D was tried on the same region, floor at weight 1, on the reading that A had failed
alone only because the network it pushed against was flat. It inflates every bin by 1.3 to 1.4
over D alone and the exponent falls, 0.241 to 0.225. The floor is a size knob in every
combination and is not a shape lever. It stays as the clean replacement for the unbounded `1/d`.

The measurement of A points here. The arc target law is the one term that sets the within
block shape and it is blind to separation. Options, none built. Give `freq_to_distance` a
separation term so a long arc with few PETs targets more than a short one with the same count.
Or lift the target with a floor of its own, `max(freq_to_distance(PET), c * (s / 1000)^nu)`,
which keeps strong short arcs tight and stops weak long arcs from pinning a block into a ball.
Either is a change to the parity era law and must be opt in. Calibration would follow the same
route as A, the contact probability curve for the exponent and a bond scale for the prefactor,
and the gate is the same within block exponent that A failed.

### E. Excluded volume across blocks after the stitch. Built, opt in, under measurement

Implemented as `gnome3d/pipeline/relax.py`, the smooth kernel over the whole chromosome with
excluded volume at 1.5 bonds, bond springs at weight 10, temperature 0.1 of the smooth
maximum, anchors fixed, gated on `[relax] use_cross_block_relax`, run after the stitch. On a
toy of two overlapping coils it takes 564 cross block contacts to zero with bonds kept within
1.44. Unit checks in `harness/test_relax.py`.

Applied offline to the D, C and stitch structure, `playground/relax_offline.py`:

| | before | after |
|---|---|---|
| cross block bead pairs within a bond | 21,641 | 605 |
| beads touched | 11.9 percent | 2.0 percent |
| straight strands | 11 | 0 |
| boundary realised over target, median and worst | 1.33 and 2.64 | 1.33 and 2.64 |
| within block exponent, Hi-C SCC, Pearson, MultiMM | 0.250, 0.091, 0.196, 0.293 | unchanged |
| Rg | 57.7 | 59.0 |

Anchors do not move, so nothing the earlier passes set is touched, and the last straight
strands go because the coils around them are free to route. Of the 605 pairs left, 520 sit
between one adjacent block pair and 599 are grazing at half a bond to a bond. That pair is the
boundary with a 33 kb gap, whose edge anchors are pinned about four units apart, so the coils
around them cannot clear each other while the anchors stay put. Zero there needs either the
stitch target to respect coil size at short gaps or the anchors to move a little, and that
choice is open. Everywhere else the globules no longer touch.

The cost rules it out at genome scale as it stands, 19,788 seconds for 42,480 beads on the
laptop's numba path, and profiling says exactly why. An MC step is two local score
evaluations. The chain term looks at a bead's two neighbours and costs a constant 0.4
microseconds. The excluded volume term scans every bead in the structure:

| beads | chain | excluded volume | its share of the step | beads inside the radius | scanned over useful |
|---|---|---|---|---|---|
| 2,000 | 0.36 us | 2.68 us | 88 percent | 53 | 38 |
| 10,000 | 0.38 us | 12.91 us | 97 percent | 52 | 191 |
| 42,480 | 0.52 us | 52.87 us | 99 percent | 57 | 749 |

The scan is linear in the structure, 1.29 microseconds per thousand beads with a spread of
0.04, while the number of beads actually inside the radius stays near 55 at every size because
that is a local density. At full size the scan therefore does 749 times more work than it
needs, and a step costs 107 microseconds, which puts the run at about 185 million steps, some
4,362 moves per bead. The step count is reasonable. The per step cost is not.

A cell list, a uniform grid at the excluded volume radius rebuilt every few thousand moves,
makes the term visit the beads that are actually near instead of all of them. That takes the
step from 107 microseconds to under two, about seventy times, so this region relaxes in
minutes rather than hours and a whole chromosome becomes possible at all.

It is not only about this pass. `exclusion_apply_to_smooth` is on in the production
configuration, so the per block smooth stage runs the same scan on blocks of up to 16,384
beads, and that stage is most of the pipeline's wall time. The same neighbour list speeds up
every reconstruction, not just the relaxation. The JAX kernels have the same shape of work
vectorised; there the equivalent is a fixed size neighbour array rebuilt on the same schedule
rather than a cell list.

Once blocks are stitched, their coils interpenetrate because no term acts between beads of
different blocks. A relaxation pass over the whole chromosome with the smooth stage's excluded
volume on every pair, anchors held fixed so the arcs and the stitch are kept and only the
subanchor coils re route, is the direct fix. The smooth kernels exist; the cost is the excluded
volume scan over the whole chromosome per move, which needs a neighbour list at genome scale.
Making the stitch itself bead aware would only move rigid blocks and cannot untangle coils that
have to touch at their edges.

### F. Subanchors across block boundaries. Not built

Densify runs per block, so the gap between the last anchor of one block and the first of the
next holds no subanchor at all. On chr1:1-60 Mb every one of the ten boundaries spans 28 kb to
1.46 Mb with zero beads between its two anchors, one boundary has a single bead. The strand
drawn across a boundary is therefore one straight bond whatever the stitch does, and it is what
a stitch gap looks like. The stitch itself is satisfied to 1.0 to 1.6 times its target on nine
boundaries and misses one, the 1.46 Mb gap, at 2.6 times, which the rigid optimiser could not
close against the excluded volume of the two large blocks on either side. Inserting subanchors
across boundaries at the same density the blocks use, and letting the relaxation of option E
route them, is the fix. It also gives the chain a genomic length across the gap instead of a
bond of zero length, which the stitch target currently stands in for.

### G. One distance law for arc targets and chain bonds. Superseded by H, removed 2026-09-06

The root cause of the within block collapse, found 2026-09-05.

The arcs stage's target matrix carries two families of positive entry and they are built by
different laws. An arc's target is `freq_to_distance(PET)`, times a span factor when D is on,
and lands between 0.2 and 1.6 model units over five kb to one Mb. A consecutive arcless anchor
pair's target, which C adds, is `arcs_chain_bond_scale` times the chain law and lands between
4.0 and 135 over the same range.

| separation | an arc asks for | the chain asks for | ratio |
|---|---|---|---|
| 5 kb | 0.44 at 2 PET to 0.20 at 50 | 4.01 | 9 to 20 |
| 20 kb | 0.54 to 0.24 | 8.59 | 16 to 35 |
| 200 kb | 1.04 to 0.47 | 41.4 | 40 to 88 |
| 1 Mb | 1.65 to 0.74 | 134.9 | 82 to 182 |

The chain law's own exponent is 0.671, far steeper than the 0.285 the contact probability curves
give. The arc law's is 0.264, close to it by construction since D sets it, but its absolute
scale is a tenth to a hundredth of the chain's. An arc therefore asks two beads to sit at a
fraction of one bead's own size, which nothing downstream can undo, because the smooth stage
holds every anchor fixed.

What that produces, measured on a finished trio chr1 by matching each loop in the bedpe to its
anchor pair. 93 percent of arc joined anchor pairs end up closer than a bead's size. Those are
24 percent of the chromosome's 13,177 overlapping anchor pairs, a 5.8 times enrichment over the
4.13 percent base rate. The other 76 percent have no arc between them and are squeezed together
as the arc network collapses the block.

Under the unified law a pair sits at the chain law distance for its separation, and its PET
count only says how far in to pull from there, between `arc_target_pull` and 1. The PET law
supplies that factor rather than the distance, normalised by its own limits, which it has: it
runs from `freq_to_distance(0)` down to `count_dist_base_level`. A zero PET arc sits on the
background, a saturated one at the pull, and an arc's target now grows with separation exactly
as the chain's does. One site, `Settings.arc_expected_distance`. Keys under `[distance]`:
`use_unified_arc_target`, `arc_target_pull` (0.45). It supersedes D, which carries the same
separation dependence in a form that cannot fix the scale.

Measured offline on eight real blocks of a trio chr1, each solved from a common start, against
the production matrix rebuilt from the same anchors and the same bedpe. The one thing not
reproduced is the anchor heatmap scaling, which shrinks arc targets further, so the production
arm here is a conservative version of the real one.

| arm | overlaps per thousand anchors | exponent | against 0.285 | arc over background |
|---|---|---|---|---|
| production | 1,619 | 0.067 | 0.24 | 0.44 |
| floor at 1.5 bead sizes | 37 | 0.069 | 0.24 | 0.52 |
| unified, pull 0.9 | 0 | 0.309 | 1.08 | 0.63 |
| unified, pull 0.45 | 0 | 0.288 | 1.01 | 0.63 |
| unified, pull 0.3 | 12 | 0.272 | 0.95 | 0.62 |

Three things that table says. The overlap and the flat curve are one defect, not two, because
flooring the old targets at a bead's size removes the overlaps and leaves the exponent where it
was. Arc joined pairs still sit at 0.63 of what an unjoined pair holds at the same separation,
so the data is still doing work. And the pull barely matters between 0.9 and 0.45, which says
most of the gain is from putting arcs on the background at all rather than from how far the PET
count pulls them in.

Measured end to end on chr1:1-60 Mb, five structures per arm, against the production arm
generated the same night.

| arm | Pearson | Spearman | SCC | MultiMM | exponent | Rg | wb-aa | wb-sa | xb |
|---|---|---|---|---|---|---|---|---|---|
| production | 0.405 | 0.321 | 0.109 | 0.331 | 0.249 | 32.9 | 89.2 | 3376 | 2680 |
| unified, chain background | 0.499 | 0.206 | 0.082 | 0.256 | 0.688 | 147.0 | 0.9 | 474 | 16 |

The overlap fix carries all the way through. Anchor overlaps fall by 99 percent, and the two
other overlap columns fall with them, which says the collateral overlaps were collateral.

The cost is expansion. Rg goes up four and a half times and the distance exponent to 0.688
against a 0.285 target. Three of the four Hi-C measures fall. Pearson rises, but full matrix
Pearson is dominated by the decay trend and a steeper decay inflates it, so it should not be
read alone against three that fall.

The expansion is the chain law's own exponent of 0.671 showing through, since the unified law
rides it. `arc_target_background_exponent` puts the background on a chosen slope instead,
anchored at the separation the chain law is calibrated for.

| background slope | Pearson | Spearman | SCC | MultiMM | exponent | Rg | wb-aa | wb-sa | xb |
|---|---|---|---|---|---|---|---|---|---|
| production, no unified law | 0.405 | 0.321 | 0.109 | 0.331 | 0.249 | 32.9 | 89.2 | 3376 | 2680 |
| 0.10 | 0.436 | 0.323 | 0.115 | 0.339 | 0.276 | 35.6 | 28.1 | 3011 | 1595 |
| 0.15 | 0.455 | 0.324 | 0.112 | 0.336 | 0.305 | 39.0 | 16.1 | 2607 | 1098 |
| 0.20 | 0.479 | 0.320 | 0.110 | 0.334 | 0.345 | 46.9 | 9.1 | 2255 | 640 |
| 0.285 | 0.498 | 0.296 | 0.101 | 0.318 | 0.453 | 53.6 | 6.4 | 1690 | 284 |
| the chain law, 0.671 | 0.499 | 0.206 | 0.082 | 0.256 | 0.688 | 147.0 | 0.9 | 474 | 16 |

Every slope from 0.10 to 0.20 beats production on all four Hi-C measures and cuts the overlaps
at the same time, so the law is not a trade against contact fidelity in that range. It becomes
one above 0.20, where the overlap count keeps falling and Spearman, SCC and MultiMM start
paying.

Within 0.10 to 0.20 the Hi-C measures barely move, SCC 0.115 to 0.110 and MultiMM 0.339 to
0.334, while the anchor overlaps fall threefold, 28.1 to 9.1. So the choice inside that range
costs almost nothing on Hi-C and buys a lot on overlap. 0.15 is the balanced pick. It has the
best Spearman of any arm including production, SCC and MultiMM above production, the exponent
at 0.305 against a 0.285 target where production sits at 0.249, overlaps down 5.5 times, and it
sits in the middle of the flat region rather than on its edge.

The exponent crosses the target between 0.10 and 0.15, at 0.276 and 0.305, and is close to
linear in the slope.

The slope transfers. H1ESC, same region, same five structures per arm, read from a raw contact
map since none of that file's thirteen resolutions carries balancing weights, so these numbers
stand against each other and not against the balanced table above.

| arm | Pearson | Spearman | SCC | MultiMM | exponent | Rg | wb-aa | wb-sa | xb |
|---|---|---|---|---|---|---|---|---|---|
| production | 0.160 | 0.120 | 0.025 | 0.072 | 0.163 | 36.5 | 121.6 | 3662 | 4719 |
| 0.15 | 0.183 | 0.134 | 0.034 | 0.075 | 0.226 | 40.4 | 35.5 | 3104 | 2815 |

Every measure improves, Pearson by 14 percent, Spearman by 12, SCC by 36, MultiMM by 4, and the
anchor overlaps fall 3.4 times. So one slope is a strict improvement on two cell lines that
differ in assay and in depth.

The optimum is not the same for both. At 0.15 GM12878 overshoots the exponent at 0.305 and
H1ESC undershoots at 0.226, so a per dataset calibration would do better than one constant. It
is not needed to adopt the law, since 0.15 improves both and harms neither.

What is still open. The trio, whose data lives on the cluster and cannot be measured on the
workstation. And whether to make the law default on, which turns on for all fifteen ensemble
configs at once, the nine trio ones included, and those are the unmeasured half.

### H. One distance law, in bead units, with its exponent measured from the input. Adopted 2026-09-06, the law

G fixed the arcs against the chain bonds and left the structure of the problem in place. The
settings reference made that structure visible, 2026-09-06. Fifteen free constants for one
physical relation. Eleven copied from the reference authors' GM12878 config with no derivation
and byte identical across every cell line. Three laws setting one quantity on three unrelated
absolute scales in a unit with no meaning, which is what let a hundredfold disagreement exist.
And the one derived number, 0.285, a mean over three of our own Hi-C files typed in as a
default that then applied to every input.

Fitted on the inputs this project runs, that exponent is 0.275 for GM12878, 0.299 for H1ESC,
0.192 for HFFC6 and 0.072 for a trio sample. The trio's production structures came out at
0.067. They tracked their own data the whole time; the constant was what did not fit. The
ChIA-PET singletons files are not a decay at all, slopes -0.23, +0.20 and -0.08, since they are
enrichment filtered around CTCF, so a fit has to be able to refuse.

The law. One bead is the distance at `target_bp_per_subanchor`, and everything is in beads.
No contact, `max(1, (s / s0) ^ nu)`. A loop of strength `q`, `1 + (background - 1) / (1 + q /
q_half)`, touching at saturation and never inside, `q` being the PET count over the typical
count at that span fitted on the run's own arcs. A heatmap cell, the background times observed
over expected to the minus third, the expectation taken within the heatmap at that separation.
`nu` is read off the singletons at load and the run says what it measured, or why it could not
and that it fell back to 0.285. Three keys replace eighteen. Flag off is byte exact.

What it retires if it holds: A, D and G, and the fifteen constants. What it does not touch: B,
C and E, which act on block placement and on subanchors and are unchanged.

Measured 2026-09-06, chr1:1-60 Mb, five structures per arm, contact maps read raw so the three
cells sit on one footing. Raw numbers compare the two arms of one cell with each other and not
with the balanced tables above. Overlap columns are each structure on its own radii.

| cell | arm | Pearson | Spearman | SCC | MultiMM | exponent | measured nu | wb-aa | wb-sa | xb |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | production | 0.433 | 0.324 | 0.202 | 0.353 | 0.249 | | 90.2 | 3424 | 2734 |
| GM12878 | polymer | 0.487 | 0.342 | 0.215 | 0.335 | 0.388 | 0.272 | 1.6 | 1478 | 286 |
| H1ESC | production | 0.161 | 0.118 | 0.027 | 0.072 | 0.163 | | 122.2 | 3675 | 4731 |
| H1ESC | polymer | 0.251 | 0.173 | 0.037 | 0.073 | 0.385 | 0.299 | 4.6 | 1635 | 504 |
| HFFC6 | production | 0.239 | 0.166 | 0.096 | 0.211 | 0.165 | | 102.3 | 2774 | 3004 |
| HFFC6 | polymer | 0.275 | 0.178 | 0.123 | 0.208 | 0.293 | 0.197 | 3.7 | 1655 | 794 |

On every cell Pearson, Spearman and SCC rise and MultiMM is level, the worst change being
0.018 down on GM12878. Anchor overlaps fall 96 to 98 percent, subanchor overlaps halve, cross
block overlaps fall four to ten times. HFFC6, whose own exponent is furthest from the old
constant, gains the most on SCC, 0.096 to 0.123.

Two things learned in the measuring. The battery had pinned its overlap radii from the first
arm, which is right within one model unit and wrong across two: the polymer law's bead is 0.62
of the old chain law's bond, so the polymer arm's subanchor overlaps were scored at its whole
bond and reported as rising when they had halved. Each structure now scores on its own radii.
And the realised exponent runs 1.3 to 1.5 times the input on every cell, 0.388 against 0.272,
0.385 against 0.299, 0.293 against 0.197, so HFFC6 lands on 0.285 by coincidence. The
arcs, the confinement and the excluded volume steepen what the background asks for, by a
factor that looks the same across cells, which a single correction on the background exponent
would absorb. Not built. It does not gate adoption, since every Hi-C measure already holds
without it.

The residual, what it turned out to be, and what did not fix it. The realised exponent ran
1.3 to 1.5 times the measured input on every cell. Fitted by band it is a kink, not a slope:
0.09 to 0.20 under 100 kb, 0.34 to 0.48 above, against inputs of 0.20 to 0.30. Loops pull
pairs to touching and nothing holds the arcless pairs between them, and the steep long range is
the recovery. A global correction on the exponent would rotate the whole line.

The other half of tier C was tried: every arcless pair held at the background by a spring
symmetric in log distance, in place of the `1/d` repulsion. On eight real GM12878 blocks with
the solver, measured input 0.272:

| weight | chain bonds | 20 to 100 kb | 100 kb to 1 Mb | overlaps per thousand | arcs realised over target |
|---|---|---|---|---|---|
| 0.01 | off | 0.203 | 0.422 | 101 | 1.03 |
| 0.30 | off | 0.287 | 0.378 | 27 | 1.25 |
| 1.00 | off | 0.312 | 0.355 | 23 | 1.38 |
| 0.30 | on | 0.284 | 0.365 | 587 | 0.98 |

Then the three cell battery, five structures each, against the polymer arms of the same day:

| cell | arm | Pearson | Spearman | SCC | MultiMM | 20 to 100 kb | 100 kb to 1 Mb | Rg | wb-aa | xb |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | polymer | 0.462 | 0.316 | 0.191 | 0.335 | -0.228 | 0.431 | 31.5 | 1.6 | 286 |
| GM12878 | background springs | 0.329 | 0.231 | 0.130 | 0.278 | 0.248 | 0.200 | 16.8 | 67.7 | 3800 |
| H1ESC | polymer | 0.258 | 0.181 | 0.033 | 0.073 | -0.008 | 0.442 | 33.6 | 4.6 | 504 |
| H1ESC | background springs | 0.119 | 0.079 | 0.020 | 0.041 | 0.110 | 0.164 | 18.8 | 47.4 | 13445 |
| HFFC6 | polymer | 0.267 | 0.175 | 0.113 | 0.208 | 0.051 | 0.343 | 26.2 | 3.7 | 794 |
| HFFC6 | background springs | 0.181 | 0.133 | 0.070 | 0.173 | 0.104 | 0.135 | 14.1 | 132.5 | 8773 |

A loss on every cell and every statistic. The kink is gone and both bands agree, but at 0.20
against an input of 0.27, with Rg halved and overlaps up. The reason is embedding. A distance
matrix that asks every pair for `s^nu` with `nu` below a third cannot be realised in three
dimensions, so an all pairs spring network settles into the least squares compromise, a mean
field blob. The `1/d` repulsion never constrained an arcless pair and so never met this. The
sweep could not see it because every isolated block came out at Rg 3.8 whatever the weight,
which was the sign and was read as stability. Reverted the same day.

What that leaves. The kink is real and its cause is known. A fix has to hold only the pairs
that collapse, under about 100 kb, and leave the long range free to be set by the arcs and the
polymer scale. That is a short range background, not an all pairs one, and it has not been
tried.

### C. Chain bonds between consecutive anchors in the arcs MC. Built, opt in, under measurement

The most direct statement of the missing constraint. It was left last because it competes with
the arc springs, and under D the arcs target sensible distances so that competition is the right
one. Implemented as `build.add_chain_bonds`, consecutive arcless pairs entered into the arcs
target matrix at `genomic_length_to_distance` of their gap after the heatmap scaling, arc pairs
kept, gated on `[springs] use_arcs_chain_bonds`. Same spring constants as the arcs; a separate
weight would need a per pair weight in the kernels. Unit checks in `harness/test_arc_target.py`.
Measured offline on the largest block, 1227 anchors, 1188 bonds, median bond target 1.27,
under D with and without C:

| | D | D and C |
|---|---|---|
| consecutive pairs more than 3 times the chain law apart | 749 of 1226 | 149 |
| more than 10 times | 274 | 0 |
| more than 20 units apart | 267 | 0 |
| 90th percentile of realised over chain law | 54 | 3.1 |
| median consecutive distance | 5.82 | 2.93 |
| core Rg | 32.2 | 29.3 |
| island centroid distance from core, median and 90th | 74 and 88 | 38 and 77 |

C removes the spokes. No consecutive pair is left more than ten times the chain law apart,
where D alone had 274, so no subanchor strand is drawn taut. The small islands come home and the
large ones stay at the block's edge, now reached by satisfied chains rather than lines. Arcs are
as satisfied as under D alone, realised over target 2.28 in both.

On the region, D with C alone, same seed and executor as every other arm:

| | off | D | D and C |
|---|---|---|---|
| straight subanchor strands, chord over contour above 0.9 | 35 | 57 | 12 |
| median 3D length of those strands | 41 | 68 | 4.9 |
| consecutive pairs more than 10 times the chain law apart | | 574 | 0 |
| median consecutive arcless distance | 3.18 | 5.35 | 2.73 |
| within block, 3 to 5 kb | 3.19 | 5.75 | 3.73 |
| within block, 562 kb to 1 Mb | 6.89 | 13.18 | 11.34 |
| within block exponent, 20 kb to 1 Mb | 0.208 | 0.241 | 0.321 |
| simulated contact probability slope | 0.41 | 0.50 | 1.03 |
| arcs realised over target, 10 to 30 kb and 300 kb to 1 Mb | 2.60 and 3.16 | 1.85 and 2.62 | 1.80 and 2.92 |

The spokes are gone, the twelve strands left are five units long. The chain bonds pull the
short range back toward the parity values while the long range keeps most of D's lift, so the
curve steepens past the target on both measures, 0.321 against 0.285 and 1.03 against 0.855,
where D alone fell short on both. That overshoot is a weight or a target law to tune. The
undershoot with spokes was a missing term.

`arcs_chain_bond_scale` multiplies the bond target. Swept on the largest block under D:

| scale | block exponent | 3 to 10 kb | 562 kb to 1 Mb | consecutive pairs beyond 3 and 10 times the chain law | median consecutive |
|---|---|---|---|---|---|
| D alone | 0.144 | 6.19 | 12.92 | 749 and 274 | 5.87 |
| 1 | 0.223 | 3.70 | 10.93 | 149 and 0 | 2.79 |
| 1.5 | 0.183 | 4.56 | 11.40 | 460 and 2 | 3.78 |
| 2 | 0.165 | 5.08 | 11.88 | 602 and 3 | 4.39 |
| 3 | 0.144 | 5.88 | 12.39 | 694 and 29 | 5.21 |

The block's own exponent maps onto the region's, D alone 0.144 against 0.241 and scale 1
0.223 against 0.321, so the region target of 0.285 sits near 0.187 on the block. Scale 1.5 lands
there with two pairs left beyond ten times the chain law; at 3 the exponent is back at D's and
the spokes return. 1.5 is the value carried into the production run.

With the stitch, at scale 1 on the region:

| arm | straight strands | jump at 562 kb to 1 Mb | Rg | overlapping block pairs | block Rg max | within block exponent |
|---|---|---|---|---|---|---|
| stitch alone | 34 | 0.98 | 46 | 8 | 24.5 | 0.207 |
| D and stitch | 56 | 0.95 | 56 | 5 | 41.3 | 0.240 |
| D, C and stitch | 11 | 0.76 | 58 | 7 | 39.4 | 0.320 |

The spokes go from 56 to 11 and Rg holds. Two small regressions come with it, overlapping pairs
5 to 7 and the jump 0.95 to 0.76, the stitch now pulling boundary pairs a little inside the
interior curve, and the exponent is the scale 1 overshoot.

Scale 1.5 on the production base, Hi-C singletons and the JAX arcs kernel, with D, C and the
stitch, on the same region:

| | D, C and stitch, ChIA-PET base, scale 1 | tuned, production base, scale 1.5 |
|---|---|---|
| boundary realised over target, median and worst | 1.33 and 2.64 | 1.04 and 1.20 |
| straight strands | 11 | 12 |
| overlapping block pairs | 7 | 9 |
| cross block bead contacts, beads touched | 21,641, 11.9 percent | 27,900, 16.4 percent |
| Rg | 58 | 33 |
| contact probability slope | 1.03 | 0.71 |
| Hi-C SCC, Pearson, MultiMM Pearson | 0.091, 0.196, 0.293 | 0.085, 0.183, 0.310 |

Scale 1.5 closes every boundary to within 20 percent, where scale 1 over pulled. The
production base is more compact and more interpenetrated, 16.4 percent of beads touching
another block, which is the case for option E.

Against the production base's own parity arm, every flag off at packing factor 0.75, same
region and seed:

| production base | parity | tuned |
|---|---|---|
| boundary realised over target, median and worst | 90 and 472 | 1.04 and 1.20 |
| straight strands | 4 | 12 |
| cross block bead contacts, beads touched | 4,786, 3.3 percent | 27,900, 16.4 percent |
| Rg | 348 | 33 |
| within block exponent | 0.100 | 0.161 |
| Hi-C SCC, Pearson, MultiMM Pearson | 0.025, 0.062, 0.222 | 0.085, 0.183, 0.310 |

The production base starts worse than the ChIA-PET base, a boundary jump of 90 against 44 and
an exponent of 0.10 against 0.16, and the tuned configuration lifts it further in relative
terms: the exponent by 1.6 times, SCC by 3.4 times, Pearson by 3 times, the MultiMM Pearson by
1.4 times. The eight extra straight strands and the fivefold rise in cross block contacts are
the same two costs as on the ChIA-PET base, and E is the answer to both.

## Validation

Against the cell line's own Hi-C, GM12878 4DN mcool at 25 kb on chr1:1-60 Mb, anchors of one
structure, contact radius 5 model units, with the validation package's metrics:

| arm | SCC | Pearson | insulation | MultiMM Pearson | contact probability exponent | distance exponent |
|---|---|---|---|---|---|---|
| parity | 0.070 | 0.160 | 0.240 | 0.239 | -0.33 | 0.23 |
| stitch | 0.070 | 0.159 | 0.244 | 0.242 | -0.50 | 0.13 |
| D and stitch | 0.087 | 0.187 | 0.264 | 0.242 | -0.55 | 0.12 |
| D, C and stitch | 0.091 | 0.196 | 0.284 | 0.293 | -0.40 | 0.20 |
| D alone | 0.087 | 0.187 | 0.264 | 0.245 | -0.42 | 0.20 |
| D and C alone | 0.092 | 0.198 | 0.286 | 0.298 | -0.34 | 0.30 |

The ordering matches the geometry. The stitch moves blocks and leaves Hi-C agreement where it
was, D lifts every Hi-C measure, and C lifts it again, most on the MultiMM Pearson, 0.24 to
0.29. These are single structure numbers at 25 kb, so the magnitudes are low and the ordering
is the result. The distance exponent of the stitched arms mixes cross block pairs into an
anchors only fit and is not the within block number above. The contact probability exponent at
this radius stays far from the Hi-C value of -0.86 in every arm.

Self collision. After the stitch nothing acts between the beads of different blocks, the
smooth stage's excluded volume being per block and the stitch guarding centroids only, so two
coils can pass through each other. Beads with a bead of another block within one bond length,
on all 42480 beads with a spatial index:

| arm | cross block bead pairs within a bond | beads touched |
|---|---|---|
| parity | 5,765 | 3.3 percent |
| stitch | 14,037 | 9.6 percent |
| D and stitch | 6,469 | 8.0 percent |
| D, C and stitch | 21,640 | 11.9 percent |

The arm that agrees best with Hi-C is the most interpenetrated, four times the parity value,
and at genome scale the same mechanism lets chromosomes pass through each other. This is a gate
from here on, and the fix is a term, not a weight: excluded volume across blocks after the
stitch, option E.

| gate | tool | at 0.15 | at 0.75 |
|---|---|---|---|
| cross block exponent, finished | `playground/boundary_wide.py` | 0.017 | 0.183 |
| within block exponent, 20 kb to 1 Mb | `playground/calibrate_beta.py` | 0.205 | 0.208 |
| simulated contact probability slope | `playground/calibrate_beta.py` | 0.40 | 0.41 |
| boundary jump at 562 kb to 1 Mb | `playground/boundary_wide.py` | 33 | 59, with stitch 0.98 |
| block layout exponent, coarse | `playground/ib_confine_ablate.py` | 0.021 | 0.214 |
| enhancer promoter expression | `enhancer3d/playground/beyond_linear.py` | 86 percent of v4 | not yet run |
| byte exact parity, flag off | see AGENTS.md parity gate | must stay identical | |

Targets are 0.285 for the two exponents and 0.86 for the contact probability slope, from the
Hi-C. The jump should approach 1.

## Decisions worth making before building

- A and B are divergences from the reference and from cudaMMC. They must be opt in, default
  off, and recorded in the AGENTS.md divergences section. The parity baseline does not move.
- The packing factor change is configuration, not code, and moves no parity baseline. Whether it
  should be 0.75 or 1.0 is decided by segments with many blocks, not by the 60 Mb region.
- Better geometry may not improve the expression statistic. If that happens it should be
  reported as evidence the statistic is limited by annotation, not used as a reason to tune
  `beta` or `nu` until it moves.
- The existing ensembles and the running trio array carry 0.15. Regenerating is expensive and is
  a decision to take on the finished structure evidence above, not on the general worry.

## Order

1. Contact probability curve and `beta`. Done.
2. Widen the boundary measurement. Done, and it split the problem into three.
3. Packing factor. Changed to 0.75, confirmed end to end. Decide 0.75 against 1.0 on a many
   block segment.
4. B, the boundary stitch. Built and measured through the pipeline, jump 58.9 to 0.98.
5. A, the genomic floor. Built and measured. It sets block size, not shape, and stays as a
   clean replacement for the unbounded `1/d`.
6. D, the separation aware arc target. Built. Measured on chr1:1-60 Mb, target alone and with
   the stitch, against the flag off structure from the same seed. The gates are the within
   block exponent, arcs realised over target per span bin, the simulated contact probability
   slope, and with the stitch, Rg and the block overlap count.
7. Decide, on those numbers, whether the existing ensembles and the running trio array are
   regenerated with the working flags on.
8. Only then consider C.
9. G, one distance law for both families. Built 2026-09-05 after the two laws were found to
   disagree by one to two orders of magnitude. Offline it takes the overlap rate to zero and the
   exponent to 0.288 against a 0.285 target. The gate is the whole region battery against the
   production arm, Hi-C correlation first, then the exponent, then the overlap columns. If it
   holds it supersedes D and makes A unnecessary, since a floor is only needed while the targets
   are below a bead's size.
