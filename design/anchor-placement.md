# Anchor placement

Reconstructions come out as compact balls of anchors scattered in space and joined by thin
strands. This records what causes that, what has been measured, what has been ruled out, and
which changes are worth making.

Status. Diagnosis complete and quantified on finished structures. One cause was a configuration
value and has been changed. Options A and B are built and opt in. A is measured and does not fix the within block shape. The next lever is D, the arc target law.

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

### A. Excluded volume floor that grows with genomic separation. Built, opt in

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

### D. Separation aware arc targets. Built, opt in, not yet measured

Implemented as `util.arc_target_with_separation`, `Settings.arc_expected_distance` and
`build.arc_expected_matrix`, gated on `[distance] use_separation_arc_target`, multiplicative above
a 10 kb pivot. Unit checks in `harness/test_arc_target.py`.

The measurement of A points here. The arc target law is the one term that sets the within
block shape and it is blind to separation. Options, none built. Give `freq_to_distance` a
separation term so a long arc with few PETs targets more than a short one with the same count.
Or lift the target with a floor of its own, `max(freq_to_distance(PET), c * (s / 1000)^nu)`,
which keeps strong short arcs tight and stops weak long arcs from pinning a block into a ball.
Either is a change to the parity era law and must be opt in. Calibration would follow the same
route as A, the contact probability curve for the exponent and a bond scale for the prefactor,
and the gate is the same within block exponent that A failed.

### C. Chain bonds between consecutive anchors in the arcs MC

The most direct statement of the missing constraint. Left last because it competes with the arc
springs and needs a weight balance that A and B do not.

## Validation

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
