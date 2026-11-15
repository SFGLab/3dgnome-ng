# Gap fill

A gap is the stretch between two consecutive anchors of a block, and a hole is the stretch between
two blocks. Inside a gap the run has bond springs, excluded volume and the start, and no data. A
hole has no beads at all. Holes are 13 percent of chr1, mostly B compartment, and touch 27 of
23,575 loops. This note collects the options for both and records the trials.

## Findings so far

Two trials on 2026-11-13 aimed every conformation at the law's mean distance, a spring on bead
pairs in the smooth stage and a start that matched the law bead by bead. Both lost Pearson and
MultiMM on every region, recorded in the AGENTS.md divergences. The smooth stage holds bonds and
nothing inside a gap, and a gap ended at the size its start gave it, 1.2 times the law under
`start = law`. A contact map is an average over cells, and contacts come from the spread between
conformations. One structure at the mean has no spread.

The full depth anchor table with the two sided background at z 3, 2026-11-13, held under 0.1
percent of anchor pairs on chr1 and moved SCC by 0.007 to 0.022 on whole chr1 with nothing else
changed, `playground/square_blocks/acxc/`. Pair counts at the anchor level do not reach the gaps.
Pooled over windows of about 10 kb inside a gap, a full depth map holds enough counts to call a
pair depleted as well as enriched, `playground/gap_window_balance.py`.

## Options

### 1. Maximum entropy bridge

HIPPS and DIMES derive an ensemble of structures from a contact map or a distance map with no
energy function. Shi and Thirumalai, Phys Rev X 11, 011051 (2021), doi 10.1103/physrevx.11.011051,
and Nat Commun (2023), doi 10.1038/s41467-023-36412-4. Constraints on mean squared pair distances
give a Gaussian chain of greatest entropy, sampled directly with no Monte Carlo, and the spread
between draws follows from the constraints.

For a gap the targets are the law's background, scaled by window contacts from the full depth map
as observed over expected to the minus third, the rule every heatmap distance follows. The
Gaussian is conditioned on the two anchors the arcs stage placed, which has a closed form. Every
conformation draws its own bridge, and the smooth stage with the hard wall removes overlaps.

One set of springs cannot hold every pair at a power law with an exponent under a third in three
dimensions, which is why the all pairs background was rejected on 2026-09-06. A distribution can.
The law's mean squared distance is the variogram of fractional Brownian motion with Hurst
exponent nu, a valid covariance for any nu up to one. The unit check confirms it on the law at nu
0.34.

The same code fills a hole once a hole has beads.

Built 2026-11-14 as `[simulation_arcs_smooth] start = maxent` with `[data] contact_map`,
`gnome3d/pipeline/ib/maxent.py`, unit checks in `harness/test_maxent_start.py`. Choices
written beside the code. Windows of up to 10 kb. A pseudocount of 9 on observed and expected,
the count at which a zero is three Poisson deviations under its expectation. The expectation is a
power law in separation fitted on the block's own windows. The variance per coordinate is pi d^2
over 8 for a mean distance d. One draw per gap and conformation. Keeping the clearest of twenty
draws, as the coil start does per bead, put the start at 1.14 to 1.24 of the law on GM12878
chr1:20-25 Mb against 0.92 to 1.08 for one draw, and is no longer the maximum entropy ensemble.

### 2. Loop extrusion per conformation

ChIA-PET records the stable loops at CTCF. Inside a gap cohesin is partway through extruding,
with nested loops that differ between cells. Fudenberg et al., Cell Rep 2016, doi
10.1016/j.celrep.2016.04.085. Sanborn et al., PNAS 2015, doi 10.1073/pnas.1518552112. LoopSage,
Korsak and Plewczynski, Methods 2024, doi 10.1016/j.ymeth.2024.01.015, is an energy Monte Carlo
of extrusion from the lab that wrote 3D-GNOME. Wang and Beer, Nat Commun 2021, doi
10.1038/s41467-021-21368-0, reproduce ChIA-PET loops with a simple extrusion and competition
model.

A one dimensional extrusion run per conformation would add a few transient arcs inside each gap,
with barriers at the CTCF sites and motif orientations the run already loads. It extends loop
dropout, each conformation getting different loops, and gives a gap the internal enrichment and
stripes a random coil lacks. To follow option 1.

### 3. Micro-C

Micro-C, and Micro Capture-C ultra at base pair resolution, Li et al., Cell 2025, doi
10.1016/j.cell.2025.10.013, hold far more short range counts than Hi-C. H1ESC and HFFC6 Micro-C
are on the workstation at `/mnt/storagelinux/_hic/<cell>/microc/` as `.hic` files and need a
conversion to cooler before `contact_map` can read them. It feeds option 1 with more counts and
is no model change by itself.

### 4. Holes as heterochromatin

Phase separation models, the strings and binders switch, Barbieri et al., PNAS 2012, doi
10.1073/pnas.1204799109, and MiChroM, Di Pierro et al., PNAS 2017, doi 10.1073/pnas.1714980114,
treat B compartment stretches as compact self attracting globules. Conte et al., Nat Commun 2020
and 2022, doi 10.1038/s41467-020-17141-4 and 10.1038/s41467-022-31856-6, find that phase
separation and extrusion coexist in single molecule imaging. The compartment term restricted to
hole beads is an option once holes have beads. The global compartment term costs SCC and
MultiMM.

### 5. Predicted contacts

C.Origami, Tan et al., Nat Biotechnol 2023, doi 10.1038/s41587-022-01612-8, Orca, Zhou, Nat
Genet 2022, doi 10.1038/s41588-022-01065-4, and Akita, Fudenberg et al., Nat Methods 2020, doi
10.1038/s41592-020-0958-x, predict maps from sequence with CTCF and accessibility. The cell lines
have deep maps. For the trios, with no Hi-C per person at depth, a predicted map could be the
`contact_map` of option 1.

### Validation

The 4DN model comparison project, PLoS Comput Biol 2025, doi 10.1371/journal.pcbi.1013358, is
worth reading against the battery.

## Trial of option 1, 2026-11-14

`slurm/ensemble/gap_maxent_trial.sh`, workstation, results in `out/gapme` and copied to
`playground/square_blocks/gapme/`. Configs `slurm/ensemble/<cell>_xc_R.ini`. Production and the
z3 arm come from the cross cell run in `out/acxc`, and a rerun of GM12878 chr1:20-25 Mb at
production checks the coil start is unchanged.

| Arm | Start | Contact data |
|---|---|---|
| prod | coil | production |
| fbm | maxent | law alone |
| maxent | maxent | window contacts at 1 kb |
| maxent_z3 | maxent | window contacts and the anchor table at z 3 |

GM12878, H1ESC and HFFC6, chr1:20-25, 10-15 and 35-40 Mb at five structures, whole chr1 at three
scored on 1-60 Mb. The gap readout is realised over law for subanchor pairs at 20 kb to 1 Mb,
where `start = law` and the coil differed. The battery is the decision.

The gate rerun was byte identical. Battery on raw counts, region columns the mean over the three
5 Mb regions, chr1 the 1-60 Mb slice of whole chr1.

| Cell | Arm | Pearson regions | SCC regions | MultiMM regions | Pearson chr1 | SCC chr1 | MultiMM chr1 | exponent over Hi-C chr1 |
|---|---|---|---|---|---|---|---|---|
| GM12878 | prod | 0.443 | 0.220 | 0.673 | 0.262 | 0.282 | 0.587 | 1.18 |
| GM12878 | fbm | 0.404 | 0.195 | 0.687 | 0.237 | 0.255 | 0.590 | 1.10 |
| GM12878 | maxent | 0.399 | 0.220 | 0.686 | 0.234 | 0.281 | 0.585 | 1.09 |
| H1ESC | prod | 0.431 | 0.176 | 0.658 | 0.292 | 0.190 | 0.590 | 1.18 |
| H1ESC | fbm | 0.397 | 0.137 | 0.635 | 0.265 | 0.166 | 0.580 | 1.11 |
| H1ESC | maxent | 0.390 | 0.178 | 0.631 | 0.262 | 0.200 | 0.578 | 1.11 |
| HFFC6 | prod | 0.408 | 0.234 | 0.605 | 0.268 | 0.234 | 0.557 | 1.15 |
| HFFC6 | fbm | 0.379 | 0.202 | 0.626 | 0.255 | 0.200 | 0.597 | 1.09 |
| HFFC6 | maxent | 0.379 | 0.211 | 0.617 | 0.251 | 0.239 | 0.577 | 1.09 |

On chr1 the z3 arms read Pearson 0.259, 0.293 and 0.268 for production and 0.235, 0.263 and 0.251
for maxent_z3, SCC 0.297, 0.212 and 0.241 against 0.298, 0.219 and 0.243.

The start loses Pearson on every cell and region, 0.01 to 0.05 raw and 0.03 to 0.06 balanced.
SCC is level with production once the window contacts scale the targets and 0.02 to 0.04 under
it on the law alone, so the window contacts carry signal, maxent over fbm on SCC in eleven of
twelve cell and region pairs. MultiMM is up 0.01 to 0.04 on GM12878 and HFFC6 regions and down on
H1ESC. The distance exponent moves toward Hi-C, 1.15 to 1.18 times its fit to 1.09 to 1.11.
Cross block overlaps rise on GM12878 chr1, 18 to 29 per thousand, and on HFFC6 chr1, 1.4 to 3.3.

The gap readout names the mechanism. On GM12878 chr1 the start draws gaps at 0.92 to 1.08 of the
law and they end at 1.24, 1.21 and 1.13 at 25-50, 50-100 and 100-200 kb, where the coil ends at
1.03, 0.87 and 0.77. The smooth stage widens the drawn bridge by about a fifth and the gaps then
hold fewer short range contacts than the coil's, which raw Pearson weighs most. Anchorless bins
stay at a model observed over expected of 0.4 to 0.6 in every arm against 0.85 to 0.92 in Hi-C,
so none of the arms reaches the contact deficit those bins carry, holes included.

Not adopted. The start stays opt in as the source of a hole's beads if holes get them, and as the
place a denser map enters.

## Smooth stage trace, 2026-11-14

`playground/smooth_widening_trace.py` keeps each block's start, its start after the restart noise
and its final chain, `slurm/ensemble/smooth_widening_trace.sh`, GM12878 chr1:20-25 Mb, one
structure per arm, logs in `playground/square_blocks/widen/`. Pairs inside one gap, realised over
law.

| Arm | State | 1-5 kb | 5-25 kb | 25-50 kb | 50-100 kb | 100-200 kb | mean bend | close pairs per thousand |
|---|---|---|---|---|---|---|---|---|
| maxent | start | 0.98 | 0.99 | 1.05 | 1.01 | 1.03 | 0.587 | 1170 |
| maxent | after noise | 1.40 | 1.15 | 1.12 | 1.07 | 1.07 | 0.670 | 858 |
| maxent | final | 1.42 | 1.33 | 1.24 | 1.17 | 1.13 | 0.269 | 8 |
| no bend term | final | 1.22 | 1.27 | 1.21 | 1.13 | 1.06 | 0.423 | 11 |
| no wall | final | 1.40 | 1.30 | 1.23 | 1.13 | 1.11 | 0.259 | 250 |
| no wall, no excluded volume | final | 1.41 | 1.23 | 1.16 | 1.05 | 1.11 | 0.247 | 695 |
| none of the three, no sphere | final | 1.01 | 1.04 | 1.07 | 1.01 | 1.05 | 0.531 | 1084 |
| coil | start | 1.08 | 0.99 | 0.79 | 0.67 | 0.55 | 0.497 | 918 |
| coil | final | 1.45 | 1.34 | 1.03 | 0.84 | 0.70 | 0.273 | 12 |

Mean bend is one minus the cosine between consecutive bonds over two, the quantity the bend term
scores as its cube, a half for a random walk. Close pairs are non neighbour pairs under the
excluded volume radius of 0.7 bonds.

The bonds and the springs keep a drawn gap on the law. The bend term straightens the chain, mean
bend 0.59 to 0.27, and the excluded volume swells it, and either alone widens a gap by a fifth to
two fifths under 50 kb. The law itself is not reachable under both. A chain on the law at nu 0.34
folds back more often than a random walk and holds 1,170 pairs per thousand beads under 0.7 bonds,
which the overlap count of the battery reads as defects. The coil ends at the same 1.45 and 1.34
under 25 kb, so production is as stiff at short range, and below the law beyond 50 kb because the
coil starts compact there and nothing in the stage moves a gap's far pairs.

The restart noise moves every subanchor by up to one bond per coordinate before the stage starts,
bonds 1.58 times their target, and the stage recovers the bonds. It is a cost of the start, not of
the end state.

## Bend term and hole beads, 2026-11-14, regions

`slurm/ensemble/gap_angle_holes_trial.sh`, results in `out/gapah` and
`playground/square_blocks/gapah/`. Gate rerun byte identical. Battery on raw counts, Pearson and
SCC.

| Cell, region | prod | noangle | maxent | maxent_noangle | maxent_bare | holes |
|---|---|---|---|---|---|---|
| GM12878 10-15 | 0.367, 0.377 | 0.356, 0.360 | 0.330, 0.350 | 0.319, 0.349 | 0.317, 0.308 | 0.346, -0.038 |
| GM12878 20-25 | 0.492, 0.081 | 0.477, 0.081 | 0.445, 0.110 | 0.433, 0.103 | 0.417, 0.066 | no hole |
| GM12878 35-40 | 0.471, 0.203 | 0.460, 0.189 | 0.423, 0.201 | 0.411, 0.188 | 0.396, 0.147 | 0.475, 0.171 |
| H1ESC 10-15 | 0.357, 0.281 | 0.349, 0.273 | 0.316, 0.238 | 0.309, 0.237 | 0.308, 0.234 | 0.298, 0.012 |
| H1ESC 20-25 | 0.488, 0.145 | 0.475, 0.155 | 0.440, 0.155 | 0.427, 0.155 | 0.412, 0.165 | no hole |
| H1ESC 35-40 | 0.448, 0.102 | 0.433, 0.087 | 0.415, 0.141 | 0.401, 0.141 | 0.387, 0.152 | no hole |
| HFFC6 10-15 | 0.372, 0.378 | 0.366, 0.365 | 0.335, 0.348 | 0.325, 0.340 | 0.321, 0.282 | 0.372, 0.267 |
| HFFC6 20-25 | 0.430, 0.108 | 0.415, 0.092 | 0.420, 0.104 | 0.404, 0.086 | 0.387, 0.107 | no hole |
| HFFC6 35-40 | 0.421, 0.217 | 0.423, 0.206 | 0.381, 0.181 | 0.374, 0.182 | 0.357, 0.150 | no hole |

Without the bend term Pearson falls 0.01 to 0.015 on every arm, and gaps held on the law with no
wall and no excluded volume score lowest of all with 540 to 850 overlapping pairs per thousand
beads. A gap at the law is not what the contact map rewards, so option 1 is not recovered by the
smooth stage.

Hole beads bring the anchorless bins' observed over expected to the Hi-C level, 0.75 to 0.93
against 0.90 to 0.93 on GM12878 chr1:10-15 Mb, and cost SCC everywhere a hole lies, to below the
null without structure on GM12878 and H1ESC. The holes there, 12.66 to 13.64 and 13.85 to
14.52 Mb, hold fewer contacts than the blocks around them at every separation, a ratio of 0.30 to
0.52 per diagonal in raw counts and 0.47 to 0.80 balanced, with 12 of 66 bins unmappable. The
bridges give them 1.2 to 2.2 times the blocks' contacts. A bridge on the law is denser at short
range than a block, which the smooth stage leaves at 1.45 of the law, and the window
expectation is fitted on each hole's own windows, so it cannot read a hole as depleted against the
chromosome. Two changes follow: the expectation taken over the chromosome from the map's 10 kb
resolution, and the hole chains run through the smooth stage's terms so their short range matches
the blocks'.

## Bend term and hole beads, 2026-11-14, chr1

Chromosome 1 at three structures, scored on 1 to 60 Mb, raw counts then balanced Pearson. Hole
beads add 7,708 beads over the 60 Mb on GM12878.

| Arm | GM12878 raw P, SCC | bal P | H1ESC raw P, SCC | bal P | HFFC6 raw P, SCC | bal P |
|---|---|---|---|---|---|---|
| prod | 0.262, 0.282 | 0.675 | 0.292, 0.190 | 0.649 | 0.268, 0.234 | 0.720 |
| maxent | 0.234, 0.281 | 0.622 | 0.262, 0.200 | 0.593 | 0.251, 0.239 | 0.689 |
| maxent_noangle | 0.226, 0.268 | 0.631 | 0.254, 0.200 | 0.599 | 0.243, 0.230 | 0.697 |
| holes | 0.269, 0.062 | 0.634 | 0.292, 0.084 | 0.629 | 0.272, 0.146 | 0.700 |
| holes_maxent_noangle | 0.236, 0.049 | 0.587 | 0.253, 0.086 | 0.575 | 0.246, 0.134 | 0.672 |

Smooth stage overlaps per thousand beads, prod against holes: 12.1 to 215, 10.4 to 127, 3.5 to
62, and across blocks 18 to 90, 16 to 51, 1.4 to 17. The bridges carry no smoothing and no
excluded volume against the blocks and sit at the law's 1,170 close pairs per thousand of
the trace. Raw Pearson holds with the holes, balanced Pearson falls 0.02 to 0.04 and SCC falls
by a factor of 1.6 to 4.5. The anchorless bins' observed over expected rises from 0.43 to 0.57
to 0.69 to 0.91 against Hi-C's 0.85 to 0.92. The first version fills the bins and overfills
them. The second version, chromosome expectation and smoothed holes, is in `out/gapv2`.

## Hole beads against the blocks, 2026-11-14

Two diagnostics on the second version, `playground/hole_contact_split.py` and
`playground/hole_contact_overlap.py`. Hole with block and two hole contacts sat under 0.7 bond in
15 percent of cases at every separation, the share for beads uniform within the contact radius,
against 2 to 5 percent inside one hole. The holes passed through the blocks. The third version
smooths every hole in one pass over the chromosome with the block beads fixed, so the excluded
volume and the wall act between them. `slurm/ensemble/gap_holes_v3_trial.sh`, results in
`playground/square_blocks/gapv3/`. Raw Pearson, SCC and cross block overlaps per thousand beads.

| Region | prod | holes2 | holes3 |
|---|---|---|---|
| GM12878 10-15 | 0.367, 0.377, 2.8 | 0.348, 0.194, 97.8 | 0.358, 0.190, 28.6 |
| H1ESC 10-15 | 0.357, 0.281, 13.4 | 0.311, 0.153, 78.9 | 0.317, 0.159, 19.7 |
| HFFC6 10-15 | 0.372, 0.378, 5.3 | 0.362, 0.322, 24.2 | 0.364, 0.329, 7.6 |
| GM12878 35-40 | 0.471, 0.203, 1.3 | 0.475, 0.187, 10.6 | 0.475, 0.192, 3.2 |

The overlaps fall three to four times and SCC stays where the second version had it. Part of the
loss is the instrument. Hi-C reads the unmappable bins as zero, production leaves them empty and
hole beads fill them. With the model zeroed there the SCC of the averaged map moves from 0.272 to
0.315 on GM12878 against 0.458 for production, 0.216 to 0.300 on H1ESC against 0.344 and 0.429 to
0.455 on HFFC6 against 0.481.

The rest is the blocks. Mean count per separation over the block with block mean at 25 to 50 kb,
balanced Hi-C against the model, chr1:10-15 Mb:

| | 25-50 kb | 50-100 | 100-200 | 200-400 | 400-1000 |
|---|---|---|---|---|---|
| GM12878 Hi-C, block pairs | 1 | 0.44 | 0.24 | 0.13 | 0.048 |
| GM12878 model, block pairs | 1 | 0.41 | 0.17 | 0.066 | 0.004 |
| GM12878 Hi-C, one hole | 0.82 | 0.33 | 0.18 | 0.11 | 0.068 |
| GM12878 holes3, one hole | 0.66 | 0.32 | 0.18 | 0.10 | 0.004 |
| H1ESC Hi-C, block pairs | 1 | 0.50 | 0.29 | 0.16 | 0.051 |
| H1ESC model, block pairs | 1 | 0.42 | 0.11 | 0.03 | 0.005 |
| H1ESC Hi-C, one hole | 1.14 | 0.51 | 0.29 | 0.26 | 0.14 |
| H1ESC holes3, one hole | 0.85 | 0.51 | 0.31 | 0.22 | 0.04 |

Inside a hole the drawn chain follows Hi-C from 50 to 400 kb. The blocks hold 1.4 to 5 times too
few contacts from 100 to 400 kb and ten times too few beyond. Against blocks that sparse at long
range a hole at the right density reads as hot, and the stratified correlation counts it against
the model. The defect the holes expose is the blocks' long range contact deficit.

The deficit is spread, not distance. `playground/contact_spread.py` prints, per separation, the
root mean square distance of pairs inside one block, their contact share and the share a Gaussian
chain at that mean square would have. GM12878 production, chr1:10-15 Mb, five structures:

| | 25-50 kb | 100-200 kb | 200-400 kb | 400-1000 kb |
|---|---|---|---|---|
| rms in bonds | 4.0 | 6.0 | 7.4 | 10.9 |
| contact share | 0.033 | 0.0072 | 0.0029 | 0.0002 |
| Gaussian share at that rms | 0.048 | 0.015 | 0.0077 | 0.0025 |

The Gaussian shares fall 19 fold from the first band to the last and Hi-C 21 fold, the blocks 170
fold. Anchor pairs inside a block never touch beyond 50 kb on GM12878 or H1ESC. Their distance over
the law has a floor: the 0.1 and 1 percent quantiles sit at 0.27 to 0.32 and 0.42 to 0.46 at every
separation from 25 kb to 3 Mb, the median at 0.88 to 1.01, where a contact at 400 kb needs 0.17.
Every conformation holds every pair near its target, and contacts come from the pairs that are not
near it. The holes are drawn from a distribution and carry the Gaussian share.

GM12878 whole chr1, three structures, scored on 1 to 60 Mb: holes3 raw Pearson 0.270 and SCC
0.179 against production's 0.262 and 0.282 and holes2's 0.266 and 0.176. Cross block overlaps 45
per thousand against holes2's 98, the hole with block contacts under 0.7 bond 8 percent against
15. Block pairs hold 0.116, 0.034 and 0.006 of their 25 to 50 kb count at 100 to 200 kb, 200 to
400 kb and 400 kb to 1 Mb, Hi-C 0.240, 0.127 and 0.045. Pairs inside a hole hold 0.191, 0.105 and
0.033, Hi-C 0.191, 0.122 and 0.076.

## Anchors drawn from a distribution, 2026-11-15

`[simulation_arcs] start = maxent` draws a chromosome's anchors once per conformation from a
Gaussian whose pair variances meet every pair's target: a loop's target, the background where it
holds a pair, the law's distance elsewhere. With `solver_iters = 0` the draw is kept. Two forms
failed first on GM12878 chr1:20-25 Mb, 434 anchors and 486 loops. The law's Gaussian conditioned
on loop springs at their targets collapsed the region to an rms of 1.3 to 2.2 bonds at every
separation, since springs in parallel pull each other's pairs closer. Fitting the loop springs to
meet their targets left 2 to 3.7 bonds, nothing holding the arcless pairs out. A DIMES iteration
over every pair diverged, every spring moving at once. A set of targets holding loops is not the
distance matrix of any Gaussian, so the form kept is a least squares one: the covariance is taken
as `y y^T`, a pair's variance per coordinate is the squared distance between its rows, and the
rows are fitted by L-BFGS to the squared relative misfit, loops and the rest weighted equally in
total. On the same region it meets loops at 1.006 and the rest at 1.000 of target in 200
iterations. Two structures, against production:

| Within one block | 25-50 kb | 100-200 kb | 200-400 kb | 400-1000 kb |
|---|---|---|---|---|
| production rms, contact share | 3.8, 0.036 | 6.0, 0.0057 | 7.5, 0.0020 | 10.8, 0.0002 |
| draw rms, contact share | 3.8, 0.045 | 5.5, 0.013 | 6.9, 0.0067 | 9.7, 0.0019 |

The draw's block contacts fall 24 fold from the first band to the last, Hi-C's 21 fold and
production's 170 fold. Anchor pairs touch at every separation. Trial `slurm/ensemble/spread_trial.sh`,
arms against production on three regions and three cells and on whole chr1 for two cells: the
draw kept, the draw and 30 solver iterations, the draw and the full solve, the repulsion's reach
at 1.0 and 0.75, loop dropout, and the kept draw with hole beads.

### Spread trial, regions, 2026-11-15

`slurm/ensemble/spread_trial.sh`, results in `playground/square_blocks/spread/`. Gate rerun
byte identical. Mean over chr1:10-15, 20-25 and 35-40 Mb on GM12878, H1ESC and HFFC6, five
structures each: raw Pearson, balanced Pearson, balanced SCC, MultiMM.

| Arm | raw P | bal P | bal SCC | MultiMM |
|---|---|---|---|---|
| prod | 0.427 | 0.707 | 0.186 | 0.678 |
| mx0, the draw kept | 0.464 | 0.598 | 0.165 | 0.685 |
| mx30, the draw and 30 solver iterations | 0.421 | 0.674 | 0.182 | 0.710 |
| mxfull, the draw and the full solve | 0.426 | 0.697 | 0.193 | 0.717 |
| reach10 | 0.450 | 0.671 | 0.189 | 0.659 |
| reach075 | 0.462 | 0.642 | 0.189 | 0.630 |
| dropout | 0.416 | 0.693 | 0.183 | 0.669 |
| mx0holes | 0.460 | 0.596 | 0.149 | 0.674 |

The kept draw raises raw Pearson, Spearman and MultiMM and costs balanced Pearson 0.11. It has no
excluded volume: anchor overlaps inside a block rise from 4.4 to 34.7 per thousand on GM12878
20-25 Mb and cross block overlaps to 40 to 220, and the 20 to 100 kb exponent falls to 0.23 against
Hi-C's 0.33. The draw followed by the full solve keeps balanced Pearson within 0.01, raises SCC
0.007 and MultiMM 0.039, up on seven of nine runs: the solve undoes the spread and keeps the
long range arrangement the draw set, where production's comes from the Hilbert curve. A shorter
repulsion reach raises raw Pearson and costs balanced Pearson and MultiMM, compacting the blocks
to an Rg 15 to 20 percent under production. Dropout loses on every column.

The arm that keeps the spread and removes the overlaps, `slurm/ensemble/spread_ev_trial.sh`,
draws and then solves with every target spring at zero, so only the excluded volume and the
arcless repulsion act. On GM12878 20-25 Mb, two structures, block contacts at 400 kb to 1 Mb
hold 0.0021 of pairs against production's 0.0002 and 25 to 50 kb 0.035 against 0.036.

### Drawn anchors with only the overlaps relaxed, 2026-11-15

`slurm/ensemble/spread_ev_trial.sh`, results in `playground/square_blocks/spreadev/`. The draw, then
up to 200 solver iterations with every target spring at zero, the arcless repulsion's reach at 0.5
or 1.0 of the mean arc target. Mean over the nine region runs, raw Pearson, balanced Pearson,
balanced SCC, MultiMM:

| Arm | raw P | bal P | bal SCC | MultiMM |
|---|---|---|---|---|
| prod | 0.427 | 0.707 | 0.186 | 0.678 |
| mx0 | 0.464 | 0.598 | 0.165 | 0.685 |
| mxev05 | 0.439 | 0.570 | 0.143 | 0.665 |
| mxev10 | 0.382 | 0.557 | 0.149 | 0.633 |

Both lose more than the kept draw. At reach 0.5 the anchor overlaps go, 34.7 to 0.5 per thousand on
GM12878 20-25 Mb, the long range spread stays, 0.0017 of block pairs in contact at 400 kb to 1 Mb,
and the subanchor coils between the drawn anchors still overlap, 150 per thousand within blocks and
103 across. At 1.0 the repulsion swells every anchor pair to an rms of 7 to 12 bonds and the
short range contacts fall under production's. The 20 to 100 kb exponent stays at 0.24 to 0.26
against Hi-C's 0.33 in every drawn arm. The raw gains of the kept draw came with its overlaps, and
the spread the draw adds at long range does not reach balanced Pearson or SCC. The whole chr1 run
was stopped. The drawn start with the full solve stays the one arm with a gain, MultiMM.

## What sets the anchor floor, 2026-11-15

`slurm/ensemble/arms_trial.sh` on frodo, GM12878 chr1:10-15, 20-25 and 35-40 Mb, five structures,
one arcs term removed per arm, production rerun beside them. Results in
`playground/square_blocks/ablate/`. `playground/contact_spread.py` now prints the anchor floor,
the 1 percent and median quantiles of anchor pair distance over the law.

| Arm | floor beyond 100 kb, 1% and median | block contacts 400 kb to 1 Mb | bal Pearson | Rg | cross block overlaps |
|---|---|---|---|---|---|
| prod | 0.42 to 0.53, 0.95 to 1.07 | 0.0002 to 0.0003 | 0.646 to 0.729 | 9 to 10 | 1 to 4 |
| no arcless repulsion | 0.17 to 0.22, 0.62 to 0.66 | 0.0022 to 0.0036 | 0.541 to 0.622 | 7.0 to 7.5 | 133 to 285 |
| no contact background | unchanged | unchanged | unchanged | | |
| no arcs excluded volume | unchanged | unchanged | unchanged | | |
| no background springs | 0.43 to 0.54, 1.3 to 2.0 | unchanged | down 0.05 to 0.10 | 10 to 12 | |
| loop springs at 0.1 | 0.45 to 0.65, 0.97 to 1.14 | lower | level | | |

Beyond 100 kb the arcless repulsion sets the floor. Without it the long range contacts rise ten
fold and the 100 kb to 1 Mb exponent falls from 0.39 to 0.46 to 0.28 to 0.33 against Hi-C's 0.33,
and the blocks compact a third and pass through each other, since nothing else holds the long
range size. The repulsion stands in for the volume of the chain between anchors, which the arcs
stage does not carry, and holds every far pair of every conformation at the same distance. Under
100 kb the background springs hold the pairs at the law, and without them the blocks swell. The
next arms take the long range arrangement and its spread from the draw and solve without the
repulsion, with the arcs excluded volume at its default radius and at a full mean arc target.
