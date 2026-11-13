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

Results to follow.
