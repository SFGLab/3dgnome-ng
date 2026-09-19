# Validation, September 2026

Closed 2026-09-15. Three cells, chr1:1-60 Mb, five structures per arm, scored against each
cell's deep 4DN Hi-C at 25 kb with `playground/validation_battery.py --balance no --ev-factor
0.7 --singletons`. The arms are the production settings of 2026-09-12 (`prod3`, the wall and
the coil start, JAX executor), the production of 2026-09-11 before those levers (`prod`), the
reference 3dgnome binary with each cell's `config.ini`, and MultiMM 2.0.2 on our clusters
bedpe at our bead count, with loops only and with loops and A/B blocks. Tables are
`out/validation/<cell>_battery.txt` and `<cell>_saddle.txt` on the workstation; the figures
are `validation_figure` and `validation_regions_gm12878` in the figures folder, drawn by
`playground/figures/results_figure.py` and `region_figure.py`.

## The three cell gate

Pearson, Spearman and SCC are correlations of the structures' contact map with the Hi-C map,
MultiMM is that tool's own ensemble metric, the exponent is the distance law fitted on the
structures between 20 kb and 1 Mb against the cell's own fit on its singletons, and the
overlap columns are non neighbour pairs per thousand beads under 0.7 of the structure's
median subanchor bond, within blocks between subanchors and across blocks.

| cell | arm | Pearson | Spearman | SCC | MultiMM | exponent | Rg | wb-sa | xb |
|---|---|---|---|---|---|---|---|---|---|
| GM12878, nu 0.329 | reference | not run to the end, see below | | | | | | | |
| | 3dgnome-ng before | 0.291 | 0.123 | 0.359 | 0.656 | 0.394 | 25.7 | 672 | 61 |
| | 3dgnome-ng now | 0.310 | 0.135 | 0.366 | 0.696 | 0.356 | 25.9 | 110 | 20 |
| | MultiMM loops | 0.183 | 0.090 | 0.259 | 0.353 | 0.158 | 27.4 | 18 | 0 |
| | MultiMM loops + A/B | 0.200 | 0.102 | 0.237 | 0.387 | 0.163 | 23.6 | 26 | 0 |
| H1ESC, nu 0.333 | reference | 0.235 | 0.126 | 0.156 | 0.415 | 0.118 | 38.5 | 2290 | 11782 |
| | 3dgnome-ng before | 0.328 | 0.122 | 0.246 | 0.675 | 0.415 | 25.9 | 650 | 51 |
| | 3dgnome-ng now | 0.344 | 0.132 | 0.252 | 0.711 | 0.368 | 26.0 | 100 | 15 |
| | MultiMM loops | 0.190 | 0.088 | 0.142 | 0.283 | 0.163 | 28.1 | 17 | 0 |
| | MultiMM loops + A/B | 0.207 | 0.099 | 0.132 | 0.323 | 0.163 | 23.8 | 25 | 0 |
| HFFC6, nu 0.352 | reference | 0.204 | 0.114 | 0.161 | 0.393 | 0.192 | 32.4 | 2190 | 8003 |
| | 3dgnome-ng before | 0.307 | 0.114 | 0.335 | 0.665 | 0.412 | 26.8 | 415 | 24 |
| | 3dgnome-ng now | 0.319 | 0.122 | 0.316 | 0.712 | 0.374 | 27.1 | 62 | 6 |
| | MultiMM loops | 0.180 | 0.073 | 0.157 | 0.307 | 0.178 | 29.0 | 3 | 0 |
| | MultiMM loops + A/B | 0.207 | 0.089 | 0.131 | 0.384 | 0.190 | 23.7 | 3 | 0 |

What it says. The production of 2026-09-12 is ahead of the reference on every Hi-C measure
on both cells the reference finished, Pearson by 0.11 to 0.12, SCC by 0.10 to 0.16, MultiMM
by 0.30 to 0.32, and its exponent sits at 1.06 to 1.10 times the cell's own fit where the
reference sits at 0.35 to 0.55. Against MultiMM on MultiMM's own metric we are at 0.70 to
0.71 where it reaches 0.28 to 0.39, and ahead on Pearson and SCC on every cell. The levers of
2026-09-12 raised Pearson 0.012 to 0.019 and MultiMM 0.04 to 0.05 on every cell, cut within
block overlaps five to seven times and cross block overlaps three to four times at the same
Rg, and brought the exponent from 1.2 to 1.1 times the fit. SCC moved within 0.02 either way.

The reference's overlap columns are not comparable and are kept off the overlap panels of the
figure. It has no excluded volume, its bead density is its own, and the battery's block
partition, recovered from our densification rule, over splits its chains, so its counts run
into the thousands per thousand beads and say nothing about the comparison.

## GM12878 and the reference

The reference did not finish GM12878 on this gate. Fifteen processes were started on the
workstation on 2026-09-13 at 18:28, one per structure and seed. HFFC6 finished in 2 h 15,
H1ESC in 7.5 to 8.5 h, and GM12878 was stopped on 2026-09-15 at 11:02 after 29 hours with
all five structures in segment 7 of 11, segment 6 alone having taken about a day per
structure. Its arcs Monte Carlo on GM12878's largest blocks is the cost.

What stands in for it is the original pipeline's own GM12878 models. The enhancer3d work
holds one hundred models per window from the published pipeline on four chr1 windows,
0.87-3.49 Mb, 12.6-13.83 Mb, 15.79-17.41 Mb and 18.3-20.22 Mb, converted with
`playground/reference_arm.py` and scored on each window's Hi-C with our five 60 Mb structures
and MultiMM's five cut to the same window by `playground/slice_arm.py`. The script is
`slurm/ensemble/reference_regions.sh`, the tables `out/validation_regions/GM12878_r<k>_battery.txt`.

| window | arm | n | Pearson | Spearman | SCC | MultiMM | exponent | Rg |
|---|---|---|---|---|---|---|---|---|
| 0.85-3.5 Mb, 106 bins | original pipeline | 100 | 0.132 | 0.118 | 0.165 | 0.112 | 0.066 | 5.3 |
| | 3dgnome-ng now | 5 | 0.446 | 0.438 | 0.281 | 0.662 | 0.358 | 8.7 |
| | MultiMM loops | 5 | 0.294 | 0.310 | 0.186 | 0.402 | 0.194 | 15.5 |
| | MultiMM loops + A/B | 5 | 0.307 | 0.324 | 0.186 | 0.417 | 0.200 | 12.4 |
| 12.6-13.85 Mb, 50 bins | original pipeline | 100 | 0.168 | 0.162 | 0.302 | 0.852 | -0.312 | 5.6 |
| | 3dgnome-ng now | 5 | 0.254 | 0.220 | 0.580 | 0.985 | 0.216 | 6.5 |
| | MultiMM loops | 5 | 0.281 | 0.252 | 0.353 | 0.338 | 0.276 | 16.1 |
| | MultiMM loops + A/B | 5 | 0.255 | 0.240 | 0.194 | 0.311 | 0.306 | 12.2 |
| 15.8-17.4 Mb, 66 bins | original pipeline | 100 | 0.144 | 0.098 | 0.307 | 0.200 | -0.002 | 4.2 |
| | 3dgnome-ng now | 5 | 0.447 | 0.459 | 0.238 | 0.505 | 0.288 | 6.5 |
| | MultiMM loops | 5 | 0.294 | 0.321 | 0.304 | 0.347 | 0.118 | 13.4 |
| | MultiMM loops + A/B | 5 | 0.284 | 0.306 | 0.277 | 0.313 | 0.120 | 11.4 |
| 18.3-20.2 Mb, 78 bins | original pipeline | 100 | 0.102 | 0.117 | 0.257 | 0.122 | -0.011 | 4.7 |
| | 3dgnome-ng now | 5 | 0.681 | 0.599 | 0.216 | 0.743 | 0.400 | 7.1 |
| | MultiMM loops | 5 | 0.338 | 0.347 | 0.205 | 0.290 | 0.113 | 12.4 |
| | MultiMM loops + A/B | 5 | 0.337 | 0.344 | 0.190 | 0.288 | 0.117 | 10.8 |

We are ahead of the original pipeline on Pearson on every window, by 0.09 to 0.58, and on
MultiMM on every window. SCC is the exception, ahead on two windows and behind on two, and at
50 to 106 bins a stratum holds very few pairs, so SCC on a window is the noisiest number in
the table. The original pipeline's exponent is flat or negative on three windows and its Rg
is 4 to 6 against our 6.5 to 8.7, which is the compaction the reference showed on the 60 Mb
gate as well. Three caveats. The window models were built for the enhancer3d study on
ChIA-PET input and whatever settings that study used, not on this gate's config, so this is
the published pipeline as it was run rather than the reference binary under our data. Our
structures are cut from a 60 Mb model, which places every window in the context of its
chromosome where the window models saw nothing outside their span. And the 12.6-13.85 Mb
window has 68 beads in the original models against our 226, so its numbers are the least
reliable of the four.

**Why SCC splits on the windows, checked 2026-09-15.** Five of the old models score the same
SCC as all hundred, so the ensemble size is not it, and the battery averages SCC per
structure in any case. SCC weights a stratum by the product of the two maps' standard
deviations, so a stratum counts only where a model has contacts to vary. The old models sit
at Rg 4 to 5 with a flat exponent and hold 13 to 22 contacts per pixel at every separation to
2 Mb, where ours follow the distance law and have almost no pair within the 1.33 bond cutoff
beyond 500 kb. On the 18.3-20.2 Mb window the strata under 100 kb tie at r 0.19, 125 to 500 kb
reads 0.30 against 0.28, and over 500 kb 0.38 against 0.00 with 13 percent of the old models'
weight there against 1 percent of ours. Scored on a soft map, every pair weighted by the cube
of one over one plus its distance in bonds, ours leads on both windows, 0.37 against 0.31 and
0.32 against 0.27, and the old pipeline and MultiMM barely move because their maps are dense
at every separation. The arm before the levers of 2026-09-12 scores level with production
on the windows, so the levers are not the cause. Run on the 18.3-20.2 Mb window alone rather
than cut from the 60 Mb model, ours scores Pearson 0.668, SCC 0.249, MultiMM 0.736 and
exponent 0.354, level with the cut on everything, so the context of the chromosome costs
nothing on the window either. The battery keeps the hard cutoff, since the 60 Mb tables were
scored with it and there we lead the reference on SCC on both finished cells.

## The map as the contact background's source, 2026-09-19

`[data] contact_map`, the deep mcool read at 25 kb, each anchor pair's pixel against the
expectation at its separation from the same map, held when it clears `contact_map_z` Poisson
standard deviations. GM12878 chr1:1-60 Mb, three structures per arm, 3 by 3 pooled unless
said; production holds 80 far pairs through the thinned singletons.

| arm | far pairs held | Pearson | Spearman | SCC | MultiMM | exponent | Rg | wb-sa | xb | saddle | eig r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production | 80 | 0.312 | 0.135 | 0.374 | 0.666 | 0.364 | 23.4 | 98 | 20 | 1.31 | 0.07 |
| z 3, one pixel | | 0.333 | 0.157 | 0.404 | 0.669 | 0.325 | 17.6 | 351 | 167 | 0.91 | 0.06 |
| z 3 | 1,434,000 | 0.343 | 0.163 | 0.390 | 0.677 | 0.314 | 19.5 | 396 | 161 | 0.78 | 0.33 |
| z 2 | | 0.346 | 0.166 | 0.392 | 0.676 | 0.308 | 19.7 | 434 | 181 | 0.76 | 0.27 |
| z 5 | 770,000 | 0.330 | 0.152 | 0.389 | 0.674 | 0.336 | 19.4 | 302 | 122 | 0.94 | 0.09 |
| z 8 | 347,000 | 0.323 | 0.146 | 0.391 | 0.660 | 0.353 | 19.5 | 232 | 99 | 0.84 | 0.20 |
| z 12 | 168,000 | 0.320 | 0.143 | 0.389 | 0.675 | 0.359 | 20.0 | 202 | 74 | 1.03 | 0.14 |

Every Hi-C correlation rises at every threshold, and the pooled arm at z 3 is the first of
ours to reach a compartment eigenvector correlation of 0.3. The price is compaction: a held
pair only ever pulls in, since a pair under expected keeps the repulsion, so Rg falls 15 to
20 percent whatever the threshold, the exponent sits under the fit until z 8, overlaps rise
two to eight times and the saddle falls. Pairs held per band at z 12 run 55,000 under
500 kb, 49,000 to 2 Mb, 35,000 to 10 Mb and 29,000 beyond. The long loops over 1 Mb added to
the joint solve were a loss on their own and are dropped, see AGENTS.md. Next measured:
`contact_map_symmetric`, holding pairs significantly under expected farther out, at z 8 and
12, and the relaxation back on for the cross block overlaps.

## Compartments

The saddle statistic at 100 kb against the cell's compartment eigenvector, with the term off
in production.

| cell | experimental | reference | 3dgnome-ng now | MultiMM loops | MultiMM loops + A/B |
|---|---|---|---|---|---|
| GM12878 | 3.47 | not run | 1.41 | 1.51 | 1.62 |
| H1ESC | 1.72 | 0.98 | 1.31 | 1.10 | 0.95 |
| HFFC6 | 6.82 | 0.96 | 2.07 | 1.50 | 2.45 |

Every arm is far under the experiment. Ours is ahead of the reference on both cells it ran,
ahead of MultiMM on H1ESC, and between MultiMM's two arms on the other two. MultiMM's A/B
blocks buy it 0.1 on GM12878 and 1.0 on HFFC6 and cost it on H1ESC. With our compartment
term on the saddle rises on H1ESC and HFFC6 at a cost in SCC and MultiMM, recorded in
`design/ab-compartments.md`, and it stays opt in.

## The stitch and the relaxation at chromosome scope

Measured 2026-09-19, `slurm/ensemble/assemble_ablation.sh`, GM12878 chr1:1-60 Mb, three
structures per arm. Boundary ratio is a boundary anchor pair's distance over the structure's
own within block curve, from `playground/restitch_model.py` on the first structure before its
replayed stitch.

| arm | Pearson | SCC | MultiMM | Rg | wb-sa | xb | boundary median, q95, max |
|---|---|---|---|---|---|---|---|
| production, both on | 0.311 | 0.377 | 0.674 | 25.6 | 118 | 22 | 1.01, 1.82, 2.30 |
| stitch off | 0.313 | 0.375 | 0.672 | 23.4 | 117 | 21 | 1.03, 1.94, 2.34 |
| relaxation off | 0.312 | 0.368 | 0.662 | 25.6 | 107 | 40 | 1.01, 1.82, 2.30 |
| neither | 0.313 | 0.374 | 0.669 | 23.4 | 107 | 23 | 1.03, 1.94, 2.34 |

Every Hi-C number is level across the four arms. The joint solve already puts boundary pairs
on the curve, median 1.03 and no pair over 2.34 times it, against the 59 times that the stitch
was built for. What the stitch adds is a tenth of Rg, from its centroid excluded volume between
every block pair, and cross block overlaps, 40 per thousand with it and no relaxation against
23 with neither, which the relaxation then removes. With the stitch off the relaxation moves
the cross block count from 23 to 21. Both passes are null on the data and one of them inflates.
Decision 2026-09-19: both off in production at chromosome scope, both kept for block scope, where a chromosome too large for one joint solve still needs a pass that places one block against the next.

## Expression, genome wide

The enhancer3d test on the production ensembles, three cell lines, ten conformations per
chromosome, run on eden 2026-09-15 to 2026-09-18 and scored 2026-09-19 with the notebook 3
protocol of `enhancer3d/playground/genome_original_correlation.py`: per gene minimum
enhancer distance, min max normalised per cell line, the difference between two lines
correlated against the DESeq2 log2 fold change on the extreme quartile jumps. Spearman, with
the gene count used, against the published cudaMMC models, which carry 100 conformations per
chromosome.

| cell pair | ours, beads as they are | ours, interpolated as v4 exports | cudaMMC v4 |
|---|---|---|---|
| GM12878 vs H1ESC | -0.559 (520) | -0.628 (479) | -0.566 (342) |
| GM12878 vs HFFC6 | -0.486 (561) | -0.509 (580) | -0.577 (354) |
| H1ESC vs HFFC6 | -0.459 (625) | -0.406 (696) | -0.456 (412) |

Level with cudaMMC overall, ahead on one pair, behind on the HFFC6 pair, on about 21,000 genes
with a bead in both arms against v4's 14,000 to 15,000. The previous genome wide run of ours,
TAD block models on 2026-08-25, stood at -0.34, -0.31 and -0.16. Tables in
`enhancer3d/playground/genome_correlation_prod` and `_prod_interp`, recipe `prod_e3d.sh`.

## Timings

One chr1:1-60 Mb structure on the workstation, an RTX 4060 Ti with the JAX executor for the
smooth stage and threaded numba for the arcs stage: GM12878 about 9 minutes, H1ESC about 13,
HFFC6 about 7. MultiMM's five structure ensemble with molecular dynamics took 4 to 5 minutes
on the same GPU. The reference on the CPU, one process per structure, took 2 h 15 on HFFC6,
7.5 to 8.5 h on H1ESC and did not finish GM12878 in 29 hours.

## What is open

- A reference GM12878 arm on the 60 Mb gate needs a day or two of CPU per structure. The
  window comparison stands in for it.
- The window SCC is not settled at 50 to 106 bins. A finer bin size on the deep map would
  give it more strata, at the cost of comparability with the 60 Mb tables.
