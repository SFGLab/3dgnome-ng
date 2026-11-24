# 3dgnome against cudaMMC and MultiMM

Measured 2026-11-16 on 36 cores and one RTX 3090, one job at a time, 3dgnome at commit
bb5baa5. Script `slurm/ensemble/tools_trial.sh`. Tables under `playground/square_blocks/tools/`
for cudaMMC and MultiMM and `playground/square_blocks/tools3/` for 3dgnome.

## Setup

Three cell lines, GM12878, H1ESC and HFFC6. Four scales per cell, chr1 10 to 15 Mb, 20 to 25 Mb
and 35 to 40 Mb at five structures, and the whole chr1 at three structures, scored on chr1 1 to
60 Mb.

Every tool reads the production config's inputs, the CTCF anchors and clusters and the Hi-C
singletons at 25 kb.

| Tool | Version and settings | Beads |
|---|---|---|
| 3dgnome | production config, `gnome3d/production.py` | one per 1 kb between anchors |
| cudaMMC | its template config, only the data section changed, ensemble by `-m` | its `loop_density` of 5 |
| MultiMM | 2.1.1 on CUDA, the clusters as loops, minimised and dynamics ensembles kept | our bead count over our span |

Neither of the other tools was tuned. The numbers are theirs at their defaults on our inputs.

## Hi-C agreement

Pooled ensemble against the deep 4DN map at 25 kb, `playground/ensemble_score.py`. Means over
the nine 5 Mb runs and over the three chr1 runs, with the range of the balanced SCC over mapped
bins.

| Scale | Tool | raw Pearson | raw Spearman | raw SCC | balanced Pearson | balanced SCC, mapped bins | range |
|---|---|---|---|---|---|---|---|
| 5 Mb | 3dgnome | 0.633 | 0.526 | 0.344 | 0.698 | 0.250 | 0.067 to 0.483 |
| 5 Mb | MultiMM dynamics | 0.320 | 0.226 | 0.232 | 0.586 | 0.174 | 0.042 to 0.245 |
| 5 Mb | cudaMMC | 0.281 | 0.279 | 0.158 | 0.243 | 0.100 | -0.023 to 0.183 |
| chr1 | 3dgnome | 0.411 | 0.176 | 0.387 | 0.680 | 0.253 | 0.134 to 0.346 |
| chr1 | MultiMM dynamics | 0.223 | 0.111 | 0.218 | 0.375 | 0.130 | 0.062 to 0.167 |
| chr1 | cudaMMC | 0.214 | 0.113 | 0.171 | 0.272 | 0.101 | 0.052 to 0.151 |

3dgnome leads on every statistic at both scales. Per run it leads on raw Pearson and raw
Spearman in all twelve. On SCC it loses two of the twelve to MultiMM, both on chr1 20 to 25 Mb.
GM12878 reads raw SCC 0.312 against 0.318 and HFFC6 0.175 against 0.222, where cudaMMC's 0.178
is ahead of it too. MultiMM's minimised ensemble is within 0.01 of its dynamics ensemble
everywhere.

On MultiMM's own metric, the battery's per structure number, 3dgnome reads 0.71 to 0.82,
MultiMM 0.24 to 0.44 and cudaMMC 0.30 to 0.56.

### Intervals

A paired block bootstrap, `playground/ensemble_bootstrap.py`, table in
`playground/square_blocks/tools3/bootstrap.txt`. Each region's 25 kb bins are resampled in
blocks of ten, a thousand times, every tool scored on the same resamples, and a difference
counts as a lead or a loss with its 95 percent interval clear of zero.

| Score | against | 3dgnome leads | tie | 3dgnome loses |
|---|---|---|---|---|
| raw Pearson | MultiMM | 12 | 0 | 0 |
| raw Pearson | cudaMMC | 12 | 0 | 0 |
| raw Spearman | MultiMM | 12 | 0 | 0 |
| raw Spearman | cudaMMC | 12 | 0 | 0 |
| raw SCC | MultiMM | 5 | 7 | 0 |
| raw SCC | cudaMMC | 6 | 6 | 0 |
| balanced SCC, mapped bins | MultiMM | 3 | 9 | 0 |
| balanced SCC, mapped bins | cudaMMC | 5 | 7 | 0 |

The two runs 3dgnome loses on SCC by the point value are ties, GM12878 chr1 20 to 25 Mb at
-0.006 with an interval of -0.133 to +0.132 and HFFC6 at -0.048 with -0.248 to +0.108. A 5 Mb
region is 200 bins and its SCC interval is 0.3 to 0.4 wide, so SCC does not separate the tools
at that scale. On chr1 1 to 60 Mb 3dgnome leads on raw SCC on all three cells against both
tools, and on balanced SCC on GM12878 and HFFC6.

The bootstrap keeps every pair's true separation, where the pooled table closes the gaps left
by unmapped bins, so its balanced SCC differs from the table's by up to 0.03 on a region with
unmapped bins.

### Regions of 20 Mb

A 5 Mb region cannot separate the tools on SCC, so three regions of 20 Mb were added, the
windows with the most CTCF loops in all three cells at once and on three chromosomes,
chr12 40 to 60 Mb, chr11 60 to 80 Mb and chr17 35 to 55 Mb, with 1,232 to 2,972 loops against a
median of 328 to 622 for a 20 Mb window. Five structures per tool and cell, the same bootstrap.
Tables in `playground/square_blocks/tools20/`.

| Score | against | 3dgnome leads | tie | 3dgnome loses |
|---|---|---|---|---|
| raw Pearson | MultiMM | 9 | 0 | 0 |
| raw Pearson | cudaMMC | 9 | 0 | 0 |
| raw Spearman | MultiMM | 9 | 0 | 0 |
| raw Spearman | cudaMMC | 9 | 0 | 0 |
| raw SCC | MultiMM | 7 | 2 | 0 |
| raw SCC | cudaMMC | 9 | 0 | 0 |
| balanced SCC, mapped bins | MultiMM | 5 | 4 | 0 |
| balanced SCC, mapped bins | cudaMMC | 9 | 0 | 0 |

Raw SCC reads 0.39 to 0.61 for 3dgnome, 0.25 to 0.47 for MultiMM and 0.18 to 0.38 for cudaMMC,
and the point difference favours 3dgnome on all nine runs against both, by 0.09 to 0.22 over
MultiMM and 0.16 to 0.30 over cudaMMC. The two ties with MultiMM are GM12878 chr11 at +0.089
with an interval from -0.010 and HFFC6 chr12 at +0.086 with one from -0.042.

### Ten structures on chr1 1 to 60 Mb

The region reconstructed on its own, ten structures per tool and cell, the same bootstrap.
Script `slurm/ensemble/night_queue.sh`, tables in `playground/square_blocks/toolpage/toolpage60/`.

| Cell | Score | 3dgnome | MultiMM | cudaMMC |
|---|---|---|---|---|
| GM12878 | raw SCC | 0.509 | 0.299 | 0.220 |
| H1ESC | raw SCC | 0.340 | 0.165 | 0.145 |
| HFFC6 | raw SCC | 0.436 | 0.181 | 0.146 |
| GM12878 | raw Pearson | 0.458 | 0.244 | 0.247 |
| H1ESC | raw Pearson | 0.494 | 0.252 | 0.258 |
| HFFC6 | raw Pearson | 0.450 | 0.237 | 0.210 |
| GM12878 | raw Spearman | 0.224 | 0.125 | 0.148 |
| H1ESC | raw Spearman | 0.216 | 0.122 | 0.146 |
| HFFC6 | raw Spearman | 0.182 | 0.099 | 0.102 |
| GM12878 | MultiMM's own metric | 0.784 | 0.354 | 0.513 |
| H1ESC | MultiMM's own metric | 0.765 | 0.286 | 0.484 |
| HFFC6 | MultiMM's own metric | 0.775 | 0.310 | 0.535 |

3dgnome leads both tools on raw Pearson, raw Spearman and raw SCC on all three cells, every
interval clear of zero. The raw SCC difference is +0.175 to +0.254 over MultiMM and +0.195 to
+0.290 over cudaMMC. On balanced SCC over mapped bins it leads both on GM12878 and HFFC6 and
ties both on H1ESC, +0.094 with an interval from -0.013 against MultiMM and +0.093 with one
from -0.031 against cudaMMC. The distance exponent over Hi-C's is 0.97, 1.02 and 1.11 for
3dgnome, 0.48 to 0.51 for MultiMM and 1.66 to 2.31 for cudaMMC.

## Physics

The validation battery per structure, means over the same runs. Overlaps are non neighbour
pairs under 0.7 of the structure's own median bond, per thousand beads. The exponent is the
distance exponent over the one fitted on the cell's Hi-C.

| Scale | Tool | exponent over Hi-C | Rg in bonds | anchor overlaps within blocks | subanchor overlaps within blocks | overlaps across blocks |
|---|---|---|---|---|---|---|
| 5 Mb | 3dgnome | 1.04 | 9.6 | 0.7 | 9.9 | 1.8 |
| 5 Mb | MultiMM dynamics | 0.59 | 18.5 | 0.0 | 1.9 | 0.0 |
| 5 Mb | cudaMMC | 1.46 | 26.4 | 5,948 | 2,378 | 11,820 |
| chr1 | 3dgnome | 1.05 | 24.4 | 2.1 | 20.8 | 7.0 |
| chr1 | MultiMM dynamics | 0.51 | 23.0 | 0.0 | 61.7 | 0.0 |
| chr1 | cudaMMC | 1.94 | 100.9 | 6,427 | 2,464 | 23,356 |

3dgnome's exponent sits at 0.99 to 1.15 of Hi-C's on all twelve runs. MultiMM's is 0.47 to 0.67
of it, a flatter curve than the data on every run. cudaMMC's runs from 0.67 to 2.83.

MultiMM has the fewest overlaps on the 5 Mb runs, a molecular dynamics excluded volume. On chr1
its subanchor overlaps are 12 to 104 per thousand against 5 to 36 for 3dgnome.

cudaMMC has no excluded volume and its counts are in the thousands per thousand beads. The
count is taken against its own median bond, and its bonds are uneven, since it places five
beads between every pair of anchors whatever their distance. Read its count as anchors piling
up. It is not the same yardstick as on an evenly spaced chain.

## Run time as shipped

Wall clock for the whole ensemble, from the command to the last structure, in seconds.

| Run | 3dgnome | cudaMMC | MultiMM |
|---|---|---|---|
| 5 Mb, five structures, nine runs | 30 to 65 | 16 to 38 | 30 to 43 |
| beads on those runs | 2,034 to 4,254 | 336 to 1,936 | as 3dgnome |
| GM12878 chr1, three structures | 673 at 124,657 beads | 521 at 50,631 beads | 2,528 at 124,657 beads |
| H1ESC chr1, three structures | 526 at 114,276 beads | 412 at 35,881 beads | 2,083 at 114,276 beads |
| HFFC6 chr1, three structures | 385 at 109,082 beads | 436 at 28,616 beads | 1,842 at 109,082 beads |
| GM12878 chr1 1 to 60 Mb, ten structures | 634 at 42,480 beads | 561 at 17,321 beads | 1,208 at 42,480 beads |
| H1ESC chr1 1 to 60 Mb, ten structures | 687 at 40,570 beads | 494 at 14,229 beads | 1,118 at 40,570 beads |
| HFFC6 chr1 1 to 60 Mb, ten structures | 374 at 36,765 beads | 503 at 10,050 beads | 978 at 36,765 beads |

On a 5 Mb region 3dgnome takes 1.0 to 1.5 times MultiMM's time at the same bead count. On a
chromosome it is 3.8 to 4.8 times faster than MultiMM at the same bead count. cudaMMC runs at
0.17 to 0.46 of our bead count on the regions and 0.26 to 0.41 on chr1, and at those counts it
is the fastest on the regions and on two of the three chromosomes.

3dgnome's smooth stage and relaxation pass run on the CPU cores and its arcs solve on the GPU,
`design/kernel-performance.md` section 14. cudaMMC and MultiMM run as their defaults place them.

## Run time against bead count

GM12878 chr1, three structures, each tool at several bead counts. `slurm/ensemble/tools_scaling.sh`,
table in `playground/square_blocks/toolscale/timing.tsv`. 3dgnome's count is set by
`target_bp_per_subanchor`, 1,000, 2,000 and 4,000 bp. cudaMMC's is set by `loop_density`, 5, 10,
14 and 20, where 14 is the density nearest 3dgnome's production count. MultiMM takes the count
directly.

| Beads | 3dgnome | cudaMMC | MultiMM |
|---|---|---|---|
| about 50,000 | 386 s at 49,387 | 521 s at 50,631 | 487 s at 49,387 |
| about 87,000 | 528 s at 81,665 | 572 s at 92,686 | 1,181 s at 81,665 |
| about 125,000 | 673 s at 124,657 | 641 s at 126,330 | 2,528 s at 124,657 |
| 176,796 | | 713 s | |

At matched beads 3dgnome and cudaMMC are within 35 percent of each other over the whole range.
3dgnome is the faster below about 100,000 beads and cudaMMC above. cudaMMC's time grows as the
bead count to the power 0.25, 3dgnome's to 0.6 and MultiMM's to 1.8, fitted between each tool's
smallest and largest count. cudaMMC's default on this chromosome is 50,631 beads and 3dgnome's
is 124,657, and the first table gives each at its own default.



## Whole genome

GM12878, chr1 to chr22, one structure. `slurm/ensemble/tools_genome.sh` and
`slurm/ensemble/tools_genome_perchr.sh`, tables in `playground/square_blocks/toolgenome/`.

| Tool | Setting | Beads | Seconds |
|---|---|---|---|
| 3dgnome | production, one job | 1,312,597 | 2,784 |
| cudaMMC | its defaults, 22 runs added up | 458,445 | 1,882 |
| MultiMM | 1,312,597 beads | | out of memory on the 24 GB card |
| MultiMM | 656,298 beads | | out of memory |
| MultiMM | 328,149 beads | | stopped at 5,400, in dynamics after a minimisation of 1,672 |

cudaMMC's genome mode builds a chromosome level heatmap from contacts between chromosomes and
stops when it finds none, `LooperSolver.cpp:1113`. The singletons every tool reads here hold
contacts within a chromosome only, and 3dgnome's genome run places its chromosomes without them
too. cudaMMC was therefore run one chromosome at a time. Its chromosomes take 18 to 192 seconds
at 2,380 to 50,631 beads.

cudaMMC's count is 0.35 of 3dgnome's and its time 0.68. Per bead 3dgnome takes 2.1 ms and
cudaMMC 4.1 ms. MultiMM ran without compartments, as on every other run here, and its own log
says a genome run wants them.

## The uniform chain, 2026-11-23

The rows of 3dgnome measured again with the uniform chain, `slurm/ensemble/redo_tools.sh` and
`redo_tools_readouts.sh`, on the machine of the first measurement at 7cfc081. The config is
`slurm/ensemble/<cell>_hic_uniform.ini`: the production inputs, the keys of `UNIFORM_CHAIN` and the
contact cells from half a of the deep map, `playground/thin_cooler.py --chromosomes`. Every tool is
scored against half b. The structures of cudaMMC and MultiMM are those of the first measurement,
and MultiMM keeps the bead count of the chain of loop anchors. Tables in
`data/figures/redo-2026-11-23/tools/`, outside git.

### Run time

The Results page of the site states the run times of the chain of loop anchors, the default chain of the form,
by a decision of 2026-11-23. The table has the uniform chain beside it.

| Run | beads | seconds | chain of loop anchors |
|---|---|---|---|
| GM12878 chr1 11.5 to 12.6 Mb, one structure | 767 | 13.2 | 760 beads, 9.0 s |
| GM12878 chr1 1 to 60 Mb, ten structures | 54,429 | 1,018 | 42,480 beads, 634 s |
| H1ESC chr1 1 to 60 Mb, ten structures | 53,707 | 1,052 | 40,570 beads, 687 s |
| HFFC6 chr1 1 to 60 Mb, ten structures | 51,285 | 779 | 36,765 beads, 374 s |
| GM12878 chr1, one structure | 184,570 | 358 | 124,657 beads, 384 s |
| GM12878 chr1, three structures | 184,570 | 1,015 | 124,657 beads, 673 s |
| H1ESC chr1, three structures | 178,059 | 916 | 114,276 beads, 526 s |
| HFFC6 chr1, three structures | 171,214 | 689 | 109,082 beads, 385 s |
| GM12878 chr1 to chr22, one structure | 2,091,330 | 3,676 | 1,312,597 beads, 2,784 s |
| GM12878 chr1 from 10 Mb, one structure, 1, 2, 5, 10, 20, 50 and 100 Mb | 811 to 83,099 | 14.3, 18.6, 20.2, 27.3, 48.2, 98.2, 152.5 | 662 to 56,458 beads, 9.1 to 124.4 s |

The uniform chain has 1.3 to 1.6 times the beads of the chain of loop anchors on a chromosome. One
structure of chr1 takes the same time. Further structures cost more: 329 s each after the first,
where the chain of loop anchors took 144 s.

### Hi-C agreement against the other tools

Pooled raw SCC on chr1 1 to 60 Mb, ten structures: 0.481, 0.353 and 0.465 for GM12878, H1ESC and
HFFC6. cudaMMC has 0.220, 0.143 and 0.145 and MultiMM 0.297, 0.163 and 0.181 against the same half.

The paired block bootstrap over the 21 runs, leads by the 95 percent interval:

| Score | against | 5 Mb and chr1, 12 runs | 20 Mb, 9 runs | chain of loop anchors, 12 and 9 |
|---|---|---|---|---|
| raw Pearson | MultiMM | 12 | 9 | 12 and 9 |
| raw Pearson | cudaMMC | 12 | 9 | 12 and 9 |
| raw Spearman | MultiMM | 12 | 9 | 12 and 9 |
| raw Spearman | cudaMMC | 12 | 9 | 12 and 9 |
| raw SCC | MultiMM | 5 | 2 | 5 and 7 |
| raw SCC | cudaMMC | 7 | 5 | 6 and 9 |
| balanced SCC, mapped bins | MultiMM | 5 | 7 | 3 and 5 |
| balanced SCC, mapped bins | cudaMMC | 9 | 9 | 5 and 9 |

The same bootstrap with the contacts per bead pair, `bootstrap_pair.txt` of 2026-11-24, the score of the
Results page since that day. `playground/ensemble_bootstrap.py` divides the count of a cell by the product of
the bead counts of its two bins and prints a two sided p value.

| Runs | against | higher value | better by the interval |
|---|---|---|---|
| 5 Mb, 9 runs | MultiMM | 9 | 2 |
| 5 Mb, 9 runs | cudaMMC | 9 | 6 |
| 20 Mb, 9 runs | MultiMM | 7 | 1 |
| 20 Mb, 9 runs | cudaMMC | 8 | 4 |
| chr1 1 to 60 Mb, three structures, 3 runs | MultiMM | 3 | 3 |
| chr1 1 to 60 Mb, three structures, 3 runs | cudaMMC | 3 | 3 |

Over the 21 runs the uniform chain is better than MultiMM in 6 and better than cudaMMC in 13 by the interval.
No run is lost to another tool. On chr1 1 to 60 Mb at ten structures the SCC per bead pair is 0.472, 0.349 and
0.454 for GM12878, H1ESC and HFFC6. cudaMMC has 0.269, 0.171 and 0.204 and MultiMM 0.297, 0.163 and 0.180. No
resample of 1,000 has another tool ahead, p under 0.002.

The last column is the first measurement, scored against the whole map. No run is lost to another
tool on any score. On the 20 Mb runs the raw SCC of the uniform chain is
0.32 to 0.48, where the chain of loop anchors had 0.39 to 0.61, and its point value is above both
tools in all nine. On chr1 1 to 60 Mb at three structures it leads both tools on all three cells.

### Bands of separation

The same ten structures of chr1 1 to 60 Mb by band of separation, 2026-11-25. The score is the SCC per bead pair
at 10 kb against the raw half b over the pairs with both bins inside the region,
`playground/rnapii/short_range_scc.py --region 0-60000000`. All three tools read CTCF loops. The three values of
a cell are GM12878, H1ESC and HFFC6. The table is `design/figures/redo-2026-11-23/tools_bands.tsv`.

| Band | uniform chain | cudaMMC | MultiMM |
|---|---|---|---|
| under 100 kb | 0.40, 0.26, 0.37 | 0.25, 0.16, 0.18 | 0.18, 0.10, 0.09 |
| 100 to 500 kb | 0.33, 0.29, 0.37 | 0.22, 0.15, 0.20 | 0.18, 0.10, 0.11 |
| 0.5 to 2 Mb | 0.24, 0.17, 0.24 | 0.08, 0.06, 0.04 | 0.06, 0.02, 0.04 |

The uniform chain is above both tools in every band on every line. Its lead is largest in the band of 0.5 to
2 Mb, three to seven times the score of the other tools. The two variants of MultiMM, after the minimisation
and after the dynamics, differ by 0.002 at most. The scores at 25 kb over the whole range, 0.475, 0.350 and
0.455, repeat the scores of the bootstrap above within 0.003.

The whole genome has cudaMMC and the uniform chain on the same halves, `design/uniform-chain.md`. The score is
weighted by the length of the chromosome.

| Band | cudaMMC | uniform chain, CTCF loops | uniform chain, CTCF and RNAPII loops |
|---|---|---|---|
| under 100 kb | 0.28, 0.20, 0.20 | 0.45, 0.39, 0.45 | 0.55, 0.42, 0.52 |
| 100 to 500 kb | 0.25, 0.15, 0.19 | 0.35, 0.31, 0.39 | 0.28, 0.28, 0.28 |
| 0.5 to 2 Mb | 0.10, 0.05, 0.07 | 0.15, 0.12, 0.20 | 0.14, 0.11, 0.15 |

With CTCF loops the uniform chain is above cudaMMC on each of the 22 chromosomes of each line in each band.
With RNAPII loops cudaMMC is higher in the band of 100 to 500 kb on 6, 1 and 2 of the 22 chromosomes. On chr19
the scores of that band are 0.11, 0.14 and 0.16 with RNAPII loops, 0.29, 0.25 and 0.31 with CTCF loops and
0.28, 0.14 and 0.20 for cudaMMC. The loss is of the model with RNAPII loops against the model without them on
the chromosomes dense in RNAPII loops. `design/uniform-chain.md` has its trace.

### The two chains on the same runs

The structures of the two chains of each run against half b, the same bootstrap,
`chain_bootstrap.txt`. 24 runs: the nine of 5 Mb, the nine of 20 Mb, chr1 1 to 60 Mb at three
structures and at ten.

| Score | uniform higher | tie | loop anchors higher |
|---|---|---|---|
| raw Pearson | 20 | 4 | 0 |
| raw Spearman | 16 | 8 | 0 |
| raw SCC | 0 | 20 | 4 |
| balanced SCC, mapped bins | 6 | 18 | 0 |

The four runs with a lower raw SCC are GM12878 chr12 40 to 60 Mb, 0.474 against 0.610, H1ESC
chr12, 0.415 against 0.567, H1ESC chr17 35 to 55 Mb, 0.321 against 0.524, and GM12878 chr1 1 to
60 Mb at three structures, 0.399 against 0.454. Three of the four are regions of 20 Mb chosen for
their count of loops. On the balanced map the same runs are ties. The cause is the measure: the count of a
cell of a pooled map grows with the beads of its two bins, which on the chain of loop anchors follow the raw
coverage of Hi-C. Per bead pair the uniform chain is higher on all nine runs of 20 Mb, 0.416 against 0.228 of raw
SCC, `design/uniform-chain.md`.

### Micro-C loops

The loops of chr1 0 to 60 Mb placed above moved loops, signal floor 15: 0.29, 0.53 and 0.51 for
GM12878, H1ESC and HFFC6, where the chain of loop anchors had 0.26, 0.38 and 0.45. Beads at both
ends of a loop: 0.98, 0.95 and 0.97 of the loops.

Tests of the uniform chain against each tool on these loops, `playground/rnapii/microc_tools_test.py` on the
table of `microc_loop_compare.py` with the ratios of the moved loops, `toolpage60/microc_moved/`. The share
placed above the moved loops, 10,000 resamples of the loops and of the moved loops: against cudaMMC +0.141,
+0.319 and +0.354 on GM12878, H1ESC and HFFC6, p under 0.001 each. Against MultiMM +0.032 with p 0.21, +0.372
and +0.361 with p under 0.001. The share of loops with beads at both ends, exact sign test over the loops: over
cudaMMC on every line, under MultiMM by 22, 19 and 19 loops, p under 0.001 each.
