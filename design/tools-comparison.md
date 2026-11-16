# 3dgnome against cudaMMC and MultiMM

Measured 2026-11-16 on frodo, 36 cores and one RTX 3090, one job at a time, 3dgnome at commit
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
| 3dgnome | production config, `validation/core/config.py` | one per 1 kb between anchors |
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

On a 5 Mb region 3dgnome takes 1.0 to 1.5 times MultiMM's time at the same bead count. On a
chromosome it is 3.8 to 4.8 times faster than MultiMM at the same bead count. cudaMMC runs at
0.17 to 0.46 of our bead count on the regions and 0.26 to 0.41 on chr1, and at those counts it
is the fastest on the regions and on two of the three chromosomes.

3dgnome's smooth stage and relaxation pass run on the CPU cores and its arcs solve on the GPU,
`design/kernel-performance.md` section 14. cudaMMC and MultiMM run as their defaults place them.
