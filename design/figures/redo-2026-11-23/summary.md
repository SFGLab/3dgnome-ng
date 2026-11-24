# The results on the uniform chain, 2026-11-23

Chromosomes 1 to 22 of GM12878, H1ESC and HFFC6, ten structures a chromosome, built on the cluster by
`slurm/ensemble/crosscell_arms.sh` with `ARMS="C UC UR"` at 2ff5766 and read out on a workstation by
`playground/rnapii/redo_pull.sh` and `redo_readouts.sh`. The uniform arms read half a of the deep map of a line.
Every score against Hi-C is against half b, for every arm. The tables per chromosome, the inputs of the figures
and the tables of the tools are in `data/figures/redo-2026-11-23/`, outside git. The figures are drawn by
`playground/rnapii/figures_two.py` with `ARM_NAMES` naming the fourth arm.

Arms: the cudaMMC models of 3D-GNOME 3.0, the uniform chain with CTCF loops, the uniform chain with CTCF and
RNAPII loops, the chain of loop anchors with CTCF loops.

## Agreement with Hi-C per bead pair at 5, 10 and 25 kb, the measure of the Results page

Decided on 2026-11-24: the page states the agreement at three resolutions with the contacts counted per bead
pair. The count of a cell of the pooled map is divided by the product of the bead counts of its two bins, the
step `resolutions` of `playground/rnapii/readouts_crossline_genome.sh`. SCC against the raw map of half b on the
pairs up to 2 Mb apart, mean over the chromosomes by length. The tables per chromosome are in
`data/figures/redo-2026-11-23/genome_pair/`, outside git.

| line | resolution | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF |
|---|---|---|---|---|---|
| GM12878 | 5 kb | 0.183 | 0.322 | 0.392 | 0.246 |
| GM12878 | 10 kb | 0.241 | 0.408 | 0.471 | 0.325 |
| GM12878 | 25 kb | 0.297 | 0.491 | 0.510 | 0.410 |
| H1ESC | 5 kb | 0.109 | 0.259 | 0.287 | 0.195 |
| H1ESC | 10 kb | 0.157 | 0.351 | 0.373 | 0.281 |
| H1ESC | 25 kb | 0.197 | 0.432 | 0.433 | 0.376 |
| HFFC6 | 5 kb | 0.130 | 0.337 | 0.383 | 0.264 |
| HFFC6 | 10 kb | 0.177 | 0.424 | 0.464 | 0.341 |
| HFFC6 | 25 kb | 0.214 | 0.507 | 0.500 | 0.437 |

Wilcoxon signed rank over the 22 chromosomes, two sided. The uniform chain is higher than cudaMMC and higher
than the chain of loop anchors on 22 of 22 chromosomes of every line at every resolution, p 5e-07. With RNAPII
loops against without, on every line: higher at 5 kb with p 5e-06 or lower. Higher at 10 kb with p 0.004 or
lower. At 25 kb higher on GM12878 with p 0.036. No difference at 25 kb on H1ESC with p 0.50 or on HFFC6 with
p 0.52. Against the balanced map the
order of the arms is the same at every resolution, 0.346, 0.267 and 0.405 for the uniform chain at 25 kb against
0.299, 0.252 and 0.366 for the chain of loop anchors.

The same pairs without the division, the uniform chain and the chain of loop anchors at 25 kb: 0.502 and 0.532
on GM12878, 0.442 and 0.454 on H1ESC, 0.516 and 0.473 on HFFC6. Both chains have beads on the same 65 % of
the bins of 25 kb of GM12878. Among the bins with beads the uniform chain has 25 beads in 96 % and a rank
correlation of the bead count with the raw coverage of 0.00 to 0.03 on the three lines. The chain of loop anchors
has 1 to 68 beads with a median of 13 or 14. Its correlation is 0.53, 0.37 and 0.27.

## Agreement with Hi-C, raw SCC of the pooled map at 25 kb, mean over the chromosomes by length

The measure of the page before 2026-11-24, the whole chromosome without the division.

| line | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF |
|---|---|---|---|---|
| GM12878 | 0.191 | 0.495 | 0.514 | 0.524 |
| H1ESC | 0.133 | 0.435 | 0.436 | 0.446 |
| HFFC6 | 0.149 | 0.513 | 0.504 | 0.468 |

Wilcoxon signed rank over the 22 chromosomes, two sided. The uniform chain against cudaMMC: higher on 22 of 22
on every line, p 5e-07. The uniform chain against the chain of loop anchors: lower on GM12878, higher on 8 of
22, p 0.010; no difference on H1ESC, p 0.84; higher on HFFC6, on 21 of 22, p 1e-06. With RNAPII loops against
without: higher on GM12878, p 0.038, no difference on H1ESC, p 0.52, and HFFC6, p 0.44.

At 10 kb by separation, median over the chromosomes, CTCF loops to CTCF and RNAPII loops on the uniform chain:

| separation | GM12878 | H1ESC | HFFC6 |
|---|---|---|---|
| under 100 kb | 0.453 to 0.547, p 3e-06 | 0.376 to 0.414, p 0.003 | 0.462 to 0.510, p 7e-06 |
| 100 to 500 kb | 0.346 to 0.278, p 1e-06 | 0.295 to 0.286, p 2e-04 | 0.390 to 0.279, p 1e-06 |
| 0.5 to 2 Mb | 0.143 to 0.135, p 0.003 | 0.124 to 0.110, p 5e-05 | 0.208 to 0.141, p 1e-06 |

The same bands, the chain of loop anchors to the uniform chain with CTCF loops: under 100 kb 0.469 to 0.453,
0.395 to 0.376 and 0.397 to 0.462, p 0.85, 0.11 and 1e-06; 100 to 500 kb 0.398 to 0.346, 0.332 to 0.295 and
0.383 to 0.390, p 2e-06, 0.054 and 0.70.

## The two sets of runs of the uniform chain

Decided on 2026-11-23: every comparison with Hi-C is from the runs with the loop dropout, the config of
production. The enhancer to promoter distance, the link to expression and the Micro-C loops are from runs
without it, `[springs] loop_dropout = no`, the arms UCN and URN of `slurm/ensemble/crosscell_arms.sh`, 132 tasks
at 1a7bb2e with the model code of 2ff5766. `playground/rnapii/redo_merge.py` joins the two sets into the inputs
of the figures. The figures and the Results page state it. The tables of the runs without the dropout are
in `data/figures/redo-2026-11-23/nodrop/genome/`, the joined inputs in `merged/`, outside git.

Without the dropout the raw SCC against half b is 0.486, 0.431 and 0.507 with CTCF loops, against 0.495, 0.435
and 0.513 with it. With CTCF and RNAPII loops it is 0.479, 0.412 and 0.462 against 0.514, 0.436 and 0.504.

## Enhancer to promoter distance, pairs joined by an RNAPII loop and by no CTCF loop, median in bead units

Runs without the dropout.

| line | pairs | uniform, CTCF | uniform, CTCF and RNAPII | closer | with the dropout |
|---|---|---|---|---|---|
| GM12878 | 23,237 | 4.01 | 3.32 | 17 % | 4.09 to 3.51, 14 % |
| H1ESC | 33,360 | 3.90 | 3.42 | 12 % | 3.98 to 3.58, 10 % |
| HFFC6 | 16,431 | 4.34 | 3.73 | 14 % | 4.43 to 3.96, 11 % |

Signed rank over the pairs and Mann-Whitney against the reference pairs: p under 0.001 on every line. On the
chain of loop anchors the pass of 2026-11-19 had 19, 24 and 17 % closer.

## Link to expression, minus the Spearman correlation of the change of distance with the change of expression

Runs without the dropout.

| pair | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF | with the dropout, CTCF and with RNAPII |
|---|---|---|---|---|---|
| GM12878 and H1ESC | 0.139 | 0.230 | 0.306 | 0.227 | 0.244 and 0.314 |
| GM12878 and HFFC6 | 0.144 | 0.243 | 0.279 | 0.243 | 0.251 and 0.296 |
| H1ESC and HFFC6 | 0.134 | 0.198 | 0.242 | 0.192 | 0.193 and 0.242 |

Median of 2,000 block bootstrap draws over the genes. With RNAPII loops against without, 10,000 draws:
+0.077, +0.037 and +0.046, p 1e-04, 0.001 and 1e-04. The uniform chain against the chain of loop anchors:
+0.003, +0.000 and +0.006, p 0.77, 0.96 and 0.57.

## Micro-C loops placed above moved loops

Runs without the dropout.

| line | loops | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF | with the dropout, CTCF and with RNAPII |
|---|---|---|---|---|---|---|
| GM12878 | 25,783 | 0.121 | 0.257 | 0.334 | 0.253 | 0.248 and 0.302 |
| H1ESC | 10,046 | 0.118 | 0.395 | 0.448 | 0.379 | 0.376 and 0.423 |
| HFFC6 | 19,192 | 0.102 | 0.442 | 0.525 | 0.434 | 0.418 and 0.488 |

Loops that an RNAPII loop supports and no CTCF loop does: 0.136 to 0.652 of 2,130, 0.189 to 0.666 of 586 and
0.238 to 0.702 of 1,318, CTCF loops to CTCF and RNAPII loops. With the dropout the same loops were at 0.538,
0.529 and 0.587 with RNAPII loops. Loops that a CTCF loop supports: 0.741 to 0.791, 0.762 to 0.812 and 0.747 to
0.819. Every rise has p under 0.001 over the 22 chromosomes. The uniform chain against the chain of loop
anchors over all loops: p 0.09, 0.03 and 0.25, the uniform chain higher on H1ESC.

## Not redone

The nine people of the trios were modelled on the chain of loop anchors in the pass of 2026-11-19 and are not
part of this pass. Figure 2 has no panel g. The tools comparison is `design/tools-comparison.md`.
