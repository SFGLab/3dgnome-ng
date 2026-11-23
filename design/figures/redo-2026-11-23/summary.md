# The results on the uniform chain, 2026-11-23

Chromosomes 1 to 22 of GM12878, H1ESC and HFFC6, ten structures a chromosome, built on the cluster by
`slurm/ensemble/crosscell_arms.sh` with `ARMS="C UC UR"` at 2ff5766 and read out on a workstation by
`playground/rnapii/redo_pull.sh` and `redo_readouts.sh`. The uniform arms read half a of the deep map of a line.
Every score against Hi-C is against half b, for every arm. The tables per chromosome, the inputs of the figures
and the tables of the tools are in `data/figures/redo-2026-11-23/`, outside git. The figures are drawn by
`playground/rnapii/figures_two.py` with `ARM_NAMES` naming the fourth arm.

Arms: the cudaMMC models of 3D-GNOME 3.0, the uniform chain with CTCF loops, the uniform chain with CTCF and
RNAPII loops, the chain of loop anchors with CTCF loops.

## Agreement with Hi-C, raw SCC of the pooled map at 25 kb, mean over the chromosomes by length

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

## Enhancer to promoter distance, pairs joined by an RNAPII loop and by no CTCF loop, median in bead units

| line | pairs | uniform, CTCF | uniform, CTCF and RNAPII | closer |
|---|---|---|---|---|
| GM12878 | 23,237 | 4.09 | 3.51 | 14 % |
| H1ESC | 33,360 | 3.98 | 3.58 | 10 % |
| HFFC6 | 16,431 | 4.43 | 3.96 | 11 % |

Signed rank over the pairs and Mann-Whitney against the reference pairs: p under 0.001 on every line. On the
chain of loop anchors the pass of 2026-11-19 had 19, 24 and 17 % closer. A structure of the uniform chain keeps
a loop with the probability q / (q + 1).

## Link to expression, minus the Spearman correlation of the change of distance with the change of expression

| pair | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF |
|---|---|---|---|---|
| GM12878 and H1ESC | 0.139 | 0.244 | 0.314 | 0.227 |
| GM12878 and HFFC6 | 0.144 | 0.251 | 0.296 | 0.243 |
| H1ESC and HFFC6 | 0.134 | 0.193 | 0.242 | 0.192 |

Median of 2,000 block bootstrap draws over the genes. With RNAPII loops against without, 10,000 draws:
+0.071, +0.045 and +0.050, p 1e-04 each. The uniform chain against the chain of loop anchors: +0.017, +0.008
and +0.001, p 0.15, 0.44 and 0.93.

## Micro-C loops placed above moved loops

| line | loops | cudaMMC | uniform, CTCF | uniform, CTCF and RNAPII | loop anchors, CTCF |
|---|---|---|---|---|---|
| GM12878 | 25,783 | 0.121 | 0.248 | 0.302 | 0.253 |
| H1ESC | 10,046 | 0.118 | 0.376 | 0.423 | 0.379 |
| HFFC6 | 19,192 | 0.102 | 0.418 | 0.488 | 0.434 |

Loops that an RNAPII loop supports and no CTCF loop does: 0.127 to 0.538 of 2,130, 0.160 to 0.529 of 586 and
0.210 to 0.587 of 1,318, CTCF loops to CTCF and RNAPII loops. Loops that a CTCF loop supports: 0.719 to 0.759,
0.748 to 0.792 and 0.720 to 0.803. Every rise has p under 0.001 over the 22 chromosomes. The uniform chain
against the chain of loop anchors over all loops: p 0.19, 0.97 and 0.001, the chain of loop anchors higher on
HFFC6.

## Not redone

The nine people of the trios were modelled on the chain of loop anchors in the pass of 2026-11-19 and are not
part of this pass. Figure 2 has no panel g. The tools comparison is `design/tools-comparison.md`.
