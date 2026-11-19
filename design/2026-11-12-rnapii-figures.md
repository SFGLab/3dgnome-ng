# The RNAPII figures, redesigned

The first main figure, `rnapii_main` of 2026-11-12, put four arms, cell lines and line pairs on
shared axes, labelled with letters of our own making and an expression axis where lower meant
better. A reader could not find the two points the work makes. This document replaces it with
two figures, one per point, and moves the rest to the supplement. The numbers are those of
`2026-11-11-rnapii-release.md`, Results.

## The two sentences

1. gnome3d matches chromatin structure better than cudaMMC, and it reads data cudaMMC cannot.
2. Adding RNAPII loops helps the functional reading of the models, not necessarily the
   structure itself.

Figure 1 argues the first, Figure 2 the second. A panel that argues neither goes to the
supplement.

## Rules for both figures

- No abbreviations of our own making. The arms are named "original models (cudaMMC)",
  "gnome3d, CTCF loops" and "gnome3d, CTCF and RNAPII loops", and in the supplement "gnome3d,
  RNAPII ends merged into CTCF anchors". Metrics are named by what they measure, for example
  "correlation with Hi-C, stratum adjusted" rather than SCC, "loop pile-up" rather than APA,
  "link between change in 3D distance and change in expression" rather than r_total.
  Established names stay: Hi-C, Micro-C, CTCF, RNAPII, enhancer, promoter.
- Every axis reads up or right as better. The expression link is negative when it is strong,
  so it is plotted with its sign flipped and the axis says so.
- Distributions are shown as joy plots, ridges stacked by line, where there is a distribution
  to show: distances over genes or pairs, bootstrap draws. Single values with few points are
  dumbbells or paired lines, never bars.
- One colour per arm in both figures: original grey, CTCF blue, CTCF and RNAPII orange. Every
  mark is also identifiable without colour, by position, label or legend.
- HFFc6 is labelled as having its RNAPII loops from BJ.

## Figure 1: structure, and the new data

- **a. Inputs.** A schematic: CTCF loops and RNAPII loops on a genome track, the Hi-C contacts
  under them, and optionally structural variants, going into one engine; out comes an ensemble
  drawn as smooth tubes. It states which inputs cudaMMC took and which are new.
- **b. Contact maps.** One 5 Mb window of chr1 as three triangles on one colour scale: the Hi-C
  map, the original models' simulated map and gnome3d's (CTCF loops).
- **c. Agreement with contact maps.** Dumbbells per cell line from the original models to
  gnome3d, CTCF loops. Two columns: correlation with Hi-C at 25 kb, correlation with Micro-C at
  5 kb. GM12878 has no Micro-C and says so in its row.
- **d. What only RNAPII input makes possible.** Joy plot per cell line of the 3D distance
  between an enhancer and a promoter that are joined by an RNAPII loop and by no CTCF loop. Two
  ridges per line, gnome3d with CTCF loops and with CTCF and RNAPII loops, and a grey control
  ridge: enhancer to promoter pairs at the same genomic separation joined by no loop, in the
  CTCF and RNAPII models. The models are the ten chr1 conformations per arm on the workstation
  under `~/xc_readout`.

## Figure 2: function, not structure

- **e. Link to expression.** Joy plot per line pair of the bootstrap distribution of the link
  between the change in enhancer to promoter distance and the change in expression across the
  two lines, CTCF loops against CTCF and RNAPII loops, sign flipped. The overlap of the ridges is
  the honest reading on one chromosome and stays visible.
- **f. Agreement with contact maps.** The dumbbells of c, now from CTCF loops to CTCF and RNAPII
  loops, on the same axes, so the small drop reads against the gain over cudaMMC in c.
- **g. Nine people, each against their own Hi-C.** Paired lines per person from CTCF loops to
  CTCF and RNAPII loops, two columns: loop pile-up at RNAPII loops relative to the person's map,
  up in 9 of 9; correlation with the person's whole map, down slightly.
- **h. One locus.** A gene whose enhancer comes close to its promoter only with RNAPII loops:
  two renders, CTCF loops and CTCF and RNAPII loops, smooth chain, promoter and enhancer as boxes
  around their stretch of the chain, the rendering of the website viewer.

## Supplement

- The merged anchors arm against the other three, on the readouts of c, e, f.
- The grid of every dataset by every metric, `rnapii_grid`, relabelled by the rules above.
- The loop pile-ups as images per line and arm.
- The compartment saddle and the trio table.

## Scope

chr1 now, ten conformations per arm. The figure script reads its inputs from a manifest, so the
autosomes, running since 2026-11-12, enter by a new manifest when they finish. Arm E exists on
the autosomes only where the enhancer3d models do.

## Where the inputs are

On the workstation under `~/xc_readout`. Models: `out/xc/<LINE>_<ARM>` for every line and arm,
the cudaMMC arm included, with `_60mb` slices beside them. The enhancer3d package is
`e3dx/enhancer3d`, the code the expression readout ran.

- d takes enhancers from `annotations.enhancer_set(line, "atlas")`, the set the expression
  readout used, and judges a pair against loops the way `annotations.annotate_pairs` does,
  each element within 2.5 kb of a loop's anchor. "Joined by no CTCF loop" uses the same rule on
  the line's CTCF loops. The loops are the run's own inputs, `loops/` beside the models. HFFc6
  stays in with its BJ label. If a line has too few pairs on chr1 for a ridge, its row says the
  count and shows points instead.
- e is the total correlation, the statistic with the published intervals, resampled the way
  `stats.boot_r` does: a block bootstrap over the `cluster` column of the per gene tables
  `readouts/e3d_<ARM>/genes_<pair>.tsv`, 2,000 draws. The draws are computed from those tables;
  the CLI is not rerun. A recomputed point estimate must reproduce `summary.tsv` before the draws
  are trusted.
- c and f take Hi-C from the battery JSON and Micro-C from the Micro-C JSON under
  `design/figures/rnapii-2026-11-12/readouts/`, original models included.
- b is computed from the models: contacts within the battery's contact distance, summed over
  the ten conformations, binned at 25 kb, beside the deep Hi-C map's same window.
- h renders one conformation per arm, the medoid of the ten by RMSD after alignment, with the
  smooth chain of `gnome3d/geometry.py::smooth_chain` and the element boxes of
  `web/gnome3d_web/viewer.py`, drawn with matplotlib for print.
- a is a drawing only. Structural variants appear in it as an input the engine takes, with no
  data behind the panel.

## Left to the builder

- The 5 Mb window of b and the locus of h, chosen from `figure_rnapii.py --loci` candidates and
  named in the figure legend.
- The exact pairing rule of d: an enhancer and a promoter (TSS plus or minus 1 kb, as on the
  website) each overlapping an opposite end of one loop; distance taken between the nearest
  beads within 5 kb, mean over conformations; the separation matching of the control.
- The number of bootstrap draws in e and the gene resampling, matching the enhancer3d CLI's.
- Figure size and type, for a journal page.

## Results on chromosomes 1 to 22, 2026-11-19

The figures are in `design/figures/rnapii-2026-11-19/genome/figures/` and every number below is
in `design/figures/rnapii-2026-11-19/genome/readouts/`. Four model sets a cell line, ten
structures a chromosome: the cudaMMC models, 3D-GNOMEᴺᴳ with CTCF loops, with CTCF and RNAPII
loops, and with RNAPII ends merged into CTCF anchors. Tests are two sided. Figures 1 and 2 draw
a comparison only under p 0.05. The supplement draws every comparison and marks the ones over
it ns.

**3D-GNOMEᴺᴳ against cudaMMC, figure 1c.** Stratum adjusted correlation of the summed contact
map with Hi-C at 25 kb: 0.19 to 0.53 for GM12878, 0.13 to 0.45 for H1 and 0.15 to 0.47 for
HFFc6. Share of Micro-C loops with the two ends brought together, above the same loops shifted
away: 0.12 to 0.23, 0.12 to 0.35 and 0.10 to 0.39. Every one has p under 0.001 in a signed rank
test over the 22 chromosomes.

**RNAPII loops bring enhancers to their promoters, figure 1d.** For the pairs an RNAPII loop
joins and no CTCF loop does, the median distance falls from 4.3 to 3.5 bead units on GM12878,
22,586 pairs, from 4.0 to 3.1 on H1, 31,537 pairs, and from 4.6 to 3.8 on HFFc6, 15,878 pairs.
Each has p under 0.001 against the same pairs without the loops and against the reference pairs.

**The link to expression is stronger with RNAPII loops, figure 2e.** Minus the Spearman
correlation between the change in enhancer to promoter distance and the change in expression:
0.19 to 0.24 for GM12878 and H1, p 0.002, 0.23 to 0.27 for GM12878 and HFFc6, p under 0.001,
0.17 to 0.21 for H1 and HFFc6, p 0.001. On chr1 alone no pair was significant.

**RNAPII loops recover Micro-C loops, figure 2f.** Share of loops with the two ends brought
together, above shifted loops: 0.23 to 0.31, 0.35 to 0.40 and 0.39 to 0.47, p under 0.001 on
every line. For the Micro-C loops an RNAPII loop supports and no CTCF loop does: 0.13 to 0.66 of
2,130 loops, 0.18 to 0.58 of 586 and 0.23 to 0.64 of 1,318.

**RNAPII loops help Hi-C agreement under 100 kb and cost it from 100 to 500 kb, supplement e.**
Stratum adjusted correlation with Hi-C at 10 kb by genomic separation, median of the 22
chromosomes, CTCF loops to CTCF and RNAPII loops:

| separation | GM12878 | H1 | HFFc6 |
|---|---|---|---|
| under 100 kb | 0.477 to 0.505, p 0.042 | 0.415 to 0.385, p 0.22 | 0.408 to 0.472, p 0.002 |
| 100 to 500 kb | 0.412 to 0.367, p under 0.001 | 0.345 to 0.302, p under 0.001 | 0.395 to 0.296, p under 0.001 |
| 0.5 to 2 Mb | 0.173 to 0.170, p 0.12 | 0.126 to 0.115, p under 0.001 | 0.209 to 0.144, p under 0.001 |

The gain is at the range the loops span and on two lines of three. The loss from 100 to 500 kb
is on every line and on 21 or 22 of 22 chromosomes each. The two cancel in the score to 2 Mb,
0.452 to 0.470, 0.390 to 0.357 and 0.406 to 0.436 with p 0.14, 0.054 and 0.008, and in the
whole map score at 25 kb, 0.53 to 0.52, 0.45 to 0.42 and 0.47 to 0.47 with p 0.35, 0.028 and
0.50. This is a cost of the RNAPII loops as they enter the model now. Its mechanism is not
traced. The set with RNAPII ends merged into CTCF anchors loses less from 100 to 500 kb, 0.412
to 0.406, 0.345 to 0.322 and 0.395 to 0.354, and is higher than CTCF loops alone to 2 Mb on
GM12878, 0.452 to 0.501, and HFFc6, 0.406 to 0.449, both p under 0.001.

**Nine people, figure 2g.** The models reach 0.34 to 0.41 of the contact enrichment each
person's own Hi-C has at that person's RNAPII loops with CTCF loops alone and 0.51 to 0.63 with
RNAPII loops, higher in 9 of 9, exact signed rank p 0.004.
