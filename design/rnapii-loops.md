# RNAPII ChIA-PET in the model, plan of 2026-09-19

Request: recompute the trio models on ChIA-PET CTCF, then on CTCF together with RNAPII loops,
and use the RNAPII data in the expression work. Status: planned, nothing built.

## What the trios were built on

The trio models already use the ChIA-PET CTCF of the shared Drive folder
(`1e-hXljTx--kd_AHADU7AgM9AAsWcVTn3`): `playground/trio/trio_fetch.py` takes the depth
matched high quality PET3+ loops from `downsampling/<family>/<sample>_CTCF/`, whose anchors
both overlap a CTCF peak of the family, the CTCF peaks, and the CTCF ChIA-PET `.hic` for the
singletons. There was no HiChIP in the pipeline. The CTCF rerun is therefore the rerun on the
production of 2026-09-19, already staged in `slurm/ensemble/trio_ensemble.sh` with
`OUT=out/trio_prod`. The one open choice is the loop set, depth matched as before or the
merged full depth `Loops/<family>/CTCF/<sample>/merged/`, and the depth matched set is the
one the family comparisons rest on.

The RNAPII data is in the same folder in the same shape, per sample: merged PET3+ loops under
`Loops/<family>/RNAPOL2/<sample>/merged/`, a depth matched set and a high quality set whose
anchors overlap RNAPII peaks under `downsampling/<family>/<sample>_RNAP2/`, phased sets, and
the peaks under `Peaks/RNAPOL2/`. `trio_fetch.py` already takes `--factor RNAPOL2`.

## What RNAPII loops are, and what the literature models

CTCF loops are cohesin extrusion held at convergent CTCF sites, which is what the arcs stage
and the orientation term encode. RNAPII ChIA-PET loops are contacts among active promoters
and enhancers held by the transcription machinery. Four results bear on how to put them in.

- Tang et al., "CTCF-mediated human 3D genome architecture reveals chromatin topology for
  transcription", Cell 2015, <https://doi.org/10.1016/j.cell.2015.11.024>, the paper 3D-GNOME
  came out of. CTCF and cohesin anchors are the structural foci, and RNAPII interacts within
  those structures, drawing cell type specific genes toward the CTCF foci. RNAPII loops sit
  inside CTCF loops far more often than chance, so a first check on the trio data is the
  nesting fraction. The reference implementation carries the leftover: `[data] factors` and
  `InteractionArcs.cpp:98-141`, which sorts arcs by factor and, for an anchor pair carried by
  more than one factor, writes a summary arc with `eff_score` zero. Our loader reads one file
  and every arc is factor 0.
- Brackley, Taylor, Papantonis, Cook and Marenduzzo, "Nonspecific bridging-induced attraction
  drives clustering of DNA-binding proteins and genome organization", PNAS 2013,
  <https://doi.org/10.1073/pnas.1302950110>. A multivalent protein that binds chromatin at
  many sites brings those sites together without any specific pairwise contact, and
  polymerase and its factors are such a protein. Transcription factories follow from it. The
  same group's HiP-HoP model, Buckle, Brackley, Boyle, Marenduzzo and Gilbert, "Polymer
  simulations of heteromorphic chromatin predict the 3D folding of complex genomic loci",
  Molecular Cell 2018, <https://doi.org/10.1016/j.molcel.2018.09.016>, puts extrusion, a
  bridging attraction among active sites and a heteromorphic fibre together and reproduces
  Capture-C at single loci. A 2025 follow up, "Cluster size determines morphology of
  transcription factories in human cells", eLife, <https://elifesciences.org/articles/103955>,
  is the multicolour form.
- Banigan et al., "Transcription shapes 3D chromatin organization by interacting with loop
  extrusion", PNAS 2023, <https://doi.org/10.1073/pnas.2210480120>. Transcribing polymerase
  is a moving barrier that relocates cohesin, which gives the contact patterns around active
  genes. Not something this model can encode, since it has no extrusion, only its result.
- Cho et al., "Mediator and RNA polymerase II clusters associate in transcription-dependent
  condensates", Science 2018, <https://doi.org/10.1126/science.aar4199>. RNAPII clusters are
  transient, seconds, where CTCF loops are minutes. A ChIA-PET RNAPII loop is therefore a
  population average of a weaker and more dynamic contact than a CTCF loop with the same PET
  count, which argues for a weaker spring.

So there are two encodings with a literature behind them, and they are not exclusive.

**A. RNAPII loops as springs of a second class.** Data driven, the ChIA-PET loops as arcs
between the anchors holding their ends, with no orientation term, PET counts normalised by
their own span fit since the libraries differ in depth, and an own spring weight, expected
below CTCF's. Their anchors become beads, so promoters and enhancers are explicit and the
enhancer3d distances read off real beads. Blocks come from the union of both loop sets. This
is what the reference's `factors` was for.

**B. A bridging attraction among RNAPII anchors.** Physics driven, the Brackley term: every
RNAPII bound anchor attracts every other within a radius, weakly, so active sites cluster
into factories without being told which pairs touch. The compartment affinity term already in
the kernels, `use_compartments`, is a pairwise attraction within a radius between beads of
one type and can carry it with two types, RNAPII bound and the rest, at its own weight. It
adds clustering that A cannot, the contacts the library did not sample, at the risk of the
same compaction every attractive term has shown here.

**C. Both**, springs for the observed loops and a weak bridging among all RNAPII anchors.

## Plan

1. **Loader and settings.** `[data] clusters` takes two files and `factors` names them, as
   the reference's format did; each arc carries its factor; the arc strength fit and the
   spring weight run per factor, `[springs] arc_weight_<factor>` with CTCF at the current
   value; the orientation term reads only CTCF anchors, which is already what an anchor with
   no motif does. Anchors are the union. One to two days.
2. **Validate where there is Hi-C.** GM12878: deep Hi-C, CTCF ChIA-PET in the repo, and the
   public RNAPII ChIA-PET of Tang et al. 2015, GEO GSE72816, on hg19 and lifted. Arms on
   chr1:1-60 Mb, three structures each: CTCF only, CTCF with RNAPII anchors as beads and no
   springs, A at CTCF's weight and at a third of it, B at the compartment term's weight, C.
   Battery and saddle for structure, then the chr1 enhancer to promoter against expression
   test of `enhancer3d/playground/chr1_ep_distance_expression_go.py`, which is the measure an
   RNAPII term should move. One day after the code.
3. **Trios.** Two arms, nine samples, ten conformations, chr1 first then the genome: CTCF on
   production, and CTCF with RNAPII in the form step 2 picked. Then the trio enhancer3d
   recipe on both, `enhancer3d/playground/trio_poly_e3d.sh`, with the lab's RNAPII gene body
   signal, 4,266 genes at Spearman over 0.5 with expression, as the comparison feature. With
   prefetch and the grid a genome arm is about 600 GPU hours on eden against 2,800 before.

## Decisions taken and open

Taken, 2026-09-19: encoding A. RNAPII loops enter as their own factor, never merged into
CTCF's counts, the depth matched high quality RNAPII set is the input, the same rule as CTCF,
and the CTCF rerun stays on the depth matched set. GM12878 with the public RNAPII ChIA-PET is
the test bed. B waits for A's result. The first pass carries no per factor spring weight: the
arms are CTCF only, RNAPII anchors as beads without springs, and RNAPII springs at CTCF's
weight; a weight comes if the springs move the expression test and cost structure.

## First pass on GM12878, 2026-09-20

Whole chr1, five structures per arm, the battery on the 1-60 Mb slice and the chr1 enhancer
to promoter against expression test of enhancer3d on 854 genes. The RNAPII set is Tang et al.
2015 lifted to hg38, 96,037 loops of three or more PETs, 57,258 anchors of 1 kb added beside
the CTCF set; 37 percent of RNAPII loop ends already sit on CTCF anchors. On chr1 the joint
solve grew from 9,195 anchors in 52 blocks to 15,976 in 92.

| arm | Pearson | SCC | MultiMM | exponent | Rg | wb-aa | wb-sa | xb | saddle | eig r | expression Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CTCF only | 0.309 | 0.364 | 0.682 | 0.385 | 25.1 | 2.7 | 91 | 25 | 1.15 | 0.12 | -0.178 |
| CTCF + RNAPII at CTCF's strength | 0.278 | 0.280 | 0.587 | 0.386 | 24.0 | 9.1 | 15 | 19 | 0.96 | 0.25 | -0.234 |

The springs move the expression test the right way and double the eigenvector correlation,
and they cost SCC 0.08, MultiMM 0.10 and triple the anchor overlaps. A loop read at CTCF's
strength is pulled to a bead or two like a cohesin loop, and RNAPII's contacts are not held
that way. Next: `[springs] factor_strength`, a multiplier on the factor's loop strength in the
law, at 0.3, 0.1 and 0, the last holding RNAPII loops at the background so its anchors are
beads with no pull, the null arm the first pass could not run.

Decision criterion, set by the user 2026-09-20: the RNAPII arm exists to study expression
and the behaviours around active genes, so the expression test is the objective and the
Hi-C battery is the cost it is read against, not a gate. A strength is chosen for the trios by
how much expression signal it keeps per unit of structure it costs; full strength stays a
candidate if the gain is monotone in it. The CTCF arm keeps the battery as its gate as before.

## The strength sweep, 2026-09-20

Same chr1, three structures per strength arm, the two first pass arms at five. Strength 0
holds every RNAPII loop at the background, so its anchors are beads with no pull.

| RNAPII strength | n | Pearson | SCC | MultiMM | Rg | wb-aa | wb-sa | xb | saddle | eig r | expression Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CTCF only | 5 | 0.309 | 0.364 | 0.682 | 25.1 | 2.7 | 91 | 25 | 1.15 | 0.12 | -0.178 |
| 0 | 3 | 0.263 | 0.259 | 0.538 | 24.8 | 7.0 | 11 | 16 | 0.82 | 0.22 | -0.149 |
| 0.1 | 3 | 0.267 | 0.260 | 0.547 | 24.5 | 7.6 | 12 | 17 | 0.83 | 0.13 | -0.188 |
| 0.3 | 3 | 0.270 | 0.265 | 0.552 | 24.3 | 8.0 | 13 | 19 | 0.90 | 0.13 | -0.224 |
| 1 | 5 | 0.278 | 0.280 | 0.587 | 24.0 | 9.1 | 15 | 19 | 0.96 | 0.25 | -0.234 |

Two things separate cleanly. The Hi-C cost is the anchor set's, not the pull's: at strength 0
SCC, MultiMM, Pearson and the saddle are already at their lowest and the anchor overlaps are
already at 7 per thousand, and every one of them recovers a little as the strength rises. The
expression signal is the pull's: it is monotone in the strength from -0.149, worse than CTCF
alone, to -0.234 at full strength. There is no trade to make between the two, so the trios
run RNAPII at full strength, `[springs] factor_strength = 1,1`.

What the anchor set costs is a separate question, and it is the one to take up if the RNAPII
arm needs its Hi-C back: 57,258 anchors of 1 kb, dense around promoters, take the joint solve
from 52 blocks to 92 on chr1 and put anchors within a bead of each other where the CTCF set
had none. The remedy would be in how those ends become beads, wider anchors or ends merged
with a neighbouring CTCF anchor within some distance, not in the springs. Not built.

## The trio arm

Built 2026-09-20, on the laptop, `playground/trio/README.md` has the sequence. The same Drive
folder holds each sample's RNAPOL2 ChIA-PET, and its `downsampling/<family>/<sample>_RNAP2/`
folder holds a filtered `wyniki_*RNAP*.BE3` set, the counterpart of the CTCF arm's high quality
set. `trio_fetch.py --factor RNAPOL2` fetches it under a `_rnapol2` tag beside the CTCF files,
`trio_prepare.py --factor RNAPOL2` writes `<S>_rnapol2_clusters_3+.bedpe` and the shared
anchor set `<S>_anchors_ctcf_rnapol2.bed` by the GM12878 rule, now in
`playground/rnapii/anchor_union.py` and byte identical to the file the GM12878 script wrote,
and `trio_configs.py --factor RNAPOL2` writes `<s>_trio_rnapol2.ini` at full strength. The
array takes `CONFIG_TAG=_trio_rnapol2 OUT=out/trio_rnapol2` and its guard now checks every
cluster file, so the second file failing to travel cannot silently run the CTCF arm.

| sample | RNAPOL2 loops | ends on a CTCF anchor | new anchors | shared anchors | on own RNAPOL2 peak |
|---|---|---|---|---|---|
| HG00512 | 155,906 | 54% | 74,878 | 335,270 | 76% |
| HG00513 | 82,771 | 52% | 49,038 | 305,142 | 77% |
| HG00514 | 15,708 | 50% | 12,548 | 271,288 | 70% |
| HG00731 | 49,245 | 52% | 32,609 | 363,045 | 75% |
| HG00732 | 37,201 | 58% | 19,872 | 353,705 | 83% |
| HG00733 | 101,829 | 48% | 58,556 | 387,872 | 75% |
| GM19239 | 193,929 | 46% | 98,390 | 299,855 | 78% |
| GM19238 | 120,490 | 48% | 70,670 | 270,801 | 74% |
| GM19240 | 158,862 | 50% | 72,587 | 272,437 | 81% |

The RNAPOL2 depth was not matched within families at first, and the table above is that
state. The providers' filtered sets were uneven by ten times inside CHS, and the cause was
their draw, not the libraries: HG00514's RNAPOL2 library is the deepest of its family by four
times. The user's decision, 2026-09-20: resample from the rawest data the folder holds and
depend on no one else's draw, for both factors, and give a family one RNAPOL2 bead set.

## Depth resampled from the full libraries, 2026-09-20

`playground/trio/trio_resample.py`, the sequence and the table in `playground/trio/README.md`.
The merged PET 1 and up cluster files are the input. The providers' rule was recovered from
their sets and reproduces them at 100 percent: span at most 1 Mb, anchors on the family's
peak union, both for CTCF and at least one for RNAPOL2. Depth is the intra chromosomal PET
total and every cluster's count is thinned binomially to the family minimum, which is the
cluster level image of drawing reads. Both arms are rebuilt on it, so the CTCF trio arm is no
longer the set the 2026-09-09 runs were made on; those runs stay under their own trees.

Family wide RNAPOL2 anchors, option 2 of the discussion: `trio_prepare.py --factor RNAPOL2`
takes the RNAPOL2 loop ends of all three members, so a family shares its RNAPOL2 beads up to
each member's own CTCF anchors, which absorb the ends that fall on them, and members differ in
which loops pull. This follows from the sweep, where the anchor set carried the Hi-C cost and
the pull carried the expression signal.

What equal depth leaves: HG00514 keeps a fifth of its RNAPOL2 PETs and yields 34,034 loops
against its father's 194,853 at the same depth, because its library puts fewer of its PETs
into clusters on peaks. That is a property of the library and is left visible rather than
hidden by matching loop counts. The anchor totals grew, 3.4 M CTCF anchors over the nine
against 2.4 M on the providers' sets and 4.6 M with RNAPOL2, so the runs cost accordingly.

## The trios on chr1, 2026-09-20

Arrays 1808796, CTCF only in `out/trio_prod`, and 1808803, CTCF with RNAPOL2 in
`out/trio_rnapol2`, on eden: chr1, ten conformations, nine samples, both on the resampled
inputs, both with the arcs solver on the CPU at 200 iterations. The checkout moved to the
device solver while the second array ran and the running tasks kept their settings, every
logged solve says 200. Wall per conformation on an A100 from the task times, setup included,
8.3 minutes for the CTCF arm at 151 thousand beads and 13.3 with RNAPOL2 at 172 thousand, so
a genome arm is about 150 and 240 GPU hours.

The enhancer3d recipe is `enhancer3d/playground/trio_arms_e3d.sh`, the chr1 arm of
`trio_poly_e3d.sh` with the population contrast left out, since at chr1 it keeps under ten
genes. Each arm's control reads the anchors that arm was built on, `TRIO_ANCHORS` and
`TRIO_ANCHOR_FILE` in `trio_control.py`. Outputs `playground/trio_prod` and
`playground/trio_rnapol2` under `~/enhancer3d` on the workstation, the tables and figures
copied to `~/Desktop/enhancer3d/playground/` under the same names, and
`trio_arms_compare.py` writes the table. The first column is the 2026-09-09 polymer run's
chr1 arm on the providers' inputs, also ten conformations, on the same genes.

| statistic, chr1 | providers' inputs, 2026-09-09 | CTCF, resampled | CTCF with RNAPOL2 |
|---|---|---|---|
| genes per sample with a model and a count | 2,268 to 2,467 | same | same |
| rho(3D distance, expression), mean of nine | -0.286 | -0.333 | -0.362 |
| rho(linear distance, expression), same genes | -0.365 | -0.365 | -0.365 |
| partial, 3D against expression controlling linear | -0.098 (sd 0.015) | -0.134 (sd 0.025) | -0.177 (sd 0.029) |
| family separation, 3D, within minus between median rho | +0.044 (0.648 / 0.605), p 0.013 | +0.042 (0.730 / 0.688), p 0.004 | +0.013 (0.726 / 0.713), p 0.020 |
| family separation, anchor input | +0.080 | +0.104 | +0.160 |
| 3D separation over the anchors' | 0.55 | 0.41 | 0.08 |
| anchors on chr1, median TSS to nearest anchor | | 32 to 54 thousand, 10 to 14 kb | 44 to 64 thousand, 1.7 to 2.2 kb |
| inheritance ratio, child over unrelated against mid parent, CHS PUR YRI | 1.07 0.83 0.97 | 0.94 0.92 0.83 | 1.09 0.92 0.88 |
| rho(structural deviation, expression deviation) | +0.03 -0.01 +0.02 | -0.01 -0.04 -0.01 | -0.02 -0.02 -0.03 |

Three readings.

The RNAPOL2 loops carry expression signal in the trios as they did on GM12878. The partial
deepens from -0.134 to -0.177 and the raw 3D correlation reaches the linear one, -0.362
against -0.365, on eight of nine individuals; the ninth is HG00514, the sample whose library
yields a fifth of its family's loops at equal depth, level at -0.131 against -0.136. The CTCF
arm on the resampled inputs is itself deeper than the providers' inputs gave, -0.134 against
-0.098 on the same genes, so the resampling was worth its own step.

The family signal goes the other way. The CTCF arm keeps it, +0.042 at p 0.004 and 0.41 of
what its anchors carry. The RNAPOL2 arm's structures are alike across all nine, related or
not, 0.726 within against 0.713 between where the CTCF arm has 0.730 against 0.688, so its
separation is a third of the CTCF arm's and a twelfth of its own anchors'. Two things in the
construction bear on that row and it should not be read as a ranking of the loop sets. The
RNAPOL2 anchors are family wide, so within family anchor similarity is raised by design,
Jaccard 0.970 against 0.951, and the anchor row is not comparable across arms. And the
anchors sit at a median 2 kb from a TSS, so a gene's nearest enhancer distance is set by
beads every individual shares and the loops move it little. A per sample RNAPOL2 anchor set,
`--own-anchors`, is the arm that would separate the two.

Inheritance is null on both arms, as on every earlier run.

Not done: the lab's RNAPII gene body signal, 4,266 genes at Spearman over 0.5 with
expression, is a described result and not a table this side holds. The comparison needs
their per gene, per sample gene body counts.

### Per individual, and the jump statistic per contrast, 2026-09-20

Nothing averaged over people from here on. `enhancer3d/playground/trio_arms_jump.py`
writes three figures to `playground/trio_arms/` on the workstation, copied to
`~/Desktop/enhancer3d/playground/trio_arms/`: `trio_arms_individual.png`, every individual's
raw and partial correlation on the three arms; `trio_arms_jump.png`, the enhancer3d
statistic on every contrast; `trio_arms_jump_ridges.png`, the fold change ridges for the
child against parents contrasts. The table is `trio_arms_jump.csv`.

Per individual the RNAPOL2 arm deepens the partial on eight of nine and the CTCF arm on nine
of nine against the providers' run; HG00514 is level between the two new arms.

The enhancer3d statistic is the one the paper uses between cell lines: genes whose enhancer
distance jumps a tertile class between the two sides of a contrast, small to large or back,
and whose expression differs by at least |log2FC| 1, then the Spearman of fold change
against distance change and the median fold change of genes that came closer minus that of
genes that moved away, both expected negative and positive. Contrasts are each child
against its mid parent, the nine within family pairs and the 27 between family pairs. On
chr1 a contrast keeps 11 to 92 genes, so one contrast is noise and the reading is across
contrasts.

| |log2FC| at least 1 | providers' inputs | CTCF | CTCF with RNAPOL2 |
|---|---|---|---|
| between family pairs, 27: median rho, right sign, p under 0.05 with it | -0.04, 18, 0 | -0.17, 25, 6 | -0.10, 18, 3 |
| between family pairs, median closer minus farther log2FC | 0.0 | +1.5 | +0.2 |
| within family pairs, 9: median rho, right sign | +0.05, 2 | -0.11, 7 | -0.08, 6 |
| child against parents, 3: rho | 0.00, -0.09, -0.01 | +0.39, -0.48, -0.12 | -0.44, +0.09, -0.54 |

The providers' models have no jump signal at all. The CTCF arm has it on 25 of 27 between
family contrasts, six of them significant on their own, and 7 of 9 within. The RNAPOL2 arm
has it more strongly where it has it, the CHS and YRI children against their parents at
p 0.03 and 0.02 within GM19238 against GM19240, and reversed on the contrasts that involve
HG00512, the CHS father, three of them significantly positive. At |log2FC| 2 a contrast keeps
5 to 21 genes and the arms cannot be ranked. What would settle it is the genome, where a
contrast keeps ten times the genes.

### The transition idea within one person, 2026-09-20

enhancer3d's result between cell types is that large expression differences go with
proximity shifts, up to r about -0.6, while the genome wide relation is weak. Between the nine
individuals there are no large differences to condition on, so
`enhancer3d/playground/trio_individual_transitions.py` asks it within a person, three ways,
each per individual and never averaged, `trio_individual_transitions.png` and
`trio_individual_vs_panel.png` with their csvs beside them.

The one axis of large expression difference a lymphoblastoid individual carries is its cell
type's own programme, so the cell line DE tables, GM12878 against H1ESC and against HFFC6,
give each gene an LCL fold change, and the individual's own model gives the distance. The raw
Spearman of distance against that fold change sharpens with the threshold exactly as the
paper's does, on the RNAPOL2 arm from -0.27 on all genes to -0.52 on the 401 chr1 genes at
|log2FC| 4 against H1ESC, the CTCF arm -0.26 to -0.48, and the individual's own expression on
those genes reaches -0.50. The partial controlling linear distance to the nearest enhancer does
not move, -0.05 to -0.18 at every threshold on every arm. The sharpening is carried by linear
proximity: the LCL programme's genes sit beside the LCL enhancer atlas on the chromosome,
which is partly circular since the atlas is GM12878's, and the models place them close. So
the claim ports as a raw correlation and not as a claim about the structure beyond the
sequence.

The two tails of a person's own expression sharpen the partial a little, keeping 40 percent of
genes takes the RNAPOL2 arm from -0.13 to -0.21 across the nine to -0.12 to -0.26 and the CTCF
arm from -0.10 to -0.16 to -0.09 to -0.20, and 10 percent, 300 genes, is noise. Each
individual against the median of the other eight keeps 4 to 19 genes on chr1 and says nothing.

What would carry the idea properly within a person is a transition the person has on both
sides: the two haplotypes. The folder holds phased loops for every sample, `_Maternal`,
`_Paternal` and `_Crossed`, and allele specific expression from the same RNA-seq would give
each gene a within nucleus fold change with the trans environment held fixed. Haplotype models
are not built.
