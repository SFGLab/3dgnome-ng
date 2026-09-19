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

Taken: RNAPII loops enter as their own factor, never merged into CTCF's counts, and the
depth matched high quality RNAPII set is the input, the same rule as CTCF. Open, for the
user: depth matched or merged CTCF loops for the rerun; whether GM12878 with the public RNAPII
ChIA-PET is the accepted test bed, since the trios have no independent Hi-C; and whether B is
worth its arm on the first pass or waits for A's result.
