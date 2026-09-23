# Expression from structure

The question, 2026-09-20: can the 3D models explain more of a person's RNA-seq expression than
they do now, and what limits them. The setting is the one that stands after the RNAPII work
in `rnapii-loops.md`: the CTCF with RNAPOL2 trio arm, each of the nine people on their own
model and their own expression, nothing averaged over people and no fold change across cell
lines, since a person has one sample. chr1 at ten conformations until the genome arms run.
This note holds the numbers to beat, what has been ruled out, and the ideas in the order they
are to be tried, each with its status, so it is the tracker.

## Where it stands, 2026-09-22

Twenty ideas in, the answer to the question of 2026-09-20 is this. Per person on chr1 at ten
conformations, out of fold, a model of a person's expression reaches 0.55 from the gene alone,
0.65 with promoter CpG, 0.70 with the linear map of the person's active elements, 0.75 to 0.77
with the person's RNAPOL2 loops at the promoter, and the 3D ensemble adds -0.005 to +0.009 on
top of that. The number did not move under any change tried: the element set, atlas or the
person's own, which is adopted; the contact form; the polymer residual; the law's half
saturation; the target, level, on and off, or the residual after sequence; the arm, CTCF, CTCF
with RNAPOL2, or the same loops on the person's own independent Hi-C; the ensemble spread,
promoter hubs, gene looping, the person's compartments.

What the models carry is real. The nearest active element is -0.40 to -0.46 with expression in
every person and every arm, the arms agree per gene at 0.83 to 0.91, and the quintile, tail
and tertile pictures replicate nine times. On the genes where a person differs most from the
others, over one log2, the person's spatial deviation tracks the expression deviation at +0.15
to +0.19 on every arm and the person's compartments at +0.20, the enhancer3d claim ported to the
individual; out of fold on those genes the compartments add +0.07 beyond the loops and the
models +0.02 (idea 23). Putting the person's compartments into the energy changed none of it
(idea 24). The person specific part is biology: a public LCL eQTL map with the
nine's genotypes predicts it at +0.29, +0.35 on the large deviations, and the structural data
do not track that genetic part at all, so the loops and compartments carry a separate,
epigenomic part (idea 22). The person specific part exists in the data and is
small: the loops at the promoter carry +0.10 of it on the deviation statistic and the person's
compartments +0.08, both from data alone; the models keep +0.05 to +0.06; and the lab's counts
turned out to be the HGSVC libraries themselves, so no public replicate can say how much of
that deviation is biology. Structural variants on chr1 in nine people are too few and too small
to show, and the phased loops are 2 to 4 percent of a person's loops on single reads.

So on this design the 3D models explain no more of a person's RNA-seq than the loops at the
promoter and the linear map of active elements already explain, and the reason is not
processing, the law, the feature or the background but the design: nine people whose
expression agrees at 0.97, with loops that agree at 0.87 and structures that agree at 0.62.
What is left needs different data or a different run: haplotypes (7) with allele counts from
the RNA-seq now on the workstation, inter chromosomal neighbours (18) with a whole genome per
process and the trans contacts, the Enformer form of 15, and the genome arms for ten times the
genes. A second engine on the same loops, MultiMM, idea 21, lands a little under ours on one person and was stopped there. The engine's own two levers, a converged
solve (32) and a spring per loop by its strength (37), each left every expression number where it
was. The lab meeting of the same evening set the paper: the trio ChIA-PET paper for the Nature
Genetics 3D call, deadline the end of November 2026, with this line as its 3D section; the
sixth list below holds what it needs. Its first result, idea 41: the largest axis of the
person specific expression, a third of its variance, is the EBV latency program, EBNA-1's
level in the RNA-seq, a trans effect of the line's state that predicts a person's deviation
out of fold at +0.22 and that no cis modality can carry.

## Summary, 2026-09-21

What the numbers say, per person on chr1 at ten conformations, RNAPOL2 arm. The nearest
enhancer distance reaches 0.34 to 0.39 raw and the loops at the promoter alone 0.43 to 0.57.
Out of fold the 3D block adds nothing beyond the loops and the linear map. The design has no
person specific expression to explain, 0.97 agreement between people, and what person specific
part the loops carry, +0.10 on the deviation statistic, the models keep 0.02 to 0.06 of.

Tried and closed: enhancer load and hubs (1), contact frequency (2), polymer residual (3), the
person's own elements (4, adopted as the element set), a cross validated model (5), the
expression side (6), the ceiling in the forming data (8). Blocked on data: haplotypes (7).

What is left, in the order to try it:

1. Done and dropped, 2026-09-21: the saturation lever, idea 19. The law's target at the
   promoter relays the same +0.08 of the input's +0.10 at every half saturation value, so the
   loss is downstream of the law and no array was run.
2. Done and adopted, 2026-09-21: the person's own elements as the element table, idea 20.
   Raw up in every person on both arms, the person specific part doubled to +0.05..+0.06,
   the 3D part beyond the set's own linear proximity unchanged at -0.14..-0.17.
3. Done, 2026-09-22: the HGSVC Hi-C arm. An independent contact map reproduces the ChIA-PET
   map arm at 0.83 to 0.91 per gene and adds nothing beyond the loops.
4. Done, 2026-09-22, the data questions: expression reliability (10: the lab's counts are
   the HGSVC libraries themselves, two quantifications agree on the deviation at 0.91, a true
   replicate does not exist in public for six of the nine) and the structural variant pre test
   (9, negative: too few and too small on chr1 in nine people).
5. Done and dropped, 2026-09-21: gene looping (11), promoter hubs (12), ensemble spread (13),
   on and off before level (14), CTCF minus RNAPOL2 (17). None adds out of fold.
6. Done in their cheap forms, 2026-09-21: the sequence residual (15, CpG; Enformer open) and
   the person's compartments (16, nine of nine; in the energy, 24, no change).
7. The fourth list, 2026-09-22: the trace survives its null (25); the person's enhancer
   activity, the eQTL variant as the element, the loops at the variant add nothing (26 to 28);
   one model of the person specific part reaches +0.20 to +0.31 from data with nothing unique
   from 3D (29). Only 30, fifty conformations, is left of it, and the deferred four. Still open: the genome arms for inter
   chromosomal neighbours (18), and haplotypes (7) once the lab gives per haplotype PET counts
   and there is allele specific expression.

## Where it stands

Per person, chr1, genes the person expresses at all, 2,268 to 2,467. `rnapol2_individual.py`
and `trio_diag.py` in `enhancer3d/playground`, outputs under `playground/trio_rnapol2/` on the
workstation and `~/Desktop/enhancer3d/playground/trio_rnapol2/`.

| feature against the person's expression, Spearman, range over the nine | rho |
|---|---|
| mean distance to the nearest atlas enhancer, the current feature | -0.34 to -0.39 |
| the same with the silent genes kept | -0.36 to -0.44 |
| the same on protein coding genes only, about 1,650 | -0.38 to -0.44 |
| the same controlling linear distance to the nearest enhancer | -0.13 to -0.21 |
| mean or maximum distance over all enhancers | 0.00 |
| the input alone: RNAPOL2 loops with an anchor at the TSS | +0.43 to +0.57 |
| the input alone: RNAPOL2 PET count at the TSS | +0.42 to +0.55 |
| CTCF PET count at the TSS | +0.25 to +0.33 |
| the current feature controlling the RNAPOL2 PET at the TSS | -0.08 to -0.20 |

Three baselines every idea is judged against, per person: the input alone, the linear partial,
and the current feature. The number that answers the question is what a 3D feature adds beyond
the input and beyond linear proximity, out of fold where a fit is involved.

## What is ruled out

The colleague's hypothesis was the processing and the normalisation. Checked 2026-09-20:

- The nine RNA-seq samples agree with each other at Spearman 0.90 or better, no outlier.
- Counts are raw, so expression correlates with gene length at +0.35, but the distance does
  not, at 0.00 to -0.10, and the partial controlling length is the raw number.
- 46 percent of modelled chr1 genes have no count under their name, all of them non coding;
  2,004 of 2,056 protein coding genes are covered.
- Keeping the silent genes and restricting to protein coding each add a few hundredths.

So the processing is worth a few hundredths and is not the ceiling.

## The answer so far

2026-09-20, from idea 5. Out of fold, per person on chr1, a model of expression reaches 0.55 to 0.57 from the gene alone, 0.69 to 0.71 with the linear map of elements around it, 0.75 to 0.77 with the loops at its promoter, and 0.76 to 0.77 with everything the 3D ensemble gives on top. The models add nothing beyond their input and the linear map. Ideas 1 to 4 each moved the raw correlation and none moved this number. What would move it has to put information into the models that is not in the loops at the promoter: the haplotypes of idea 7, or a second data type in the energy.

## What limits amplification, 2026-09-21

The signal is there in every person and nothing tried amplified it, so three tests asked where
the limit sits: in the data, in the engine, or in the feature. `cross_person.py`,
`pooled_panel.py`, and `expression_model.py` on the CTCF arm.

**The data design has no person specific expression to explain.** The nine expression
profiles agree at Spearman 0.97 on the genes all nine express. Another person's model predicts
a person's expression as well as their own, 0.33 to 0.38 against 0.34 to 0.39 on the RNAPOL2
arm and the same on the CTCF arm, and the person specific part of the distance against the
person specific part of expression is 0.00 to 0.05. What the models read is the shared
lymphoblastoid map, which every person carries alike. That is a property of healthy trios and
not of the modelling: the cross cell type design has expression differences of many log units
to work with, and this design has none outside the haplotypes of idea 7.

**The engine relays its loops and adds nothing of its own, on both arms.** On the CTCF arm,
where the loops are not a transcription readout, the out of fold gain of the 3D block beyond
the loops at the promoter and the linear map is -0.01 to +0.01, the same as on the RNAPOL2
arm. Nothing emergent, transitive proximity, hubs or compartment level packing, reaches
expression. That is by construction: a pair with no loop sits at the polymer law, and the
contact background these runs carry comes from a Hi-C built from the same ChIA-PET, so it
holds no long range information the loops do not.

**The ensemble is noisy and that part is recoverable.** Models of different people agree at
0.62 to 0.64 where the expression agrees at 0.97. Averaging the nine people's models, ninety
conformations of one shared map, raises the correlation from 0.30 to 0.37 on the RNAPOL2 arm
and 0.28 to 0.34 on the CTCF arm on the common genes; averaging expression changes nothing.
Ten conformations undersample the map by about 0.06 of correlation.

**The feature saturates.** Corrected for the noise on both sides, the nearest enhancer
distance tops out at 0.39 on the RNAPOL2 arm and 0.35 on the CTCF arm. The rest of expression
is not in enhancer proximity as these models represent it, and ideas 1 to 4 showed no
transformation of that proximity does better.

So the order is: the design first, the ensemble second, the engine's information third, the
feature last. What would move each. A target with person specific variance, the allelic
ratio of idea 7 or an RNA-seq of the same cells under a stimulus, which is a data request.
Fifty conformations a person on chr1, `PER_TASK=50` on the same `OUT`, about 0.06 for the
price of forty conformations a person. Independent long range data in the energy, a real deep
Hi-C as the contact background instead of the ChIA-PET derived map, GM12878's 4DN map for all
nine as a shared prior, which is the one engine side lever that puts information into the
models the loops do not hold, measurable as the 3D gain beyond the input on one arm.

## The contact map, and an independent one, 2026-09-21

The trio runs' contact singletons come from `ChIA-PET_hg38_<S>_merged_allres.hic`, which the
lab built with juicer from the CTCF ChIA-PET read pairs of each sample for hicrep, its README
and `make_hic.sh` say so, and the copies in the Drive folder `1QkrLRi7Xc92z_bquS8QcY5nt-tQEIgnm`
are the files already fetched, same names and sizes. On chr1 a run loads 0.96 to 1.8 million
of these pairs a sample and fits its exponent on them, 0.34 to 0.42. They are the same
library as the loops, so the contact background and the segment heatmap hold nothing the loops
do not.

The Human Genome Structural Variation Consortium sequenced these nine lymphoblastoid lines
independently, and all of it is public under
`ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/hgsv_sv_discovery/`:

- Hi-C, Ren lab 2016, GRCh38, two biological replicates a sample, several hundred million
  read pairs a sample. BAMs under `working/20160822_HiC_bam_files/`, 18 to 47 GB each, 460 GB
  in all, mapping quality 10 and duplicates removed; fastqs at ENA PRJEB11418; normalised 40 kb
  dense matrices per chromosome under `working/20160817_HiC_contact_matrices/`, 128 MB for
  chr1.
- Haplotype resolved genotypes from that Hi-C, `working/20170302_hic_phase/`, one VCF a sample.
- Strand specific mRNA-seq, paired, HiSeq 2500, one library a sample, `illumina_rna.sequence.index`,
  fastqs at ENA ERP012633.

The last two are what idea 7 was blocked on: allele specific expression per gene follows from
the RNA-seq reads against the phased VCF, and the ChIA-PET reads could be phased on the same
VCF with per haplotype counts.

The first is the engine side lever, and 4DN has reprocessed that Hi-C: one released GRCh38
mcool a sample on the open data bucket, 3.8 to 6.1 GB, accessions 4DNFIM8SM3SD, 4DNFI66KARTU,
4DNFIEWJXIW4, 4DNFIUZJP1ED, 4DNFIFXDDDJ6, 4DNFIHGNUBH5, 4DNFIIRNP38T, 4DNFIO7M1D22 and
4DNFIM8KVPS6 for the nine in the order above, with compartment and insulation tracks beside
each. They go through the cell lines' own path, `slurm/ensemble/prep_singletons.py` at 25 kb
thinned to 5,000 contacts a megabase, into `<S>_hgsvc_25kb_singletons.bedpe`, and
`trio_configs.py --factor RNAPOL2 --singletons hgsvc` writes `<s>_trio_rnapol2_hgsvc.ini`,
which differs from the RNAPOL2 arm's config in the singletons line alone. The arm is chr1 at
ten conformations on the nine, `CONFIG_TAG=_trio_rnapol2_hgsvc OUT=out/trio_rnapol2_hgsvc`,
and the judge is idea 5's model on it, the 3D block's gain beyond the input. The mcools are
on the workstation under `/mnt/storagelinux/_hgsvc/mcool/`.

**Run and measured, 2026-09-22.** Eden job 1809232, nine samples at ten conformations, 5 to 7
minutes a conformation against 13 on the ChIA-PET derived map; the law measured 0.25 to 0.26 on
every person's Hi-C where the ChIA-PET map gave 0.40. Models on the workstation under
`/mnt/storagelinux/models_trio_rnapol2_hgsvc/`, enhancer3d with the person's own elements in
`playground/trio_rnapol2_hgsvc/`, `hgsvc_arm.sh`, the pairs window capped at 3 Mb since the
box held other jobs. Against the same arm on the ChIA-PET map, per person on chr1:

| statistic, own elements | ChIA-PET map | HGSVC Hi-C |
|---|---|---|
| raw rho(3D, expr) | -0.40 to -0.46 | -0.38 to -0.46 |
| partial beyond the set's linear distance, mean | -0.169 | -0.192 |
| person specific deviation, 1,903 genes | +0.062 | +0.052 |
| family separation, 3D | +0.030, p 0.005 | +0.025, p 0.014 |
| nearest distance, agreement between the two arms | | 0.83 to 0.91 |
| idea 5, the 3D block alone | 0.52 to 0.55 | 0.55 to 0.60 |
| idea 5, the 3D block beyond the linear map | -0.001 to +0.011 | +0.006 to +0.025 |
| idea 5, the 3D block beyond the loops at the promoter | -0.005 to +0.009 | -0.003 to +0.007 |

So an independent contact map changes little: the two arms agree per gene at 0.83 to 0.91,
the structures on the Hi-C are a shade more informative on their own and a shade deeper beyond
linear, and the gain beyond the loops at the promoter is zero on both. The background is
background.

## Why the ceiling is the feature

RNAPOL2 ChIA-PET is a polymerase occupancy readout, so its loop count at a promoter predicts
expression at 0.55 with no model at all. The model turns that input into one number, the mean
over conformations of the distance to the nearest enhancer of the GM12878 atlas, and that
number keeps 0.36. A model cannot out-explain its own input on a per promoter measure. What
it can add is geometry, which enhancers a promoter is brought together with and how often,
and the current feature reads almost none of that: every summary over more than the nearest
enhancer sits at zero.

## Ideas, in the order to try them

The judge for each is the table above, per person. Status is one of open, running, done and
adopted, done and dropped. A result line gets the numbers and the date.

| # | idea | what to build | status | result |
|---|---|---|---|---|
| 1 | Contact weighted enhancer load, the multi enhancer hub in one feature | Per gene the sum over enhancers of the atlas activity times a contact term, the ensemble frequency P(d < r) or exp(-d / d0), in the ABC model's form; the number of enhancers within r as the plain hub size. The pairwise mean distances are computed inside `genome_ep_distances.py` and only min, mean and max are kept, so the change is to keep the pairs. | done and dropped as a replacement, 2026-09-20 | `enhancer_load.py`, all 4,710 chr1 enhancers within 3 Mb of each TSS, 140 per gene, from the ten conformations. No form beats the nearest distance: raw rho over the nine, nearest 0.34 to 0.40; hub size within 3 units 0.31 to 0.37, within 2 units 0.28 to 0.35; exp(-d/d0) load 0.27 to 0.35 at d0 1 and worse at every other d0, the atlas score changes nothing, 1/(1+d)^3 0.27 to 0.35. Controlling linear distance the nearest keeps 0.12 to 0.22 and the best hub 0.11 to 0.19. What a hub adds beyond the nearest distance is +0.06 to +0.14 for the count within 3 to 5 units, so hub size goes into idea 5 as a second feature and not as a replacement. |
| 2 | Contact frequency instead of mean distance | A loop is on in some conformations and off in others and the mean blurs it. P(d < r) over the ensemble for the nearest enhancer and inside idea 1. Ten conformations is thin for it; the genome arms give more. | done and dropped, 2026-09-20 | `enhancer_load.py` at ten conformations. The hard frequency, the fraction of conformations with the nearest enhancer under r, peaks at r 2.5 units at 0.31 to 0.36 against 0.34 to 0.40 for the mean distance, and takes eleven values at ten conformations. The soft form settles the direction without that cap: the mean over conformations of exp(-d / d0) for the nearest enhancer is continuous, and it loses more the more contact like it is, 0.28 to 0.33 at d0 0.5, 0.31 to 0.36 at 1, 0.33 to 0.38 at 2, 0.33 to 0.39 at 4, climbing towards the mean distance as d0 grows and never past it. Controlling the mean distance every contact form is negative. More conformations would sharpen a form that already loses. Reopen only if the genome arms give fifty or more conformations for free, with `PER_TASK=50` on the same `OUT`, since the array skips members that exist. |
| 3 | Distance relative to the polymer expectation | The law gives the expected distance at any genomic separation, so observed over expected says closer than the chain would be, which is the loop signal itself. Replaces the linear partial, which approximates it crudely. | done and dropped, 2026-09-20 | `polymer_residual.py`, each person's own measured exponent, 0.34 to 0.42, s0 1 kb, and also the person's own model curve, the median distance per separation bin, which sits within 15 percent of the law over 30 kb to 3 Mb. Observed over expected for the nearest enhancer correlates at -0.03 to 0.16 with the law and 0.23 to 0.32 with the model's own curve, against 0.34 to 0.40 for the raw distance; the smallest ratio among the enhancers, the count under 0.5 or 0.7 of expectation and the summed pull are all under 0.21. Controlling the raw distance every residual adds nothing, -0.11 to +0.03. Expression follows how close the enhancer is, not how much closer than the chain would put it; dividing by the expectation removes the part that carries the signal. |
| 4 | The person's own regulatory elements | All nine use the GM12878 atlas. Each sample's RNAPOL2 peaks are fetched, and RNAPOL2 bound non promoter sites are that person's active elements; a hub of RNAPOL2 peaks within a 3D radius is the transcription factory view. Ideas 1 to 3 on that set. | done, adopted as the element set for idea 5, 2026-09-20 | `own_elements.py`, the same nearest distance feature against eight element sets per person, the atlas and sets from the person's RNAPOL2 peaks and loop anchors, under the current rule that a gene's own span is among its candidates and under the stricter rule that it is not. The element set matters and activity in the person is what matters in it: the atlas enhancers that overlap one of the person's RNAPOL2 peaks, a third to a half of them, give 0.39 to 0.46 against 0.34 to 0.40 for the whole atlas and add +0.18 to +0.23 beyond it, while the atlas enhancers no peak touches give 0.03 to 0.15 and subtract. The person's distal RNAPOL2 loop anchors give 0.36 to 0.42, the distal peaks 0.32 to 0.39, both adding +0.21 to +0.29 beyond the atlas; intergenic peaks alone 0.08 to 0.19. Every RNAPOL2 peak with promoters in, 0.44 to 0.54, is the gene's own polymerase and is not an element set. Two limits. The linear distance to the same set does nearly as well as the 3D distance in every set, 0.39 to 0.44 for the active atlas, so the gain is an active element near the gene and the 3D part beyond linear stays 0.07 to 0.20. And beyond the input, the RNAPOL2 PET at the TSS, no set beats the atlas, 0.13 to 0.22 against 0.15 to 0.24. With the gene's own span excluded every number falls, atlas 0.12 to 0.21, active atlas 0.17 to 0.24, distal anchors 0.24 to 0.31, and beyond the input all sit at zero: what the current feature reads is mostly an element inside or at the gene, which the loops have already placed. |
| 5 | A model, not one correlation | Per person, cross validated regression of log expression on the 3D features, the linear features, the promoter RNAPOL2 PET, CTCF, gene length and type; the out of fold gain from the 3D features is the answer to how much more. | done, 2026-09-20 | `expression_model.py`, gradient boosting, five folds by three repeats, per person on the genes the person expresses, out of fold Spearman of the prediction against expression, ranges over the nine. Gene alone, length, type and enhancer density, 0.55 to 0.57. With the linear distances to the four element sets 0.69 to 0.71. With the input at the TSS, RNAPOL2 PET, loop count, peak signal and CTCF PET, 0.75 to 0.77, R squared 0.59 to 0.62. With the 3D block on top, seventeen features, the nearest distances and hub counts to the four sets under both rules and the contact fraction, 0.76 to 0.77: the 3D block adds -0.01 to +0.01 beyond the input and +0.00 to +0.03 beyond the linear distances, while the input adds +0.05 to +0.07 beyond linear. The 3D block alone reaches 0.55 to 0.59, what the gene block alone reaches. A ridge on ranks gives the same picture 0.03 lower. This is the number the question asked for: on chr1 at ten conformations the models explain nothing of a person's expression that the loops at the promoter and the linear map do not already explain. |
| 6 | The expression side | Protein coding only, silent genes kept or on and off fitted separately from level, length normalisation for the regression. Worth a few hundredths, see above. | done, 2026-09-20 | Silent genes kept in the idea 5 model, 3,022 genes a person: every fit rises, gene alone 0.59 to 0.62, with linear 0.72 to 0.74, with the input 0.78 to 0.79 at R squared 0.66 to 0.68, with 3D 0.78 to 0.79, and the 3D block adds -0.00 to +0.01 beyond the input and +0.01 to +0.02 beyond linear. Gene length and type sit in the gene block of every fit already. The expression side moves every number by a few hundredths and the gain of 3D by none. `playground/trio_rnapol2/model_zeros/`. |
| 7 | Haplotypes, the within person transition | The folder holds phased loops for every sample. A maternal and a paternal model of one nucleus with allele specific expression from the same RNA-seq gives each gene a fold change with the trans environment fixed, the port of enhancer3d's between cell type result that stands. Needs haplotype models and allele specific counts, neither built. | open, blocked on data, inventory done 2026-09-20 | The folder's phased sets, `<S>_Maternal.txt`, `_Paternal.txt`, `_Crossed.txt` beside `_nonPhased.txt`, for both factors, `playground/phased_inventory.py`. On chr1 a person has 220 to 750 CTCF and 11 to 620 RNAPOL2 loops labelled maternal and as many paternal, two to four percent of the loops, and 0 to 13 loops in both files, so the labels are exclusive. They are also thin: the label column reads `M_NA` or `NA_M`, one anchor phased and the other not, at a median PET of 4 to 6, so a label is one or two phaseable reads and not a per haplotype count, and a loop on both chromosomes with one phased read reads as maternal. A maternal model would be the unphased set plus a few hundred such loops. Genes with a TSS on a maternal labelled anchor, 24 to 228 per person and factor, 8 to 139 protein coding, and as many paternal. The expression side does not exist here: the count sheet is per gene per sample. Two things are needed before the arm is worth building. From the lab, per haplotype PET counts per loop or the phased BAMs, so a loop can be called allele specific on counts rather than on a label; and allele specific expression per gene from the RNA-seq reads against the phased genotypes, which for these nine are the public 1000 Genomes phased VCFs. With both, the arm is two haplotype cluster files per sample from `trio_prepare.py`, an eden array of two haplotypes by nine samples, chr1 at 13 minutes a conformation, and the test is per person the maternal minus paternal distance against the allelic log ratio on the genes with a phased loop at the promoter, a few hundred a person at the genome. Expression side located 2026-09-22: HGSVC's `201707_ASE_RES_Trios` holds WASP corrected SNP allele specific expression for all nine on phased SNPs (`data/_hgsvc/201707_Table_1_ASE_SNP_RES_Trios.xlsx`, GRCh38), but it is the table of significant sites only, 6,011 rows over the nine, 30 to 58 chr1 genes a person at 20 reads or more; against 24 to 228 genes with a phased loop at the TSS the overlap is a handful, so the arm still needs allele counts at every het SNP, which the HGSVC RNA-seq now downloading can give against the phased VCFs.  First test, 2026-09-23, on idea 42's allelic ratio: the panel's h1 is the paternal haplotype in all three children at every informative site, so the lab's phased loops, one phased read each, and the allelic ratio can be read in the same maternal minus paternal orientation, `ase_phased.py`. RNAPOL2 phased loops at the promoter agree in sign with the allele's expression in 168 of 306 gene child pairs, 55 percent, p 0.10, and in GM19240 alone, the child with the most phased loops, 123 of 210, 59 percent, p 0.016; CTCF phased loops 142 of 290, nothing. A weak signal in the right direction on a thin input; the haplotype models want haplotype resolved loops, not one read per loop.|

Ideas 1 to 3 are features from the pairwise distances the pipeline already computes, about a
day of work with 5 on top, and they run on the models there are.

## Ideas, second list, 2026-09-21

From the limit analysis and the literature. The 3D-GNOME line itself points at one of them:
Szalaj 2016 (Genome Research, 10.1101/gr.205062.116) built the engine on ChIA-PET, and
Sadowski 2019 (Genome Biology, 10.1186/s13059-019-1728-x) and 3D-GNOME 2.0 (Wlasnowolski 2020,
NAR, 10.1093/nar/gkaa388) used it to model structural variants of 1000 Genomes individuals
and relate the altered structure to expression. Delaneau 2019 (Science, 10.1126/science.aat8266)
found in 317 lymphoblastoid lines that regulatory activity co-varies within 3D domains and
mediates genetic effects on expression, at a sample size two orders above nine. Gorkin 2019
(Genome Biology, 10.1186/s13059-019-1855-4) measured Hi-C across 20 Yoruba lines and found
person to person 3D variation real and modest. Li 2012 (Cell, 10.1016/j.cell.2011.12.014),
the RNAPII ChIA-PET paper, found promoters in multigene complexes more highly expressed. Zuin
2022 (Nature, 10.1038/s41586-022-04570-y) found transcription a nonlinear, saturating function
of enhancer contact probability. GraphReg (Karbalayghareh 2022, Genome Research,
10.1101/gr.275870.121) predicts expression from 3D contacts and 1D epigenome and gains from
3D mainly for distal elements, with element activity from the cell's own marks.

| # | idea | what it adds | cost | status |
|---|---|---|---|---|
| 8 | The data ceiling, before any model: the forming data at the promoter | The loops are the forming data and Hi-C the background, so the ceiling of anything person specific is in the person's own RNAPOL2 loops at the promoter, not in their Hi-C. Per gene per person the input feature's deviation from the panel against the expression deviation, `input_ceiling.py`. | hours | done, 2026-09-21: the forming data carries a person specific signal and the models keep a fraction of it. On the 1,903 chr1 genes all nine express, the deviation of the RNAPOL2 PET count at the TSS against the expression deviation is +0.10 over the nine, -0.03 to +0.18, the loop count +0.08, the peak signal +0.05, the CTCF PET +0.02, while the linear distance is the noise floor at +0.01. Every 3D feature keeps less: the nearest atlas enhancer +0.016, the nearest distal RNAPOL2 anchor +0.058, hubs of the person's own elements +0.04 to +0.05. The models move with their input only weakly, the PET deviation against the distance deviation +0.08 for the atlas and +0.13 to +0.20 for the person's own anchors and their hub, so between the loop list and the structure most of the person specific variation is lost, which is what the law's saturation does: a loop at three PETs and one at thirty both sit at one bead. The loops themselves agree between people at 0.87 where expression agrees at 0.97, so part of their variation is depth and noise. Two levers follow: elements from the person's own loops rather than the atlas, which keeps three times more, and a target distance that does not saturate in the PET count, `contact_half_saturation`, to be measured on this deviation statistic. |
| 9 | Structural variants, the 3D-GNOME 2.0 design | HGSVC's haplotype resolved SV calls for these nine. Genes whose enhancer or loop anchor a person's deletion, duplication or inversion removes or moves; that person's expression deviation at those genes, sign by SV type. Person specific by construction and the lab's own lineage. A data only pre test needs no model. | a day for the pre test, a week with the engine | pre test done and negative, 2026-09-22; the engine arm not built. HGSVC2 freeze 4 (`data/_hgsvc/variants_freeze4_sv_insdel_alt.vcf.gz`, GRCh38, phased, all nine genotyped), `sv_pretest.py` in `enhancer3d/playground`. chr1 holds 7,422 insertions and deletions, median 186 bp, 3,666 varying among the nine; at a kilobase or more 1,043, 384 varying. Per person per gene, an SV the person carries and someone lacks in the gene body, the promoter, an own element within 50 kb of the TSS, or the far anchor of an RNAPOL2 loop at the TSS, against the person's expression deviation: every class and type within 0.03 log2 of the non carriers, p 0.13 to 0.93, on 185 to 2,054 carrier pairs; at a kilobase or more the same, on 24 to 323 pairs. Per gene and SV pair with variation, the Spearman of dosage against the expression deviation over the nine sits on its permutation null in every class, means -0.04 to +0.04 against null -0.05 to +0.02, the one exception 65 promoter deletions at +0.10 with the wrong sign for a deletion. So on chr1 in nine people the SVs that would move an element or an anchor are too few and too small to show in expression, and a modelled arm would have nothing to explain. Also found in HGSVC's working directory: SNP and SV allele specific expression tables for the three trios (`201707_ASE_RES_Trios`, WASP corrected, phased SNPs), which is idea 7's expression side. |
| 10 | Reliability of the person specific expression | HGSVC's mRNA-seq is a second measurement of each person. Quantified against the lab's counts, the agreement of the two on each person's deviation from the panel is the ceiling of anything person specific. | a day, salmon on the workstation | done, 2026-09-22, and it turned out not to be a second measurement. ENA ERP012633, the nine runs at 56 to 92 million pairs, salmon 1.10 against GENCODE v40 on the workstation, `/mnt/storagelinux/_hgsvc/salmon/`, `rnaseq_quant.sh`, `rnaseq_compare.py`, copies in `enhancer3d/playground/rnaseq/`; mapping 86 to 90 percent. Summed to gene names and normalised as the lab's counts are, on the 16,409 genes expressed in every person in both: levels agree per person at 0.956 to 0.962, and each person's deviation from the panel agrees between the two quantifications at 0.88 to 0.93, mean 0.91, with the deviation's sd 0.5 to 0.9 log2. But the lab's column totals are 1.44 to 1.52 times ENA's read pair counts in every sample, one to one, so the lab's table is a count of these same HGSVC libraries and the 0.91 is two pipelines on one set of reads, not two experiments. It says the person specific deviation is a stable property of the reads and a quantifier does not lose it; it cannot say how much of it is biology. A true replicate would be another library of the same lines; Geuvadis (ERP001942, 667 runs) holds none of the nine, so in public there is none. |
| 11 | Gene looping and gene compaction | 3D distance TSS to TES and the radius of gyration of the gene body, controlling length. RNAPII recycling by gene loops is old literature; the RNAPOL2 anchors put beads inside bodies. | hours | done and dropped, 2026-09-21. `model_features2.py`, RNAPOL2 arm, own elements, 5,565 chr1 genes, 3,109 with a body of three or more beads. The TSS to TES distance correlates with expression at +0.27 and the body's radius of gyration at +0.05, both gene length: controlling log length -0.08 and -0.22, and the distance over the law's expectation for the length is 0.00. As a person specific deviation all three sit at -0.01. In the idea 5 model, with the hubs and the variability of 13, -0.004 to +0.003 out of fold beyond everything, `d2_gain.py`. |
| 12 | Promoter hubs, transcription factories | Other active promoters within a 3D radius, and promoter to promoter loops in the RNAPOL2 set, Li 2012's multigene complexes. | hours | done and dropped, 2026-09-21. Same run. Other protein coding TSSs within 2 or 3 units, mean over conformations, +0.02 to +0.06 with expression, the expressed ones only +0.04 to +0.06, and beyond the nearest element distance -0.04 to -0.05; deviation +0.01. The count of RNAPOL2 loops joining the TSS to another TSS is +0.45, and +0.29 beyond the nearest distance, but that is the input at the promoter again, the loop list and not the structure, and it adds nothing in the model either. |
| 13 | Ensemble variability as the feature | The spread over conformations of the nearest enhancer distance, stable against fluctuating contacts, a bursting reading. | hours | done and dropped, 2026-09-21. Same run, the nearest own element per conformation. The sd over conformations -0.42, the same as the mean, -0.44; beyond the mean the sd keeps -0.11, the coefficient of variation -0.14 and the minimum over conformations +0.09, all with deviations at -0.02 to -0.04, and in the idea 5 model the block of 11 to 13 together adds -0.004 to +0.003 out of fold. Ten conformations is thin for a spread; reopen only with the genome arms at fifty. |
| 14 | On and off before level | Zuin 2022's saturation says structure may set whether a gene is on more than how much. A classifier on silent against expressed with the 3D block, then level among the expressed. | hours | done and dropped, 2026-09-21. `onoff_model.py` on the idea 6 features with the silent genes, 3,022 chr1 genes a person, 78 to 82 percent expressed, gradient boosting five folds by three repeats, out of fold AUROC: the gene alone 0.73 to 0.77, with the linear map 0.78 to 0.81, with the loops at the promoter 0.80 to 0.84, with the 3D block on top 0.80 to 0.84, the 3D block beyond the input -0.006 to +0.002 and beyond the gene alone +0.04 to +0.08; the 3D block alone 0.71 to 0.79. On and off is the same picture as level. |
| 15 | Expression residual to sequence | Take out what promoter sequence predicts, CpG class as the cheap proxy and Enformer as the real one, and ask what structure explains of the residual. Changes the target rather than the feature. | a day cheap, a week with Enformer | cheap form done and dropped, 2026-09-21; the Enformer form open. `sequence_residual.py`, hg38 chr1 from UCSC on the workstation, CpG observed over expected and GC in TSS +- 500 bp, 1,746 of 5,565 chr1 promoters CpG island like. CpG alone +0.41 to +0.46 with expression, the gene block out of fold 0.54 to 0.57 to 0.64 to 0.66 with it; with the linear map and the loops 0.75 to 0.78 and the 3D block on top -0.001 to +0.006. Against the residual after gene and sequence the 3D features keep +0.17 to +0.24 of their raw +0.34 to +0.44, which the loops and the linear map explain as before. |
| 16 | The person's compartments | 4DN ships a compartment and an insulation track per person from their Hi-C. Compartment deviation from the panel against expression deviation, and the tracks as features. | hours | done, 2026-09-21, nine of nine. The validation tracks study on each person's 4DN mcool, chr1 at 100 kb, `slurm/ensemble/tracks_hgsvc_ws.sh`; the study phases each chromosome on its own and two people came out flipped, so `compartment_phase.py` phases every person to the panel's mean expression, one bit each; `compartment_person.py` runs the model. E1 at the TSS agrees between people at 0.98 once phased and correlates with expression at +0.17 to +0.23 in every person; its person specific deviation against the expression deviation is +0.078, -0.07 to +0.15 over the nine, on 1,710 genes, the same order as the loops' +0.10 and from an independent experiment. In the model it adds -0.001 to +0.005 beyond the gene, the linear map and the loops, seven people, and the 3D block adds nothing beyond it either. So a person's compartments are a second data type with a person specific trace and no out of fold gain. |
| 17 | What the RNAPOL2 loops did to the structure | Per gene the CTCF arm distance minus the RNAPOL2 arm distance, the geometry the second factor added. Both arms exist. | hours | done and dropped, 2026-09-21. `factor_delta.py` on the own element tables of both arms, 5,565 chr1 genes. The arms agree per person at 0.61 to 0.88 on the nearest distance; the difference between them against expression is -0.006 on average, -0.04 to +0.06, and as a person specific deviation +0.004; on the atlas tables +0.03 and +0.005. Each arm's distance beyond the other's keeps -0.16 (CTCF beyond RNAPOL2) and -0.22 (RNAPOL2 beyond CTCF), so the second factor made the same feature a little stronger rather than a different one. The fifth of genes the RNAPOL2 arm pulled in by over a unit are the genes that sat far from any own element on the CTCF arm, and they are the lower expressed ones, so the pull is not a signal either. |
| 18 | Inter chromosomal | Li 2012's complexes cross chromosomes. The genome arms model chromosomes together at the top level, so a promoter's neighbours on other chromosomes are a feature only the genome run can give. | with the genome arms | open |
| 19 | The saturation lever, from idea 8 | `[distance] contact_half_saturation` sets the loop strength, in multiples of a typical loop at that span, at which a pair sits halfway from the background to touching; at the default 1 a typical loop is already halfway and a ten times stronger loop at a tenth, so the person to person differences, which are in the counts of loops everyone has, compress into a fraction of a bead. One chr1 RNAPOL2 arm per value, 3 and 10 first, nine samples at ten conformations, `trio_configs.py` writing `<s>_trio_rnapol2_qh<value>.ini`. Judged on three numbers per person: the PET deviation against the distance deviation from `input_ceiling.py`, which has to rise from +0.08 before anything else is asked; the idea 5 gain of the 3D block beyond the input; and the battery's Hi-C on the cell lines, since a weaker pull on every loop can cost it. | an hour to prepare, one eden array, an afternoon to judge | done and dropped at the data level, 2026-09-21, no array run. `qhalf_ceiling.py` puts the law itself at the promoter: per person the target the law assigns to the RNAPOL2 loops at each TSS, on the 1,903 genes and the deviation statistic of idea 8, at `contact_half_saturation` 0.3, 1, 3 and 10. The input's PET deviation carries +0.10; the law's summed pull at the promoter carries +0.076, +0.077, +0.078 and +0.079, the closest loop as a fraction of its background +0.063, +0.076, +0.086 and +0.086, the same within noise at every value, against the models' +0.016 to +0.058. The premise was wrong on the data: at the default a typical loop, strength 1.0 by construction, sits halfway and only 2 to 4 percent of loops sit within a quarter bead of touching, so the target is already most sensitive to the count of a typical loop, and moving the half point to 3 or 10 spreads the strong tail while weakening the pull on every loop, HG00512 chr1 `qhalf_scan.py`. The person specific deviation sits in loops of strength 1 to 3, 651 genes at +0.09, and 3 or more, 179 genes at +0.14, and the law relays both at every value. So the loss between the loop list, +0.10, and the structure, +0.02 to +0.06, is not in the law; it is downstream, in the engine's compromise between loops and in the feature, since the person's own anchors keep +0.058 of the law's +0.08 and the atlas +0.016. The generator keeps `--half-saturation` for the record. |
| 20 | The person's own elements as the only feature set | Ideas 4 and 8 both put the person's active elements, the atlas enhancers under one of their RNAPOL2 peaks and their distal RNAPOL2 anchors, above the atlas, and they keep three times more of the person specific part. `genome_ep_distances.py` takes an enhancer table per cell, so the change is one table per person built from `own_elements.py`'s sets, and every downstream script then reads the person's distances. Measured together with 19, on the arm 19 produces, since each is what the other needs to show up. | hours | done and adopted, 2026-09-21. `own_element_beds.py` writes per person, genome wide, the atlas enhancers under one of their RNAPOL2 peaks, 13,000 to 24,000, and their merged distal RNAPOL2 loop anchors, 22,000 to 132,000, the thin HG00514 library at the low end; `genome_ep_distances.py --enhancers own` reads it, `trio_expression.py` and `rnapol2_individual.py` take `TRIO_ELEMENTS` for the matching linear control. Both chr1 arms rerun on the models there are, `playground/trio_rnapol2_own/`, `trio_prod_own/`. RNAPOL2 arm, atlas to own: raw rho -0.34..-0.39 to -0.40..-0.46 in every person; the linear distance to the set alone -0.33..-0.38 to -0.44..-0.49; the partial beyond the set's own linear distance -0.13..-0.21 to -0.11..-0.22, mean -0.18 to -0.17; the person specific deviation +0.030 to +0.062 on the 1,903 genes; family separation +0.013 to +0.030, p 0.02 to 0.005. CTCF arm: raw -0.30..-0.36 to -0.37..-0.44, partial mean -0.134 to -0.136, deviation +0.021 to +0.051, family +0.042 to +0.036. So the person's own elements are the better table on every raw number and double the person specific part, and they are the right table, since an inactive atlas enhancer is inert in the person; what they add is linear proximity to an active element, and the 3D part beyond that is what it was. Adopted as the element set for the trios from here; the atlas tables stay for the record. |
| 21 | MultiMM's models on the same loops, a one off | A second engine on the same input: MultiMM 2.0.2 on each person's CTCF and RNAPOL2 loops on chr1, `playground/multimm_arm.py --n-beads 50000`, 5 kb beads, ten members, loops only and loops with the person's own compartment track, its minimised and its after dynamics ensembles. Then the same pipeline, own elements, the per person scripts, the deviation statistic and the idea 5 model. If a different energy and a cleaner ensemble keep more of the person specific part or add out of fold, the engine was the limit; if they land where ours do, the input was. | a day of GPU on the workstation, an afternoon to judge | done and dropped on one person, 2026-09-22, by decision: the test only had to show whether MultiMM's structures explain expression better, and they do not, so the other eight and the compartment arm were not run. HG00512, chr1, ten members at 50,000 beads, 68 minutes on OpenCL, `slurm/ensemble/multimm_trio_ws.sh`, the distance pipeline on own elements, `playground/trio_multimm_smoke*/`. Nearest own element distance against expression: ours -0.401, MultiMM after dynamics -0.376, minimised -0.379; the partial beyond the set's linear distance -0.144, -0.116, -0.119; MultiMM's distances agree with ours per gene at 0.72. A second engine on the same loops lands a little under ours on the same measure, so the engine is not the limit. |

8, 10 and 9's pre test are data questions and come first, since a negative on them ends the
line. 11 to 14 and 17 are afternoon tests on the models there are.

## Ideas, third list, 2026-09-22

What is left after 21, in the order to try them. The first three need no new models.

| # | idea | what to build | cost | status |
|---|---|---|---|---|
| 23 | The large deviations only | The deviation statistic of idea 8 ran over all 1,903 genes; enhancer3d's lesson was that the relation lives in the large changes. The same statistic, per person, on the genes where the person deviates from the panel by over one log2, and by over half, on the own element tables of all three arms and on the input features, with the sign test and the Mann-Whitney of the jump statistic. | an afternoon on tables that exist | done, 2026-09-22, and it is the one positive person specific result. `large_deviations.py`, 1,710 chr1 genes expressed in all nine with every feature; per person 481 genes deviate from the panel by over half a log2 and 213 by over one. The deviation statistic sharpens on the large deviations exactly as enhancer3d's did between cell types: the nearest own element distance, all genes +0.05 to +0.07, over half +0.10 to +0.12, over one +0.15 to +0.19 on the CTCF, RNAPOL2 and HGSVC arms alike, Mann-Whitney of genes up against genes down pooled p 1e-10 to 1e-16; the loops at the promoter +0.10 to +0.16; the person's compartment E1 +0.08 to +0.20; the linear distance stays at 0.00 and CTCF PET under 0.11. Controlling the PET deviation on the over one set the 3D keeps +0.15 and E1 +0.18, in sample. Out of fold on gene person pairs with the deviation as the target and folds cut by gene: over one, 1,918 pairs, the input reaches +0.16, with E1 +0.24, with 3D +0.18, with both +0.26, so E1 adds +0.07 and the 3D block +0.02 beyond the loops; on all 15,390 pairs +0.14, +0.16, +0.14, +0.16, E1 +0.02 and 3D +0.003. So on the genes where a person differs most from the others, spatial organisation tracks the difference, the ported enhancer3d claim, and what carries it beyond the loops is the person's own compartments, with the models a little behind them. |
| 22 | The genetic explanation of the person specific part | The deviation is stable to quantification at 0.91 and all nine are 1000 Genomes samples with phased genotypes. Public LCL eQTLs, GTEx v8 or Geuvadis, give a predicted deviation per gene per person from genotype alone. Two questions on existing tables: how much of the observed deviation the eQTLs explain, and whether the person's loops at the promoter and the nearest element distance track the eQTL genotype, Delaneau 2019's framing. The one test that says whether the person specific part is biology the structure could ever reach. | a day, data only | done, 2026-09-22. `eqtl_person.py`, GTEx v8 EBV lymphocyte eGenes at q under 0.05, 450 on chr1, the lead variant each with its slope, genotypes of the nine read with pysam from the 1000 Genomes high coverage phased chr1 panel, 435 of 439 lead variants found with matching alleles; 358 of the 1,903 genes have an eQTL, about 270 per person after the zero dosage deviations. The genotype predicted deviation correlates with the observed expression deviation at +0.29 over the nine, +0.09 to +0.39, and +0.35 on the genes deviating by over one log2, so the person specific part is biology and in good part cis genetic, from a public map and nine people. The structural data do not track it: the loops at the promoter against the genotype prediction +0.04, the nearest own element distance -0.01, the compartment E1 +0.01. Out of fold on the eQTL gene person pairs, target the observed deviation: genotype +0.19, loops +0.10, both +0.24, with the 3D block +0.24, with E1 +0.27; on the large deviations, 310 pairs, genotype and loops +0.32, with 3D +0.35, with E1 +0.40. So the person specific expression has a genetic part the eQTL map predicts and an epigenomic part the loops and the compartments carry, the two nearly independent, and the models add +0.03 on the large deviations beyond both. The structure could reach the genetic part only through the variants themselves, which is idea 9's design at the scale of SNPs. |
| 24 | The person's compartments in the energy | Each person has an E1 track from their own Hi-C now, `data/<S>/<S>_compartments.bedGraph`, and the compartment term has never run on a person's own track. One chr1 array on eden with `use_compartments` on the RNAPOL2 arm, judged on the deviation statistic and idea 5. Expected to give what the feature gave, +0.08 in the data and nothing out of fold; last among the cheap ones. Idea 22 found the compartments carry an epigenomic part beyond the loops and the genotype on the large deviations, so the arm is worth its array. | one eden array | done and dropped, 2026-09-22. Eden job 1809339, 9 to 16 minutes a conformation with the term, `comp_arm.sh`, `playground/trio_rnapol2_hgsvc_comp/`. Against the HGSVC arm without the term, per person on chr1: raw rho 0.348 to 0.350, the partial beyond the set's linear distance -0.192 to -0.192, the person specific deviation +0.052 to +0.058, on the genes deviating by over one log2 +0.155 to +0.157, family separation +0.025 to +0.024, the 3D block beyond the loops out of fold -0.003 to +0.009 both, the two arms' nearest distances one feature (+0.05 each beyond the other). The number the arm was for, the models' deviation against the person's E1 deviation, is +0.023 with the term against +0.031 without, while the level stays at +0.54. The term at weight 0.5 does not move the structure toward the person's compartments where they differ from the panel; the loops and the chain set that. Prepared as: On the HGSVC map arm, so the background and the compartments come from the same Hi-C, at weight 0.5 as on the cell lines, phased by anchor density: `trio_configs.py --factor RNAPOL2 --singletons hgsvc --compartments 0.5` wrote `<s>_trio_rnapol2_hgsvc_comp0.5.ini`, the nine tracks are on eden under `data/<S>/`, and the guard admits the term on the sample's own track. `CONFIG_TAG=_trio_rnapol2_hgsvc_comp0.5 OUT=out/trio_rnapol2_hgsvc_comp CHROMS=chr1 PER_TASK=10 sbatch --array=0-8%9 --time=24:00:00 slurm/ensemble/trio_ensemble.sh`; judged against `trio_rnapol2_hgsvc` on the deviation statistic, the large deviations and idea 5. Before the run, `e1_vs_3d.py`: the models without the term track the compartment level, closeness to an active element against E1 +0.55 to +0.56 in every person, but not its person specific deviation, +0.02 to +0.03, so the term has the room the feature showed. |
| 7 | Haplotypes | See the first list. Buildable now, a day of alignment for allele counts against the phased VCFs, but a handful of genes per person on chr1. Only with the genome arms. | a day, then the genome arms | open, deferred |
| 18 | Inter chromosomal | See the second list. Trans contacts as inter chromosomal singletons and a whole genome per process, days per conformation. A design change, not for this question. | a design change | open, deferred |
| 15 | The Enformer form | See the second list. A week; the cheap form showed sequence lifts the gene block without touching the 3D gain. | a week | open, deferred |
| genome arms | Ten times the genes for every number in this note, and the prerequisite for 7. The user submits; about 240 GPU hours on the ChIA-PET map or half on the HGSVC map. | 120 to 240 GPU hours | waits for the sbatch |

Also open from `rnapii-loops.md`, not about expression: the per sample RNAPOL2 anchor arm, to
separate the family signal drop from the shared bead set.

## Ideas, fourth list, 2026-09-22

From what 22 to 24 taught: the person specific part is part genetic and part epigenomic, the
structure tracks neither at the gene level, and its one trace is on the large deviations. The
first is a control on that trace and comes before anything built on it.

| # | idea | what to build | cost | status |
|---|---|---|---|---|
| 25 | A null for the large deviation trace | The +0.15 to +0.19 of idea 23 is a correlation of two deviations from the same panel of nine, and the Mann-Whitney was pooled. Permute the person labels of the model deviations within each gene and recompute the statistic on the same gene sets, a thousand times, for the 3D distance, the loops, the compartments and the linear floor. What survives the null is the result. | an hour on tables that exist | done, 2026-09-22, the trace survives. `deviation_null.py`, 1,710 genes, a thousand permutations of the person labels within each gene. The null mean is 0.000 with sd 0.009 on all genes, 0.017 over half a log2 and 0.025 over one, for every feature. Observed against that: the 3D nearest own element on the RNAPOL2 arm +0.065, +0.121, +0.185 at z 7.1, 7.2, 7.3; CTCF arm +0.049, +0.098, +0.153 at z 5.9 to 6.0; HGSVC arm +0.053, +0.104, +0.155 at z 5.9 to 6.3; the RNAPOL2 PET at the TSS +0.100, +0.144, +0.155 at z 12.4, 9.1, 7.1; the person's compartment E1 +0.078, +0.141, +0.200 at z 9.5, 8.7, 7.9; every p under 0.001. The linear floor has no person deviation and sits on its null exactly. So the person specific trace, and its sharpening on the large deviations, are not an artefact of nine deviations from one panel. |
| 26 | The person's own enhancer activity, ABC style | Idea 1 weighted the atlas by GM12878's activity score and idea 4 used the person's RNAPOL2 peaks as positions. Not yet tried: the person's own RNAPOL2 signal at each element as its activity, times the contact from the person's model, summed over elements, the ABC form with person specific activity; and its deviation from the panel, since the enhancer's activity in the person is the epigenomic part the loops at the promoter do not see. | an afternoon on the models there are | done and dropped as a model gain, 2026-09-22, with one finding. `own_activity.py`, RNAPOL2 arm, own elements within 3 Mb, activity the person's RNAPOL2 peak signal over the element, 31 to 59 percent of elements carry one, contact exp(-d/2). Raw per person: the best activity times contact over elements +0.45, level with the nearest distance +0.44; the sum +0.38; the nearest element's activity +0.26; the linear only form +0.33. Beyond the nearest distance in sample the best pair keeps +0.16 and the sum +0.12, but out of fold in the idea 5 model the four activity features add -0.004 to +0.004 beyond the gene, the linear map and the loops, and the same beyond the 3D block. The finding is on the person specific side: the linear only form, the person's peak signal at elements near the gene with no model, carries +0.084 on the deviation statistic at z 11, level with the loops' +0.10, while the contact weighted forms carry +0.045 to +0.058, so the model's contact dilutes the person's activity rather than sharpening it. The person's own enhancer activity is a second data only carrier of the epigenomic part, beside the loops and the compartments. |
| 27 | The eQTL variant as the element | For every eQTL gene the variant marks the regulatory element that matters for that gene, by genetics rather than by an atlas. The 3D distance from the TSS to the eQTL variant's bead in each person's model, its level against expression and its deviation against the genotype and the expression deviation, beside the nearest element distance. | an afternoon | done and dropped, 2026-09-22. `eqtl_variant.py`, 440 eGenes in the model gene set with a genotyped lead variant, 358 expressed in all nine, the lead variant a median 28 kb from the TSS. Closeness of the TSS to the variant's bead against expression -0.08 where the nearest own element on the same genes gives +0.34; as a person specific deviation +0.01 against expression and -0.01 against the genotype, where the nearest element gives +0.11; per gene across the nine the carriers of the expression raising allele hold the variant no closer, share positive 0.42 against a null of 0.43. The genetic element is not what the models bring near. |
| 28 | Loops at the variant, the mechanism test | Idea 22 asked whether the loops at the promoter track the genotype and found 0.04. The finer test: for eQTL variants that lie in a loop anchor or an own element, does the person's PET count at that anchor track the genotype? A loop that follows the variant is the mechanism 3D-GNOME 2.0 assumed; a loop that does not says the genetic part never enters this data. | an afternoon, data only | done and dropped, 2026-09-22. Same script, 25,591 significant pairs on chr1 with genotype variation among the nine, 20,283 variants over 406 genes, genotypes from the local 1000 Genomes panel. Against the dosage of the expression raising allele: the person's RNAPOL2 PET over loop anchors covering the variant +0.024 on 10,413 pairs where someone has a loop, null 0.000, share positive 0.51 against 0.48; CTCF PET +0.012 on 5,455, null +0.006; the RNAPOL2 peak signal at the variant +0.010 on 5,297, null -0.006. The loops and peaks do not follow the variant, so the genetic part of a person's expression never enters the data these models are built from, which is why idea 22 found the structure blind to it. |
| 29 | One model of the person specific part | Everything together, genotype, loops, peak signal, compartments and the 3D deviations, out of fold on gene person pairs with folds by gene, on all genes and on the large deviations: the fraction of the person specific deviation that is explained, and each part's share. The summary number for the write up. | an afternoon on tables that exist | done, 2026-09-22. `person_model.py`, 1,710 chr1 genes with every quantity, 15,390 gene person pairs, 236 genes with an eQTL, gradient boosting out of fold with folds cut by gene. Everything together predicts a person's deviation from the panel at +0.20 on all genes, +0.25 on the genes deviating by over half a log2 and +0.31 over one. Each block alone, and its gain when added to all the others, on the large deviations: the loops at the promoter +0.16 alone and +0.05 unique, the person's compartments +0.16 and +0.06, the peaks +0.16 and +0.02, the genotype +0.04 alone on the 236 eQTL genes and +0.03 unique, the 3D distances on the three arms +0.08 alone and -0.003 unique; on all genes the same order with the 3D gain +0.001. So the person specific part of expression is predictable at about 0.2 to 0.3 from data alone, in shares of loops, compartments, genotype and peaks, and the 3D models contribute nothing unique to it. |
| 30 | Fifty conformations for one person | Idea 2's reopen condition. Ten conformations cost about 0.06 of the correlation by attenuation, and the ensemble spread of 13 was thin. One person on chr1 at fifty on eden, the deviation and idea 5 numbers against the ten. Low expectation, since the ceiling from attenuation was 0.39 against 0.36 kept. | 7 GPU hours, the user submits | open, last |

Then the deferred: 7 haplotypes, 18 inter chromosomal, 15 in its Enformer form, and the genome arms.

## Ideas, fifth list, the engine, 2026-09-22

The question turned to the engine: what change to 3dgnome itself could raise expression
prediction, and would long range interactions. What the numbers say about where the engine
stands. The law's target at the promoter keeps +0.08 of the loops' +0.10 person specific part
(idea 19) and the realised structure keeps +0.02 to +0.06 (idea 8), so the loss is in the
solve, not the law. The arcs solve on a trio chromosome hits its 800 iteration cap on every
conformation, `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` in every log of job 1809232, 23,080
anchors on chr1. Loops beyond `max_pet_length`, 1 Mb, leave the arcs for the segment heatmap,
and the contact background reads the thinned singletons, 930,000 pairs on chr1, so it holds
few far pairs; the person's compartments, which are the same Hi-C at 100 kb, carry +0.06
unique out of fold on the large deviations (idea 29) and the models track their person
specific deviation at +0.02 (ideas 16 and 24). Every conformation of an ensemble solves the
same targets from a different start, so the ensemble spread is annealing noise, and a loop's
PET count, which is a frequency across cells, enters only as a target distance. RNAPOL2 loops
enter as pairwise springs like CTCF's, with no many body form.

Three ceilings to keep in view. A perfect relay of the promoter loops gives +0.10 person
specific and 0 beyond the loops out of fold, since the loops are the input. The one source
of person specific information not in the loops that the models could carry is the person's
Hi-C beyond the loops, worth at most about +0.06 on 200 genes a person by idea 29. And the
ensemble at ten conformations costs about 0.06 by attenuation.

| # | idea | what to build | cost | status |
|---|---|---|---|---|
| 31 | Loop realisation fidelity, the diagnostic | On the models there are, per loop: the law's target distance against the realised mean distance over conformations, by strength, span and factor; and per person the deviation of the target from the panel against the deviation of the realised distance, which is where the +0.08 becomes +0.03. Names the loops the solve loses, weak, long or crowded, and decides between 32 and 33. | an afternoon | done, 2026-09-22, and it locates the loss in the solve. `loop_fidelity.py`, HGSVC arm, 28,000 to 47,000 loops a person on chr1, the law rebuilt from each person's own fit and measured exponent. Target against realised mean distance, Spearman per person: 0.54 to 0.66 over all loops, but that is mostly span; within a span band 0.26 to 0.44, and by strength tertile 0.69 to 0.79 for the weak and middle thirds against 0.35 to 0.49 for the strongest third, whose targets sit near one bead and whose realised distances do not: the median realised over target is 1.3 to 1.65, so every loop is under pulled and the strong ones most. The person specific chain per gene, mean Spearman over the nine on 1,903 genes: the PET deviation at the promoter against the law's summed pull deviation +0.60, the law relays the counts; the law's pull deviation against the realised pull of the same loops +0.19, the solve keeps a fifth; PET against the realised pull +0.22 and against the nearest own anchor feature +0.18. Against expression: PET +0.10, the law's pull +0.08, the realised pull +0.02, the nearest anchor +0.05. On the ChIA-PET map arm, exponents 0.34 to 0.42, the same diagnostic reads far better: target against realised 0.76 to 0.83 over all loops, 0.44 to 0.66 within a span band, 0.63 to 0.73 for the strongest third, realised over target 1.24 to 1.41; the solve relays the law's pull deviation at +0.63 and the PET deviation reaches the realised pull at +0.44. Yet against expression that arm's realised pull is +0.048 and its nearest anchor +0.058, where the HGSVC arm's are +0.021 and +0.054, the same feature numbers from a relay three times better. So the solve does lose loops, most on the HGSVC map whose lower exponent leaves the loops less room between background and touching and where the 800 iteration cap binds, and converging it (32) is the right first test there; but the relay is not what caps expression, since the arm that relays three times better lands on the same expression numbers, which is the ceiling again: the loops' +0.10 is the input and a structure can only approach it. |
| 32 | Converge the arcs solve | The cap of 800 iterations binds on every trio chromosome. One chr1 arm at 3,000 or until converged, judged on 31's fidelity and the deviation statistics. Cheap if the device solve stays at seconds an iteration. | one eden array | done, 2026-09-22, the tail is inert; the energy stop rule adopted anyway. The 3,000 iteration arm, job 1809444, `<s>_trio_rnapol2_hgsvc_it3000.ini`, still ends on the limit: HG00513's first solve 3,063 evaluations in 200 s against 30 s at 800, so neither cap is a convergence. On the user's decision the cap is replaced by a stop rule on the energy: `[simulation_arcs] solver_tol` (0fa598b), L-BFGS-B's own relative improvement test, with the iteration count a safety; at zero the options are unchanged and the parity gate holds. `GNOME3D_ARCS_TRACE` records every evaluation's energy, The traced solve, HG00512 chr1, 23,080 anchors, 8,000 iterations in 705 s on the workstation, still ends on the limit. The energy at 800 is 1.7 percent above the value at 8,000, at 3,000 0.7 percent, at 5,000 0.4 percent, and it is still creeping at 8,000: the relative improvement per iteration falls to 1e-5 by 830 and to 1e-6 by 2,400 and then sits flat near 1e-6 for the remaining 5,600, a long tail with no plateau. So a tolerance is a choice of where to leave the tail: 1e-5 stops near 800, what the cap did; 1e-6 stops near 2,400 within one percent of the 8,000 value at three times the cost; nothing under 1e-7 stops in 8,000. The 3,000 iteration arm is level with 800 on everything: loop fidelity 0.535 to 0.658 against 0.535 to 0.660, the strong tertile 0.35 to 0.49 both, realised over target 1.29 to 1.65 both, the solve's relay of the law's pull +0.186 against +0.185, raw rho 0.350 against 0.348, the partial -0.197 against -0.192, the person specific deviation +0.057 against +0.052, on the large deviations +0.161 against +0.155, the 3D block beyond the loops -0.002 to +0.006, `playground/trio_rnapol2_hgsvc_it3000/`. So the solve's tail is inert: the under realisation of the loops is the energy's compromise among competing loops, not an unfinished descent, which is what idea 37 addresses. The stop rule on the energy is adopted regardless, by decision, as the principled stop, `solver_tol` 1e-6 with a 5,000 cap (8b1992b); it changes no result. Done. |
| 33 | Loop presence sampled per conformation | A PET count is a frequency across cells. Each conformation draws each loop with a probability from its strength, so the ensemble mean over conformations encodes the frequency and a person's count of a shared loop survives into the mean distance, which the target distance alone cannot carry once it saturates. Opt in, the ensemble then needs more members. Judged on the deviation statistics and idea 5. | a few days of engine work, one array | open , 2026-09-22: idea 43's 2D form puts the shared loop strength at +0.03 to +0.06 of the loops' +0.10 person specific part, the rest being loop presence the models already carry, so the ceiling this idea can reach is that much|
| 34 | The person's Hi-C far pairs, the long range answer | Loops carry up to 1 Mb and the background reads a thinned map. The person's full 4DN map at anchor resolution as the contact background across blocks in the joint solve, the denser anchor level map that is open since September, on the HGSVC arm. The only person specific information beyond the loops that the models could take in; judged on whether the models' deviation tracks the person's compartment deviation, on the large deviations and on idea 5. | a few days, one array | open |
| 35 | RNAPOL2 as a many body attraction | Encoding B of `rnapii-loops.md`: RNAPOL2 anchors attract as a group, transcription factories, rather than as pairwise springs, so promoter hubs form in the model. Judged on idea 12's hub features and on expression. | a week of engine work | open |
| 37 | Loop stiffness by strength | Idea 31 found the strong loops the least faithfully realised, and 33 would fix that by changing what a conformation is. The smaller change: scale each loop's spring by its strength, so a 30 PET loop wins a competition a 3 PET loop loses, with every target and every loop kept where they are and exponent zero byte exact. Pre test on the fidelity tables, `loop_crowding.py`: the realised over target ratio rises with the number of loops sharing an anchor at +0.24 to +0.37 in every person and every strength tertile on both arms, and the strong loops' median ratio goes from 1.3 at low crowding to 1.7 to 2.3 at high, so it is competition that beats the strong loops. A day in the arcs energy and gradient on both backends, `[springs] arc_weight_exponent`, judged on 31's strong tertile fidelity and the deviation statistics. | a day of engine work, one array | done, 2026-09-22, null on its target and on expression, left at 0. Job 1809564, `<s>_trio_rnapol2_hgsvc_aw1.ini`, exponent 1 on the HGSVC map arm under the stop rule, `playground/trio_rnapol2_hgsvc_aw1/`. Loop fidelity against the arm without the weight: target against realised over all loops 0.546 to 0.674 from 0.535 to 0.660, up 0.01 to 0.02 in every person, the CTCF loops up 0.02 and the RNAPOL2 loops level; within a span band up 0.01 to 0.06, most between 100 and 500 kb; the median realised over target 1.26 to 1.63 from 1.29 to 1.65. But the strongest third, the loops the weight was built for, is level at 0.33 to 0.49 from 0.35 to 0.49, within 0.02 in every person and down in four, while the weak and middle thirds lose 0.005 to 0.02; so the weight reorders the loops a little across strengths and does not let a strong crowded loop close. The solve relays the law's person specific pull at +0.191 from +0.185, PET against the realised pull +0.224 from +0.22, and against expression the realised pull +0.019 and the nearest anchor +0.053, as before. Expression, own elements: raw 0.347 from 0.348, partial -0.194 from -0.192, the within against between family separation +0.031 from +0.025, the person specific deviation +0.056 from +0.052, on the large deviations +0.148 from +0.155 with the top decile of feature deviation +0.135 from +0.091, noise at 171 genes; out of fold on the large deviations the 3D block beyond the loops +0.008 to +0.014 from -0.003 to +0.016, and idea 5's 3D block beyond the loops -0.004 to +0.010 from -0.003 to +0.007. One instrument note: `large_deviations.py` reads its out of fold 3D block from the three fixed arms, so the out of fold line in an extra arm's chain log is the HGSVC arm's, and the number above comes from a rerun with the weighted table in that slot. The reading: a strong loop's competition is with the background springs and the repulsion among its anchor's partners, which want those partners a bead or more apart while the loops want each within a bead, and the weight does not touch those terms; a larger exponent would only trade the weak loops away. Left at 0, opt in, with its tests and the parity gate. |
| 36 | More conformations, idea 30 | The attenuation ceiling. Fifty conformations for one person. | 7 GPU hours | open |

Long range interactions, then, are 34 and only 34: loops within a megabase are already in and
the joint solve places blocks by the Hilbert start and a thin background, and the person's
compartments show what the full map could add, about +0.06 on the large deviations and
nothing on the rest. The larger lever for the person specific part is 33, since it is the
only way a shared loop's person specific count reaches the structure.

## Ideas, sixth list, after the lab meeting of 2026-09-22

The colleague's modality survey on the same nine people (slides, `LabMeetingPietryga`) and the
meeting. His two axes are ours: the global correlation across genes within a person, which is
our raw number, and the gene wise correlation across people for one gene, which is our person
specific deviation in another form. His table on the trios: methylation of CpG islands global
-0.43 and gene wise 0; RNAPOL2 occupancy over the gene body, counts from the ChIA-PET RNAPOL2
BAMs between TSS plus 300 bp and the gene end divided by the gene length, global 0.72 to 0.78
in eight people and 0.36 in GM19238, gene wise mean +0.17 with 4,266 genes over 0.5; CTCF
peak strength 0.05 and 0; the nearest enhancer distance on his 3DGnome models from the HiChIP
CTCF loops -0.28 and 0; EBV expression in the HPRC panel gene wise +0.10 to +0.12. Two notes
for his table. Under the null at eight people 11 percent of genes pass 0.5 by chance, so of
about 14,000 genes 1,500 would, and his 4,266 is three times that, real but a smaller count
than it reads. And GM19238 is not an outlier in our numbers, raw -0.44 and partial -0.22
among the best of the nine, so the outlier is that library's gene body coverage, not its
loops. On our side the same axes read: nearest own element -0.40 to -0.46 global and +0.05
to +0.06 gene wise, the RNAPOL2 loops at the promoter +0.43 to +0.57 and +0.10, the person's
compartments +0.08 gene wise, the genotype through GTEx eQTLs +0.29 gene wise.

What the meeting settled. Loop strength enters the engine only as a target distance; 37 was
the first answer to that and is null, 33 is the second. The colleague normalises loop counts
on the 2D side and we were to try it on 2D and 3D. He gets the loops with PET counts. The
paper is the trio ChIA-PET paper for the Nature Genetics 3D call, deadline the end of
November, where the 3D part is ours, and its story is the NAR result ported to individuals:
structure tracks the large transitions, not the level (idea 23). Follow up meeting Monday
2026-09-28, then he is away a week.

| # | idea | what to build | cost | status |
|---|---|---|---|---|
| 38 | RNAPOL2 over the gene body, the colleague's feature | His table of gene body occupancy per gene per person, or the BAMs to count it from, since we hold peaks and loops but no BAMs. Then the feature goes into idea 29's person model and idea 5's model with the 3D block beside it. The honest reading first: RNAPOL2 over a gene body is transcription measured again, so as an explanation of expression it is near circular, and its value here is as the covariate that sets the ceiling for every cis modality, the loops and the models included. | an hour once the table arrives | one off reproduction, 2026-09-23, half way. We hold no BAMs; the rawest RNAPOL2 file the Drive gives is each merged library's cis PET clusters at PET 1 and up, 5.7 to 37 million clusters a person, and their ends are a read coverage. `polii_body.py` bins both ends at 100 bp, sums the body from TSS plus 300 bp to the gene end, strand aware, per kb, `playground/trio_rnapol2_own/polii/`. His gene wise axis reproduces to the third decimal: per gene across the nine, mean +0.175 against his +0.174, 3,662 of 17,139 genes over 0.5 against a null share of 0.08, his 4,266; without GM19238 +0.172. His global axis does not: per person across genes the body density reads 0.47 to 0.52 on the genes the person expresses and 0.55 to 0.62 with the silent genes kept, the promoter window 0.58 to 0.61 and 0.65 to 0.69, against his 0.72 to 0.78; and GM19238 is no outlier for us, 0.50 and 0.58. The expression side is not the difference: counts, counts per kb of gene and salmon TPM as the target all land at 0.45 to 0.64. The reason is what the cis file holds: no pair under 8 kb, 0.3 percent between 1 and 8 kb, so the self ligation pairs, the 1D ChIP coverage that dominates a BAM, are not in it, and our ends are the loop ends alone; the peak calls' signal over the body per kb gives 0.60, the same ceiling. The juicer .hic the Drive gives per person keeps those pairs, 19 percent of chr1 contacts on the 5 kb diagonal and 14 percent at 1 to 8 kb, so its row sums are the 1D coverage at 5 kb, and the folder holds one .hic per person, `hic_files/ChIA-PET_hg38_<S>_merged.hic`, which is the CTCF library: on HG00512 chr1 at 5 kb its coverage is 2,230 at CTCF only peaks, 995 at RNAPOL2 only peaks and 402 elsewhere. No RNAPOL2 read coverage exists in the folder. So the global 0.74 needs his gene body count table or the RNAPOL2 BAMs, or a bedGraph of their read coverage, and the ask is with the user. What the reproduction says for the 3D question: his number is Pol II occupancy over the transcribed body, transcription measured on the same molecules, and it is 1D coverage the loop data do not carry; a model built from loops reaches the loops' ceiling, +0.47 at the promoter, and would reach 0.74 only by taking the coverage in as an input, which is expression predicting expression. Its use stays what 38 said: the covariate that bounds every cis modality, once the table arrives. Then the user's note that he normalised the RNA-seq counts by gene length as well, and the grid with both sides per kb: on the protein coding genes a person expresses, expression counts per kb against Pol II per kb of the gene reads 0.696, 0.68 to 0.74 across the nine, and against Pol II per kb of the body 0.678, which is his 0.72 to 0.78 within the coverage we lack; the same two quantities raw, counts against the body count, read 0.415. The difference is the divisor. Pol II counts shuffled across genes and then divided by length, against expression per kb, correlate at 0.25 in every person, 0.23 to 0.27, since expression per kb carries 1/length at 0.35 and so does anything else divided by it. So of his 0.74 about 0.25 is the shared gene length and the substantive correlation of Pol II occupancy with expression is about 0.45, which is where the loops at the promoter, +0.47, and the nearest own element, -0.40 to -0.46, already stand. The data ceiling the models are measured against is not 0.74. |
| 39 | Gene length normalised expression as the target | The lab's counts are counts and our target is their log. Salmon TPM for the nine exists from idea 10, `/mnt/storagelinux/_hgsvc/salmon/quant/<S>`. Rerun the baselines on TPM: raw, deviation, idea 5. Expected: the deviation unchanged, since a gene's length cancels in a deviation from the panel, and the raw number moves a little at most, since gene length has no correlation with the distance (the processing check of 2026-09-20). Promised in the meeting. | an hour | done, 2026-09-22, dropped, the counts stay the target. `tpm_table.py` sums salmon TPM per gene into the lab's layout, `data/trio_gene_tpm.csv`, and `tpm_arm.sh` reruns the CTCF with RNAPOL2 own element arm on both, `playground/trio_rnapol2_own_tpm/`. TPM against counts: raw -0.38 to -0.42 from -0.40 to -0.46, the own linear control -0.39 to -0.42 from -0.44 to -0.49, the partial -0.164 from -0.169, the person specific deviation +0.046 from +0.062 on 2,554 genes against 1,903, since salmon's fractional assignment leaves fewer zeros. Idea 5 out of fold: the gene block 0.54 to 0.57 level, loops and linear 0.70 to 0.73 from 0.75 to 0.77, the 3D block alone 0.46 to 0.48 from 0.55 to 0.59, its gain beyond the loops +0.002 to +0.011 from -0.006 to +0.009. So a length normalised target lowers every fit and moves the 3D gain by nothing; the colleague's gain came from normalising his feature, the RNAPOL2 coverage over the gene body, by the gene's length, not from the target. |
| 40 | The colleague's gene wise statistic on our features | Per gene Spearman across the nine people, mean over genes and the count over 0.5 against its null, for the nearest own element distance on each arm, the linear distance, the loops at the promoter, the compartments and the genotype prediction, so the two tables merge on one statistic. Same for his statistic's null: about 1,500 of 14,000 genes at n 8. | an hour | done, 2026-09-22, `genewise_axis.py`, table below. On his statistic our 3D nearest own element is +0.06 to +0.09 gene wise where his models read 0, the RNAPOL2 loops at the promoter +0.165 match his gene body occupancy's +0.17, the linear map of active elements +0.12 sits above the 3D, and the genotype through GTEx eQTLs is the strongest per gene at +0.26 on the 278 eQTL genes. Every feature's share of genes over 0.5 is read against a within gene permutation null of about 0.08 at nine people; his 0.30 is against 0.11 at eight. |
| 41 | EBV load as a trans covariate | The HGSVC RNA-seq on the workstation aligned to the EBV genome, NC_007605, total and BHRF1 per person, CPM. Test whether the person specific deviation tracks it, and put it in idea 29's model. A part of the deviation that is trans can never be carried by loops, compartments or models, so this sets the cis ceiling below +0.10 if it holds. On HPRC he finds 931 genes over 0.5 on 206 people. | half a day on the workstation | done, 2026-09-22, and it names the largest axis of the person specific expression. `ebv_genes.py` cuts the 86 genes of NC_007605.1 from its GenBank record, `ebv_quant.sh` builds one salmon index of GENCODE v40 plus those and requantifies the nine so a read chooses between human and viral transcripts, `/mnt/storagelinux/_hgsvc/salmon/ebv/`, `ebv_person.py` and `playground/trio_rnapol2_own/ebv/`. The load, EBV reads per million mapped: 2,551 in HG00512 to 13,940 in HG00732, a factor of five, mostly lytic genes, BHLF1, BALF2, BLLF1, BORF2, BSLF2/BMLF1, so the lines differ in how many cells are in the lytic cycle. On the 1,903 chr1 genes expressed in all nine, per gene across the nine people and per person out of fold from the other eight's slopes: the total load and the lytic genes read negative gene wise, mean -0.16 with 20 percent of genes under -0.5 against 6 percent under the null, out of fold +0.10 and +0.11 against a permutation null with a 95th percentile of +0.16, and within families nothing, -0.04; both are also library covariates, Spearman -0.73 with the mapping rate and -0.83 with the person's RNAPOL2 loop count on chr1, which is either host shutoff in the lytic lines or the libraries, and n 9 cannot say which. EBNA-1 is different. Its level predicts a person's expression deviation out of fold at +0.222, +0.04 to +0.38 in every person, above all 100 permutations of the load over people, whose 95th percentile is +0.12 and maximum +0.219; the same within families, +0.23; 20 percent of genes over 0.5 against 9 under the null, 27 percent of the deviation variance against 12; no library covariate, mapping rate +0.38, loops +0.18, the lab's totals 0; and it sits on the first principal axis of the deviation matrix at -0.92, an axis holding 32 percent of the deviation variance where the second holds 16. LMP-1 reads nothing. The family control, a person's deviation against the mean deviation of the two family members, is -0.09, so the deviations carry no family structure to confound this. So a third of the person specific expression in these LCLs is one axis, and that axis is the EBV latency program, EBNA-1's level, a trans effect of the cell line's state that no cis modality, loops, compartments or models, can carry; beside it the genotype through eQTLs carries +0.29 on its 278 genes and the loops at the promoter +0.10. In idea 29's gene fold model the EBV block scores +0.10 alone and +0.15 and +0.19 on the deviations over 0.5 and 1, with a unique gain of +0.01, +0.04 and +0.04, and a control block of the person's identity, one column per person, scores the same to the third decimal, +0.10, +0.15, +0.19 and gains +0.01, +0.05, +0.06; so with folds by gene a per person value carries only the person's offset over genes, and the EBV specific evidence is the across people fold above, where a permutation of the load over people is the null. The 3D block's gain stays at 0 with either. The paper's decomposition of the person specific part: a trans axis, EBNA-1, +0.22 across people; the genotype, +0.29 on the eQTL genes and +0.36 in the allelic form (42); the loops at the promoter +0.10; the models +0.05, nothing beyond the loops. The RNAPOL2 loop count against the lytic load is worth one look on the genome arms. ||
| 42 | Allelic expression, the trans free target, idea 7's missing input | The colleague's formalism: within a person the ratio of the two haplotypes' expression cancels the trans term, `y = log(E_h1 / E_h2)`, regressed on haplotype differences `X = x_h1 - x_h2`. The RNA-seq and the 1000G phased chr1 panel for the nine are on the workstation; the allele counts come from a diploid transcriptome per person with bcftools consensus and salmon, or from an aligner and counts at heterozygous sites. The 3D side of this stays thin, phased loops are 2 to 4 percent and the chr1 SVs few, so the ASE table is first a second view of the deviation with the trans part removed, and only then idea 7's target. | a day | done, 2026-09-23 00:40, the target exists and the cis part of the large deviations is measured. `panel_exonic_snps.py` makes one pass over the 1000G high coverage phased chr1 panel for the nine, 5.76 million records in six minutes, keeping 33,442 exonic SNVs heterozygous in at least one of them and 2,059 GTEx LCL lead eQTL variants with the phased alleles; `diploid_transcripts.py` rebuilds every chr1 transcript from the reference exons, zero mismatches against the reference at the sites, and writes a person's heterozygous ones twice with the phased alleles substituted, 7,365 to 10,339 diploid transcripts of 21,800 per person, beside the other chromosomes' GENCODE transcripts; `ase_quant.sh` indexes and quantifies each person in five to six minutes, `/mnt/storagelinux/_hgsvc/salmon/ase/`; `ase_person.py`, `playground/trio_rnapol2_own/ase/`. Indels are left out. The target `y = log2(E_h1 / E_h2)` at 20 or more allelic reads: 1,043 to 1,384 informative chr1 genes per person, 201 in all nine, 1,990 in at least one, a third of them imbalanced by over one log2, median |y| 0.41 to 0.53. The genotype in its allelic form, `slope * (alt on h1 - alt on h2)` at the heterozygous lead eQTL of the gene: 700 gene person pairs on 251 genes, Spearman +0.36 with the ratio, sign agreement 0.63, per person +0.16 to +0.59 and +0.36 on average; at 100 or more allelic reads +0.43 and 0.67. The same genotype against the total deviation was +0.29 (idea 22), so the ratio is the cleaner target the formalism promises. And the total deviation is cis in good part: per person the |ratio| tracks the |deviation| at +0.14 on 840 to 1,138 genes, and on the genes a person deviates by over one log2 the median |ratio| is 0.8 to 1.2 against 0.4 elsewhere, two to three times the imbalance. So the large person specific deviations that idea 23 found the models tracking at +0.15 to +0.19 are largely one allele's doing, which is the case for the haplotype models of idea 7 once there is a haplotype input; the phased loops of the three children are its first cheap test, see the log. ||
| 43 | Loop strength normalised across people, 2D then 3D | Per loop, the PET count over the person's depth and over the typical count at its span, the law's q, as a person specific strength deviation. 2D: the strength deviation at the promoter against the expression deviation, which idea 19's ceiling analysis put at +0.10 with the law relaying +0.08. 3D: the engine's three forms, the saturating target (in), stiffness by strength (37, null), presence sampled per conformation (33, open, needs 50 to 100 conformations). Our resampled depth supersedes the providers' downsampling, which is his HG00512 anomaly. Promised in the meeting. | 2D an afternoon; 3D is idea 33 | 2D done, 2026-09-22, null and it reads on 33. `loop_strength_norm.py`, chr1, 1,903 genes, the RNAPOL2 loops with an end within 2 kb of the TSS, seven forms of the strength summed there, on the global, deviation, large deviation and gene wise statistics. The raw log PET is the best or level on every statistic: global +0.450, deviation +0.100, on the large deviations +0.158, gene wise +0.164. Per million of the person's chr1 PETs +0.450, +0.087, +0.142, +0.115; the law's q, the count over the typical count at the span, +0.451, +0.101, +0.160, +0.161; the rank among the person's loops +0.461, +0.073, +0.131, +0.160; the bare loop count at the promoter +0.467, +0.077, +0.138, +0.165. So no normalisation across people adds to the count as it comes, and the count of loops carries as much gene wise as their PETs. What the split by presence says: matched at 5 kb on both ends, 377 RNAPOL2 loops are in all nine people, 3 percent of a person's, and 6,043 in at least five. The loops in at least five carry a deviation of +0.060 by PET and +0.037 by count; the loops in under five, the person's own, +0.086 by PET and +0.074 by count; the loops in all nine +0.033. The person specific part of the loops at the promoter is which loops a person has, not how strong a shared loop is, and the shared loop's strength is worth +0.03 to +0.06 of the +0.10. CTCF the same at a fifth of the size, raw +0.201 global, +0.019 deviation, the gene wise -0.03 on every form. This bears on 33: sampling loop presence by strength exists to carry a shared loop's count into the ensemble, and that count is the smaller part; the larger part, the loop set, is already in every model. |
| 44 | The loop package for the colleague | Nine people, CTCF and RNAPOL2, genome wide bedpe with PET counts, the hq filter and the resampled depth, the anchor sets, and a README on the resampling, from `data/<S>/<S>_clusters_3+.bedpe` and `<S>_rnapol2_clusters_3+.bedpe`. Klaudiusz packages, the user sends. Promised for 2026-09-23. | an hour | built 2026-09-22 on the laptop, `~/Desktop/trio_loops_2026-09-22.tar.gz`: per sample the two loop files as the runs read them, the two anchor files, the two SPP peak files, a README with the provenance, the resampling rule, the depth table and how the engine reads a PET count, and an md5 manifest. The loops are the depth resampled sets of 2026-09-20 from the merged PET 1 and up libraries, not the providers' downsampled or high quality sets. Awaiting the send. |
| 45 | Genome arms for the paper | Nine people genome wide on the CTCF plus RNAPOL2 arm with the HGSVC map, the user's sbatch, `--array=0-206%12` on `_trio_rnapol2_hgsvc`, then the own element distances genome wide on the workstation at one worker capped at 3 Mb, then ideas 23, 25, 29 and 40 on ten times the genes. The paper's 3D numbers. | a day or two on eden, a day on the workstation | open, the user submits |
| 46 | The paper's 3D section | The NAR result ported to individuals: on the genes where a person differs from the panel by over one log2, the person's spatial deviation tracks the expression deviation at +0.15 to +0.19 and the compartments at +0.20 (23), the trace survives the null at z 6 to 12 (25), the genetic part is separate (22), the level is not tracked; one model of the person specific part from loops, compartments, genotype, peaks and, after 38 and 41, RNAPOL2 occupancy and EBV load (29). Figures from `large_deviations.py` and `person_model.py`. | writing | open |
| 47 | HPRC, the gene wise axis at 206 people | The design's limit is nine people. HPRC has 206 people of 27 ancestries with phased assemblies, methylation, Hi-C and Kinnex expression published, Fiber-seq for 38, and no ChIA-PET or HiChIP. Loops called from each person's Hi-C would give the engine its arcs, the law its exponent, and a model per person; the gene wise statistic then has 206 points per gene instead of nine and the eQTL, EBV and methylation covariates exist for all of them. A different project in compute and data, and the only way past n 9. | weeks; loop calling on 206 maps and 206 genomes of models | open, recorded |

Order: 44 and 40 first, both for the Monday meeting; 39 and 43's 2D form the same day; 38 and
41 when the table and the workstation allow; 45 as soon as eden is free, since everything in
46 wants the genome; 42 after. The engine list, 33 to 36, stands behind 45.

### Meeting record, 2026-09-22 evening

His processing of the RNAPOL2 modality, since it is what lifted his number from nothing to
0.74. The V3 files the lab distributes, Michał Waśniewski's processing, gave him a
correlation near zero with expression; the raw bigWig signal too. What worked, following the
published practice for RNAPOL2 occupancy: the BAMs of the ChIA-PET RNAPOL2 libraries mapped
to hg38, reads counted from TSS plus 300 bp to the gene end so the paused polymerase at the
promoter is excluded, the count divided by the gene length so genes compare within a person,
GM19238 removed as an outlier at 0.36 against 0.72 to 0.78. Dariusz's stance is that the
distributed processing is correct and not worth redoing; the colleague's view, and ours, is
that a normalisation that turns 0 into 0.74 is worth owning. The same held for the loops:
the providers' downsampling left HG00512 with the most loops before and the fewest after,
which is why our inputs are resampled from the raw pairs.

His other modalities on the trios. HiChIP CTCF, the strongest peak within the TSS: global
0.05, gene wise 0, symmetric distributions, nothing. His 3D models, 3DGnome on the HiChIP
CTCF loops with the nearest enhancer distance: global -0.26 to -0.29, gene wise 0. Methylation
of CpG islands within 500 bp of the TSS on HPRC: global -0.43, gene wise 0, 28 genes under
-0.5. EBV on HPRC, counts per EBV gene from the Kinnex BAMs mapped to hg38 plus EBV, CPM: total
EBV expression gene wise +0.10 with 218 genes over 0.5, BHRF1 +0.12 with 1,865. His Xformer,
a sequence to expression model on 1 Mb windows with cross attention between the person's
haplotype and hg38: 0.5 to 0.6 on training chromosomes for seen and unseen people, 0 on held
out chromosomes, so it generalises to people and not to regions. His causal order, top down:
expression, RNAPOL2, methylation, TF binding, sequence; a variant breaks a motif, the factor
does not bind, methylation changes as a side effect, RNAPOL2 follows; CTCF and structure sit
between methylation and RNAPOL2. His data table: the trios have everything, nine people of
three ancestries; HPRC has the published layers for 206 people of 27 ancestries and none of
the lab's ChIA-PET or HiChIP.

His haplotype formalism, slides 24 to 26, which is idea 7 written as a regression. For a
sample s, each haplotype's expression is a shared trans term times a haplotype specific cis
term, `E_h1 = T_s C_h1` and `E_h2 = T_s C_h2`, so the ratio `E_h1 / E_h2 = C_h1 / C_h2`
cancels trans effects, TF abundance, EBV load, batch and depth. With additive cis effects on
the log scale, `log(E_h1 / E_h2) = sum_i beta_i (x_h1,i - x_h2,i)`, a linear model with the
allelic ratio as `y` and the haplotype difference of every feature as `X`, no expression
normalisation needed. Total expression `E_s = T_s (C_h1 + C_h2)` keeps both. His ERAP2 example
on HPRC: dosage alone R2 0.72, haplotype 0.82, haplotype plus TF 0.83.

What we said. The models copy the loops and cannot pass the data ceiling, which on the nearest
element is -0.40 to -0.46; every manipulation of the 3D feature, hubs, own elements, RNAPOL2
peaks as elements, lands there. The NAR result: over all genes the distance change between
cell types correlates with the fold change at nothing, and on the genes that jump a proximity
tertile it reaches -0.6, structure tracks the transitions and not the level; the same on the
individuals, idea 23. Loop strength enters only as a target distance; stiffness by strength
was tried the same day and is null (37); the second idea, each conformation drawing each loop
with a probability from its PET count so the ensemble mean carries the frequency (33), needs
50 to 100 conformations and the GPU time we do not have. Compute as it stands: chr1 for nine
people at ten conformations in about an hour on eight A100s, a genome a day or two, 20 Mb in
ten minutes on the workstation; the engine is latency bound and slower than MultiMM, and more
exact.

Agreed. He gets the loops with PET counts, CTCF and RNAPOL2, via Klaudiusz's package, by
2026-09-23 (44), and normalises them on 2D; we try the same on 2D and 3D (43) and the length
normalised target (39). The presentation of the ideas tried went to him on Zulip. Dariusz
wants the modalities combined in one model, a graph network over the counts in his plan; ours
is idea 29, which already holds loops, compartments, genotype and peaks, and takes 38 and 41
next. The paper: the Hi-C goes into Abhishek's paper, the ChIA-PET into the colleague's, where
the 3D part is ours and a significant share; the target is the Nature Genetics call for 3D
genome papers, open over a year with its deadline moved several times and now the end of
November 2026, with three further journals in the call and NAR as the fallback. He is in
Warsaw in person on Thursdays and for a week or more in November with the visiting professor;
we meet any day but Monday for the user, and the follow up is Monday 2026-09-28 before his
week away.

The NAR manuscript, `NAR26_manuscript`, is the template for 46. Its numbers on the three cell
lines from enhancer3D, 100 conformations a chromosome: all genes r -0.05 to -0.16, proximity
transitions between the small and large tertiles about -0.3 with no expression threshold,
|log2FC| over 2 up to -0.595 (GM12878 against H1ESC -0.591, H1ESC against HFFc6 -0.559,
HFFc6 against GM12878 -0.580); Reactome enrichment of the genes that move to the small tertile
and go up by over two log2 gives immune signalling in the lymphoblastoid, matrix and adhesion
in the fibroblast, neuronal and developmental programs in the stem cell. The individual level
version is ideas 23 and 25 with the person's deviation from the panel in place of the fold
change, and 22 for the part the genotype explains, which the cell line paper has no analogue of.

### Idea 40, the two axes on our features, chr1, nine people, 2026-09-22

`genewise_axis.py`, output `playground/trio_rnapol2_own/genewise_axis_chr1.csv`. Global is
the per person Spearman across genes, mean of the nine; gene wise is the per gene Spearman
across the nine people, mean and median over genes, with the share of genes over +0.5 and
under -0.5 and the share over +0.5 under a null that permutes the people within each gene,
twenty draws. Every feature is signed so that positive is the direction expression is
expected to follow, so a distance reads positive. Genes are the chr1 genes expressed in all
nine that carry the feature in all nine.

| feature | genes | global | gene wise mean / median | over +0.5 | under -0.5 | null over +0.5 |
|---|---|---|---|---|---|---|
| 3D nearest own element, CTCF arm | 1,903 | +0.336 | +0.064 / +0.083 | 0.120 | 0.069 | 0.078 |
| 3D nearest own element, CTCF with RNAPOL2 arm | 1,903 | +0.360 | +0.089 / +0.117 | 0.130 | 0.062 | 0.081 |
| 3D nearest own element, HGSVC map arm | 1,903 | +0.348 | +0.080 / +0.083 | 0.126 | 0.048 | 0.082 |
| 3D hub of own elements within 3 units | 1,903 | +0.327 | +0.081 / +0.104 | 0.135 | 0.075 | 0.091 |
| linear nearest active atlas element | 1,903 | +0.357 | +0.118 / +0.137 | 0.170 | 0.054 | 0.096 |
| linear nearest distal RNAPOL2 anchor | 1,903 | +0.260 | +0.078 / +0.100 | 0.127 | 0.060 | 0.079 |
| RNAPOL2 PET at the TSS, log | 1,903 | +0.450 | +0.164 / +0.200 | 0.180 | 0.042 | 0.084 |
| RNAPOL2 loops at the TSS | 1,903 | +0.467 | +0.165 / +0.197 | 0.188 | 0.041 | 0.088 |
| CTCF PET at the TSS, log | 1,903 | +0.201 | -0.028 / -0.035 | 0.107 | 0.136 | 0.085 |
| RNAPOL2 peak signal at the TSS | 1,903 | +0.464 | +0.121 / +0.150 | 0.178 | 0.071 | 0.081 |
| own enhancer activity, ABC sum | 1,903 | +0.309 | +0.109 / +0.133 | 0.154 | 0.068 | 0.081 |
| own enhancer activity, linear sum | 1,903 | +0.260 | +0.161 / +0.200 | 0.224 | 0.065 | 0.081 |
| compartment E1 at the TSS, own Hi-C | 1,710 | +0.118 | +0.056 / +0.067 | 0.120 | 0.063 | 0.080 |
| genotype, GTEx LCL eQTL prediction | 278 | +0.066 | +0.262 / +0.274 | 0.320 | 0.036 | 0.091 |
| his RNAPOL2 over the gene body, eight people | ~14,000 | +0.74 | +0.17 | 0.30 | | 0.11 |
| his 3D models, HiChIP CTCF loops | | -0.28 | 0 | | | |
| his methylation of CpG islands, HPRC | | -0.43 | 0 | | | |

Three readings. The gene wise axis orders the features differently from the global one:
the RNAPOL2 peak signal is second on the global axis and fifth on the gene wise, the
genotype is last on the global and first on the gene wise, so the two axes are two
questions, as his slides say. Everything cis and epigenomic sits between +0.06 and +0.17
gene wise, his gene body occupancy included, and only the genotype is above. And the CTCF
PET at the promoter is the one feature that reads negative gene wise, more genes under -0.5
than over, a small number worth one look in the genome arms.

## Log

- 2026-09-20. Question raised, diagnostics run, baselines set, list written.
- 2026-09-20. Idea 1 tried on the RNAPOL2 arm, dropped as a replacement for the nearest distance, hub size kept for idea 5. Idea 2 first look from the same run, open until more conformations. Figure `playground/trio_rnapol2/load/enhancer_load_chr1.png`, table beside it.
- 2026-09-20. Idea 3 tried, dropped: the residual from the polymer expectation loses to the raw distance and adds nothing beyond it. `playground/trio_rnapol2/residual/`.
- 2026-09-20. Idea 2 closed on the soft contact at ten conformations: the more contact like the feature, the worse, monotonically. Dropped.
- 2026-09-20. Idea 4 tried: the person's own active elements raise the raw number, the active atlas 0.39 to 0.46 and the distal loop anchors 0.36 to 0.42, the unbound atlas is inert, but nothing beats the atlas beyond the input and most of every set's signal is an element inside the gene's own span. Adopted as the element set for idea 5. `playground/trio_rnapol2/own_keep/` and `own/`.
- 2026-09-20. Idea 5 done: out of fold, the 3D block adds -0.01 to +0.01 beyond the input and +0.00 to +0.03 beyond linear, in every person. `playground/trio_rnapol2/model/`.
- 2026-09-20. Idea 6 done in the same model with the silent genes kept: every fit up a few hundredths, the gain of 3D unchanged.
- 2026-09-21. Summary section written; ideas 19 and 20 are the next arm, one eden array.
- 2026-09-23, 02:00. Idea 38 closed: with expression per kb as well, protein coding, the loop ends give 0.70 against his 0.74; a quarter of it is the shared length divisor, the substantive part about 0.45, level with the loops and the nearest element.
- 2026-09-23, 01:30. Idea 38's one off: the gene wise axis reproduces from the cis clusters' ends, +0.175 against his +0.174; the global reaches 0.58 to 0.67 against his 0.74 because the cis clusters hold no pair under 8 kb, the 1D coverage; a RNAPOL2 .hic, his table or the BAMs would close it.
- 2026-09-23, 00:55. Idea 7's first test on the allelic ratio: the children's phased RNAPOL2 loops agree with the allele's expression in 55 percent of pairs, 59 percent and p 0.016 in GM19240; CTCF nothing. Thin.
- 2026-09-23, 00:40. Idea 42 done: the allelic ratio exists for 1,000 to 1,400 chr1 genes a person; the genotype reaches it at +0.36 (+0.43 at depth) against +0.29 on the total deviation; the large deviations carry two to three times the allelic imbalance, so they are cis in good part. Idea 41's control: in the gene fold model the EBV block equals person identity; the across people fold is the evidence.
- 2026-09-23, 00:05. Idea 41 done: EBNA-1's level is the first axis of the person specific expression, a third of its variance, out of fold +0.22 above every permutation and within families; the lytic load reads on the RNAPOL2 libraries; a trans part no cis modality can carry.
- 2026-09-22, 23:30. Ideas 41 and 42 running on the workstation: EBV load from a combined salmon index, then the diploid chr1 quantification for the allelic ratio, chained.
- 2026-09-22, late. Idea 39 done, dropped: TPM as the target lowers every number a little and leaves the 3D gain at nothing; counts stay.
- 2026-09-22, late. Idea 43's 2D form done, null: no strength normalisation beats the raw count; the person specific part of the promoter loops is loop presence (+0.086), the shared loops' strength +0.03 to +0.06; 33's ceiling is that.
- 2026-09-22, late. Idea 40 done: on the colleague's gene wise statistic our 3D reads +0.06 to +0.09, the loops at the promoter +0.165 level with his gene body occupancy, the genotype +0.26; table in the note.
- 2026-09-22, evening. Idea 44: the loop package built on the laptop with its README, awaiting the send.
- 2026-09-22, evening. Sixth list after the lab meeting: 38 RNAPOL2 over the gene body, 39 length normalised target, 40 the colleague's gene wise statistic on our features, 41 EBV load, 42 allelic expression, 43 loop strength normalised across people, 44 the loop package, 45 genome arms, 46 the paper's 3D section. Deadline the Nature Genetics call, end of November.
- 2026-09-22. Idea 37 done, null: exponent 1 lifts loop fidelity a hundredth or two and the strongest third not at all; every expression number level; left at 0.
- 2026-09-22. Idea 32 done: 3,000 iterations level with 800 on fidelity, deviation and idea 5; the tail is inert. Idea 37 built: `[springs] arc_weight_exponent`, weights in the solver on both backends and the numba annealer, tests, parity at zero.
- 2026-09-22. Idea 37 added with its pre test: the under realised loops are the crowded ones; stiffness by strength is the targeted lever, 33 the distribution changing one.
- 2026-09-22. The traced solve: the energy creeps at 1e-6 per iteration with no plateau; 800 is 1.7 percent and 3,000 is 0.7 percent above 8,000. Candidate `solver_tol` 1e-6 with a 5,000 cap, pending the 3,000 arm.
- 2026-09-22. Idea 32 running at 3,000, which still hits the limit; the cap is replaced by an energy stop rule, `solver_tol`, chosen from a traced solve.
- 2026-09-22. Idea 31 done: the solve keeps a fifth of the law's person specific target variation and under realises the strong loops, realised over target 1.3 to 1.65; the loss is in the solve.
- 2026-09-22. Fifth list, the engine: 31 loop fidelity diagnostic, 32 converge the arcs solve, 33 loop presence sampled per conformation, 34 the person's full Hi-C as the long range background, 35 RNAPOL2 many body, 36 more conformations.
- 2026-09-22. Ideas 27 and 28 done and dropped: the eQTL variant is not what the models bring near, and the loops and peaks at the variant do not follow the genotype; the genetic part never enters the data.
- 2026-09-22. Idea 29 done: one model of the person specific part reaches +0.20 to +0.31 out of fold from loops, compartments, genotype and peaks; the 3D block's unique gain is 0.
- 2026-09-22. Idea 26 done: the person's own enhancer activity matches the nearest distance raw and adds nothing out of fold; its linear only form carries +0.08 of the person specific part, the contact weighted forms less.
- 2026-09-22. Idea 25 done: the person specific trace survives a within gene permutation null at z 6 to 12 on every feature and gene set.
- 2026-09-22. Fourth list written: 25 a null for the large deviation trace, 26 the person's own enhancer activity, 27 the eQTL variant as the element, 28 loops at the variant, 29 one model of the person specific part, 30 fifty conformations.
- 2026-09-22. Idea 24 done and dropped: the compartment term at 0.5 in the energy leaves every number where the HGSVC arm had it; the models' deviation tracks the person's E1 deviation at +0.02 with or without it.
- 2026-09-22. Idea 24 prepared: the compartment term on each person's own track, HGSVC map arm, configs and tracks on eden, waits for the sbatch.
- 2026-09-22. Idea 22 done: the person specific deviation is cis genetic in good part, GTEx eQTLs and the nine's genotypes predict it at +0.29; the loops, distances and compartments do not track the genetic part; E1 adds beyond genotype and loops on the large deviations.
- 2026-09-22. Idea 23 done: on the large person specific deviations the spatial deviation tracks expression at +0.15 to +0.19 and the compartments at +0.20; out of fold beyond the loops E1 +0.07, 3D +0.02.
- 2026-09-22. Third list written: 23 the large deviations, 22 the genetic explanation, 24 compartments in the energy, then 7, 18, 15 and the genome arms.
- 2026-09-22. Idea 21 stopped after one person by decision: MultiMM -0.38 raw and -0.12 partial against ours -0.40 and -0.14, agreement 0.72. The engine is not the limit.
- 2026-09-22. Write up of where the line stands at the top of the note; idea 21, MultiMM on the same loops, added and started.
- 2026-09-22. Idea 10 done: the lab's counts are a count of the HGSVC libraries (column totals 1.44 to 1.52 times ENA's pairs in every sample); two quantifications agree on each person's deviation at 0.91.
- 2026-09-22. The HGSVC arm measured: agrees with the ChIA-PET map arm at 0.83 to 0.91 per gene, partial -0.19 against -0.17, deviation +0.05 against +0.06, the 3D block beyond the loops -0.003 to +0.007. The background is background.
- 2026-09-22. The HGSVC chr1 arm ran on eden, job 1809232, the law measuring 0.25 to 0.26 on every person's Hi-C against 0.40 on the ChIA-PET derived map, 5 to 7 minutes a conformation; models on the workstation, enhancer3d running. Idea 9's pre test negative. HGSVC's trio ASE tables found for idea 7.
- 2026-09-21, late. All nine 4DN mcools on the workstation, chr1 singletons at 25 kb on eden under `data/<S>/`, compartment tracks for nine; idea 16 rerun with nine. The HGSVC arm waits for the sbatch.
- 2026-09-21. Idea 15 cheap form and idea 16 done: promoter CpG lifts the gene block and changes nothing for 3D; the person's own compartments carry a +0.08 person specific trace and add nothing out of fold.
- 2026-09-21. Ideas 11 to 14 done and dropped on the RNAPOL2 arm with own elements: length proxies, promoter hubs at zero, variability tracks the mean, on and off the same as level; nothing adds out of fold.
- 2026-09-21. Idea 17 done and dropped: the difference between the arms carries nothing; the RNAPOL2 arm is the same feature a little stronger.
- 2026-09-21. Idea 20 done and adopted: own element beds per person, both chr1 arms rerun; raw and deviation up, the partial beyond the set's linear proximity unchanged.
- 2026-09-21. Idea 19 dropped at the data level: the law relays the same person specific signal at every `contact_half_saturation`, the loss is downstream. `trio_configs.py --half-saturation` exists, unused.
- 2026-09-21. Idea 8 reframed and done: the person specific ceiling is in the RNAPOL2 loops at the promoter, +0.10 on the deviation statistic; the nearest atlas distance keeps +0.016 of it, the person's own anchors +0.06; the law's saturation and the atlas lose it.
- 2026-09-21. The Drive folder's Hi-C is the ChIA-PET pair map already in use. HGSVC has independent Hi-C, RNA-seq and phased VCFs for the nine; the HGSVC chr1 arm is prepared and waits for the sbatch.
- 2026-09-21. Where the limit sits: the design has no person specific expression (0.97 agreement, another person's model predicts as well as one's own), the engine relays the loops on both arms, the ensemble costs about 0.06, the feature saturates at 0.35 to 0.39. Three levers named.
- 2026-09-20. Idea 7 inventoried: the phased loops are two to four percent of a person's loops and single read labels, and there is no allele specific expression. Blocked on two data requests to the lab.
