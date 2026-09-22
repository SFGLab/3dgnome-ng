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
and tertile pictures replicate nine times. The person specific part exists in the data and is
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
genes. One more arm is queued as idea 21, MultiMM's models on the same loops, as a one off.

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
   the person's compartments (16, seven of nine). Still open: the genome arms for inter
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
| 7 | Haplotypes, the within person transition | The folder holds phased loops for every sample. A maternal and a paternal model of one nucleus with allele specific expression from the same RNA-seq gives each gene a fold change with the trans environment fixed, the port of enhancer3d's between cell type result that stands. Needs haplotype models and allele specific counts, neither built. | open, blocked on data, inventory done 2026-09-20 | The folder's phased sets, `<S>_Maternal.txt`, `_Paternal.txt`, `_Crossed.txt` beside `_nonPhased.txt`, for both factors, `playground/phased_inventory.py`. On chr1 a person has 220 to 750 CTCF and 11 to 620 RNAPOL2 loops labelled maternal and as many paternal, two to four percent of the loops, and 0 to 13 loops in both files, so the labels are exclusive. They are also thin: the label column reads `M_NA` or `NA_M`, one anchor phased and the other not, at a median PET of 4 to 6, so a label is one or two phaseable reads and not a per haplotype count, and a loop on both chromosomes with one phased read reads as maternal. A maternal model would be the unphased set plus a few hundred such loops. Genes with a TSS on a maternal labelled anchor, 24 to 228 per person and factor, 8 to 139 protein coding, and as many paternal. The expression side does not exist here: the count sheet is per gene per sample. Two things are needed before the arm is worth building. From the lab, per haplotype PET counts per loop or the phased BAMs, so a loop can be called allele specific on counts rather than on a label; and allele specific expression per gene from the RNA-seq reads against the phased genotypes, which for these nine are the public 1000 Genomes phased VCFs. With both, the arm is two haplotype cluster files per sample from `trio_prepare.py`, an eden array of two haplotypes by nine samples, chr1 at 13 minutes a conformation, and the test is per person the maternal minus paternal distance against the allelic log ratio on the genes with a phased loop at the promoter, a few hundred a person at the genome. Expression side located 2026-09-22: HGSVC's `201707_ASE_RES_Trios` holds WASP corrected SNP allele specific expression for all nine on phased SNPs (`data/_hgsvc/201707_Table_1_ASE_SNP_RES_Trios.xlsx`, GRCh38), but it is the table of significant sites only, 6,011 rows over the nine, 30 to 58 chr1 genes a person at 20 reads or more; against 24 to 228 genes with a phased loop at the TSS the overlap is a handful, so the arm still needs allele counts at every het SNP, which the HGSVC RNA-seq now downloading can give against the phased VCFs. |

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
| 21 | MultiMM's models on the same loops, a one off | A second engine on the same input: MultiMM 2.0.2 on each person's CTCF and RNAPOL2 loops on chr1, `playground/multimm_arm.py --n-beads 50000`, 5 kb beads, ten members, loops only and loops with the person's own compartment track, its minimised and its after dynamics ensembles. Then the same pipeline, own elements, the per person scripts, the deviation statistic and the idea 5 model. If a different energy and a cleaner ensemble keep more of the person specific part or add out of fold, the engine was the limit; if they land where ours do, the input was. | a day of GPU on the workstation, an afternoon to judge | open, running |

8, 10 and 9's pre test are data questions and come first, since a negative on them ends the
line. 11 to 14 and 17 are afternoon tests on the models there are.

## Log

- 2026-09-20. Question raised, diagnostics run, baselines set, list written.
- 2026-09-20. Idea 1 tried on the RNAPOL2 arm, dropped as a replacement for the nearest distance, hub size kept for idea 5. Idea 2 first look from the same run, open until more conformations. Figure `playground/trio_rnapol2/load/enhancer_load_chr1.png`, table beside it.
- 2026-09-20. Idea 3 tried, dropped: the residual from the polymer expectation loses to the raw distance and adds nothing beyond it. `playground/trio_rnapol2/residual/`.
- 2026-09-20. Idea 2 closed on the soft contact at ten conformations: the more contact like the feature, the worse, monotonically. Dropped.
- 2026-09-20. Idea 4 tried: the person's own active elements raise the raw number, the active atlas 0.39 to 0.46 and the distal loop anchors 0.36 to 0.42, the unbound atlas is inert, but nothing beats the atlas beyond the input and most of every set's signal is an element inside the gene's own span. Adopted as the element set for idea 5. `playground/trio_rnapol2/own_keep/` and `own/`.
- 2026-09-20. Idea 5 done: out of fold, the 3D block adds -0.01 to +0.01 beyond the input and +0.00 to +0.03 beyond linear, in every person. `playground/trio_rnapol2/model/`.
- 2026-09-20. Idea 6 done in the same model with the silent genes kept: every fit up a few hundredths, the gain of 3D unchanged.
- 2026-09-21. Summary section written; ideas 19 and 20 are the next arm, one eden array.
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
