# Expression from structure

The question, 2026-09-20: can the 3D models explain more of a person's RNA-seq expression than
they do now, and what limits them. The setting is the one that stands after the RNAPII work
in `rnapii-loops.md`: the CTCF with RNAPOL2 trio arm, each of the nine people on their own
model and their own expression, nothing averaged over people and no fold change across cell
lines, since a person has one sample. chr1 at ten conformations until the genome arms run.
This note holds the numbers to beat, what has been ruled out, and the ideas in the order they
are to be tried, each with its status, so it is the tracker.

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
| 7 | Haplotypes, the within person transition | The folder holds phased loops for every sample. A maternal and a paternal model of one nucleus with allele specific expression from the same RNA-seq gives each gene a fold change with the trans environment fixed, the port of enhancer3d's between cell type result that stands. Needs haplotype models and allele specific counts, neither built. | open, blocked on data, inventory done 2026-09-20 | The folder's phased sets, `<S>_Maternal.txt`, `_Paternal.txt`, `_Crossed.txt` beside `_nonPhased.txt`, for both factors, `playground/phased_inventory.py`. On chr1 a person has 220 to 750 CTCF and 11 to 620 RNAPOL2 loops labelled maternal and as many paternal, two to four percent of the loops, and 0 to 13 loops in both files, so the labels are exclusive. They are also thin: the label column reads `M_NA` or `NA_M`, one anchor phased and the other not, at a median PET of 4 to 6, so a label is one or two phaseable reads and not a per haplotype count, and a loop on both chromosomes with one phased read reads as maternal. A maternal model would be the unphased set plus a few hundred such loops. Genes with a TSS on a maternal labelled anchor, 24 to 228 per person and factor, 8 to 139 protein coding, and as many paternal. The expression side does not exist here: the count sheet is per gene per sample. Two things are needed before the arm is worth building. From the lab, per haplotype PET counts per loop or the phased BAMs, so a loop can be called allele specific on counts rather than on a label; and allele specific expression per gene from the RNA-seq reads against the phased genotypes, which for these nine are the public 1000 Genomes phased VCFs. With both, the arm is two haplotype cluster files per sample from `trio_prepare.py`, an eden array of two haplotypes by nine samples, chr1 at 13 minutes a conformation, and the test is per person the maternal minus paternal distance against the allelic log ratio on the genes with a phased loop at the promoter, a few hundred a person at the genome. |

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
| 8 | The data ceiling, before any model: the person's own Hi-C at the promoter | Per gene per person the contact of the promoter with its enhancers from the 4DN mcool at 10 kb, its deviation from the panel against the expression deviation. If the raw Hi-C carries no person specific enhancer signal, no model of it can. | hours once the mcools are here | open |
| 9 | Structural variants, the 3D-GNOME 2.0 design | HGSVC's haplotype resolved SV calls for these nine. Genes whose enhancer or loop anchor a person's deletion, duplication or inversion removes or moves; that person's expression deviation at those genes, sign by SV type. Person specific by construction and the lab's own lineage. A data only pre test needs no model. | a day for the pre test, a week with the engine | open |
| 10 | Reliability of the person specific expression | HGSVC's mRNA-seq is a second measurement of each person. Quantified against the lab's counts, the agreement of the two on each person's deviation from the panel is the ceiling of anything person specific. | a day, salmon on the workstation | open |
| 11 | Gene looping and gene compaction | 3D distance TSS to TES and the radius of gyration of the gene body, controlling length. RNAPII recycling by gene loops is old literature; the RNAPOL2 anchors put beads inside bodies. | hours | open |
| 12 | Promoter hubs, transcription factories | Other active promoters within a 3D radius, and promoter to promoter loops in the RNAPOL2 set, Li 2012's multigene complexes. | hours | open |
| 13 | Ensemble variability as the feature | The spread over conformations of the nearest enhancer distance, stable against fluctuating contacts, a bursting reading. | hours | open |
| 14 | On and off before level | Zuin 2022's saturation says structure may set whether a gene is on more than how much. A classifier on silent against expressed with the 3D block, then level among the expressed. | hours | open |
| 15 | Expression residual to sequence | Take out what promoter sequence predicts, CpG class as the cheap proxy and Enformer as the real one, and ask what structure explains of the residual. Changes the target rather than the feature. | a day cheap, a week with Enformer | open |
| 16 | The person's compartments | 4DN ships a compartment and an insulation track per person from their Hi-C. Compartment deviation from the panel against expression deviation, and the tracks as features. | hours | open |
| 17 | What the RNAPOL2 loops did to the structure | Per gene the CTCF arm distance minus the RNAPOL2 arm distance, the geometry the second factor added. Both arms exist. | hours | open |
| 18 | Inter chromosomal | Li 2012's complexes cross chromosomes. The genome arms model chromosomes together at the top level, so a promoter's neighbours on other chromosomes are a feature only the genome run can give. | with the genome arms | open |

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
- 2026-09-21. The Drive folder's Hi-C is the ChIA-PET pair map already in use. HGSVC has independent Hi-C, RNA-seq and phased VCFs for the nine; the HGSVC chr1 arm is prepared and waits for the sbatch.
- 2026-09-21. Where the limit sits: the design has no person specific expression (0.97 agreement, another person's model predicts as well as one's own), the engine relays the loops on both arms, the ensemble costs about 0.06, the feature saturates at 0.35 to 0.39. Three levers named.
- 2026-09-20. Idea 7 inventoried: the phased loops are two to four percent of a person's loops and single read labels, and there is no allele specific expression. Blocked on two data requests to the lab.
