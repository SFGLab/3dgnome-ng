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
| 4 | The person's own regulatory elements | All nine use the GM12878 atlas. Each sample's RNAPOL2 peaks are fetched, and RNAPOL2 bound non promoter sites are that person's active elements; a hub of RNAPOL2 peaks within a 3D radius is the transcription factory view. Ideas 1 to 3 on that set. | open | |
| 5 | A model, not one correlation | Per person, cross validated regression of log expression on the 3D features, the linear features, the promoter RNAPOL2 PET, CTCF, gene length and type; the out of fold gain from the 3D features is the answer to how much more. | open | |
| 6 | The expression side | Protein coding only, silent genes kept or on and off fitted separately from level, length normalisation for the regression. Worth a few hundredths, see above. | open | |
| 7 | Haplotypes, the within person transition | The folder holds phased loops for every sample. A maternal and a paternal model of one nucleus with allele specific expression from the same RNA-seq gives each gene a fold change with the trans environment fixed, the port of enhancer3d's between cell type result that stands. Needs haplotype models and allele specific counts, neither built. | open | |

Ideas 1 to 3 are features from the pairwise distances the pipeline already computes, about a
day of work with 5 on top, and they run on the models there are.

## Log

- 2026-09-20. Question raised, diagnostics run, baselines set, list written.
- 2026-09-20. Idea 1 tried on the RNAPOL2 arm, dropped as a replacement for the nearest distance, hub size kept for idea 5. Idea 2 first look from the same run, open until more conformations. Figure `playground/trio_rnapol2/load/enhancer_load_chr1.png`, table beside it.
- 2026-09-20. Idea 3 tried, dropped: the residual from the polymer expectation loses to the raw distance and adds nothing beyond it. `playground/trio_rnapol2/residual/`.
- 2026-09-20. Idea 2 closed on the soft contact at ten conformations: the more contact like the feature, the worse, monotonically. Dropped.
