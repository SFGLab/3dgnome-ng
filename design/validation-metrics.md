# Validation metrics

Every metric the `validation/` package computes, with its source, its definition and what it establishes. The
metrics answer three questions. The first is agreement of the reconstructed structure with the contacts that built
it. The second is agreement with independent data the model did not see. The third is physical sanity as a polymer.

The reference numbers come from the 3D-GNOME paper of 2016 (Szałaj et al., Genome Research), its supplement, the
MultiMM paper of 2024, hicrep and Lieberman-Aiden 2009. Function names below point into
`validation/metrics/structure.py` and `validation/metrics/hic.py`.

## Summary

| Metric | Function | Question | Source |
|--------|----------|----------|--------|
| self-consistency | `metrics.self_consistency` | reproduces its input | 3D-GNOME 2016 supplement |
| scaling laws, distance scaling β | `metrics.distance_scaling` | physically sane | fractal globule, Lieberman-Aiden 2009 |
| scaling laws, contact decay α | `metrics.contact_probability` | physically sane | chromatin power law |
| ensemble diversity, contact measure d_AB | `metrics.dab_matrix` | ensemble spread | 3D-GNOME 2016 supplement |
| cross-data correlation | `metrics.hic.cross_data_correlation` | agrees with Hi-C at data level | 3D-GNOME 2016 Fig. 2 |
| Overlap fraction | `metrics.overlap_fraction_binned` | physically sane | excluded volume test |
| Hi-C SCC | `metrics.hic.ensemble_hic_correlation` | agrees with Hi-C | hicrep |
| Hi-C inverse-distance Pearson | `metrics.hic.multimm_faithful_pearson` | agrees with Hi-C | MultiMM 2024 |
| Reconstruction fidelity | `metrics.contact_measure`, `metrics.rmsd_superpose` | recovers a known structure | 3D-GNOME 2016 supplement, Fig. S19-S22 |

## Self-consistency

**Definition.** Spearman rank correlation between the input interaction frequency of each contact and the 3D
distance between the two beads it maps to in the reconstructed structure. A strong contact belongs at a short
distance. A good structure gives a negative rho.

**Source.** The 2016 supplement describes self-consistency as the correlation of the input heatmap interaction
frequency against model 3D distance. This is the paper's own self-consistency check.

**Validates.** The structure honours its input contacts. The metric is fed the full ChIA-PET, the cluster loop
strengths together with the singletons. The cluster PET counts carry the real dynamic range. Singletons alone are
almost all count one or two and give a near-binary signal that goes to noise on small regions.

## Scaling laws, distance scaling exponent β

**Definition.** Mean 3D distance R grows with genomic separation s as a power law R(s) proportional to s to the β.
We fit β on the log-log means of distance against separation over the scaling window, excluding the sub-resolution
small separations and the saturated tail. `LAW_BANDS` sets the plausible band 0.15 to 0.60 with canonical value 1/3.

**Source.** Polymer physics. A fractal globule has β near 1/3. An ideal chain has β near 1/2.

**Validates.** Large-scale folding. β near 1/3 is the signature of knot-free fractal-globule chromatin. A value well
below the band means the structure is over-compact. The exponent is only meaningful over a large or whole-chromosome
region where several decades of separation exist. A single small interaction block is one globule and reads flatter.

## Scaling laws, contact probability exponent α

**Definition.** The probability that two beads are within a contact radius decays with genomic separation s as P(s)
proportional to s to the minus α. Fit the same way as β, reported as a positive α. Band 0.50 to 1.60 with canonical
value 1.0.

**Source.** The contact-probability power law measured in Hi-C. Chromatin is near α equal to 1.

**Validates.** Contact frequency falls off with genomic distance at the biological rate. Together with β and the
bond-length coefficient of variation it forms the physical-sanity check. A bond CV above 3 marks a degenerate or
exploded structure and invalidates β and α.

## Ensemble diversity, contact measure d_AB

**Definition.** A distance between two structures A and B. For every bead pair take the difference of their two 3D
distances, divide by an expected distance ED, square and average.

```
d_AB = mean over i<j of  ((D_A[i,j] - D_B[i,j]) / ED[i,j])^2
```

ED is the known reference distance for a synthetic run, otherwise the mean of the two distances. The measure compares
distance matrices and is blind to rotation, translation and chirality. `ensemble_diversity` reports the median
off-diagonal d_AB across the ensemble.

**Source.** The 2016 supplement defines this measure and argues it suits chromatin better than RMSD, which reports
large values for mirror-image structures that a contact map cannot tell apart.

**Validates.** Ensemble spread. ChIA-PET data is a population average and the model produces an ensemble. A low
median d_AB means the structures agree, a high one means real conformational diversity.

## Cross-data correlation, ChIA-PET against Hi-C

**Definition.** Correlate the input ChIA-PET contact heatmap against experimental Hi-C on a common bin grid.
`cross_data_correlation` reports the Pearson correlation of log-transformed contacts and, separately, an
observed-over-expected correlation that divides each entry by its genomic-distance diagonal mean before correlating.

**Source.** The 2016 paper Fig. 2. Panel A is inter-chromosomal at chromosome level with rho 0.73. Panel B is
intra-chromosomal on chromosome 3 binned at 1 Mb with rho 0.67.

**Validates.** The input ChIA-PET carries the same genome-wide signal as Hi-C. It is a data-level check with no
structure involved and is identical across the reference, parity and tuned variants. Our intra-chromosomal
cross-data correlation is compared at the paper's 1 Mb resolution, where the raw Pearson is about 0.76 across GM12878
regions. At 100 kb a CTCF ChIA-PET map fills only about 7 percent of bin pairs and the correlation is underpowered.
The observed-over-expected number strips the shared distance decay and reports the structure-only agreement, which is
a genuine 0.2 to 0.3 for a loop-specific assay against all-contact Hi-C.

**Panel A is unavailable in our data.** The GM12878 ChIA-PET has zero inter-chromosomal contacts and the panel A
comparison cannot be reproduced here.

## Overlap fraction

**Definition.** The fraction of non-bonded bead pairs closer than a contact radius. `overlap_fraction` measures it
at the native bead resolution. `overlap_fraction_binned` first coarse-grains the structure to a fixed base-pair bin.
Two structures at different bead densities then compare on equal terms.

**Source.** The physical excluded-volume and confinement test. The 2016 paper noted that its models produced
physically impossible overlaps between loops, a consequence of not enforcing excluded volume.

**Validates.** Steric plausibility. Fewer overlaps mean fewer beads occupying the same space. Use the
resolution-normalized value for any comparison across variants. The tuned model has finer beads and its raw overlap
is inflated by density.

## Hi-C stratum-adjusted correlation

**Definition.** Build a simulated contact map by counting reconstructed bead pairs within a contact radius, binned to
the Hi-C grid, then correlate against experimental Hi-C with the stratum-adjusted correlation coefficient. The
coefficient correlates within each genomic-distance stratum and averages, which removes the distance decay that would
otherwise dominate a plain correlation.

**Source.** hicrep. It is the standard decay-aware contact-map similarity.

**Validates.** The reconstructed structure reproduces Hi-C beyond the trivial distance decay.
`ensemble_hic_correlation` aggregates over the whole ensemble first, as Hi-C is a population average.

## Hi-C inverse-distance Pearson

**Definition.** Place a bead centroid per Hi-C bin, form a simulated map as 1 over (d plus 1) cubed between
centroids, neutralise the main diagonal and its five neighbours on both maps and take the Pearson correlation against
ICE-balanced Hi-C.

**Source.** MultiMM 2024. This reproduces the metric MultiMM actually computes, following its validation code and not
the looser wording in its text.

**Validates.** Decay-retained agreement with Hi-C. The value scales with region size and is compared between variants
at the same geometry, not against an absolute target.

## Reconstruction fidelity

**Definition.** Reconstruct a known synthetic structure from its own contact heatmap and measure how close the result
is to the truth. `contact_measure` is the ensemble-diversity d_AB against the ground truth, with ED set to the
ground-truth distances. `rmsd_superpose` is the Kabsch superposition RMSD, taken as the minimum over the reflection.
A mirror image scores as a match. Structures are scale-matched to the truth before scoring, since the model works in
normalized units.

**Source.** The 2016 supplement section III and IV, figures S19 and S20 for synthetic structures and S21 and S22 for
real chromosomes.

**Validates.** The engine recovers a structure it should be able to recover. This is the paper's core model
validation and the only one that can measure absolute fidelity, the true structure being known. Noise robustness is
the same measurement across increasing Gaussian noise on the heatmap. Fidelity that degrades gently with noise shows
the solution is stable. See `validation/studies/synthetic.py`.
