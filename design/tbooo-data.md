# tbooo data for the 3D-GNOME site

Written 2026-11-18, before any of the work below was started. It records the contents of tbooo, the
parts the site can use, the changes tbooo itself needs, and the order of the work.

tbooo is The Biobank Of Our Own, https://github.com/SFGLab/tbooo. It rebuilds the data layout of a
biobank from the 1000 Genomes Project and the Simons Genome Diversity Project.

## Copies

| Copy | Path | Commit | Notes |
|---|---|---|---|
| laptop | `~/Desktop/biobank/tbooo` | 2d7669f | Newest code. Holds the expression data and no genotypes. |
| eden | `/mnt/evafs/groups/sfglab/data/tbooo` | ef082eb | 8.8 TB of raw data. |
| frodo | `/mnt/raid/nik/tbooo` | 10bbebf | The same raw data and 8.4 TB of intermediates in `data/tmp`. |

The raid on frodo was full on 2026-11-18, with 0 bytes free of 44 TB.

## Contents

The table states the contents of the files. The README of tbooo differs in two places, listed in the
next section.

| Data | Path in `data/` | Size | Contents |
|---|---|---|---|
| 1000 Genomes 30x cohort files | `raw/1kg/pvcf/nygc_c*.pvcf.gz` | 37 GB | GRCh38, phased, 3,202 samples. SNVs and indels. No symbolic structural variant. About 2,200 alleles over 50 bp on chr22, with their sequence. Sample columns are numeric ids. |
| Sample table | `metadata/eid_map_1kg.tsv` | small | Numeric id, sample, population, superpopulation, sex. 26 populations. |
| Pedigree | `raw/1kg/20130606_g1k_3202_samples_ped_population.txt` | small | Father and mother of each sample. 612 samples have a parent listed. |
| 1000 Genomes Phase 3 | `raw/1kg/ALL.chr*.vcf.gz` | 17 GB | GRCh37, 2,504 samples. |
| SGDP per sample | `raw/sgdp/vcf/` | 7.6 TB | 277 samples, 27 GB each. One record per base of the genome. |
| SGDP merged | `raw/sgdp/pvcf/` | 1.1 TB | The same records merged per chromosome. |
| GEUVADIS | `raw/geuvadis/` | 91 MB on eden, 6.2 GB on the laptop | RPKM per gene for 462 lymphoblastoid lines, all of them 1000 Genomes samples. |
| MAGE | `raw/mage/`, laptop only | 107 MB | TMM expression of 20,154 genes for 731 lymphoblastoid lines of 26 populations, GRCh38. |
| GTEx v10 | `raw/gtex/`, laptop only | 381 MB | TPM for whole blood, muscle, lung and thyroid. Not 1000 Genomes samples. |

## Differences from the README of tbooo

1. The SGDP files are on hs37d5, which is GRCh37. The genotypes are unphased. The README says GRCh38 and
   phased. The files are the project's own release, `*.annotated.nh2.vcf.gz`, with a record for every base.
2. tbooo holds no structural variants. It downloads the release of 2020-10-28,
   `20201028_3202_phased`, which has SNVs and indels alone. The release of 2022-04-22,
   `20220422_3202_phased_SNV_INDEL_SV`, phases the structural variants together with them. The old
   3D-GNOME server took its per sample files from that release, and the site's 1KGP call set was merged
   from those files on 2026-11-18.

The README also counts 300 SGDP samples. The directory holds 277.

## Work on the site

Each step is separate and can be released alone. The order is the order of the sections.

### 1. Sample table for the 1000 Genomes call set

The sample field takes a typed sample and suggests from 3,202 bare names. A reader who wants a Yoruba
sample or a trio has to know the names.

Data. One file beside the call set, `data/svs/1KGP.samples.tsv`, with the columns sample, sex,
population, superpopulation, father and mother. It is built from the two small tbooo files above.

Server. A call set of the catalog gets an optional `samples_table`. A new route returns the table of
one call set. The catalog route keeps the bare list. Every page load reads the catalog, and the
table is needed in the variants step alone.

Interface. A suggestion shows the sample with its population and sex. Typing a population code or a
superpopulation lists its samples. A line below the field states the parents of a child and the
children of a parent, each as a link that puts that sample in the field. The job page shows the same
facts beside the sample.

Checks. The table has 3,202 rows and every sample of the call set is in it. The pedigree gives 602
trios in the paper and 612 samples with a parent listed in the file, and the difference is read before
the trio line is written.

### 2. Small variants in CTCF motifs

The site applies structural variants. A person's SNVs change CTCF sites far more often. The variant
package already has the method in `snv.py`. For each anchor it scores the CTCF motifs on a haplotype
with that haplotype's SNVs put in. An anchor loses its loops on a haplotype where every motif falls
below the threshold, and the rules rewire its partners. `rewrite.apply` takes these anchors as
`knockouts`.

Data. The phased panel of 2022-04-22 from the 1000 Genomes server, about 40 GB for chromosomes 1 to
22, in `data/variants/1KGP/` on the raid of aragorn. It carries the sample names, the SNVs, the indels
and the structural variants of all 3,202 samples in one phasing. The tbooo cohort files are the older
release under numeric ids and are not used for the site.

Server. A reader for the SNVs of one sample in the anchors of a region, through tabix. The job applies
the knockouts before the structural variants. The status counts the anchors lost. The variants dialog
lists each SNV that breaks a motif with the anchor and the loops it removes.

Interface. One switch in the variants step, off for a job without a hosted sample. An uploaded VCF
with SNVs works the same way.

Long indels. The panel writes alleles over 50 bp with their sequence, where the structural variant
records give an insertion as a length. An inserted sequence can carry a CTCF motif. These alleles are
read as insertions and deletions with sequence where no symbolic record overlaps them. This part is
second and is measured first: the count per sample, and the count that overlaps a symbolic record.

Checks. Unit tests on a motif with one base changed. The count of anchors lost per sample over one
chromosome, against the count triosformer found for its nine samples. One job with a known motif
breaking variant.

Once the panel is on aragorn, the site's 1KGP structural variant file can be cut from it too. The
merged file from the old server's copies stays as the check: the two must agree record for record.

### 3. Measured expression beside a structure

Data. MAGE first, 731 samples on GRCh38 with every 1000 Genomes population. GEUVADIS adds samples
MAGE does not have. Both are lymphoblastoid lines, the cell type of GM12878. One table per source in
`data/expression/`, gene by sample.

Server. The gene table of a job with a hosted sample gets the sample's expression of each gene and its
rank among the samples of that source. A job without a sample, or with a sample neither source has,
shows nothing.

Interface. One more column in the gene table and one line in the side panel that states the source.
The Cite page lists both sources with their accessions, which are confirmed at that time.

Open point. The two sources use different units and annotations. The rank within a source is
comparable between them and the value is not.

## Work on tbooo

This work is in the tbooo repository. Jobs on eden are submitted by the maintainer. The code emits
batch scripts and runs nothing itself, the way `cohorts/sv.py` does.

### 4. The 1000 Genomes source

Switch the download from `20201028_3202_phased` to `20220422_3202_phased_SNV_INDEL_SV`. The cohort
files then hold the structural variants of all 3,202 samples. The steps that turn the cohort files
into array, imputed and exome data are checked against symbolic alleles and skip them where the format
has no place for them.

### 5. SGDP on GRCh38, phased

The raw files stay as the input and a new stage makes the usable set.

1. Per sample, keep the records with an alternate allele and drop the rest. This turns 27 GB into
   a file of the size of a normal VCF.
2. Normalise and merge the 277 samples into one file per chromosome.
3. Lift the positions from GRCh37 to GRCh38 with the Ensembl chain. Sites that do not lift, or that
   come out with the alleles swapped, are counted and dropped.
4. Phase against the 2022 panel as the reference, with the genetic maps tbooo already holds.
5. Rename the samples to their numeric ids, as for 1000 Genomes.

The README is corrected in the same change. The merged all site files, 1.1 TB, are no longer needed
after it. The raw per sample files can stay on eden alone.

Checks. SGDP includes samples that 1000 Genomes also has, HG00126 for one. Their phased genotypes are
compared with the panel's, which gives a switch error rate. The count of sites lost to the lift.

### 6. Structural variants for SGDP

The SGDP files have no structural variants, and calling them needs the reads, 14 TB that tbooo does
not download. Two routes do not need the reads.

Imputation. Step 4 of the stage above can impute as it phases. The 2022 panel has the structural
variants as markers among the SNVs, and a sample's haplotypes then get the structural variants that
the panel's matching haplotypes carry. This works for variants that are common in the panel's
populations. It is weak for rare variants and for the populations 1000 Genomes lacks, which are the
reason to use SGDP. Every imputed genotype carries its quality, and the site would apply those above
a cut and label the call set as imputed.

Long indels. The alleles over 50 bp in the SGDP calls themselves, as in step 2 of the site work.
These are observed and not imputed, and they are few and short.

A third route is a different data set. The Human Genome Diversity Project has 929 samples of 54
populations at high coverage on GRCh38, a phased panel joint with 1000 Genomes, and published
structural variant calls. This is stated from memory and is checked before any decision. If it holds,
HGDP gives measured structural variants for many of the populations SGDP was chosen for.

### 7. Space on frodo

`data/tmp` holds 8.4 TB of per sample splits of chromosomes 1 to 3 and other intermediates. The raid
is full. Nothing was removed. The maintainer decides which of it goes.

## What was done, 2026-11-19

Step 5 ran on eden for the 22 autosomes. The SGDP calls turned out to hold single base variants
alone, no insertion or deletion of any length, so the long indel route of step 6 gives nothing.
The maintainer asked for SGDP on the site, and the set was imputed from the 2022 panel:
35,434 symbolic records at DR2 of 0.8 or more. The check on the 23 people of both projects is in
`web/deploy/aragorn.md`. HGDP was not looked into. The panel and the phased SGDP files are on
aragorn under `data/variants/`, which is what step 2 waited for.

## Order

1. Sample table. Small data, no modelling.
2. Panel of 2022 to aragorn, then small variants.
3. Expression.
4. tbooo source of 1000 Genomes.
5. SGDP stage in tbooo, run on eden by the maintainer.
6. SGDP or HGDP on the site, after the checks of steps 5 and 6 are read.

## Decisions for the maintainer

- The SGDP route: imputed structural variants, long indels alone, or HGDP in its place.
- The quality cut for an imputed variant, once step 5 gives the numbers.
- The 8.4 TB of intermediates on frodo.
- The restricted cohorts of tbooo carry structural variant files of their own. They are private and
  are not part of this plan.
