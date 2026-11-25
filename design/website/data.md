# Data directory of the site

```
DATA_DIR/
  catalog.json
  cell_types/<CELL>/   anchors, loops per factor, singletons, segments, centromeres,
                       config.ini, genes.tsv, enhancers.bed
  svs/<NAME>.vcf.gz    one bgzipped VCF per call set, with its .tbi
  svs/<NAME>.samples.tsv  optional: sex, population and parents of the call set's samples
  expression/<NAME>.tsv   optional: one table of genes by samples per RNA-seq set
```

A cell type from this repository's prepared inputs, the cross line RNAPII prep of
`playground/rnapii/crosscell_prep.py`:

```bash
python web/scripts/make_dataset.py data $DATA_DIR GM12878 \
  --gtf gencode.v40.basic.annotation.gtf --enhancers GM12878_enhancers_hg38.bed \
  --label "GM12878, lymphoblastoid" --lymphoblastoid \
  --singletons /path/to/GM12878_hic_25kb_singletons.bedpe
python web/scripts/make_dataset.py data $DATA_DIR HFFC6 ... \
  --note "RNAPII loops from BJ, a different foreskin fibroblast line"
```

`genes.tsv` has gene_id, name, chrom, tss, strand, start and end, the span 0 based with the end left out. A
table made before 2026-11-24 lacks the span, which the tracks of the genes read:
`python web/scripts/add_gene_spans.py $DATA_DIR gencode.v40.annotation.gtf.gz` adds it and keeps the table as
it was beside it. The site draws a gene of a table without spans at its TSS.

The script copies the files and adds the catalog entry. It refuses a cell type that already
exists. `--lymphoblastoid` marks the lines HiCDiffusion was trained on. For the others the form
and the result page show the out of domain notice.

An SV call set:

```bash
python web/scripts/add_sv_set.py $DATA_DIR HGSVC variants_freeze4_sv_insdel_alt.vcf.gz \
  --label "HGSVC freeze 4 insertions and deletions"
```

A table of the call set's samples, from the pedigree file of the 1000 Genomes 30x panel:

```bash
python web/scripts/add_sample_table.py $DATA_DIR 1KGP 20130606_g1k_3202_samples_ped_population.txt
```

With it the sample field suggests each sample with its population and sex, lists the samples of
a population code typed in place of a name and states the parents and children of the chosen
sample. For the 30x panel the script reports 3,202 samples, 602 of them with both parents in the
set.

Records need SVTYPE of DEL, INS, INV or DUP, with END or SVLEN. The ALT sequence of an insertion
is used when it has one. Other types are skipped and counted on the result page. Many large call
sets are fine, as a job reads only its region through tabix.

The call set of the Human Genome Diversity Project is written from the files of the project, Almarri et al.,
Cell 2020, `https://ngs.sanger.ac.uk/production/hgdp/hgdp_structural_variation/`:

```bash
python web/scripts/make_hgdp_sv_set.py HGDP.metadata RAW/ HGDP.vcf HGDP.samples.tsv \
  del.manta.vcf.gz ins.manta.vcf.gz dup.manta.vcf.gz inv.manta.vcf.gz
bgzip HGDP.vcf
python web/scripts/add_sv_set.py $DATA_DIR HGDP HGDP.vcf.gz --label "HGDP, 911 samples"
python web/scripts/add_sample_table.py $DATA_DIR HGDP HGDP.samples.tsv
```

The files of the project have the type and the size of a call in its ALT field and the samples under their
accessions. The script writes the fields the site reads and the names of the project. It keeps a record that
passed the filters with a carrier whose own genotype passed: 74,965 of 102,367, for 911 samples, 57,081
deletions, 10,047 insertions, 6,259 duplications and 1,578 inversions, 3,599 to 6,342 a sample. An insertion
has its sequence from the raw calls of the project. The 8,893 insertions of unknown length are left out. The
genotypes are unphased. The site puts an unphased variant on the first haplotype.

## Gene expression

A table has the genes in its first column, by Ensembl id or by name, an optional column `length`
with the gene lengths in bp and one column for each sample. The catalog lists the tables under
`expression`:

    "ENCODE": {"label": "ENCODE RNA-seq", "file": "expression/ENCODE.tsv", "kind": "counts",
               "cell_types": {"GM12878": ["GM12878_ENCFF800HIP", "GM12878_ENCFF991KKX"]}},
    "MAGE":   {"label": "MAGE RNA-seq", "file": "expression/MAGE.tsv", "kind": "values", "unit": "TMM",
               "people": true, "lymphoblastoid": true}

| key | meaning |
|---|---|
| `kind` | `counts` for read counts, `values` for normalised values |
| `unit` | the unit of normalised values, shown in the Genes tab |
| `cell_types` | the columns of each cell type, its replicates |
| `people` | every column is a person, under the name the call sets give their samples. A job with a hosted sample is offered the column of that sample |
| `lymphoblastoid` | the samples are lymphoblastoid cell lines. The form warns when the table is taken beside a cell type that is not lymphoblastoid |
| `feature` | a feature of `config.json`. The table is neither offered nor taken while the feature is off |

The tables of the site come from two scripts. The first reads the published files and the
second puts its output into the data directory, with a copy of the catalog as
`catalog.json.before-expression`:

```bash
python web/scripts/make_expression_tables.py tables \
  --encode GM12878=gm12878_read_count.txt --encode H1ESC=h1esc_read_count.txt --encode HFFC6=hffc6_read_count.txt \
  --trios trio_gene_counts.csv --mage TMM.filtered.TSS.MAGE.v1.0.bed.gz --geuvadis GD660.GeneQuantRPKM.txt.gz
python web/scripts/add_expression.py $DATA_DIR tables
```

The ENCODE tables are read counts of featureCounts over GENCODE release 29, one for each cell
line. The table of the trios has the feature `trios`. The RNA-seq of the nine people is not
published and stays hidden until its release is cleared.

A job with expression keeps the columns it took as `expression.tsv` with `expression.json`
beside it. The Genes tab shows the mean level of a gene over those columns: TPM for counts with
gene lengths, counts per million for counts without and the mean for normalised values. The
comparison of two jobs runs DESeq2 through PyDESeq2 when both hold read counts of two samples
or more, over the genes with ten reads or more. Any other pair of one kind gives a log2 ratio
of the levels and no p-value. Counts are not compared with normalised values and the values of
two tables are not compared with each other. One table by Ensembl id and one by gene name are
compared by name, with the names of the gene table of the site. A result is kept under
`RESULTS_DIR/expression` by the hash of the two job tables.

The trio individuals are not hosted until their release is cleared.
