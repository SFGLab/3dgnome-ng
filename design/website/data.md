# Data directory of the site

```
DATA_DIR/
  catalog.json
  cell_types/<CELL>/   anchors, loops per factor, singletons, segments, centromeres,
                       config.ini, genes.tsv, enhancers.bed
  svs/<NAME>.vcf.gz    one bgzipped VCF per call set, with its .tbi
  svs/<NAME>.samples.tsv  optional: sex, population and parents of the call set's samples
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

The script copies the files and adds the catalog entry. It refuses a cell type that already
exists. `--lymphoblastoid` marks the lines HiCDiffusion was trained on; for the others the form
and the result page carry the out of domain notice.

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
a population code typed in place of a name, and states the parents and children of the chosen
sample. For the 30x panel the script reports 3,202 samples, 602 of them with both parents in the
set.

Records need SVTYPE of DEL, INS, INV or DUP, with END or SVLEN; an insertion's ALT sequence is used
when it has one. Other types are skipped and counted on the result page. Many large call sets
are fine: a job reads only its region through tabix.

The trio individuals are not hosted until their release is cleared.
