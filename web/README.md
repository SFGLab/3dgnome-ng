# 3D-GNOME website

A scientist picks a cell type and a region, chooses CTCF loops or CTCF and RNAPII loops,
optionally applies structural variants from a hosted call set or an uploaded VCF, and gets an
ensemble of gnome3d structures with promoters and enhancers marked, an enhancer to promoter
distance table and the CIF files.

Three processes: the web app (FastAPI), one worker per GPU (rq), and Redis. No database. The
hosted data is a read only directory with a `catalog.json`; jobs and their results are files
under a results directory.

## Deploy with Docker

From the repository root's `web/` directory, on a machine with Docker and the NVIDIA container
toolkit:

```bash
DATA_DIR=/srv/gnome3d/data RESULTS_DIR=/srv/gnome3d/results PORT=8000 docker compose up -d --build
```

The image builds from the repository root, so the engine it runs is the checkout's. `GPU=0`
builds a CPU image; then remove the worker's `deploy` block and see "CPU only" below.

## Populate the data directory

```
DATA_DIR/
  catalog.json
  cell_types/<CELL>/   anchors, loops per factor, singletons, segments, centromeres,
                       config.ini, genes.tsv, enhancers.bed
  svs/<NAME>.vcf.gz    one bgzipped VCF per call set, with its .tbi
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

Records need SVTYPE of DEL, INS or INV, with END or SVLEN; an insertion's ALT sequence is used
when it has one. Other types are skipped and counted on the result page. Many large call sets
are fine: a job reads only its region through tabix.

The trio individuals are not hosted until their release is cleared.

## Settings

All by environment, see `gnome3d_web/settings.py`. The ones that matter:

| variable | default | meaning |
|---|---|---|
| `DATA_DIR` | `/data` | the hosted data |
| `RESULTS_DIR` | `/results` | jobs, `jobs/` shared, `private/` from uploads |
| `QUEUE` | `rq` | `rq` with `REDIS_URL`, `thread` or `inline` for development |
| `RETENTION_DAYS` | 14 | a private job is deleted after this many days, as the form says |
| `ENGINE_CMD` | `python -m gnome3d.cli` | how the worker starts the engine |

HiCDiffusion is offered only when `HICDIFFUSION_DIR`, `HICDIFFUSION_PYTHON`,
`HICDIFFUSION_MODELS` and `REFERENCE_FASTA` are all set, the checkout, its own interpreter (its
package pins clash with the engine's), the directory with `hicdiffusion.ckpt` and
`hicdiffusion_encoder_decoder.ckpt` (triosformer's `models/manifest.tsv` lists the archive), and
the hg38 FASTA with its `.fai`. Mount them into the worker and set the variables. Its map is
a contact background from sequence spliced into the cell type's map over a 2 Mb window from the
region's start; the site never presents it as individual specific.

## Caching and privacy

A request on hosted inputs is keyed by a hash of the normalised request, so an identical request
finds the finished job instead of running again. A request with an uploaded VCF gets a random
link under `/p/`, never enters the shared cache, and is removed by the retention sweep the app
runs at start. Shared jobs are never removed by the site.

## CPU only

The production configs run the smooth stage on JAX (`mc_executor_* = batch`). Without a GPU set
those keys to `threaded` in each cell type's `config.ini`; a 2 Mb region then takes several
times longer.

## Development

```bash
python -m venv venv && venv/bin/pip install -r web/requirements.txt numpy numba scipy jax
PYTHONPATH=web:web/vendor DATA_DIR=... RESULTS_DIR=/tmp/r QUEUE=thread ENGINE_ROOT=$PWD \
  venv/bin/uvicorn gnome3d_web.app:create_app --factory --reload
for t in web/tests/test_*.py; do venv/bin/python $t; done
```

`personal3d` under `vendor/` is copied from triosformer, see `vendor/VENDORED.md`.
