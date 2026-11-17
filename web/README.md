# 3D-GNOME website

A scientist picks a cell type and a region, chooses CTCF loops or CTCF and RNAPII loops,
optionally applies structural variants from a hosted call set or an uploaded VCF, and gets an
ensemble of gnome3d structures with promoters and enhancers marked, an enhancer to promoter
distance table and the CIF files.

Three processes: the web app (FastAPI), one or more workers per GPU (rq), and Redis. No database. The
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

## Workers and GPUs

docker-compose.yml has one worker group per GPU, `worker-gpu0` and `worker-gpu1`, each worker
pinned to its card. `WORKERS_PER_GPU` sets the workers per card, 2 by default, and
`GPU_POOL_GIB` the memory pool of each, 1.5 GiB by default. On a machine with one GPU, remove
`worker-gpu1`.

A job of 20 Mb and 10 structures needs 0.65 GiB of GPU memory and runs at the same speed in any
pool from 1 GiB up. The largest array is the bead distance matrix, which grows with the square
of the window. The pools of the workers on a card and 0.5 GiB per worker for its CUDA context
add up to less than the card. A worker whose pool does not fit the free memory refuses to start
its job and says how much is free.

Measured 2026-11-12 on an RTX 3090 with the same job:

| Jobs at once on one card | Jobs per hour | With MPS |
|---|---|---|
| 1 | 39.3 | 42.8 |
| 2 | 37.3 | 46.5 |
| 3 | 39.4 | 47.3 |

Without MPS the processes on a card take turns and a second worker gains nothing. With the
NVIDIA MPS daemon running on the host, kernels of several processes run at once and two workers
per card give 18 percent more than one. A third adds nothing. Start the daemon on the host with
`nvidia-cuda-mps-control -d` and add docker-compose.mps.yml, which gives the workers the host
IPC namespace and the daemon's pipe directory:

```bash
docker compose -f docker-compose.yml -f docker-compose.mps.yml up -d
```

Without MPS set `WORKERS_PER_GPU=1`. The host needs the NVIDIA container toolkit; Docker's
`nvidia` runtime entry alone does not give the containers a GPU.

Measured 2026-11-12 on team-frodo, RTX 3090, with `web/scripts/docker_mps_check.sh`: Redis, the
web container and two workers on GPU 0 with this overlay took six jobs posted at once, GM12878,
CTCF and RNAPII, 10 structures, 3 Mb windows. Both engines ran on the one MPS server, 1.7 GiB
each on the card, all six done in 226 s, 95.6 jobs per hour.

Without Docker, `QUEUE=thread` runs jobs in the web process on the slots of `GPU_SLOTS`, one GPU
id per slot, `0,0,1,1` for two jobs at once on each of two cards. Waiting jobs are lost when the
process stops.

## Frontend

The pages are a React app in `web/frontend/`, built with Vite to `web/frontend/dist/`. FastAPI
serves the build: every client route returns `dist/index.html`, and the API, the job pages, the
private job pages and the old form at `/model` keep their own handlers. Without a build the app
answers 503 and says how to build. `FRONTEND_DIST` points the app at another build. The Docker
image builds the frontend in a Node stage.

Development needs Node 18 or newer:

```bash
cd web/frontend
npm ci
npm test            # Vitest, the pages and their text against design/writing-rules.md
npm run dev         # Vite on port 5173, passing /api, /jobs, /p and /model to FastAPI on 8000
npm run build       # writes dist/ for FastAPI
```

A site under a path of a host, `https://host/new/`, needs `BASE_PATH=/new` twice, at the build
and on the server:

```bash
BASE_PATH=/new npm run build                 # in web/frontend
BASE_PATH=/new uvicorn gnome3d_web.app:create_app --factory
```

The build writes the path into its asset links, its router and its API requests. The server
answers every route under the path and at the root, and the proxy in front may strip the path or
pass it on. The server answers 503 with both paths named when the build and the server disagree.
With Docker Compose, `BASE_PATH=/new` in the environment sets both. A proxy rule for `/new/`
does not match `/new`, and a second rule redirects `/new` to `/new/`.

The page text, the colours and the figures come from the approved design in web/design/site.pen.
A change of wording goes into the design first.

The Model a region page at `/model` reads the JSON API: `GET /api/catalog`, `GET /api/region`,
`GET /api/preview` for the loop arcs of a region, `GET /api/settings` for the engine settings a
user may change, `POST /api/settings/ini` to read an uploaded config, and `POST /api/jobs`, which
answers with the job's url or the field a request fails on. The settings a user may change, with
their ranges, are listed in `gnome3d_web/settings_schema.py`; every other setting keeps the cell
type's value. The old form is at `/model/classic`.

A job's page at `/jobs/<key>`, or `/p/<token>` for a private job, is the explorer: the Structure
tab in 3D with hover, two point measurement and a gene card, and the Genes, Contact map, Loops
and Log tabs, and a compare view of two jobs. It reads `GET /api/jobs/<key>/` followed by
`summary`, `structure/<i>`, `annotations`, `picks`, `distance?a=&b=`, `genes`, `contacts`,
`loops`, `log` and `files.zip`, the same under `/api/p/<token>/`, and `GET /api/compare?a=&b=`.
Results that read every structure, the picks and the gene statistics, are written to the job
directory as `explore_<name>.json`. The old job pages are at `.../classic`.

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
| `MAX_REGION_MB` | 20 | the longest region the form accepts |
| `GPU_SLOTS` | `0` | the slots of the `thread` queue, one GPU id per slot |
| `GNOME3D_GPU_POOL_GIB` | unset | the GPU memory pool of each engine process in GiB |
| `MAX_STRUCTURES` | 100 | the most structures a job may ask for |
| `UPLOAD_MAX_MB` | 200 | the largest upload; set the proxy's body limit to match |
| `BASE_PATH` | unset | the path the site is served under, `/new` for a site at `/new/`; the frontend is built with the same value |
| `LINK_ALLOW_PRIVATE` | off | `1` lets a contact map link point at a private address, for tests only |
| `LINK_READS` | 4 | contact map links read at once |

HiCDiffusion is offered only when `HICDIFFUSION_DIR`, `HICDIFFUSION_PYTHON`,
`HICDIFFUSION_MODELS` and `REFERENCE_FASTA` are all set, the checkout, its own interpreter (its
package pins clash with the engine's), the directory with `hicdiffusion.ckpt` and
`hicdiffusion_encoder_decoder.ckpt` (triosformer's `models/manifest.tsv` lists the archive), and
the hg38 FASTA with its `.fai`. Mount them into the worker and set the variables. Its map is
a contact background from sequence spliced into the cell type's map over a 2 Mb window from the
region's start; the site never presents it as individual specific.

## How long a job takes

Measured 2026-11-11 on an RTX 3090 with a warm JAX cache: GM12878 chr1:1-3 Mb, CTCF and
RNAPII loops, HG00512's first haplotype from HGSVC (69 variants), ten structures, 58 s from
submission to the finished page. Jobs run on the queue and the page refreshes itself, so a user
waits about a minute. The first job after a restart also pays the JAX compile, which the cache
under `RESULTS_DIR/.jax-cache` keeps for the next.

## Caching and privacy

A request on hosted inputs is keyed by a hash of the normalised request. An identical request
finds the finished job instead of running again. A request with an uploaded VCF or any of your
own data gets a random link under `/p/`, never enters the shared cache, and is removed by the
retention sweep the app runs at start. Uploads waiting for a job sit under `RESULTS_DIR/uploads`
and are swept after the same retention period. Shared jobs are never removed by the site.

## Uploads

The Your own data dialog takes loops, one seven column BEDPE per factor, anchors in BED with an
optional orientation column, contacts and enhancers in BED. The structural variants dialog takes
a VCF compressed with bgzip. `POST /api/uploads` stores each file in chunks under the
`UPLOAD_MAX_MB` cap and checks it on arrival; the dialog shows the check and holds the job until
every file passes. A job copies the files it uses into its own directory.

Contacts come by one of three routes, and each ends as the region's bin pairs at 25 kb without
the diagonal, the way the hosted maps hold them.

| route | where the window is read |
|---|---|
| link to a .hic or .mcool | on the server, `POST /api/uploads/link`, over HTTP range requests: hic-straw for .hic, h5py on an fsspec file for .mcool, each in a child process that reads through a range proxy on the loopback interface. Every connection resolves the host once and goes only to the public address it checked, redirects included. A read is stopped after 2 minutes, and at most `LINK_READS` run at once; one more is refused at once. |
| local .hic file | in the browser, with the hic-straw JavaScript reader, uploaded as a BEDPE |
| BEDPE of the region | on the user's computer; `gnome3d-ng extract --mcool FILE --region chr:start-end --out region.bedpe` writes it from a local .mcool, with `pip install gnome3d-ng[extract]` for cooler |

In production the worker's network should also refuse private ranges, as a second guard
behind the address check: with Docker, a network whose egress rule drops 10.0.0.0/8,
172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16 and the host's own addresses.

At job build the pairs are thinned to the contacts per Mb of the cell type's hosted map on the
region's chromosome, by drawing contacts without replacement. A deep map and a shallow one then
reach the engine at the density its maps are read at. The whole remote file never comes to the
server.

Checked 2026-11-12 on team-frodo through the page: an own data job with CTCF and RNAPII loops
and contacts read from a local .hic, a job with an uploaded HGSVC VCF, and a job with contacts by
link to ENCODE ENCFF718AWL.hic, 37 GB, whose 2 Mb window read in about 9 s.

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
