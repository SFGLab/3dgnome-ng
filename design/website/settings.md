# Settings of the site

The site reads its settings from the environment, see `web/gnome3d_web/settings.py`. The table has
the main ones.

| variable | default | meaning |
|---|---|---|
| `DATA_DIR` | `/data` | the hosted data |
| `RESULTS_DIR` | `/results` | jobs, `jobs/` shared, `private/` from uploads |
| `QUEUE` | `rq` | `rq` with `REDIS_URL`, `thread` for a pool in the web process or `inline` for development. Another value is refused at start |
| `RETENTION_DAYS` | 14 | a private job is deleted after this many days, as the form states |
| `ENGINE_CMD` | `python -m gnome3d.cli` | the command that starts the engine |
| `MAX_REGION_MB` | 20 | the longest region the form accepts, unless `config.json` sets `limits.region_mb` |
| `MAX_STRUCTURES` | 100 | the most structures a job may ask for |
| `UPLOAD_MAX_MB` | 200 | the largest upload. Set the body limit of the proxy to match |
| `BASE_PATH` | unset | the path the site is served under, `/new` for a site at `/new/`. The frontend is built with the same value |
| `LINK_ALLOW_PRIVATE` | off | `1` lets a contact map link point at a private address, for tests only |
| `LINK_READS` | 4 | contact map links read at once |

## Workers

Four variables set the number of jobs at once and the share of each job. `Settings` of
`web/gnome3d_web/settings.py` reads them for both queues. Every compose file and
`web/scripts/serve_local.sh` pass them on under these names.

| variable | default | meaning |
|---|---|---|
| `GPUS` | `0` | the cards the jobs run on, their ids separated by commas |
| `WORKERS_PER_GPU` | 1 | the jobs at once on one card. The site runs the number of cards times this number of jobs at once |
| `WORKER_CORES` | the cores of the server divided by the jobs at once | the cores the engine of one job runs on |
| `GPU_POOL_GIB` | from the size of the card, see below | the GPU memory pool of the engine of one job, in GiB |

A server with two cards and 36 cores runs four jobs at once with `GPUS=0,1` and
`WORKERS_PER_GPU=2`. Each job then has 9 cores. One worker on a card keeps the pool JAX takes,
0.75 of the card. For more workers the site reads the size of the card with nvidia-smi, leaves
each engine 0.5 GiB beside its pool for its CUDA context and divides nine tenths of the rest
between the workers: 3.15 GiB each for two workers on a card of 8 GiB and 1.35 GiB each for
four. Without nvidia-smi the share of JAX is divided by the workers.

The `thread` queue has one slot for each job at once. The first slots go to one card each, the
order of the slots of two cards with two workers is 0, 1, 0, 1. Each slot has a set of cores of
its own, the next `WORKER_CORES` cores of the server. The engine of its job is started on that
set. An engine takes as many threads as it has cores. Two jobs at once share no core. When the
cores do not divide evenly the first slots get one core more: eight slots on 36 cores are four
of 5 cores and four of 4. A count above the share of a job makes the sets overlap. The number of
cores of the server gives every job all of them.

With the `rq` queue a worker is one slot. docker-compose.yml starts `WORKERS_PER_GPU` workers on
each card and a worker has the card of its container. A worker takes a free slot for each job,
a lock on a file under `RESULTS_DIR/.slots`. Its engine runs on the cores of that slot. The
workers of one server then share no core either.

A change of these variables needs a restart of the site. The site does not start on a value it
cannot use. It does not start with `GPU_SLOTS` either, the variable these replace.
`design/website/deployment.md` has the measurements of jobs at once on one card.

## config.json

`config.json` beside the catalog, or the file of `SITE_CONFIG`, holds the feature flags and
the limits of a job. The server reads the file again when it changes. An edit needs no restart
and no build.

    {"features": {"results": true, "method": true, "trios": false, "uniform_chain": true},
     "limits": {"region_mb": 20, "hicdiffusion_mb": 2}}

| key | default | meaning |
|---|---|---|
| `features.results`, `features.method` | on | the two pages. Off, the page leaves the top bar and its address leads to the front page |
| `features.trios` | on | the nine people on the Results page and their RNA-seq: an expression table with `"feature": "trios"` in the catalog, its row of the Cite page and its sentence of the Help page |
| `features.uniform_chain` | off | the choice of the chain on the form, two cards at the top of step 6. On, the form opens on the chain of one bead per kilobase and a job may ask for it with `chain=uniform`. Off, the form shows no choice, a job is built on loop anchors and a request for the other chain is refused. The Method page follows the feature: off, it has no section on the chain of one bead per kilobase and numbers its sections and figures without it. It is the one feature that is off unless the file turns it on |
| `limits.region_mb` | `MAX_REGION_MB`, 20 | the longest region of a job, in Mb |
| `limits.hicdiffusion_mb` | 2 | the longest region of a job with a HiCDiffusion background, in Mb, 2.097152 at most |

HiCDiffusion predicts one window of 2,097,152 bp of the haplotype. A longer region would get the
prediction on its first part alone. A file with a higher limit is refused. The default
of 2 Mb leaves room for insertions. The form, the title page and the refusals of the server
state the limits of the file.

A site does not start on a file with an unknown key, feature or limit. A broken edit of a
running site is logged and the configuration it had stays.

HiCDiffusion is a step of a job that asks for it. It predicts the contact background of a 2 Mb
window from the sequence of the haplotype. The map is spliced into the map of the cell type from
the start of the region. The step runs the encoder decoder of HiCDiffusion, which is
deterministic, on the GPU of the queue slot of the job before the engine starts. Its output opens
the log of the job. Measured 2026-11-17 on an RTX 3090 the prediction takes 2 s and 1.1 GiB of
GPU memory, 5 s on the CPU. The step takes 12 s of a job with the start of its interpreter and
the load of the model. The site does not present the map as individual specific.

A job keeps the predicted map in its directory as `hicdiffusion.bedpe`, one row per pair of
8,192 bp bins with the predicted count, in the coordinates of the haplotype. The file is listed
with the files of the job and is in its archive. The Contact map tab offers the prediction as a
half of the map, put onto the grid of the tab by the share of each predicted bin inside a bin of
the grid. The tab goes to 10 kb bins when the prediction is chosen, the resolution nearest to the
bins of the model. A job without the file does not offer it.

HiCDiffusion runs in its own interpreter. Its package pins clash with the pins of the engine. The
Docker image holds its checkout at a fixed commit with that interpreter and sets the four
variables of the step. Two parts are data and go below `DATA_DIR`:

| variable | value in the image | part |
|---|---|---|
| `HICDIFFUSION_DIR` | `/opt/hicdiffusion` | the checkout, in the image |
| `HICDIFFUSION_PYTHON` | `/opt/hicdiffusion-venv/bin/python` | its interpreter, in the image |
| `HICDIFFUSION_MODELS` | `/data/hicdiffusion` | `hicdiffusion.ckpt` and `hicdiffusion_encoder_decoder.ckpt`. `models/manifest.tsv` of triosformer lists the archive |
| `REFERENCE_FASTA` | `/data/reference/hg38.fa` | the hg38 FASTA with its `.fai` beside it |

The site offers the step once every part is in place. Until then the form states that it is not
set up. A job that asks for it on such a server is refused with the missing part in the message.
`HICDIFFUSION=0` at the build leaves the step out of the image, which is then about 5 GB
smaller.
