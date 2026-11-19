# Settings of the site

All by environment, see `web/gnome3d_web/settings.py`. The ones that matter:

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

HiCDiffusion is a step of a job that asks for it. It predicts the contact background of a 2 Mb
window from the haplotype's sequence, and the map is spliced into the cell type's map from the
region's start. The step runs HiCDiffusion's encoder decoder, which is deterministic, on the GPU
of the job's queue slot before the engine starts, and its output opens the job's log. Measured
2026-11-17 on an RTX 3090 the prediction takes 2 s and 1.1 GiB of GPU memory, 5 s on the CPU,
and the step takes 12 s of a job with the start of its interpreter and the load of the model.
The site never presents the map as individual specific.

A job keeps the predicted map in its directory as `hicdiffusion.bedpe`, one row per pair of
8,192 bp bins with the predicted count, in the haplotype's coordinates. The file is listed with
the job's files and is in its archive. The Contact map tab offers the prediction as a half of
the map, put onto the grid of the tab by the share of each predicted bin inside a bin of the grid.
The tab goes to 10 kb bins when the prediction is chosen, the resolution nearest to the model's.
A job that ran before the file was kept does not offer it.

HiCDiffusion runs in its own interpreter, since its package pins clash with the engine's. The
Docker image holds its checkout at a fixed commit and that interpreter, and sets the four
variables of the step. Two parts are data and go below `DATA_DIR`:

| variable | value in the image | part |
|---|---|---|
| `HICDIFFUSION_DIR` | `/opt/hicdiffusion` | the checkout, in the image |
| `HICDIFFUSION_PYTHON` | `/opt/hicdiffusion-venv/bin/python` | its interpreter, in the image |
| `HICDIFFUSION_MODELS` | `/data/hicdiffusion` | `hicdiffusion.ckpt` and `hicdiffusion_encoder_decoder.ckpt`; triosformer's `models/manifest.tsv` lists the archive |
| `REFERENCE_FASTA` | `/data/reference/hg38.fa` | the hg38 FASTA with its `.fai` beside it |

The site offers the step once every part is in place, and the form names it as not set up
until then. A job that asks for it on such a server is refused with the missing part named.
`HICDIFFUSION=0` at the build leaves the step out of the image, which is then about 5 GB
smaller.
