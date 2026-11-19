# 3D-GNOME website

A scientist picks a cell type and a region, chooses CTCF loops or CTCF and RNAPII loops,
optionally applies structural variants from a hosted call set or an uploaded VCF, and gets an
ensemble of gnome3d structures with promoters and enhancers marked, an enhancer to promoter
distance table and the CIF files.

Three processes: the web app (FastAPI), one or more workers per GPU (rq), and Redis. No database. The
hosted data is a read only directory with a `catalog.json`; jobs and their results are files
under a results directory.

## Run with Docker

From the repository root's `web/` directory, on a machine with Docker and the NVIDIA container
toolkit:

```bash
DATA_DIR=/srv/gnome3d/data RESULTS_DIR=/srv/gnome3d/results PORT=8000 docker compose up -d --build
```

The image builds from the repository root and runs the engine of the checkout. `GPU=0` builds a
CPU image.

## Development

```bash
python -m venv venv && venv/bin/pip install -r web/requirements.txt numpy numba scipy jax
PYTHONPATH=web:web/vendor DATA_DIR=... RESULTS_DIR=/tmp/r QUEUE=thread ENGINE_ROOT=$PWD \
  venv/bin/uvicorn gnome3d_web.app:create_app --factory --reload
```

The tests of the server run with pytest, from the repository root:

```bash
venv/bin/pip install -r web/requirements-dev.txt
venv/bin/pytest web/tests
```

The frontend needs Node 18 or newer:

```bash
cd web/frontend
npm ci
npm test            # Vitest, the pages and their text against design/writing-rules.md
npm run dev         # Vite on port 5173, passing /api, /jobs, /p and /model to FastAPI on 8000
npm run build       # writes dist/ for FastAPI
```

`personal3d` under `vendor/` is copied from triosformer, see `vendor/VENDORED.md`.

## Documentation

The notes are under `design/website/`.

| Note | Content |
|---|---|
| [deployment.md](../design/website/deployment.md) | The compose files, one container, the registry on the host, workers and GPUs, MPS, a site without a GPU |
| [data.md](../design/website/data.md) | The data directory, cell types, SV call sets and sample tables |
| [settings.md](../design/website/settings.md) | The environment variables and the HiCDiffusion step |
| [frontend.md](../design/website/frontend.md) | The build, the base path and the API each page reads |
| [jobs.md](../design/website/jobs.md) | The time of a job, caching and privacy, sync codes and uploads |
