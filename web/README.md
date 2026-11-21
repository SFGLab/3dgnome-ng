# Website of 3D-GNOMEᴺᴳ

A scientist picks a cell type and a region and chooses CTCF loops or CTCF and RNAPII loops.
Structural variants from a hosted call set or an uploaded VCF are optional. The job returns an
ensemble of structures with promoters and enhancers marked, an enhancer to promoter distance
table and the CIF files.

The site runs as one container with a pool of worker threads, or as three kinds of process: the
web app, rq workers and Redis. It has no database. The hosted data is a read only directory with
a `catalog.json`. Jobs and their results are files under a results directory.

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
| [code.md](../design/website/code.md) | The layout of the server and of the frontend code, their rules and the tests that guard the interface |
| [deployment.md](../design/website/deployment.md) | The compose files, one container, the registry on the host, workers and GPUs, MPS, a site without a GPU |
| [data.md](../design/website/data.md) | The data directory, cell types, SV call sets, sample tables and expression tables |
| [settings.md](../design/website/settings.md) | The environment variables, the workers, config.json and the HiCDiffusion step |
| [frontend.md](../design/website/frontend.md) | The build, the base path and the API each page reads |
| [jobs.md](../design/website/jobs.md) | The duration of a job, caching and privacy, sync codes and uploads |
