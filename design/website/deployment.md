# Deployment of the site

The compose files are under `web/`.

## Deploy with Docker

From the repository root's `web/` directory, on a machine with Docker and the NVIDIA container
toolkit:

```bash
DATA_DIR=/srv/gnome3d/data RESULTS_DIR=/srv/gnome3d/results PORT=8000 docker compose up -d --build
```

The image builds from the repository root, so the engine it runs is the checkout's. `GPU=0`
builds a CPU image; then remove the worker's `deploy` block and see "CPU only" below.

The build context has no git history. `GNOME3D_VERSION=$(git describe --tags --abbrev=0)` at the
build records the latest tag as the version of the engine in the image. Without it the version is
0.0.0.

## One container

docker-compose.single.yml runs the site and the engine in one container of the same image. The
jobs run in the web process on the slots of `GPU_SLOTS`, one GPU id per slot.

```bash
cd web
SITE_ROOT=/srv/3dgnome BASE_PATH=/new PORT=8770 GPU_SLOTS=0,1 \
  SITE_USER=$(id -u):$(id -g) docker compose -f docker-compose.single.yml up -d --build
```

`SITE_ROOT` holds two directories. The container reads `data/` and never writes it. It writes
`results/` as `SITE_USER`, and the JAX and Numba caches go there too. The port is bound to the
loopback of the host, for a proxy on the host. A job that waits in the queue is lost when the
container stops; docker-compose.yml keeps its queue in Redis.

A path in `catalog.json` must exist inside the container. `data/` is mounted at `/data`, and a
`contact_map` is written as `/data/maps/...`. A link that leaves `data/` does not resolve in the
container. A large set kept elsewhere is mounted below `/data` in an override file, passed with
a second `-f`:

```yaml
services:
  site:
    volumes: ["/srv/tbooo/release:/data/svs/tbooo:ro"]
```

## A registry on the host

docker-compose.registry.yml runs an image registry on the deployment host, with its storage in
`REGISTRY_ROOT`. The image is then built on any machine, pushed under a tag, and the host pulls
the tag it should run. An earlier tag stays in the registry for a way back.

```bash
# on the host, once
REGISTRY_ROOT=/srv/registry REGISTRY_USER=$(id -u):$(id -g) \
  docker compose -f docker-compose.registry.yml up -d

# on the machine that builds, from web/, with a tunnel to the host's registry
ssh -N -L 5000:localhost:5000 HOST &
TAG=$(git rev-parse --short HEAD)
SITE_ROOT=. IMAGE=localhost:5000/gnome3d-web:$TAG BASE_PATH=/new \
  docker compose -f docker-compose.single.yml build
docker push localhost:5000/gnome3d-web:$TAG

# on the host, with the variables of "One container" and the same IMAGE
docker compose -f docker-compose.single.yml pull
docker compose -f docker-compose.single.yml up -d --no-build
```

The registry listens on the loopback of the host and has no login. Docker accepts a registry
on localhost without TLS, on the host directly and on the building machine through the tunnel.
`BASE_PATH` is fixed at the build, so an image serves the path it was built for.

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

Measured 2026-11-12 on an RTX 3090 with `web/scripts/docker_mps_check.sh`: Redis, the
web container and two workers on GPU 0 with this overlay took six jobs posted at once, GM12878,
CTCF and RNAPII, 10 structures, 3 Mb windows. Both engines ran on the one MPS server, 1.7 GiB
each on the card, all six done in 226 s, 95.6 jobs per hour.

Without Docker, `QUEUE=thread` runs jobs in the web process on the slots of `GPU_SLOTS`, one GPU
id per slot, `0,0,1,1` for two jobs at once on each of two cards. Waiting jobs are lost when the
process stops.

## CPU only

The production configs run the smooth stage on JAX (`mc_executor_* = batch`). Without a GPU set
those keys to `threaded` in each cell type's `config.ini`; a 2 Mb region then takes several
times longer.
