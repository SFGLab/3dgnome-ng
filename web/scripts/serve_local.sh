#!/bin/bash
# The site without Docker or Redis: uvicorn with the thread queue on fixed GPU slots and the
# NVIDIA MPS daemon for the engines when MPS=1. Run from the repository root.
#
#   SITE=~/site-data RESULTS=~/results VENV=~/venv PORT=8766 GPUS=0 WORKERS_PER_GPU=2 MPS=1 \
#     bash web/scripts/serve_local.sh
#
# With MPS=1 the daemon runs as this user in a pipe directory under RESULTS and stops when the site
# stops.
set -euo pipefail
SITE="${SITE:?set SITE to the site data directory}"
RESULTS="${RESULTS:?set RESULTS to the results directory}"
VENV="${VENV:?set VENV to the virtual environment}"
PORT="${PORT:-8766}"
ROOT="$PWD"

export PYTHONPATH="$ROOT/web:$ROOT/web/vendor" DATA_DIR="$SITE" RESULTS_DIR="$RESULTS" QUEUE=thread
export GPUS="${GPUS:-0}" WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}" WORKER_CORES="${WORKER_CORES:-}" GPU_POOL_GIB="${GPU_POOL_GIB:-}"
export ENGINE_PYTHON="$VENV/bin/python" ENGINE_ROOT="$ROOT" GNOME3D_JAX_CACHE="$RESULTS/.jax-cache"
mkdir -p "$RESULTS"

if [ "${MPS:-0}" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY="$RESULTS/.mps/pipe" CUDA_MPS_LOG_DIRECTORY="$RESULTS/.mps/log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  # A daemon already running in this pipe directory is reused and left running.
  if nvidia-cuda-mps-control -d; then
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
  fi
fi

"$VENV/bin/uvicorn" gnome3d_web.app:create_app --factory --host 127.0.0.1 --port "$PORT"
