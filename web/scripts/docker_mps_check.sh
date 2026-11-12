#!/bin/bash
# Runs the site in Docker on one GPU with the MPS overlay, posts six jobs at once and reports
# whether the engines ran on the MPS server. Run from web/ on a host with the gnome3d-web image.
#
#   SITE=~/site-data RESULTS=~/results PORT=8767 bash scripts/docker_mps_check.sh
#
# The MPS daemon runs as this user in a pipe directory under RESULTS and stops at the end, and
# the containers are stopped and removed with `docker compose down`.
set -uo pipefail
SITE="${SITE:?set SITE to the site data directory}"
RESULTS="${RESULTS:?set RESULTS to the results directory}"
PORT="${PORT:-8767}"
PY="${PY:-python3}"
export DATA_DIR="$SITE" RESULTS_DIR="$RESULTS" PORT="127.0.0.1:$PORT" WORKERS_PER_GPU=2 GPU_POOL_GIB=1.5
export MPS_PIPE_DIRECTORY="$RESULTS/.mps/pipe" MPS_USER="$(id -u):$(id -g)"
mkdir -p "$RESULTS" "$MPS_PIPE_DIRECTORY" "$RESULTS/.mps/log"

CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY" CUDA_MPS_LOG_DIRECTORY="$RESULTS/.mps/log" nvidia-cuda-mps-control -d
compose=(docker compose -p g3dmps -f docker-compose.yml -f docker-compose.mps.yml)
"${compose[@]}" up -d --no-build redis web worker-gpu0
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/" > /dev/null && break; sleep 2; done

"$PY" scripts/load_test.py "http://127.0.0.1:$PORT" "$RESULTS" GM12878 \
  chr1:1000000-4000000 chr1:5000000-8000000 chr1:10000000-13000000 chr1:20000000-23000000 \
  chr2:5000000-8000000 chr3:5000000-8000000 &
load=$!
sleep 40
echo "MPS servers while jobs run: $(echo get_server_list | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY" nvidia-cuda-mps-control)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
wait "$load"
grep -h "GPU memory pool" "$RESULTS"/jobs/*/engine.log | sort | uniq -c
grep -l -E "Traceback|Error" "$RESULTS"/jobs/*/engine.log

"${compose[@]}" down
echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY" nvidia-cuda-mps-control
