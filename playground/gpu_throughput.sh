#!/bin/bash
# Jobs per hour on one GPU with K web sized jobs at once, with and without MPS.
#
#   playground/gpu_throughput.sh <tree_root> <out_dir> <K list> [mps]
#
# Each job is GM12878 chr1:1-20 Mb, 10 structures, production config, through
# playground/gpu_share_run.py. The K jobs start together, each with an equal share of the card.
# With "mps" the MPS control daemon runs for the measurement in a pipe directory under out_dir,
# and stops afterwards. Other users' processes do not use that pipe directory.
set -u
root=$1; out=$2; ks=$3; mode=${4:-plain}
py=${PYTHON:-$HOME/3dgnome-ng/.venv/bin/python}
region=${REGION:-chr1:1-20000000}
n=${N_STRUCTURES:-10}
mkdir -p "$out"
if [ "$mode" = mps ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$out/mps/pipe CUDA_MPS_LOG_DIRECTORY=$out/mps/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
fi
for k in $ks; do
  share=$(awk -v k="$k" 'BEGIN { printf "%.3f", 0.9 / k }')
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$out/util_${mode}_k$k.csv" &
  smi=$!
  t0=$(date +%s.%N)
  for j in $(seq "$k"); do
    XLA_PYTHON_CLIENT_MEM_FRACTION=$share "$py" "$root/playground/gpu_share_run.py" "$root" "$out/${mode}_k${k}_j$j.npz" "$region" "$n" \
      > "$out/${mode}_k${k}_j$j.log" 2>&1 &
  done
  wait $(jobs -p | grep -v "^$smi$")
  t1=$(date +%s.%N)
  kill "$smi"
  awk -v k="$k" -v m="$mode" -v a="$t0" -v b="$t1" \
    'BEGIN { w = b - a; printf "%s K=%d wall %.0f s, %.1f jobs per hour\n", m, k, w, 3600 * k / w }' | tee -a "$out/summary.txt"
done
if [ "$mode" = mps ]; then
  echo quit | nvidia-cuda-mps-control
fi
