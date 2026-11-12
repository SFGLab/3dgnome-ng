#!/bin/bash
# The smallest GPU pool the largest web job runs in, and its wall time per pool.
#
#   playground/gpu_floor.sh <tree_root> <out_dir> <pools in GiB>
#
# The job is GM12878 chr1:1-20 Mb, 10 structures, production config. Each pool is turned into
# the fraction of the card's memory that XLA_PYTHON_CLIENT_MEM_FRACTION takes.
set -u
root=$1; out=$2; pools=$3
py=${PYTHON:-$HOME/3dgnome-ng/.venv/bin/python}
mkdir -p "$out"
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
for g in $pools; do
  f=$(awk -v g="$g" -v t="$total" 'BEGIN { printf "%.4f", g * 1024 / t }')
  XLA_PYTHON_CLIENT_MEM_FRACTION=$f "$py" "$root/playground/gpu_share_run.py" "$root" "$out/floor_$g.npz" chr1:1-20000000 10 \
    > "$out/floor_$g.log" 2>&1
  echo "pool $g GiB, exit $?: $(grep -h "structures of" "$out/floor_$g.log")" | tee -a "$out/floor_summary.txt"
done
