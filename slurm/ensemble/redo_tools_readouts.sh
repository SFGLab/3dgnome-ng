#!/bin/bash
# The readouts of the tools comparison on the uniform chain, after redo_tools.sh, on a workstation.
#
#   DATA=~/gapfill-data HALVES=~/gapfill-data/_hic_halves NEW=out/utools LOOPS=<dir of the Micro-C loop lists> \
#     bash slurm/ensemble/redo_tools_readouts.sh
#
# 1. The paired block bootstrap of the pooled scores, our tool against MultiMM and cudaMMC, for the
#    twelve runs of NEW/tools, the nine of NEW/tools20 and the three of NEW/toolpage60, against half b.
#    Table NEW/<step>/bootstrap_pair.txt, with the score per bead pair and the p values. The tables
#    bootstrap.txt of the first pass have neither.
# 2. The three tools of NEW/toolpage60 at the Micro-C loops of chr1:0-60 Mb with the signal floor of 15,
#    tables NEW/toolpage60/microc/<LINE>_5000.{json,txt}. LOOPS holds <LINE>_5000.bedpe for H1ESC and HFFC6,
#    GM_LOOPS the list of GM12878.
# A table that is there is not made again.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
HALVES="${HALVES:-$DATA/_hic_halves}"
NEW="${NEW:-out/utools}"
LOOPS="${LOOPS:?the folder of the Micro-C loop lists}"
GM_LOOPS="${GM_LOOPS:?the Micro-C loop list of GM12878 at 5 kb}"
PY=.venv/bin/python
R5="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:1-60000000"
R20="chr12:40000000-60000000 chr11:60000000-80000000 chr17:35000000-55000000"
boot() {  # step regions
  [ -f "$NEW/$1/bootstrap_pair.txt" ] && return
  for C in GM12878 H1ESC HFFC6; do
    for R in $2; do
      T=$(echo "$R" | tr ':-' '__'); [ "$R" = chr1:1-60000000 ] && [ "$1" = tools ] && T=chr1_60mb
      $PY playground/ensemble_bootstrap.py "$HALVES/b/$C.mcool" "$R" "$NEW/$1/${C}_${T}_ours" "$NEW/$1/${C}_${T}_multimm_md" "$NEW/$1/${C}_${T}_cudammc" 2>&1 | grep -vE "INFO|WARN" >> "$NEW/$1/bootstrap_pair.part"
    done
  done
  mv -n "$NEW/$1/bootstrap_pair.part" "$NEW/$1/bootstrap_pair.txt"
}
boot tools "$R5" &
boot tools20 "$R20" &
boot toolpage60 "chr1:1-60000000" &
O="$NEW/toolpage60/microc"; T="$NEW/toolpage60"
mkdir -p "$O"
for L in GM12878 H1ESC HFFC6; do
  [ -s "$O/${L}_5000.json" ] && continue
  B="$LOOPS/${L}_5000.bedpe"; [ "$L" = GM12878 ] && B="$GM_LOOPS"
  PYTHONPATH=$PWD $PY playground/rnapii/microc_loop_compare.py --json "$O/${L}_5000.json.part" \
    "$DATA/microc/microc_${L}_5000.npz" "$B" \
    "ctcf=$DATA/$L/${L}_clusters_3+.bedpe" "rnapii=$DATA/$L/${L}_xc_rnapii_clusters_3+.bedpe" min_z=15 \
    "$T/${L}_chr1_1_60000000_cudammc" "$T/${L}_chr1_1_60000000_multimm_md" "$T/${L}_chr1_1_60000000_ours" > "$O/${L}_5000.txt" 2>&1
  mv "$O/${L}_5000.json.part" "$O/${L}_5000.json"
done
wait
echo "redo tools readouts DONE $(date)"
