#!/bin/bash
# The halves of the deep Hi-C maps for the published arms on the uniform chain, on a workstation.
# Each line is split by playground/thin_cooler.py --chromosomes on chr1 to chr22, the three lines
# at once on the CPU. Half a is the input of the arms UC and UR of crosscell_arms.sh, as
# data/_hic_half/a/<LINE>.mcool of the checkout that runs them. Half b is the map of the score.
#
#   bash slurm/ensemble/redo_halves.sh
#
# HIC holds <LINE>/*.mcool and the deep map of a line is the largest. OUT gets a/, b/ and logs/.
# A line whose half b exists is skipped.
set -euo pipefail
HIC="${HIC:-$HOME/gapfill-data/_hic}"
OUT="${OUT:-$HOME/gapfill-data/_hic_half}"
PY="${PY:-.venv/bin/python}"
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22"
mkdir -p "$OUT/logs"
for LINE in GM12878 H1ESC HFFC6; do
  [ -s "$OUT/b/$LINE.mcool" ] && { echo "[skip] $LINE"; continue; }
  MAP=$(ls -S "$HIC/$LINE"/*.mcool | head -1)
  echo "[start] $LINE $MAP $(date)"
  # shellcheck disable=SC2086
  "$PY" playground/thin_cooler.py --chromosomes "$MAP" "$OUT" "$LINE" $CHROMS > "$OUT/logs/$LINE.log" 2>&1 &
done
wait
for LINE in GM12878 H1ESC HFFC6; do tail -n 6 "$OUT/logs/$LINE.log"; done
echo "[done] $(date)"
