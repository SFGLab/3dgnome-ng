#!/bin/bash
# The second round of the sweep of the uniform chain at eight structures per arm.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_sweep8c.sh
#
# The first round, uniform_sweep8.sh, has the ladder of resolutions ahead at a radius of the arcs
# excluded volume of 3.0. This round joins it with the loop dropout, the threshold of 2 and the cap
# by the target, and tries the ladder up to 50 kb, the radius of 2.0 and the radius that the dropout
# trace gives for the nodes the dropout leaves, 3.85. The rows, the inputs and the comparisons are
# those of the first round. The arms are built into out/sweep8 and scored into
# out/sweep8/scores-sweep8c*. The log is out/sweep8/sweep.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
OUT=out/sweep8
LOG=$OUT/sweep.log
CELLS="GM12878 H1ESC HFFC6"
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,springs.squeeze_constant=1.0"
BASE="$U,data.contact_map=CONTACT_MAP,springs.contact_resolution=5000"
Z3="springs.contact_background_z=3"
Z2="springs.contact_background_z=2"
R3="excluded_volume.radius_arcs=3.0"
LAD="springs.contact_resolution_max=25000"
DROP="springs.loop_dropout=yes"
OLD="prod s_c5k_r3 s_c5k_drop_r3 s_c5k_z2_r3 w_r3_lad w_r3_target_drop"
NEW="x_lad_drop x_lad_z2 x_lad_z2_drop x_lad_target_drop x_lad_drop_r385 x_lad50 x_lad_r2"

score() {  # tag arm...
  local tag=$1
  shift
  for RES in 25000 10000 5000; do
    local S=$OUT/scores-$tag
    [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
    mkdir -p "$S"
    for C in $CELLS; do
      local M
      M=$(ls -S "$HIC/$C"/*.mcool | head -1)
      for R in $R6; do
        local T dirs=""
        T=$(echo "$R" | tr ':-' '__')
        for A in "$@"; do dirs="$dirs $OUT/${C}_${T}_$A"; done
        .venv/bin/python playground/ensemble_score.py --res "$RES" "$M" "$R" $dirs 2>&1 | grep -vE "INFO|WARN" > "$S/${C}_${T}_pooled.txt"
      done
    done
  done
  .venv/bin/python playground/uniform_overlap_check.py "$OUT" "$@" 2>&1 | grep -v INFO > "$OUT/scores-$tag/overlaps.txt"
  .venv/bin/python playground/uniform_stretch_check.py "$OUT" "$@" 2>&1 | grep -v INFO > "$OUT/scores-$tag/stretch.txt"
  .venv/bin/python playground/uniform_ends_check.py "$OUT" "$@" 2>&1 | grep -v INFO > "$OUT/scores-$tag/ends.txt"
  echo "scores $tag done $(date)" >> "$LOG"
}

echo "step 3 start $(date)" >> "$LOG"
OUT=$OUT TAG=sweep8c REGIONS=$R6 N=8 bash slurm/ensemble/arms_trial.sh \
  "x_lad_drop:$BASE,$Z3,$R3,$LAD,$DROP" \
  "x_lad_z2:$BASE,$Z2,$R3,$LAD" \
  "x_lad_z2_drop:$BASE,$Z2,$R3,$LAD,$DROP" \
  "x_lad_target_drop:$BASE,$Z3,$R3,$LAD,$DROP,excluded_volume.arcs_cap=target" \
  "x_lad_drop_r385:$BASE,$Z3,$LAD,$DROP,excluded_volume.radius_arcs=3.85" \
  "x_lad50:$BASE,$Z3,$R3,springs.contact_resolution_max=50000" \
  "x_lad_r2:$BASE,$Z3,$LAD,excluded_volume.radius_arcs=2.0"
echo "step 3 done $(date)" >> "$LOG"
score sweep8c $OLD $NEW
echo "sweep8c DONE $(date)" >> "$LOG"
