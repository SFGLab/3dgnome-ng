#!/bin/bash
# A sweep of the uniform chain on the rows of the larger trial at eight structures per arm.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_sweep8.sh [first_step]
#
# The larger trial has twenty structures per arm and an arm takes two hours. This sweep has eight,
# on the same six regions and three cells, with the cells at 5 kb on counts the run does not read.
# The arms of out/held4_20 are its comparisons: out/sweep8 gets links to their first eight
# structures, which have the seeds of the eight of an arm built here. An arm that wins is built at
# twenty structures afterwards. Step 1 is the radius of the arcs excluded volume, its cap, the
# dropout, the ladder and the weight of the cells. Step 2 is the subanchor heatmap, whose dry
# passes are slow. The log is out/sweep8/sweep.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
FIRST="${1:-1}"
SRC=out/held4_20
OUT=out/sweep8
LOG=$OUT/sweep.log
CELLS="GM12878 H1ESC HFFC6"
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
OLD="prod s_c5k s_c5k_r3 s_c5k_drop_r3 s_c5k_z2_r3"
U="main.uniform_chain=yes,springs.squeeze_constant=1.0"
C5="$U,data.contact_map=CONTACT_MAP,springs.contact_background_z=3,springs.contact_resolution=5000"
R3="excluded_volume.radius_arcs=3.0"
mkdir -p "$OUT"

# The first eight structures of the arms of the larger trial.
for C in $CELLS; do
  for R in $R6; do
    T=$(echo "$R" | tr ':-' '__')
    for A in $OLD; do
      mkdir -p "$OUT/${C}_${T}_$A"
      for k in 1 2 3 4 5 6 7 8; do
        f=$(ls "$SRC/${C}_${T}_$A"/*_s$k.cif 2>/dev/null | head -1)
        [ -n "$f" ] && ln -sfn "$PWD/$f" "$OUT/${C}_${T}_$A/$(basename "$f")"
      done
    done
  done
done

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

NEW1="w_r3_target w_r3_chain w_auto w_auto_target w_r3_target_drop w_r2 w_r3_lad w_r3_bg03"
if [ "$FIRST" -le 1 ]; then
  echo "step 1 start $(date)" >> "$LOG"
  OUT=$OUT TAG=sweep8a REGIONS=$R6 N=8 bash slurm/ensemble/arms_trial.sh \
    "w_r3_target:$C5,$R3,excluded_volume.arcs_cap=target" \
    "w_r3_chain:$C5,$R3,excluded_volume.arcs_cap=chain" \
    "w_auto:$C5" \
    "w_auto_target:$C5,excluded_volume.arcs_cap=target" \
    "w_r3_target_drop:$C5,$R3,excluded_volume.arcs_cap=target,springs.loop_dropout=yes" \
    "w_r2:$C5,excluded_volume.radius_arcs=2.0" \
    "w_r3_lad:$C5,$R3,springs.contact_resolution_max=25000" \
    "w_r3_bg03:$C5,$R3,springs.background_weight=0.3"
  echo "step 1 done $(date)" >> "$LOG"
  score sweep8a $OLD $NEW1
fi
if [ "$FIRST" -le 2 ]; then
  echo "step 2 start $(date)" >> "$LOG"
  OUT=$OUT TAG=sweep8b REGIONS=$R6 N=8 bash slurm/ensemble/arms_trial.sh \
    "w_r3_heat:$C5,$R3,subanchor_heatmap.use_subanchor_heatmap=yes"
  echo "step 2 done $(date)" >> "$LOG"
  score sweep8b prod s_c5k_r3 w_r3_heat
fi
echo "sweep8 DONE $(date)" >> "$LOG"
