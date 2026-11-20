#!/bin/bash
# Scores the structures of a finished trial of arms_trial.sh against another map.
#
#   DATA=out/halves/data HIC=out/halves/bw OUT=out/held SCORES=out/held/scores-held-w \
#   REGIONS="chr1:10000000-15000000 chr1:20000000-25000000" \
#     bash slurm/ensemble/rescore_trial.sh prod arm1 arm2 ...
#
# OUT holds the structures, one folder per cell, region and arm. SCORES gets the pooled scores and
# the battery of every cell and region, under the file names of arms_trial.sh. No structure is
# built.
set -u
cd "$(dirname "$0")/../.."
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$SCORES"
for C in $CELLS; do
  M=$(ls -S "$HIC/$C"/*.mcool | head -1)
  for R in $REGIONS; do
    T=$(echo "$R" | tr ':-' '__')
    dirs=""
    for A in "$@"; do dirs="$dirs $OUT/${C}_${T}_$A"; done
    for B in no yes; do
      .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 \
        --singletons "$DATA/$C/${C}_hic_25kb_singletons.bedpe" "$M" "$R" 25000 $dirs 2>&1 \
        | grep -vE "INFO|WARN" > "$SCORES/${C}_${T}_battery_balance_$B.txt"
    done
    .venv/bin/python playground/ensemble_score.py "$M" "$R" $dirs 2>&1 | grep -vE "INFO|WARN" > "$SCORES/${C}_${T}_pooled.txt"
    echo "[$C $T] scored $(date +%H:%M)" >> "$SCORES/rescore.log"
  done
done
echo "rescore DONE $(date)" >> "$SCORES/rescore.log"
