#!/bin/bash
# Confirmation of chosen arms of the uniform chain at twenty structures per arm, built in parallel.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_confirm20.sh 'arm1:sec.key=val,sec.key=val' ...
#
# The rows, the inputs and the score are those of the larger trial: three cells, six regions of
# 5 Mb, the contact cells from out/halves/a, the score against out/halves/bw. A run of arms_trial.sh
# builds one cell, region and arm after another on one core. This script builds JOBS of them at a
# time, each with JAX on the CPU and THREADS threads. REUSE names the folder of an earlier trial and
# REUSE_ARMS its arms to compare with: OUT gets links to their structures and they are not built
# again. The default chain, the arm prod, is built here unless REUSE_ARMS holds it. The arms are
# compared over the rows with playground/arm_stats.py. The scores are the pooled scores at 25, 10
# and 5 kb with the battery at 25 kb, the close pairs, the stretched pieces and the ends, in
# OUT/scores-TAG*, TAG confirm20 unless set. The log is OUT/confirm.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
export OUT="${OUT:-out/confirm20}" N="${N:-20}" THREADS="${THREADS:-4}"
JOBS="${JOBS:-8}"
export TAG="${TAG:-confirm20}"
export LOG=$OUT/confirm.log
CELLS="GM12878 H1ESC HFFC6"
export R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
mkdir -p "$OUT"
ARMS="prod"
for spec in "$@"; do ARMS="$ARMS ${spec%%:*}"; done
for A in ${REUSE_ARMS:-}; do
  case " $ARMS " in *" $A "*) ;; *) ARMS="$ARMS $A" ;; esac
  for C in $CELLS; do
    for R in $R6; do
      T=$(echo "$R" | tr ':-' '__')
      [ -e "$OUT/${C}_${T}_$A" ] || ln -s "$PWD/$REUSE/${C}_${T}_$A" "$OUT/${C}_${T}_$A"
    done
  done
done
export ARMS

OUT=$OUT TAG=confirm20 INI_ONLY=yes bash slurm/ensemble/arms_trial.sh "$@"

build() {  # cell region arm
  local C=$1 R=$2 A=$3 T D
  T=$(echo "$R" | tr ':-' '__')
  D=$OUT/${C}_${T}_$A
  mkdir -p "$D"
  [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ] && return 0
  JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS .venv/bin/python -m gnome3d.cli \
    --config "$OUT/${C}_$A.ini" --data-dir "$DATA/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
  echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l)" >> "$LOG"
}
score() {  # cell region resolution
  local S=$OUT/scores-$TAG
  [ "$3" = 25000 ] || S=${S}_$(($3 / 1000))k
  CELLS=$1 REGIONS=$2 RES=$3 SCORES=$S bash slurm/ensemble/rescore_trial.sh $ARMS
}
export -f build score

echo "$TAG build start $(date), $JOBS at a time" >> "$LOG"
for C in $CELLS; do for R in $R6; do for A in $ARMS; do echo "$C $R $A"; done; done; done \
  | xargs -P "$JOBS" -L1 bash -c 'build "$@"' _
echo "$TAG build done $(date)" >> "$LOG"

for C in $CELLS; do for R in $R6; do for RES in 25000 10000 5000; do echo "$C $R $RES"; done; done; done \
  | xargs -P "$JOBS" -L1 bash -c 'score "$@"' _
S=$OUT/scores-$TAG
.venv/bin/python playground/uniform_overlap_check.py "$OUT" $ARMS 2>&1 | grep -v INFO > "$S/overlaps.txt"
.venv/bin/python playground/uniform_stretch_check.py "$OUT" $ARMS 2>&1 | grep -v INFO > "$S/stretch.txt"
.venv/bin/python playground/uniform_ends_check.py "$OUT" $ARMS 2>&1 | grep -v INFO > "$S/ends.txt"
echo "$TAG DONE $(date)" >> "$LOG"
