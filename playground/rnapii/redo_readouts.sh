#!/bin/bash
# The readouts of the redo, for every chromosome whose nine ensembles of the three arms are in the
# readout root, on a workstation. A step whose output is there is skipped, and a second call makes
# what the first did not.
#
#   ROOT=~/xu HALVES=~/gapfill-data/_hic_halves bash playground/rnapii/redo_readouts.sh
#
# The uniform arms read half a of the deep map of a line. The scores of the structure and the
# short range steps are against half b, HALVES/b/<LINE>.mcool, for every arm. The Micro-C step
# reads the Micro-C map of the line, which no arm read. The root is that of redo_pull.sh.
set -uo pipefail
export ROOT="${ROOT:-$HOME/xu}"
export REPO="${REPO:-$ROOT/repo}"
export SCORE_MAPS="${HALVES:-$HOME/gapfill-data/_hic_halves}/b"
export ARMS="E C R M"
D="$REPO/playground/rnapii"
cd "$ROOT"
mkdir -p logs
CHROMS=""
for n in $(seq 22 -1 1); do
  have=$(ls -d out/lr/*_[CRM]/chr$n 2>/dev/null | wc -l)
  [ "$have" -eq 9 ] && CHROMS="$CHROMS chr$n"
done
echo "$(date +%H:%M:%S) chromosomes with every arm:${CHROMS:- none}"
[ -n "$CHROMS" ] || exit 0
run() {  # step parallel
  echo $CHROMS | tr ' ' '\n' | xargs -P "$2" -I{} bash -c "bash $D/readouts_crossline_genome.sh $1 {} > logs/$1_{}.log 2>&1 || echo '$1 {} failed'"
}
run expression 6
echo "$(date +%H:%M:%S) expression: $(ls out/readouts/e3d_*/chr*/summary.tsv 2>/dev/null | wc -l) summaries"
run pairs 6
echo "$(date +%H:%M:%S) pairs: $(ls out/readouts/pairs/ep_*.tsv 2>/dev/null | wc -l) tables"
run structure 3
echo "$(date +%H:%M:%S) structure: $(ls out/readouts/structure/*_pooled.txt 2>/dev/null | wc -l) pooled scores"
run shortrange 6
echo "$(date +%H:%M:%S) shortrange: $(ls out/readouts/shortrange/*.json 2>/dev/null | wc -l) tables"
for L in GM12878 H1ESC HFFC6; do
  echo $CHROMS | tr ' ' '\n' | NPROC=4 xargs -P 3 -I{} bash -c "bash $D/microc_genome.sh $L {} > logs/microc_${L}_{}.log 2>&1 || echo 'microc $L {} failed'"
  echo "$(date +%H:%M:%S) Micro-C $L: $(ls out/readouts/microc/${L}_chr*_5000.json 2>/dev/null | wc -l) chromosomes scored"
done
echo "$(date +%H:%M:%S) readouts done for:$CHROMS"
