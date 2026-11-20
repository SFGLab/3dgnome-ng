#!/bin/bash
# The seventh round for the uniform chain: squeeze_constant 1.0 on the default chain.
# Not run. The maintainer chose two canonical configs on 2026-11-20, one per chain, and the
# question of one value of the key for both chains is closed.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night7.sh
#
# No setting implies another: a uniform chain config states squeeze_constant 1.0, and the canonical
# config has one value of the key for both chains. This round measures 1.0 on the default chain,
# six regions and twenty structures, against the default chain at 0.1 in out/held4_20. The round
# runs on the checkout of uniform_night4.sh, commit f5ec43a, with the scorer of ab4b3c6. The log is
# out/held4_20/night7.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
RESCORE="${RESCORE:-slurm/ensemble/rescore_trial.sh}"
OUT=out/held4_20
LOG=$OUT/night7.log
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
ALL="prod d_sq1"
echo "step 1 sq20 start $(date)" >> "$LOG"
OUT=$OUT TAG=sq20 REGIONS=$R6 N=20 bash slurm/ensemble/arms_trial.sh "d_sq1:springs.squeeze_constant=1.0"
for RES in 25000 10000 5000; do
  S=$OUT/scores-sq20p
  [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
  OUT=$OUT SCORES=$S REGIONS=$R6 RES=$RES bash "$RESCORE" $ALL
done
.venv/bin/python playground/uniform_overlap_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-sq20p/overlaps.txt"
.venv/bin/python playground/uniform_stretch_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-sq20p/stretch.txt"
.venv/bin/python "${ENDS:-playground/uniform_ends_check.py}" "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-sq20p/ends.txt"
echo "night7 DONE $(date)" >> "$LOG"
