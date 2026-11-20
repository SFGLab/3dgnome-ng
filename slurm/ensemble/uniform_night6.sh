#!/bin/bash
# The sixth round for the uniform chain: the excluded volume radius of the arcs stage.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night6.sh [first_step]
#
# A node of the uniform chain excludes at the radius of a bead in the arcs stage, 0.7 of a bond. An
# anchor of the default chain excludes at the mean target distance, 2.6 to 3.1 bonds. With the
# radius of a bead the first and the last megabase of a region share their space in a third of the
# structures. Step 1 adds two arms at a radius of 3 bonds to the larger trial in out/held4_20, six
# regions and twenty structures per arm. Step 2 scores every arm of out/held4_20 in one table at
# 25, 10 and 5 kb, with the close pairs, the stretched pieces and the overlap of the ends. The
# round runs on the checkout of uniform_night4.sh, commit f5ec43a, with the scorer of this commit.
# The log is out/held4_20/night6.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
FIRST="${1:-1}"
RESCORE="${RESCORE:-slurm/ensemble/rescore_trial.sh}"
OUT=out/held4_20
LOG=$OUT/night6.log
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,simulation_arcs_smooth.anchor_centre=yes,springs.squeeze_constant=1.0"
C5="$U,data.contact_map=CONTACT_MAP,springs.contact_background_z=3,springs.contact_resolution=5000"
R3="excluded_volume.radius_arcs=3.0"
ALL="prod s_c5k s_c5k_drop s_lad25 s_lad25_drop s_c25k s_c5k_r3 s_c5k_drop_r3"
mkdir -p "$OUT"

if [ "$FIRST" -le 1 ]; then
  echo "step 1 ev20 start $(date)" >> "$LOG"
  OUT=$OUT TAG=ev20 REGIONS=$R6 N=20 bash slurm/ensemble/arms_trial.sh "s_c5k_r3:$C5,$R3" "s_c5k_drop_r3:$C5,springs.loop_dropout=yes,$R3"
  echo "step 1 ev20 done $(date)" >> "$LOG"
fi

if [ "$FIRST" -le 2 ]; then
  echo "step 2 all20p start $(date)" >> "$LOG"
  for RES in 25000 10000 5000; do
    S=$OUT/scores-all20p
    [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
    OUT=$OUT SCORES=$S REGIONS=$R6 RES=$RES bash "$RESCORE" $ALL
  done
  .venv/bin/python playground/uniform_overlap_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-all20p/overlaps.txt"
  .venv/bin/python playground/uniform_stretch_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-all20p/stretch.txt"
  .venv/bin/python "${ENDS:-playground/uniform_ends_check.py}" "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-all20p/ends.txt"
  echo "step 2 all20p done $(date)" >> "$LOG"
fi
echo "night6 DONE $(date)" >> "$LOG"
