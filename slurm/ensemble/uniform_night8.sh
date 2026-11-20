#!/bin/bash
# The eighth round for the uniform chain: the test of the contact cells at z of 2.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night8.sh
#
# Every arm of the larger trial holds a pair by its cell at contact_background_z of 3. This round
# adds the cells at 5 kb at z of 2 to out/held4_20, at the radius of a bead and at radius_arcs 3.0,
# six regions and twenty structures per arm, and scores them beside the same arms at z of 3. The
# round runs on the checkout of uniform_night4.sh, commit f5ec43a, with the scorer of ab4b3c6. The
# log is out/held4_20/night8.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
RESCORE="${RESCORE:-slurm/ensemble/rescore_trial.sh}"
OUT=out/held4_20
LOG=$OUT/night8.log
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,simulation_arcs_smooth.anchor_centre=yes,springs.squeeze_constant=1.0"
Z2="$U,data.contact_map=CONTACT_MAP,springs.contact_background_z=2,springs.contact_resolution=5000"
ALL="prod s_c5k s_c5k_z2 s_c5k_r3 s_c5k_z2_r3"
echo "step 1 z20 start $(date)" >> "$LOG"
OUT=$OUT TAG=z20 REGIONS=$R6 N=20 bash slurm/ensemble/arms_trial.sh "s_c5k_z2:$Z2" "s_c5k_z2_r3:$Z2,excluded_volume.radius_arcs=3.0"
echo "step 1 z20 done $(date)" >> "$LOG"
for RES in 10000 5000 25000; do
  S=$OUT/scores-z20p
  [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
  OUT=$OUT SCORES=$S REGIONS=$R6 RES=$RES bash "$RESCORE" $ALL
done
.venv/bin/python playground/uniform_overlap_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-z20p/overlaps.txt"
.venv/bin/python playground/uniform_stretch_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-z20p/stretch.txt"
.venv/bin/python "${ENDS:-playground/uniform_ends_check.py}" "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-z20p/ends.txt"
echo "night8 DONE $(date)" >> "$LOG"
