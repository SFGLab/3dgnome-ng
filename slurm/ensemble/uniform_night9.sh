#!/bin/bash
# The ninth round for the uniform chain: the cap of the excluded volume radius of a pair.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night9.sh
#
# A node excludes at the radius of the arcs stage, and [excluded_volume] arcs_cap limits the radius
# of a pair by its target or by the chain between its two beads. The round adds three arms with the
# cells at 5 kb to out/held4_20, six regions and twenty structures per arm: the radius of 3.0 with
# each cap, and the radius that the stage derives without a cap. It scores them beside the arms at
# the radius of a bead and at 3.0 without a cap. The log is out/held4_20/night9.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
OUT=out/held4_20
LOG=$OUT/night9.log
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,springs.squeeze_constant=1.0"
C5="$U,data.contact_map=CONTACT_MAP,springs.contact_background_z=3,springs.contact_resolution=5000"
R3="excluded_volume.radius_arcs=3.0"
ALL="prod s_c5k s_c5k_r3 s_c5k_r3_target s_c5k_r3_chain s_c5k_auto"
echo "step 1 cap20 start $(date)" >> "$LOG"
OUT=$OUT TAG=cap20 REGIONS=$R6 N=20 bash slurm/ensemble/arms_trial.sh \
  "s_c5k_r3_target:$C5,$R3,excluded_volume.arcs_cap=target" \
  "s_c5k_r3_chain:$C5,$R3,excluded_volume.arcs_cap=chain" \
  "s_c5k_auto:$C5"
echo "step 1 cap20 done $(date)" >> "$LOG"
for RES in 10000 5000 25000; do
  S=$OUT/scores-cap20p
  [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
  OUT=$OUT SCORES=$S REGIONS=$R6 RES=$RES bash slurm/ensemble/rescore_trial.sh $ALL
done
.venv/bin/python playground/uniform_overlap_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-cap20p/overlaps.txt"
.venv/bin/python playground/uniform_stretch_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-cap20p/stretch.txt"
.venv/bin/python playground/uniform_ends_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-cap20p/ends.txt"
echo "night9 DONE $(date)" >> "$LOG"
