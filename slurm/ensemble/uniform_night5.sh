#!/bin/bash
# The fifth round for the uniform chain: the larger trial of uniform_night4.sh with the two arms
# that the choice of the resolution of the contact cells needs. design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night5.sh [first_step]
#
# The inputs are those of uniform_night4.sh: the data of production, the contact cells from
# out/halves/a/<cell>, the score against out/halves/bw/<cell>. Step 1 adds the ladder alone and the
# cells at 25 kb to out/held4_20, six regions and twenty structures per arm. Step 2 scores every
# arm of out/held4_20 in one table at 25, 10 and 5 kb, with the close pairs and the stretched
# pieces. The log is out/held4_20/night5.log.
# The round runs on the checkout of uniform_night4.sh, commit f5ec43a. A later commit removes a
# rule of the numba kernel, and the structures of one table are built on one commit.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
FIRST="${1:-1}"
OUT=out/held4_20
LOG=$OUT/night5.log
R6="chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,simulation_arcs_smooth.anchor_centre=yes,springs.squeeze_constant=1.0"
M="$U,data.contact_map=CONTACT_MAP"
Z3="springs.contact_background_z=3"
C5="$M,$Z3,springs.contact_resolution=5000"
C25="$M,$Z3,springs.contact_resolution=25000"
LAD="$C5,springs.contact_resolution_max=25000"
ALL="prod s_c5k s_c5k_drop s_lad25 s_lad25_drop s_c25k"
mkdir -p "$OUT"

if [ "$FIRST" -le 1 ]; then
  echo "step 1 res20 start $(date)" >> "$LOG"
  OUT=$OUT TAG=res20 REGIONS=$R6 N=20 bash slurm/ensemble/arms_trial.sh "s_lad25:$LAD" "s_c25k:$C25"
  echo "step 1 res20 done $(date)" >> "$LOG"
fi

if [ "$FIRST" -le 2 ]; then
  echo "step 2 all20 start $(date)" >> "$LOG"
  for RES in 25000 10000 5000; do
    S=$OUT/scores-all20
    [ "$RES" = 25000 ] || S=${S}_$((RES / 1000))k
    OUT=$OUT SCORES=$S REGIONS=$R6 RES=$RES bash slurm/ensemble/rescore_trial.sh $ALL
  done
  .venv/bin/python playground/uniform_overlap_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-all20/overlaps.txt"
  .venv/bin/python playground/uniform_stretch_check.py "$OUT" $ALL 2>&1 | grep -v INFO > "$OUT/scores-all20/stretch.txt"
  echo "step 2 all20 done $(date)" >> "$LOG"
fi
echo "night5 DONE $(date)" >> "$LOG"
