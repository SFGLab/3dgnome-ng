#!/bin/bash
# The second round of the night of 2026-11-19 for the uniform chain: every arm reads one half of
# the contact map and is scored against the other. design/uniform-chain.md.
#
#   bash slurm/ensemble/uniform_night2.sh [first_step]
#
# out/halves holds the halves of playground/thin_cooler.py: a/<cell> is the map of the contact
# cells, b/<cell> the map of the score, data/<cell> the inputs of a run with the Hi-C singletons of
# half a. Every step is one call of arms_trial.sh into its OUT with its own TAG, and keeps its
# score files and close pairs under OUT/scores-<TAG>. The log is out/held/night.log.
set -u
cd "$(dirname "$0")/../.."
export DATA=out/halves/data HIC=out/halves/b CONTACT_HIC=out/halves/a
FIRST="${1:-1}"
LOG=out/held/night.log
R2="chr1:10000000-15000000 chr1:20000000-25000000"
R6="$R2 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
# The base of the round: the centre of mass rule and a bond that resists a squeeze, the winners of
# steps 1 to 4 of uniform_night.sh.
U="main.uniform_chain=yes,simulation_arcs_smooth.anchor_centre=yes,springs.squeeze_constant=1.0"
M="$U,data.contact_map=CONTACT_MAP"
Z3="springs.contact_background_z=3"
C5="$M,$Z3,springs.contact_resolution=5000"
LAD="$C5,springs.contact_resolution_max=25000"
DROP="springs.loop_dropout=yes"
mkdir -p out/held

step() {  # number out tag regions n arm...
  local k=$1 out=$2 tag=$3 regions=$4 n=$5
  shift 5
  [ "$k" -lt "$FIRST" ] && return
  echo "step $k $tag start $(date)" >> "$LOG"
  OUT=$out TAG=$tag REGIONS=$regions N=$n bash slurm/ensemble/arms_trial.sh "$@"
  mkdir -p "$out/scores-$tag"
  cp "$out"/*.txt "$out/$tag.log" "$out/scores-$tag/"
  names="prod"
  for spec in "$@"; do names="$names ${spec%%:*}"; done
  .venv/bin/python playground/uniform_overlap_check.py "$out" $names 2>&1 | grep -v INFO > "$out/scores-$tag/overlaps.txt"
  echo "step $k $tag done $(date)" >> "$LOG"
}

# 1. The contact cells on counts that the run did not read.
step 1 out/held held "$R2" 5 \
  "s_nocells:$U" \
  "s_c5k:$C5" \
  "s_c5k_z2:$M,springs.contact_background_z=2,springs.contact_resolution=5000" \
  "s_c10k:$M,$Z3,springs.contact_resolution=10000" \
  "s_c25k:$M,$Z3,springs.contact_resolution=25000" \
  "s_lad25:$LAD" \
  "s_c5k_drop:$C5,$DROP" \
  "s_lad25_drop:$LAD,$DROP"

# 2. The loops and the bonds on held out counts.
step 2 out/held combo "$R2" 5 \
  "s_c5k_drop:$C5,$DROP" \
  "s_c5k_dropcontact:$C5,$DROP,springs.loop_contact=yes" \
  "s_lad25_dropcontact:$LAD,$DROP,springs.loop_contact=yes" \
  "s_c5k_drop_scale05:$C5,$DROP,springs.loop_dropout_scale=0.5" \
  "s_c5k_drop_scale2:$C5,$DROP,springs.loop_dropout_scale=2.0" \
  "s_c5k_drop_sq03:$C5,$DROP,springs.squeeze_constant=0.3" \
  "s_c5k_drop_sq3:$C5,$DROP,springs.squeeze_constant=3.0" \
  "s_c5k_drop_fixed:$C5,$DROP,simulation_arcs_smooth.anchor_centre=no"

# 3. The larger trial on held out counts: six regions, twenty structures.
step 3 out/held20 large "$R6" 20 \
  "s_c5k_drop:$C5,$DROP" \
  "s_lad25_drop:$LAD,$DROP" \
  "s_c5k:$C5"
echo "night2 DONE $(date)" >> "$LOG"
