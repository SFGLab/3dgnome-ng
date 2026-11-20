#!/bin/bash
# The third round of the night of 2026-11-19 for the uniform chain: the inputs of production, the
# contact cells from one half of the contact map, the score against the other half.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night3.sh [first_step]
#
# The second round gave every run the Hi-C singletons of half a, which are 80 times deeper than the
# singletons of production and fit another exponent. Here DATA is the data of production. A run
# differs from the earlier trials in the map of its contact cells alone. out/halves/a/<cell> is
# that map. out/halves/bw/<cell> is the map of the score, half b with the balancing weights of the
# whole map. Every step is one call of arms_trial.sh into its OUT with its own TAG, and keeps its
# score files and close pairs under OUT/scores-<TAG>. The log is out/held3/night.log.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-$HOME/gapfill-data}" HIC=out/halves/bw CONTACT_HIC=out/halves/a
FIRST="${1:-1}"
LOG=out/held3/night.log
R2="chr1:10000000-15000000 chr1:20000000-25000000"
R6="$R2 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes,simulation_arcs_smooth.anchor_centre=yes,springs.squeeze_constant=1.0"
M="$U,data.contact_map=CONTACT_MAP"
Z3="springs.contact_background_z=3"
C5="$M,$Z3,springs.contact_resolution=5000"
C25="$M,$Z3,springs.contact_resolution=25000"
LAD="$C5,springs.contact_resolution_max=25000"
DROP="springs.loop_dropout=yes"
mkdir -p out/held3

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
step 1 out/held3 held3 "$R2" 5 \
  "s_nocells:$U" \
  "s_c5k:$C5" \
  "s_c10k:$M,$Z3,springs.contact_resolution=10000" \
  "s_c25k:$C25" \
  "s_lad25:$LAD" \
  "s_c5k_drop:$C5,$DROP" \
  "s_c25k_drop:$C25,$DROP" \
  "s_lad25_drop:$LAD,$DROP"

# 2. The larger trial: six regions, twenty structures.
step 2 out/held3_20 large3 "$R6" 20 \
  "s_c5k_drop:$C5,$DROP" \
  "s_c25k_drop:$C25,$DROP" \
  "s_lad25_drop:$LAD,$DROP"

# 3. The loops and the bonds.
step 3 out/held3 combo3 "$R2" 5 \
  "s_c5k_drop:$C5,$DROP" \
  "d_drop:$DROP" \
  "s_nocells_drop:$U,$DROP" \
  "s_c5k_dropcontact:$C5,$DROP,springs.loop_contact=yes" \
  "s_c5k_drop_scale05:$C5,$DROP,springs.loop_dropout_scale=0.5" \
  "s_c5k_drop_scale2:$C5,$DROP,springs.loop_dropout_scale=2.0" \
  "s_c5k_drop_sq03:$C5,$DROP,springs.squeeze_constant=0.3" \
  "s_c5k_drop_sq3:$C5,$DROP,springs.squeeze_constant=3.0" \
  "s_c5k_drop_fixed:$C5,$DROP,simulation_arcs_smooth.anchor_centre=no"
echo "night3 DONE $(date)" >> "$LOG"
