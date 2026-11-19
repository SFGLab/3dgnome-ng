#!/bin/bash
# The trials of the uniform chain of the night of 2026-11-19, one after another.
# design/uniform-chain.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/uniform_night.sh [first_step]
#
# Every step is one call of arms_trial.sh into OUT, default out/uniform, with its own TAG. An arm
# that has its structures is not run again. After a step its score files are copied to
# OUT/scores-<TAG> with the close pairs of playground/uniform_overlap_check.py, since the next
# step writes the same file names. The log is OUT/night.log. A step number skips the steps
# before it.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-data}" HIC="${HIC:-${DATA:-data}/_hic}"
export OUT="${OUT:-out/uniform}"
FIRST="${1:-1}"
R2="chr1:10000000-15000000 chr1:20000000-25000000"
R6="$R2 chr1:35000000-40000000 chr1:60000000-65000000 chr1:150000000-155000000 chr1:200000000-205000000"
U="main.uniform_chain=yes"
C5="$U,data.contact_map=CONTACT_MAP,springs.contact_background_z=3,springs.contact_resolution=5000"
CTR="$C5,simulation_arcs_smooth.anchor_centre=yes"
MID="main.contact_bead=midpoint"
mkdir -p "$OUT"

step() {  # number tag regions n arm...
  local k=$1 tag=$2 regions=$3 n=$4
  shift 4
  [ "$k" -lt "$FIRST" ] && return
  echo "step $k $tag start $(date)" >> "$OUT/night.log"
  TAG=$tag REGIONS=$regions N=$n bash slurm/ensemble/arms_trial.sh "$@"
  mkdir -p "$OUT/scores-$tag"
  cp "$OUT"/*.txt "$OUT/$tag.log" "$OUT/scores-$tag/"
  names="prod"
  for spec in "$@"; do names="$names ${spec%%:*}"; done
  .venv/bin/python playground/uniform_overlap_check.py "$OUT" $names 2>&1 | grep -v INFO > "$OUT/scores-$tag/overlaps.txt"
  echo "step $k $tag done $(date)" >> "$OUT/night.log"
}

# 1. Room for the free chain: gyration windows under 100 kb, with the nodes fixed and with the
#    centre of mass rule.
step 1 gyr "$R2" 5 "node_c5k:$C5" "node_c5k_centre:$CTR" \
  "node_c5k_g10k:$C5,springs.gyration_min_bp=10000" \
  "node_c5k_g25k:$C5,springs.gyration_min_bp=25000" \
  "node_c5k_g50k:$C5,springs.gyration_min_bp=50000" \
  "node_mid_c5k_g25k:$C5,$MID,springs.gyration_min_bp=25000" \
  "ctr_g25k:$CTR,springs.gyration_min_bp=25000" \
  "ctr_g50k:$CTR,springs.gyration_min_bp=50000"

# 2. The resolution and the test of the contact cells, with the centre of mass rule.
CZ="$U,data.contact_map=CONTACT_MAP,simulation_arcs_smooth.anchor_centre=yes"
step 2 res "$R2" 5 "node_c5k_centre:$CTR" \
  "ctr_c2k:$CZ,springs.contact_background_z=3,springs.contact_resolution=2000" \
  "ctr_c10k:$CZ,springs.contact_background_z=3,springs.contact_resolution=10000" \
  "ctr_c25k:$CZ,springs.contact_background_z=3,springs.contact_resolution=25000" \
  "ctr_c5k_z2:$CZ,springs.contact_background_z=2,springs.contact_resolution=5000" \
  "ctr_c5k_z4:$CZ,springs.contact_background_z=4,springs.contact_resolution=5000" \
  "ctr_nocells:$U,simulation_arcs_smooth.anchor_centre=yes"

# 3. The draw and the loops, with the centre of mass rule.
step 3 draw "$R2" 5 "node_c5k_centre:$CTR" \
  "ctr_mid:$CTR,$MID" \
  "ctr_dropout:$CTR,springs.loop_dropout=yes" \
  "ctr_dropout_contact:$CTR,springs.loop_dropout=yes,springs.loop_contact=yes" \
  "ctr_w05:$CTR,springs.arc_weight_exponent=0.5" \
  "ctr_nomotif:$CTR,motif_orientation.use_motif_orientation=no" \
  "ctr_motif10:$CTR,motif_orientation.weight=10"

# 4. The smooth stage and the relaxation, with the centre of mass rule.
step 4 smooth "$R2" 5 "node_c5k_centre:$CTR" \
  "ctr_squeeze1:$CTR,springs.squeeze_constant=1.0" \
  "ctr_bonds1:$CTR,springs.squeeze_constant=1.0,springs.stretch_constant=1.0" \
  "ctr_step05:$CTR,main.noise_smooth=0.5" \
  "ctr_norelax:$CTR,relax.use_cross_block_relax=no" \
  "ctr_nowall:$CTR,simulation_arcs_smooth.hard_wall=no" \
  "ctr_ev1:$CTR,excluded_volume.auto_factor_smooth=1.0"

# 5. The arcs stage, with the centre of mass rule.
step 5 arcs "$R2" 5 "node_c5k_centre:$CTR" \
  "ctr_bg03:$CTR,springs.background_weight=0.3" \
  "ctr_bg1:$CTR,springs.background_weight=1.0" \
  "ctr_gyr03:$CTR,springs.gyration_weight=0.3" \
  "ctr_gyr0:$CTR,springs.gyration_weight=0" \
  "ctr_evarcs1:$CTR,excluded_volume.radius_arcs=1.0" \
  "ctr_range50k:$CTR,springs.background_range_bp=50000"

# 6. The larger trial: six regions, twenty structures.
OUT20="${OUT}20"
OUT=$OUT20 step 6 large "$R6" 20 "node_c5k:$C5" "node_c5k_centre:$CTR" "ctr_mid:$CTR,$MID"
echo "night DONE $(date)" >> "$OUT/night.log"
