#!/bin/bash
# Finds the smooth stage term that widens the maximum entropy gaps, one structure of GM12878
# chr1:20-25 Mb per arm. design/gap-fill.md.
#
#   DATA=~/3dgnome-ng/data bash slurm/ensemble/smooth_widening_trace.sh
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/widen}"
BASE="${BASE:-out/gapme/GM12878_maxent.ini}"
R=chr1:20000000-25000000
mkdir -p "$OUT"
git log --format="smooth widening trace on %h %s" -1
t() {  # tag overrides...
  T=$1; shift
  [ -f "$OUT/$T/trace.npz" ] && { echo "[$T] kept"; return; }
  .venv/bin/python playground/smooth_widening_trace.py "$BASE" "$DATA/GM12878" "$R" "$OUT/$T" "$@" > "$OUT/$T.log" 2>&1
  echo "[$T] exit $?"; grep -A6 "blocks," "$OUT/$T.log"
}
t maxent
t maxent_noangle simulation_arcs_smooth.angle_weight=0
t maxent_nowall simulation_arcs_smooth.hard_wall=no
t maxent_noev simulation_arcs_smooth.hard_wall=no excluded_volume.apply_to_smooth=no
t maxent_bare simulation_arcs_smooth.hard_wall=no excluded_volume.apply_to_smooth=no simulation_arcs_smooth.angle_weight=0 confinement.apply_to_smooth=no
t coil simulation_arcs_smooth.start=coil data.contact_map=
echo "WIDEN DONE $(date)"
