#!/bin/bash
# The three smooth stage levers against within block overlaps, on three cells, numba kernels.
#
#   bash slurm/ensemble/overlap_levers.sh
#
# Six arms per cell, each a copy of the production ini with the smooth stage on the threaded
# numba kernel, since only that kernel carries the levers and it runs a cell in under five
# minutes on the workstation, and with the excluded volume radius at 0.7 of a bond, which is
# the radius the wall acts at and the one the battery counts at:
#
#   base        nothing else changed, the like for like baseline on this kernel
#   wall        hard_wall = yes
#   wallcap     hard_wall = yes, anchor_cap = 0.5
#   coil        start = coil
#   coilwall    start = coil, hard_wall = yes
#   coilwallcap start = coil, hard_wall = yes, anchor_cap = 0.5
#
# Structures go to out/validation/<cell>_<arm> beside the earlier arms, and the battery and the
# saddle then score every arm of the cell together, the MultiMM arms included.
set -u
cd "$(dirname "$0")/../.."
OUT="${OUT:-out/validation}"
R="${REGION:-chr1:1-60000000}"
N="${N:-5}"
HIC_ROOT="${HIC_ROOT:-/mnt/storagelinux/_hic}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
ARMS="${ARMS:-base wall wallcap coil coilwall coilwallcap}"
mkdir -p "$OUT"
git log --format="levers on %h %s" -1

for C in $CELLS; do
  L=$(echo "$C" | tr '[:upper:]' '[:lower:]')
  MCOOL=$(ls "$HIC_ROOT/$C"/*.mcool | head -1)
  SING=data/$C/${C}_hic_25kb_singletons.bedpe
  TRACK=data/$C/${C}_compartments.bedGraph
  for ARM in $ARMS; do
    D=$OUT/${C}_${ARM}
    INI=$OUT/${C}_${ARM}.ini
    .venv/bin/python - "$ARM" "slurm/ensemble/${L}_hic_arcs.ini" "$INI" <<'PY'
import configparser, sys
arm, src, dst = sys.argv[1:4]
c = configparser.ConfigParser()
c.read(src)
c["simulation_backend"]["mc_executor_smooth"] = "threaded"
c["excluded_volume"]["auto_factor_smooth"] = "0.7"
sm = c["simulation_arcs_smooth"]
sm["hard_wall"] = "yes" if "wall" in arm else "no"
sm["anchor_cap"] = "0.5" if "cap" in arm else "0"
sm["start"] = "coil" if "coil" in arm else "line"
with open(dst, "w") as fh:
    c.write(fh)
PY
    mkdir -p "$D"
    if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ]; then continue; fi
    echo "[$C $ARM] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$INI" --data-dir "data/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    echo "[$C $ARM] done $(date +%H:%M) exit $? structures $(ls "$D"/*.cif 2>/dev/null | wc -l)"
  done
  DIRS=$(for A in "$OUT"/${C}_*/; do [ -n "$(ls "$A"*.cif 2>/dev/null)" ] && echo "$A"; done | tr '\n' ' ')
  # shellcheck disable=SC2086
  .venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons "$SING" "$MCOOL" "$R" 25000 $DIRS 2>&1 \
    | grep -vE "INFO|WARN" | tee "$OUT/${C}_levers_battery.txt"
  # shellcheck disable=SC2086
  .venv/bin/python playground/saddle_offline.py "$MCOOL" "$R" 100000 "$TRACK" $DIRS 2>&1 \
    | grep -vE "INFO|WARN" | tee "$OUT/${C}_levers_saddle.txt"
done
echo "LEVERS DONE $(date)"
