#!/bin/bash
# Overnight arms on where a gap's subanchors start, GM12878 with CTCF and RNAPII loops,
# workstation, screen gapnight.
#
#   DATA=~/3dgnome-ng/data BASE=out/lawspring_run bash slurm/ensemble/gap_start_overnight.sh
#
# Arms, each the production ini from BASE with the overrides named:
#   coil          production, start = coil, hard wall
#   law           start = law
#   law_nowall    start = law, hard_wall = no
#   law_subheat   start = law, use_subanchor_heatmap = yes, the Hi-C term at the subanchor level
# Phase 1, three 5 Mb regions at five structures. Phase 2, whole chr1 at three structures for
# coil, law and law_nowall, scored on the 1-60 Mb slice. Every phase ends with the law readout
# and the battery, raw and balanced. A finished arm is not rerun.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
C=GM12878
DATA="${DATA:-data}"
BASE="${BASE:-out/lawspring_run}"
OUT="${OUT:-out/gapnight}"
MCOOL=$(ls $HIC_ROOT/$C/*.mcool | head -1)
SING=$DATA/$C/${C}_hic_25kb_singletons.bedpe
mkdir -p "$OUT"
git log --format="gap start overnight on %h %s" -1

.venv/bin/python - "$BASE/law_0.ini" "$OUT" <<'PY'
import configparser, sys
src, out = sys.argv[1:3]
arms = {
    "coil": {},
    "law": {("simulation_arcs_smooth", "start"): "law"},
    "law_nowall": {("simulation_arcs_smooth", "start"): "law", ("simulation_arcs_smooth", "hard_wall"): "no"},
    "law_subheat": {("simulation_arcs_smooth", "start"): "law", ("subanchor_heatmap", "use_subanchor_heatmap"): "yes"},
}
for arm, over in arms.items():
    c = configparser.ConfigParser()
    c.read(src)
    for k in ("law_weight", "law_range_bp"):
        c["simulation_arcs_smooth"].pop(k, None)
    for (sec, key), v in over.items():
        c[sec][key] = v
    with open(f"{out}/{arm}.ini", "w") as fh:
        c.write(fh)
PY
grep -HE "^start|^hard_wall|^use_subanchor_heatmap" "$OUT"/*.ini

run() {  # region tag n arms...
  R=$1; T=$2; N=$3; shift 3
  for A in "$@"; do
    D=$OUT/${T}_$A
    mkdir -p "$D"
    if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ]; then continue; fi
    echo "[$T $A] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$OUT/$A.ini" --data-dir "$DATA/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    echo "[$T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) smooth $(grep -oE 'smooth\[batch\]: [0-9]+ nodes in [0-9.]+s' "$D.log" | head -1)"
  done
}

score() {  # region tag dirs...
  R=$1; T=$2; shift 2
  .venv/bin/python playground/law_spring_readout.py "$MCOOL" "$R" 1000 "$@" 2>&1 | grep -vE "INFO|WARN" > "$OUT/${T}_readout.txt"
  for B in no yes; do
    .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 --singletons "$SING" "$MCOOL" "$R" 25000 "$@" 2>&1 \
      | grep -vE "INFO|WARN" > "$OUT/${T}_battery_balance_$B.txt"
  done
  echo "[$T] scored $(date +%H:%M)"
}

SMALL_ARMS="coil law law_nowall law_subheat"
for R in chr1:20000000-25000000 chr1:10000000-15000000 chr1:35000000-40000000; do
  T=$(echo "$R" | tr ':-' '__')
  run "$R" "$T" 5 $SMALL_ARMS
  score "$R" "$T" $(for A in $SMALL_ARMS; do echo "$OUT/${T}_$A"; done)
done
echo "PHASE 1 DONE $(date)"

CHR1_ARMS="coil law law_nowall"
run chr1 chr1 3 $CHR1_ARMS
for A in $CHR1_ARMS; do
  .venv/bin/python playground/slice_arm.py "$OUT/chr1_$A" "$OUT/chr1_60mb_$A" chr1:1-60000000 > /dev/null
  cp "$OUT/chr1_$A.log" "$OUT/chr1_60mb_$A.log"
done
score chr1:1-60000000 chr1_60mb $(for A in $CHR1_ARMS; do echo "$OUT/chr1_60mb_$A"; done)
echo "GAP START OVERNIGHT DONE $(date)"
