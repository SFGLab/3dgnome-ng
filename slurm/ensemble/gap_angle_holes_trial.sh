#!/bin/bash
# The bend term of the smooth stage against the maximum entropy gaps, and hole beads, on three cell
# lines. design/gap-fill.md.
#
#   DATA=~/3dgnome-ng/data bash slurm/ensemble/gap_angle_holes_trial.sh
#
# Arms, each slurm/ensemble/<cell>_xc_R.ini with overrides:
#   noangle               production with angle_weight 0
#   maxent_noangle        start maxent with the map of the cell at 1 kb, angle_weight 0
#   maxent_bare           maxent_noangle with no hard wall and no smooth excluded volume
#   holes                 production with hole beads, scaled by the map of the cell
#   holes_maxent_noangle  maxent_noangle with hole beads
# Production comes from out/acxc and the maxent arm from out/gapme. A finished arm is not rerun.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/gapah}"
ACXC="${ACXC:-out/acxc}"
GAPME="${GAPME:-out/gapme}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
git log --format="gap angle and holes trial on %h %s" -1

mcool() { ls $HIC_ROOT/$1/*.mcool | head -1; }

write_inis() {  # cell
  c=$(echo "$1" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$1" "$(mcool "$1")::/resolutions/1000" <<'PY'
import configparser, sys
src, out, cell, uri = sys.argv[1:5]
S = "simulation_arcs_smooth"
noangle = {(S, "angle_weight"): "0"}
maxent = {(S, "start"): "maxent", ("data", "contact_map"): uri}
holes = {("holes", "use_hole_beads"): "yes", ("data", "contact_map"): uri}
arms = {
    "prod": {},
    "noangle": noangle,
    "maxent_noangle": {**maxent, **noangle},
    "maxent_bare": {
        **maxent, **noangle, (S, "hard_wall"): "no", ("excluded_volume", "apply_to_smooth"): "no"
    },
    "holes": holes,
    "holes_maxent_noangle": {**maxent, **noangle, **holes},
}
for arm, over in arms.items():
    c = configparser.ConfigParser()
    c.read(src)
    for (sec, key), v in over.items():
        if sec not in c:
            c[sec] = {}
        c[sec][key] = v
    with open(f"{out}/{cell}_{arm}.ini", "w") as fh:
        c.write(fh)
PY
}

run() {  # cell region tag n arms...
  C=$1; R=$2; T=$3; N=$4; shift 4
  for A in "$@"; do
    D=$OUT/${C}_${T}_$A
    mkdir -p "$D"
    if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ]; then continue; fi
    echo "[$C $T $A] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$OUT/${C}_$A.ini" --data-dir "$DATA/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'hole beads .*' "$D.log")"
  done
}

score() {  # cell region tag arms...
  C=$1; R=$2; T=$3; shift 3
  dirs="$ACXC/${C}_${T}_prod $GAPME/${C}_${T}_maxent"
  for A in "$@"; do [ -d "$OUT/${C}_${T}_$A" ] && dirs="$dirs $OUT/${C}_${T}_$A"; done
  M=$(mcool "$C")
  .venv/bin/python playground/law_spring_readout.py "$M" "$R" 1000 $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${T}_readout.txt"
  for B in no yes; do
    .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 \
      --singletons "$DATA/$C/${C}_hic_25kb_singletons.bedpe" "$M" "$R" 25000 $dirs 2>&1 \
      | grep -vE "INFO|WARN" > "$OUT/${C}_${T}_battery_balance_$B.txt"
  done
  echo "[$C $T] scored $(date +%H:%M)"
}

slice60() {  # cell arms...
  C=$1; shift
  for A in "$@"; do
    [ -d "$OUT/${C}_chr1_60mb_$A" ] && continue
    .venv/bin/python playground/slice_arm.py "$OUT/${C}_chr1_$A" "$OUT/${C}_chr1_60mb_$A" chr1:1-60000000 > /dev/null
    cp "$OUT/${C}_chr1_$A.log" "$OUT/${C}_chr1_60mb_$A.log"
  done
}

for C in $CELLS; do write_inis "$C"; done

G=chr1_20000000_25000000
run GM12878 chr1:20000000-25000000 $G 5 prod
if cmp -s <(cat "$OUT"/GM12878_${G}_prod/*.cif) <(cat "$ACXC"/GM12878_${G}_prod/*.cif); then
  echo "GATE identical, production unchanged"
else
  echo "GATE DIFFERS, compare per file before reading the arms"
fi

ARMS="noangle maxent_noangle maxent_bare holes holes_maxent_noangle"
for C in $CELLS; do
  for R in chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000; do
    T=$(echo "$R" | tr ':-' '__')
    run "$C" "$R" "$T" 5 $ARMS
    score "$C" "$R" "$T" $ARMS
  done
done
echo "PHASE 1 DONE $(date)"

CHR_ARMS="holes maxent_noangle holes_maxent_noangle"
for C in $CELLS; do
  run "$C" chr1 chr1 3 $CHR_ARMS
  slice60 "$C" $CHR_ARMS
  score "$C" chr1:1-60000000 chr1_60mb $CHR_ARMS
done
echo "GAP ANGLE HOLES TRIAL DONE $(date)"
