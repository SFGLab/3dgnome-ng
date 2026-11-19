#!/bin/bash
# Hole beads against the chromosome expectation and through the smooth kernel, three cell lines,
# workstation, screen gapv2, in a worktree of its own. design/gap-fill.md.
#
#   DATA=~/3dgnome-ng/data REF=~/3dgnome-lawspring bash slurm/ensemble/gap_holes_v2_trial.sh
#
# Waits for REF/gapah.log to end, so one job holds the card at a time. Arms, each
# slurm/ensemble/<cell>_xc_R.ini with the overrides written:
#   holes2         production with hole beads, window contacts against the chromosome at 10 kb,
#                  each hole run through the smooth kernel
#   maxent2        start maxent with the same expectation for the gaps
# Production comes from REF/out/acxc and the first hole arm from REF/out/gapah. Phase 1,
# chr1:10-15 and 35-40 Mb at five structures, the two regions holding holes. Phase 2, whole chr1
# at three structures for holes2, scored on the 1-60 Mb slice.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
REF="${REF:-.}"
OUT="${OUT:-out/gapv2}"
ACXC="$REF/out/acxc"
GAPAH="$REF/out/gapah"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
while ! grep -q "^EXIT" "$REF/gapah.log" 2>/dev/null; do sleep 120; done
git log --format="gap holes v2 trial on %h %s" -1

mcool() { ls $HIC_ROOT/$1/*.mcool | head -1; }

write_inis() {  # cell
  c=$(echo "$1" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$1" "$(mcool "$1")::/resolutions/1000" <<'PY'
import configparser, sys
src, out, cell, uri = sys.argv[1:5]
arms = {
    "holes2": {("holes", "use_hole_beads"): "yes", ("data", "contact_map"): uri},
    "maxent2": {("simulation_arcs_smooth", "start"): "maxent", ("data", "contact_map"): uri},
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
    echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'hole beads .*' "$D.log") $(grep -m1 -oE 'maxent start: .*' "$D.log")"
  done
}

score() {  # cell region tag arms...
  C=$1; R=$2; T=$3; shift 3
  dirs="$ACXC/${C}_${T}_prod"
  [ -d "$GAPAH/${C}_${T}_holes" ] && dirs="$dirs $GAPAH/${C}_${T}_holes"
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

for C in $CELLS; do write_inis "$C"; done

for C in $CELLS; do
  for R in chr1:10000000-15000000 chr1:35000000-40000000; do
    T=$(echo "$R" | tr ':-' '__')
    run "$C" "$R" "$T" 5 holes2 maxent2
    score "$C" "$R" "$T" holes2 maxent2
  done
done
echo "PHASE 1 DONE $(date)"

for C in $CELLS; do
  run "$C" chr1 chr1 3 holes2
  if [ ! -d "$OUT/${C}_chr1_60mb_holes2" ]; then
    .venv/bin/python playground/slice_arm.py "$OUT/${C}_chr1_holes2" "$OUT/${C}_chr1_60mb_holes2" chr1:1-60000000 > /dev/null
    cp "$OUT/${C}_chr1_holes2.log" "$OUT/${C}_chr1_60mb_holes2.log"
  fi
  score "$C" chr1:1-60000000 chr1_60mb holes2
done
echo "GAP HOLES V2 TRIAL DONE $(date)"
