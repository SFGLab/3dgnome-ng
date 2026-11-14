#!/bin/bash
# Hole beads with excluded volume against the blocks, three cell lines,
# workstation, screen gapv3, in a worktree of its own. design/gap-fill.md.
#
#   DATA=/home/nik/3dgnome-ng/data REF=/home/nik/3dgnome-lawspring bash slurm/ensemble/gap_holes_v3_trial.sh
#
# One arm, slurm/ensemble/<cell>_xc_R.ini with the overrides written:
#   holes3         production with hole beads, window contacts against the chromosome at 10 kb,
#                  every hole run through the smooth kernel in one pass over the chromosome with
#                  the block beads fixed, so excluded volume acts between holes and blocks
# Production comes from REF/out/acxc, the first hole arm from REF/out/gapah and the second from
# V2/out/gapv2. Phase 1, chr1:10-15 and 35-40 Mb at five structures, the two regions holding
# holes. Phase 2, whole chr1 at three structures, scored on the 1-60 Mb slice.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
REF="${REF:-.}"
OUT="${OUT:-out/gapv3}"
ACXC="$REF/out/acxc"
GAPAH="$REF/out/gapah"
GAPV2="${V2:-.}/out/gapv2"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
git log --format="gap holes v3 trial on %h %s" -1

mcool() { ls /mnt/storagelinux/_hic/$1/*.mcool | head -1; }

write_inis() {  # cell
  c=$(echo "$1" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$1" "$(mcool "$1")::/resolutions/1000" <<'PY'
import configparser, sys
src, out, cell, uri = sys.argv[1:5]
arms = {
    "holes3": {("holes", "use_hole_beads"): "yes", ("data", "contact_map"): uri},
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
  [ -d "$GAPV2/${C}_${T}_holes2" ] && dirs="$dirs $GAPV2/${C}_${T}_holes2"
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
    run "$C" "$R" "$T" 5 holes3
    score "$C" "$R" "$T" holes3
  done
done
echo "PHASE 1 DONE $(date)"

for C in $CELLS; do
  run "$C" chr1 chr1 3 holes3
  if [ ! -d "$OUT/${C}_chr1_60mb_holes3" ]; then
    .venv/bin/python playground/slice_arm.py "$OUT/${C}_chr1_holes3" "$OUT/${C}_chr1_60mb_holes3" chr1:1-60000000 > /dev/null
    cp "$OUT/${C}_chr1_holes3.log" "$OUT/${C}_chr1_60mb_holes3.log"
  fi
  score "$C" chr1:1-60000000 chr1_60mb holes3
  v2=""; [ -d "$GAPV2/${C}_chr1_holes2" ] && v2="$GAPV2/${C}_chr1_holes2"
  .venv/bin/python playground/hole_contact_split.py "$(mcool "$C")" chr1:1-60000000 "$ACXC/${C}_chr1_prod" "$ACXC/${C}_chr1_prod" $v2 "$OUT/${C}_chr1_holes3" > "$OUT/hole_contact_split_${C}_chr1.txt" 2>&1
  .venv/bin/python playground/hole_contact_overlap.py "$ACXC/${C}_chr1_prod" "$OUT/${C}_chr1_holes3" > "$OUT/hole_contact_overlap_${C}_chr1.txt" 2>&1
done
echo "GAP HOLES V3 TRIAL DONE $(date)"
