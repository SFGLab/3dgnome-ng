#!/bin/bash
# The maximum entropy gap start against the coil start on three cell lines, CTCF and RNAPII
# loops, workstation, screen gapme. design/gap-fill.md, idea 1.
#
#   DATA=~/3dgnome-ng/data bash slurm/ensemble/gap_maxent_trial.sh
#
# Production arms come from the anchor contacts cross cell run in ACXC, out/acxc, which ran
# the same configs on the commit before this start existed. Before anything else GM12878
# chr1:20-25 Mb is rerun at production and compared with that run byte for byte, which is the
# gate that the new code leaves the coil start alone. A difference is reported and the run goes
# on, since a GPU pool of another size packs launches differently and moves the bits too. New arms, each slurm/ensemble/<cell>_xc_R.ini
# with the overrides written:
#   fbm         start maxent, the law alone
#   maxent      start maxent, the law scaled by window contacts from the cell's map at 1 kb
#   maxent_z3   maxent with the full depth anchor table and the two sided background at z 3
# Phase 1, chr1:20-25, 10-15 and 35-40 Mb at five structures on every cell. Phase 2, whole chr1
# at three structures for fbm and maxent, scored on the 1-60 Mb slice. Phase 3, whole chr1 for
# maxent_z3. Every scoring step has the law readout and the battery, raw and balanced, with the
# production and z3 arms of ACXC beside the new ones. A finished arm is not rerun.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/gapme}"
ACXC="${ACXC:-out/acxc}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
git log --format="gap maxent trial on %h %s" -1

mcool() { ls $HIC_ROOT/$1/*.mcool | head -1; }

write_inis() {  # cell
  c=$(echo "$1" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$1" "$(mcool "$1")::/resolutions/1000" \
    "$(realpath "$ACXC/${1}_anchor_contacts_1kb_chr1.npz")" <<'PY'
import configparser, sys
src, out, cell, uri, table = sys.argv[1:6]
arms = {
    "prod": {},
    "fbm": {("simulation_arcs_smooth", "start"): "maxent"},
    "maxent": {("simulation_arcs_smooth", "start"): "maxent", ("data", "contact_map"): uri},
    "maxent_z3": {
        ("simulation_arcs_smooth", "start"): "maxent",
        ("data", "contact_map"): uri,
        ("data", "anchor_contacts"): table,
        ("springs", "contact_background_z"): "3",
    },
}
for arm, over in arms.items():
    c = configparser.ConfigParser()
    c.read(src)
    for (sec, key), v in over.items():
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
    echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'maxent start: .*' "$D.log") $(grep -m1 -oE 'contact background: .*' "$D.log")"
  done
}

score() {  # cell region tag arms...
  C=$1; R=$2; T=$3; shift 3
  dirs="$ACXC/${C}_${T}_prod $ACXC/${C}_${T}_z3"
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

# Gate, the coil start at production unchanged by this commit.
G=chr1_20000000_25000000
run GM12878 chr1:20000000-25000000 $G 5 prod
if cmp -s <(cat "$OUT"/GM12878_${G}_prod/*.cif) <(cat "$ACXC"/GM12878_${G}_prod/*.cif); then
  echo "GATE identical, production coil start unchanged"
else
  echo "GATE DIFFERS, compare per file before reading the arms"
fi

REGIONS="chr1:20000000-25000000 chr1:10000000-15000000 chr1:35000000-40000000"
for C in $CELLS; do
  for R in $REGIONS; do
    T=$(echo "$R" | tr ':-' '__')
    run "$C" "$R" "$T" 5 fbm maxent maxent_z3
    score "$C" "$R" "$T" fbm maxent maxent_z3
  done
done
echo "PHASE 1 DONE $(date)"

for C in $CELLS; do
  run "$C" chr1 chr1 3 fbm maxent
  slice60 "$C" fbm maxent
  score "$C" chr1:1-60000000 chr1_60mb fbm maxent
done
echo "PHASE 2 DONE $(date)"

for C in $CELLS; do
  run "$C" chr1 chr1 3 maxent_z3
  slice60 "$C" maxent_z3
  score "$C" chr1:1-60000000 chr1_60mb fbm maxent maxent_z3
done
echo "GAP MAXENT TRIAL DONE $(date)"
