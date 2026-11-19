#!/bin/bash
# The full depth anchor contact table and the two sided contact background on three cell lines, with
# CTCF and RNAPII loops.
#
#   DATA=~/3dgnome-ng/data bash slurm/ensemble/anchor_contacts_crosscell.sh
#
# Four arms per cell, each slurm/ensemble/<cell>_xc_R.ini with overrides:
#   prod           production
#   prod_noscale   production with [anchor_heatmap] heatmap_influence 0
#   z3             the table, pairs held in or out at three Poisson deviations
#   z3_noscale     z3 with heatmap_influence 0
# CELLS picks a subset. A finished arm or table is not redone.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/acxc}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
ARMS="prod prod_noscale z3 z3_noscale"
mkdir -p "$OUT"
git log --format="anchor contacts cross cell on %h %s" -1

for C in $CELLS; do
  c=$(echo "$C" | tr 'A-Z' 'a-z')
  MCOOL=$(ls $HIC_ROOT/$C/*.mcool | head -1)
  SING=$DATA/$C/${C}_hic_25kb_singletons.bedpe
  TABLE=$OUT/${C}_anchor_contacts_1kb_chr1.npz
  if [ ! -f "$TABLE" ]; then
    .venv/bin/python slurm/ensemble/prep_anchor_contacts.py --mcool "$MCOOL" \
      --anchors "$DATA/$C/${C}_xc_anchors_ctcf_rnapii.bed" --chroms chr1 --res 1000 --out "$TABLE"
  fi
  TABLE=$(realpath "$TABLE")
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$C" "$TABLE" <<'PY'
import configparser, sys
src, out, cell, table = sys.argv[1:5]
arms = {
    "prod": {},
    "prod_noscale": {("anchor_heatmap", "heatmap_influence"): "0"},
    "z3": {("data", "anchor_contacts"): table, ("springs", "contact_background_z"): "3"},
    "z3_noscale": {
        ("data", "anchor_contacts"): table,
        ("springs", "contact_background_z"): "3",
        ("anchor_heatmap", "heatmap_influence"): "0",
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

  run() {  # region tag n
    for A in $ARMS; do
      D=$OUT/${C}_${2}_$A
      mkdir -p "$D"
      if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$3" ]; then continue; fi
      echo "[$C $2 $A] start $(date +%H:%M)"
      .venv/bin/python -m gnome3d.cli --config "$OUT/${C}_$A.ini" --data-dir "$DATA/$C" --region "$1" -n "$3" --out "$D" > "$D.log" 2>&1
      echo "[$C $2 $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'anchor contacts: [0-9,]+ of [0-9,]+' "$D.log") $(grep -m1 -oE 'contact background: .*' "$D.log")"
    done
  }

  score() {  # region tag
    dirs=$(for A in $ARMS; do echo "$OUT/${C}_${2}_$A"; done)
    .venv/bin/python playground/law_spring_readout.py "$MCOOL" "$1" 1000 $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${2}_readout.txt"
    for B in no yes; do
      .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 --singletons "$SING" "$MCOOL" "$1" 25000 $dirs 2>&1 \
        | grep -vE "INFO|WARN" > "$OUT/${C}_${2}_battery_balance_$B.txt"
    done
    echo "[$C $2] scored $(date +%H:%M)"
  }

  for R in chr1:20000000-25000000 chr1:10000000-15000000 chr1:35000000-40000000; do
    T=$(echo "$R" | tr ':-' '__')
    run "$R" "$T" 5
    score "$R" "$T"
  done
  run chr1 chr1 3
  for A in $ARMS; do
    .venv/bin/python playground/slice_arm.py "$OUT/${C}_chr1_$A" "$OUT/${C}_chr1_60mb_$A" chr1:1-60000000 > /dev/null
    cp "$OUT/${C}_chr1_$A.log" "$OUT/${C}_chr1_60mb_$A.log"
  done
  score chr1:1-60000000 chr1_60mb
  echo "$C DONE $(date)"
done
echo "ANCHOR CONTACTS CROSS CELL DONE $(date)"
