#!/bin/bash
# The full depth anchor contact table and the two sided contact background against production,
# GM12878 with CTCF and RNAPII loops, workstation, screen actrial.
#
#   DATA=~/3dgnome-ng/data BASE=out/gapnight TABLE=out/GM12878_anchor_contacts_1kb_chr1.npz \
#     bash slurm/ensemble/anchor_contacts_trial.sh
#
# Arms, each the production ini BASE/coil.ini with the overrides named:
#   prod        production, the thinned 25 kb singletons and the one sided rule
#   table_old   the anchor map from TABLE, the one sided rule
#   table_z3    the anchor map from TABLE, pairs beyond three Poisson deviations held in or out
# Phase 1, chr1:20-25, 10-15 and 35-40 Mb at five structures. Phase 2, whole chr1 at three
# structures, scored on the 1-60 Mb slice. Every phase ends with the law readout and the
# battery, raw and balanced. A finished arm is not rerun. ARMS picks a subset; COMPARE names
# an earlier OUT whose prod arms are scored beside this run's.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
C=GM12878
DATA="${DATA:-data}"
BASE="${BASE:-out/gapnight}"
OUT="${OUT:-out/actrial}"
TABLE=$(realpath "${TABLE:?the anchor contact table}")
MCOOL=$(ls $HIC_ROOT/$C/*.mcool | head -1)
SING=$DATA/$C/${C}_hic_25kb_singletons.bedpe
mkdir -p "$OUT"
git log --format="anchor contacts trial on %h %s" -1

.venv/bin/python - "$BASE/coil.ini" "$OUT" "$TABLE" <<'PY'
import configparser, sys
src, out, table = sys.argv[1:4]
arms = {
    "prod": {},
    "table_old": {("data", "anchor_contacts"): table},
    "table_z3": {("data", "anchor_contacts"): table, ("springs", "contact_background_z"): "3"},
}
for arm, over in arms.items():
    c = configparser.ConfigParser()
    c.read(src)
    for (sec, key), v in over.items():
        c[sec][key] = v
    with open(f"{out}/{arm}.ini", "w") as fh:
        c.write(fh)
PY
grep -HE "^anchor_contacts|^contact_background_z|^singletons =" "$OUT"/*.ini

run() {  # region tag n arms...
  R=$1; T=$2; N=$3; shift 3
  for A in "$@"; do
    D=$OUT/${T}_$A
    mkdir -p "$D"
    if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ]; then continue; fi
    echo "[$T $A] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$OUT/$A.ini" --data-dir "$DATA/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    echo "[$T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'anchor contacts: .*' "$D.log") $(grep -m1 -oE 'contact background: .*' "$D.log")"
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

ARMS="${ARMS:-prod table_old table_z3}"
COMPARE="${COMPARE:-}"
extra() { [ -n "$COMPARE" ] && echo "$COMPARE/${1}_prod"; }
for R in chr1:20000000-25000000 chr1:10000000-15000000 chr1:35000000-40000000; do
  T=$(echo "$R" | tr ':-' '__')
  run "$R" "$T" 5 $ARMS
  score "$R" "$T" $(extra "$T") $(for A in $ARMS; do echo "$OUT/${T}_$A"; done)
done
echo "PHASE 1 DONE $(date)"

run chr1 chr1 3 $ARMS
for A in $ARMS; do
  .venv/bin/python playground/slice_arm.py "$OUT/chr1_$A" "$OUT/chr1_60mb_$A" chr1:1-60000000 > /dev/null
  cp "$OUT/chr1_$A.log" "$OUT/chr1_60mb_$A.log"
done
score chr1:1-60000000 chr1_60mb $(extra chr1_60mb) $(for A in $ARMS; do echo "$OUT/chr1_60mb_$A"; done)
echo "ANCHOR CONTACTS TRIAL DONE $(date)"
