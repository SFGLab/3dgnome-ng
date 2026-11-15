#!/bin/bash
# Arms against production on chosen regions and cells, one machine, one job at a time.
#
#   DATA=~/gapfill-data HIC=~/gapfill-data/_hic OUT=out/arms TAG=name \
#   CELLS="GM12878 H1ESC HFFC6" REGIONS="chr1:10000000-15000000 chr1:20000000-25000000" N=5 \
#     bash slurm/ensemble/arms_trial.sh 'arm1:sec.key=val,sec.key=val' 'arm2:sec.key=val' ...
#
# Every arm is slurm/ensemble/<cell>_xc_R.ini with its overrides written, and production, the
# file as it is, runs beside them on the same machine, since another card or pool size does not
# reproduce a structure bit for bit. A value CONTACT_MAP in an override is replaced by the cell's
# deepest mcool at 1 kb. A region written chr1 runs the whole chromosome and is scored on
# chr1:1-60000000. Each region is scored on the pooled ensemble map, the headline, and by the
# battery per structure on raw and balanced counts, the readout and the contact spread, and the
# log of the whole trial is OUT/TAG.log.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
HIC="${HIC:-$DATA/_hic}"
OUT="${OUT:-out/arms}"
TAG="${TAG:-arms}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
REGIONS="${REGIONS:-chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000}"
N="${N:-5}"
mkdir -p "$OUT"
LOG="$OUT/$TAG.log"
git log --format="$TAG on %h %s" -1 >> "$LOG"

mcool() { ls -S "$HIC/$1"/*.mcool | head -1; }

ARMS="prod"
for spec in "$@"; do ARMS="$ARMS ${spec%%:*}"; done

for C in $CELLS; do
  c=$(echo "$C" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$C" "$(mcool "$C")::/resolutions/1000" "$@" <<'PY'
import configparser, sys
src, out, cell, uri, *specs = sys.argv[1:]
arms = {"prod": {}}
for spec in specs:
    name, _, body = spec.partition(":")
    over = {}
    for item in filter(None, body.split(",")):
        key, val = item.split("=", 1)
        sec, k = key.split(".", 1)
        over[(sec, k)] = uri if val == "CONTACT_MAP" else val
    arms[name] = over
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
done

for C in $CELLS; do
  M=$(mcool "$C")
  for R in $REGIONS; do
    T=$(echo "$R" | tr ':-' '__')
    SCORE_R=$R; SCORE_T=$T
    [ "$R" = chr1 ] && SCORE_R=chr1:1-60000000 && SCORE_T=chr1_60mb
    dirs=""
    for A in $ARMS; do
      D=$OUT/${C}_${T}_$A
      dirs="$dirs $OUT/${C}_${SCORE_T}_$A"
      mkdir -p "$D"
      n=$N; [ "$R" = chr1 ] && n=${N_CHR:-3}
      if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -lt "$n" ]; then
        echo "[$C $T $A] start $(date +%H:%M)" >> "$LOG"
        .venv/bin/python -m gnome3d.cli --config "$OUT/${C}_$A.ini" --data-dir "$DATA/$C" --region "$R" -n "$n" --out "$D" > "$D.log" 2>&1
        echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'maxent anchors: [0-9]+ anchors.*' "$D.log") $(grep -m1 -oE 'hole beads .*' "$D.log")" >> "$LOG"
      fi
      if [ "$R" = chr1 ] && [ ! -d "$OUT/${C}_chr1_60mb_$A" ]; then
        .venv/bin/python playground/slice_arm.py "$D" "$OUT/${C}_chr1_60mb_$A" chr1:1-60000000 > /dev/null
        cp "$D.log" "$OUT/${C}_chr1_60mb_$A.log"
      fi
    done
    .venv/bin/python playground/law_spring_readout.py "$M" "$SCORE_R" 1000 $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${SCORE_T}_readout.txt"
    for B in no yes; do
      .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 \
        --singletons "$DATA/$C/${C}_hic_25kb_singletons.bedpe" "$M" "$SCORE_R" 25000 $dirs 2>&1 \
        | grep -vE "INFO|WARN" > "$OUT/${C}_${SCORE_T}_battery_balance_$B.txt"
    done
    .venv/bin/python playground/ensemble_score.py "$M" "$SCORE_R" $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${SCORE_T}_pooled.txt"
    for d in $dirs; do
      .venv/bin/python playground/contact_spread.py "$OUT/${C}_${SCORE_T}_prod" "$d" 2>&1 | grep -vE "INFO|WARN"
    done > "$OUT/${C}_${SCORE_T}_spread.txt"
    echo "[$C $T] scored $(date +%H:%M)" >> "$LOG"
  done
done
echo "$TAG DONE $(date)" >> "$LOG"
