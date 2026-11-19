#!/bin/bash
# Drawn anchors with only their overlaps relaxed, three cell lines, workstation, screen spreadev,
# in a worktree of its own. design/gap-fill.md.
#
#   DATA=~/3dgnome-ng/data REF=~/3dgnome-lawspring SPREAD=~/3dgnome-spread2 \
#     bash slurm/ensemble/spread_ev_trial.sh
#
# Waits for SPREAD/spread.log to end, so one job holds the card at a time. Arms, each
# slurm/ensemble/<cell>_xc_R.ini with the overrides written:
#   mxev05       anchors drawn from the maximum entropy Gaussian, then up to 200 solver iterations
#                with every target spring at zero, so only the excluded volume and the arcless
#                repulsion, its reach at 0.5 of the mean arc target, act
#   mxev10       the same with the reach at 1.0
# Production comes from REF/out/acxc and the plain draw from SPREAD/out/spread. Phase 1,
# chr1:10-15, 20-25 and 35-40 Mb at five structures. Phase 2, whole chr1 on GM12878 and H1ESC at
# three structures, scored on the 1-60 Mb slice.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
REF="${REF:-.}"
SPREAD="${SPREAD:-.}"
OUT="${OUT:-out/spreadev}"
ACXC="$REF/out/acxc"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
while ! grep -q "^EXIT" "$SPREAD/spread.log" 2>/dev/null; do sleep 120; done
git log --format="spread ev trial on %h %s" -1

mcool() { ls $HIC_ROOT/$1/*.mcool | head -1; }

write_inis() {  # cell
  c=$(echo "$1" | tr 'A-Z' 'a-z')
  .venv/bin/python - "slurm/ensemble/${c}_xc_R.ini" "$OUT" "$1" "$(mcool "$1")::/resolutions/1000" <<'PY'
import configparser, sys
src, out, cell, uri = sys.argv[1:5]
ev = {
    ("simulation_arcs", "start"): "maxent",
    ("simulation_arcs", "solver_iters"): "200",
    ("springs", "stretch_constant_arcs"): "0",
    ("springs", "squeeze_constant_arcs"): "0",
    ("springs", "background_weight"): "0",
}
arms = {
    "mxev05": {**ev, ("excluded_volume", "arcs_repulsion_cutoff_factor"): "0.5"},
    "mxev10": {**ev, ("excluded_volume", "arcs_repulsion_cutoff_factor"): "1.0"},
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
    echo "[$C $T $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l) $(grep -m1 -oE 'maxent anchors: .*' "$D.log") $(grep -m1 -oE 'hole beads .*' "$D.log")"
  done
}

score() {  # cell region tag arms...
  C=$1; R=$2; T=$3; shift 3
  dirs="$ACXC/${C}_${T}_prod"
  [ -d "$SPREAD/out/spread/${C}_${T}_mx0" ] && dirs="$dirs $SPREAD/out/spread/${C}_${T}_mx0"
  for A in "$@"; do [ -d "$OUT/${C}_${T}_$A" ] && dirs="$dirs $OUT/${C}_${T}_$A"; done
  M=$(mcool "$C")
  .venv/bin/python playground/law_spring_readout.py "$M" "$R" 1000 $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${T}_readout.txt"
  for B in no yes; do
    .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 \
      --singletons "$DATA/$C/${C}_hic_25kb_singletons.bedpe" "$M" "$R" 25000 $dirs 2>&1 \
      | grep -vE "INFO|WARN" > "$OUT/${C}_${T}_battery_balance_$B.txt"
  done
  for d in $dirs; do
    .venv/bin/python playground/contact_spread.py "$ACXC/${C}_${T}_prod" "$d" 2>&1 | grep -vE "INFO|WARN"
  done > "$OUT/${C}_${T}_spread.txt"
  echo "[$C $T] scored $(date +%H:%M)"
}

for C in $CELLS; do write_inis "$C"; done

ARMS="mxev05 mxev10"
for C in $CELLS; do
  for R in chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000; do
    T=$(echo "$R" | tr ':-' '__')
    run "$C" "$R" "$T" 5 $ARMS
    score "$C" "$R" "$T" $ARMS
  done
done
echo "PHASE 1 DONE $(date)"

CHR_ARMS="mxev05"
for C in GM12878 H1ESC; do
  run "$C" chr1 chr1 3 $CHR_ARMS
  for A in $CHR_ARMS; do
    [ -d "$OUT/${C}_chr1_60mb_$A" ] && continue
    .venv/bin/python playground/slice_arm.py "$OUT/${C}_chr1_$A" "$OUT/${C}_chr1_60mb_$A" chr1:1-60000000 > /dev/null
    cp "$OUT/${C}_chr1_$A.log" "$OUT/${C}_chr1_60mb_$A.log"
  done
  score "$C" chr1:1-60000000 chr1_60mb $CHR_ARMS
done
echo "SPREAD EV TRIAL DONE $(date)"
