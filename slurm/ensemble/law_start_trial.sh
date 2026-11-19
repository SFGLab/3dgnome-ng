#!/bin/bash
# The law start for a gap's subanchors against production's compact coil, GM12878 with CTCF and
# RNAPII loops, workstation, screen lawstart.
#
#   DATA=~/3dgnome-ng/data BASE=out/lawspring_run bash slurm/ensemble/law_start_trial.sh
#
# The coil arm is the law spring trial's weight zero arm in BASE, production as it stands. The
# law arm is the same config with `start = law`. Two 5 Mb regions at five structures, then the
# law readout and the battery, raw and balanced, over both arms. SKIP_CHR1 unset runs whole
# chr1 at three structures per arm afterwards, scored on the 1-60 Mb slice.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
C=GM12878
DATA="${DATA:-data}"
BASE="${BASE:-out/lawspring_run}"
OUT="${OUT:-out/lawstart}"
MCOOL=$(ls $HIC_ROOT/$C/*.mcool | head -1)
SING=$DATA/$C/${C}_hic_25kb_singletons.bedpe
mkdir -p "$OUT"
git log --format="law start trial on %h %s" -1

# The base ini carries the removed law spring keys at weight zero; both arms drop them.
.venv/bin/python - "$BASE/law_0.ini" "$OUT" <<'PY'
import configparser, sys
src, out = sys.argv[1:3]
for arm, start in (("coil", "coil"), ("law_start", "law")):
    c = configparser.ConfigParser()
    c.read(src)
    sm = c["simulation_arcs_smooth"]
    for k in ("law_weight", "law_range_bp"):
        sm.pop(k, None)
    sm["start"] = start
    with open(f"{out}/{arm}.ini", "w") as fh:
        c.write(fh)
PY
grep -E "^start|^anchors|^clusters|^factors" "$OUT/law_start.ini"

run() {  # region tag n
  D=$OUT/${2}_law_start
  mkdir -p "$D"
  if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -lt "$3" ]; then
    echo "[$2 law start] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$OUT/law_start.ini" --data-dir "$DATA/$C" --region "$1" -n "$3" --out "$D" > "$D.log" 2>&1
    echo "[$2 law start] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l)"
  fi
}

score() {  # region tag dirs...
  R=$1; T=$2; shift 2
  .venv/bin/python playground/law_spring_readout.py "$MCOOL" "$R" 1000 "$@" 2>&1 | grep -vE "INFO|WARN" | tee "$OUT/${T}_readout.txt"
  for B in no yes; do
    .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 --singletons "$SING" "$MCOOL" "$R" 25000 "$@" 2>&1 \
      | grep -vE "INFO|WARN" > "$OUT/${T}_battery_balance_$B.txt"
  done
}

for R in chr1:20000000-25000000 chr1:10000000-15000000; do
  T=$(echo "$R" | tr ':-' '__')
  run "$R" "$T" 5
  score "$R" "$T" "$BASE/${T}_law_0" "$OUT/${T}_law_start"
done
echo "PHASE 1 DONE $(date)"
[ -n "${SKIP_CHR1:-}" ] && { echo "chr1 skipped"; exit 0; }

run chr1 chr1 3
D=$OUT/chr1_coil
mkdir -p "$D"
if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -lt 3 ]; then
  echo "[chr1 coil] start $(date +%H:%M)"
  .venv/bin/python -m gnome3d.cli --config "$OUT/coil.ini" --data-dir "$DATA/$C" --region chr1 -n 3 --out "$D" > "$D.log" 2>&1
  echo "[chr1 coil] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l)"
fi
for A in chr1_coil chr1_law_start; do
  .venv/bin/python playground/slice_arm.py "$OUT/$A" "$OUT/${A}_60mb" chr1:1-60000000 > /dev/null
  cp "$OUT/$A.log" "$OUT/${A}_60mb.log"
done
score chr1:1-60000000 chr1_60mb "$OUT/chr1_coil_60mb" "$OUT/chr1_law_start_60mb"
echo "LAW START TRIAL DONE $(date)"
