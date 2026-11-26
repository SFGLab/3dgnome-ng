#!/bin/bash
# One arm of a trace on one chromosome of a cell line, a config of slurm/ensemble with keys
# changed, ten structures, on a workstation. CELL is GM12878 and CHROM chr19 unless given.
#
#   DATA=~/gapfill-data OUT=~/3dgnome-gapfill/out/arcs_ev bash slurm/ensemble/trace_arcs_arm.sh \
#     contact gm12878_xc_UR.ini distance.contact_half_saturation=0.001
#
# The arguments are the name of the arm, the base config and the keys as section.key=value. The
# structures go to OUT/lr/<CELL>_<name>/<CHROM> and a line to OUT/arms.log. A run that is there is
# kept. REPO is the checkout that runs, the one of this script unless given.
# design/2026-11-25-rnapii-band.md has the trace.
set -u
cd "${REPO:-$(dirname "$0")/../..}"
DATA="${DATA:-data}"
OUT="${OUT:-out/arcs_ev}"
CELL="${CELL:-GM12878}"
CHROM="${CHROM:-chr19}"
NAME="$1"; BASE="$2"; shift 2
mkdir -p "$OUT/lr" "$OUT/ini"
.venv/bin/python - "slurm/ensemble/$BASE" "$OUT/ini/${BASE%.ini}_$NAME.ini" "$@" <<'PY'
import configparser, sys
c = configparser.ConfigParser(); c.read(sys.argv[1])
for item in sys.argv[3:]:
    key, value = item.split("=", 1)
    section, key = key.split(".", 1)
    if not c.has_section(section):
        c.add_section(section)
    c[section][key] = value
with open(sys.argv[2], "w") as fh:
    c.write(fh)
PY
D="$OUT/lr/${CELL}_$NAME/$CHROM"
if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -lt 10 ]; then
  .venv/bin/python -m gnome3d.cli --config "$OUT/ini/${BASE%.ini}_$NAME.ini" --data-dir "$DATA/$CELL" --region "$CHROM" --members 0-9 --out "$D" > "$OUT/lr/${CELL}_${NAME}_$CHROM.log" 2>&1
  echo "[$CELL $NAME $CHROM] exit $? $(date +%H:%M) $(ls "$D"/*.cif 2>/dev/null | wc -l) structures, $BASE $*" >> "$OUT/arms.log"
fi
