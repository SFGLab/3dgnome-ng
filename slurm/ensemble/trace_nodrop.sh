#!/bin/bash
# The uniform chain without the loop dropout, beside the runs of the redo, on a workstation, one
# run after another. design/uniform-chain.md.
#
#   DATA=~/gapfill-data HALVES=~/gapfill-data/_hic_halves OUT=out/nodrop bash slurm/ensemble/trace_nodrop.sh
#
# 1. The nine runs of 20 Mb of the tools comparison, five structures, on <cell>_hic_uniform.ini with
#    [springs] loop_dropout = no, into OUT/tools20/<cell>_<region>_nodrop.
# 2. chr19 of GM12878, ten structures, on gm12878_xc_UC.ini and gm12878_xc_UR.ini with the same key,
#    into OUT/lr/GM12878_C/chr19 and OUT/lr/GM12878_R/chr19, the layout of the readout scripts.
# The configs read half a of the deep map, data/_hic_half/a/<cell>.mcool. A run that is there is kept.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/nodrop}"
mkdir -p "$OUT/tools20" "$OUT/lr" "$OUT/ini"
ini() {  # source name
  .venv/bin/python - "$1" "$OUT/ini/$2" <<'PY'
import configparser, sys
c = configparser.ConfigParser(); c.read(sys.argv[1])
c["springs"]["loop_dropout"] = "no"
with open(sys.argv[2], "w") as fh:
    c.write(fh)
PY
}
for C in GM12878 H1ESC HFFC6; do
  c=$(echo "$C" | tr 'A-Z' 'a-z')
  ini "slurm/ensemble/${c}_hic_uniform.ini" "${c}_hic_uniform_nodrop.ini"
  for R in chr12:40000000-60000000 chr11:60000000-80000000 chr17:35000000-55000000; do
    T=$(echo "$R" | tr ':-' '__'); D="$OUT/tools20/${C}_${T}_nodrop"
    [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge 5 ] && continue
    .venv/bin/python -m gnome3d.cli --config "$OUT/ini/${c}_hic_uniform_nodrop.ini" --data-dir "$DATA/$C" --region "$R" -n 5 --out "$D" > "$D.log" 2>&1
    echo "[$C $T nodrop] exit $? $(date +%H:%M) $(ls "$D"/*.cif 2>/dev/null | wc -l) structures"
  done
done
for A in C R; do
  ini "slurm/ensemble/gm12878_xc_U$A.ini" "gm12878_xc_U${A}_nodrop.ini"
  D="$OUT/lr/GM12878_$A/chr19"
  [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge 10 ] && continue
  .venv/bin/python -m gnome3d.cli --config "$OUT/ini/gm12878_xc_U${A}_nodrop.ini" --data-dir "$DATA/GM12878" --region chr19 --members 0-9 --out "$D" > "$OUT/lr/GM12878_${A}_chr19.log" 2>&1
  echo "[GM12878 U$A chr19 nodrop] exit $? $(date +%H:%M) $(ls "$D"/*.cif 2>/dev/null | wc -l) structures"
done
echo "trace nodrop DONE $(date)"
