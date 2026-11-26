#!/bin/bash
# The uniform chain with CTCF and RNAPII loops on chr19 of GM12878 in two arms, without the
# excluded volume of the arcs stage and with its radius at the contact radius. The arms run on a
# workstation, one after another. design/uniform-chain.md, the section on RNAPII loops and the agreement at 25 kb.
#
#   DATA=~/gapfill-data OUT=~/3dgnome-gapfill/out/arcs_ev bash slurm/ensemble/trace_arcs_ev.sh
#
# The base is gm12878_xc_UR.ini, which reads half a of the deep map. Arm `noev` has
# [excluded_volume] apply_to_arcs = no. Arm `r133` has radius_arcs = 1.33, the contact radius of the
# scores in bonds, in place of the mean loop target. Ten structures each, into
# OUT/lr/GM12878_<arm>/chr19. A run that is there is kept.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/arcs_ev}"
mkdir -p "$OUT/lr" "$OUT/ini"
arm() {  # name section key value
  .venv/bin/python - "slurm/ensemble/gm12878_xc_UR.ini" "$OUT/ini/gm12878_xc_UR_$1.ini" "$2" "$3" "$4" <<'PY'
import configparser, sys
c = configparser.ConfigParser(); c.read(sys.argv[1])
c[sys.argv[3]][sys.argv[4]] = sys.argv[5]
with open(sys.argv[2], "w") as fh:
    c.write(fh)
PY
  D="$OUT/lr/GM12878_$1/chr19"
  [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge 10 ] && return
  .venv/bin/python -m gnome3d.cli --config "$OUT/ini/gm12878_xc_UR_$1.ini" --data-dir "$DATA/GM12878" --region chr19 --members 0-9 --out "$D" > "$OUT/lr/GM12878_$1_chr19.log" 2>&1
  echo "[GM12878 UR $1 chr19] exit $? $(date +%H:%M) $(ls "$D"/*.cif 2>/dev/null | wc -l) structures"
}
arm noev excluded_volume apply_to_arcs no
arm r133 excluded_volume radius_arcs 1.33
echo "trace arcs ev DONE $(date)"
