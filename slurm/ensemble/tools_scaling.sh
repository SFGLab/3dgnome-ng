#!/bin/bash
# Run time against bead count for 3dgnome, cudaMMC and MultiMM on GM12878 chr1, one machine, one
# job at a time. design/plans/2026-11-16-genomes-and-comparison.md.
#
#   DATA=~/gapfill-data TOOLS=out/tools OUT=out/toolscale bash slurm/ensemble/tools_scaling.sh
#
# The tools trial times each tool as shipped, cudaMMC at its loop density of 5 and MultiMM at our
# bead count. Here each tool runs at more bead counts, three structures each, timed by wall clock
# into OUT/timing.tsv:
#   ours       `[main] target_bp_per_subanchor` 2000 and 4000 beside production's 1000
#   cudammc    `[main] loop_density` 10 and 20, then the density whose bead count, on the line
#              through the measured ones, comes nearest ours at 1000
#   multimm    at the bead counts of our 2000 and 4000 runs
# Waits for the tools trial and the site check behind it.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
TOOLS="${TOOLS:-out/tools}"
OUT="${OUT:-out/toolscale}"
N="${N:-3}"
C=GM12878
CUDAMMC="${CUDAMMC:-$HOME/tools/cudaMMC/build/cudaMMC}"
MULTIMM="${MULTIMM:-$HOME/tools/mmenv/bin/MultiMM}"
SITE_CHECK="${SITE_CHECK:-$HOME/3dgnome-web-lr/check.log}"
mkdir -p "$OUT"
LOG="$OUT/scale.log"
TIMING="$OUT/timing.tsv"
while ! grep -q "^EXIT" "$TOOLS/tools.log" 2>/dev/null; do sleep 120; done
while [ -f "$SITE_CHECK" ] && ! grep -q "WEBCHECK_DONE" "$SITE_CHECK"; do sleep 60; done
[ -f "$TIMING" ] || printf "tool\tsetting\tstructures\tbeads\tseconds\n" > "$TIMING"
git log --format="tools scaling on %h %s" -1 >> "$LOG"

beads() { grep -c "^ATOM" "$(ls "$1"/*.cif | head -1)"; }
row() {  # tool setting dir t0
  local nb=0
  ls "$3"/*.cif > /dev/null 2>&1 && nb=$(beads "$3")
  printf "%s\t%s\t%s\t%s\t%.1f\n" "$1" "$2" "$N" "$nb" "$(echo "$(date +%s.%N) - $4" | bc -l)" >> "$TIMING"
  echo "[$1 $2] $(date +%H:%M) beads $nb" >> "$LOG"
}
ini_with() {  # src dst section key value
  .venv/bin/python -c "
import configparser, sys
c = configparser.ConfigParser(); c.optionxform = str; c.read(sys.argv[1])
c[sys.argv[3]][sys.argv[4]] = sys.argv[5]
c.write(open(sys.argv[2], 'w'))" "$@"
}

ours() {  # spacing
  local D=$OUT/ours_$1
  [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ] && return
  ini_with slurm/ensemble/gm12878_hic_arcs.ini "$OUT/ours_$1.ini" main target_bp_per_subanchor "$1"
  local t0; t0=$(date +%s.%N)
  .venv/bin/python -m gnome3d.cli --config "$OUT/ours_$1.ini" --data-dir "$DATA/$C" --region chr1 -n "$N" --out "$D" > "$D.log" 2>&1
  row ours "$1" "$D" "$t0"
}
cudammc() {  # density
  local D=$OUT/cudammc_$1 H=$OUT/cudammc_$1_hcm
  [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ] && return
  ini_with "$TOOLS/${C}_cudammc.ini" "$OUT/cudammc_$1.ini" main loop_density "$1"
  mkdir -p "$H" "$D"
  local t0; t0=$(date +%s.%N)
  "$CUDAMMC" -s "$OUT/cudammc_$1.ini" -c chr1 -o "$H/" -n cudammc -m "$N" > "$H.log" 2>&1
  .venv/bin/python playground/reference_arm.py "$DATA/$C/GM12878_anchors_3+_oriented.bed" "$H" "$D" chr1 >> "$H.log" 2>&1
  row cudammc "$1" "$D" "$t0"
}
multimm() {  # spacing of our run it matches
  local O=$OUT/ours_$1 MM=$OUT/multimm_$1
  [ "$(ls "${MM}_md"/*.cif 2>/dev/null | wc -l)" -ge "$N" ] && return
  local R; R=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'playground')
from pathlib import Path
from hole_readout import read_spans
st, en = read_spans(Path('$O'))
print(f'chr1:{st.min()}-{en.max()}')")
  local t0; t0=$(date +%s.%N)
  .venv/bin/python playground/multimm_arm.py "$DATA/$C/GM12878_clusters_3+.bedpe" "$R" "$O" "$MM" \
    --platform CUDA --multimm "$MULTIMM" --n "$N" > "$MM.log" 2>&1
  row multimm "$1" "${MM}_md" "$t0"
}

ours 2000
ours 4000
cudammc 10
cudammc 20
MATCH=$(.venv/bin/python - "$TOOLS/${C}_chr1_cudammc" "$OUT/cudammc_10" "$OUT/cudammc_20" "$TOOLS/${C}_chr1_ours" <<'PY'
import sys
from pathlib import Path
def beads(d):
    return sum(l.startswith("ATOM") for l in sorted(Path(d).glob("*.cif"))[0].read_text().splitlines())
b5, b10, b20, ours = (beads(d) for d in sys.argv[1:5])
slope = (b20 - b5) / 15.0
print(max(1, round(5 + (ours - b5) / slope)))
PY
)
echo "matched loop density $MATCH" >> "$LOG"
cudammc "$MATCH"
multimm 2000
multimm 4000
echo "scaling DONE $(date)" >> "$LOG"
echo EXIT >> "$LOG"
