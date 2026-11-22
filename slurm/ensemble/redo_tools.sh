#!/bin/bash
# Our rows of the tools comparison on the uniform chain, one run after another on the machine of
# the first measurement. design/tools-comparison.md.
#
#   DATA=~/gapfill-data HALVES=~/gapfill-data/_hic_halves OLD=~/3dgnome-gapfill/out \
#     bash slurm/ensemble/redo_tools.sh
#
# Each step is tools_trial.sh with our tool alone, on slurm/ensemble/<cell>_hic_uniform.ini, into
# out/utools/<step>. The config reads half a of the deep map, data/_hic_half/a/<cell>.mcool, and
# every tool is scored against half b, HALVES/b/<cell>.mcool. The structures of cudaMMC and MultiMM
# are those of the first measurement, linked from OLD/<step>. Their run times are not measured
# again. A step skips what exists and ends with EXIT in its redo.done.
#
#   toolpage1    GM12878 chr1:11.5-12.6 Mb, one structure
#   toolpagechr  GM12878 whole chr1, one structure
#   toolpage60   chr1:1-60 Mb, ten structures, three cells
#   toolsize     GM12878 chr1 from 10 Mb, one structure at 1 to 100 Mb
#   tools        three regions of 5 Mb at five structures and whole chr1 at three, three cells
#   tools20      three regions of 20 Mb, five structures, three cells
#   toolgenome   GM12878 chr1 to chr22, one structure
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-data}" HIC="${HIC:-${DATA:-data}/_hic}"
HALVES="${HALVES:-$DATA/_hic_halves}"
OLD="${OLD:-out}"
NEW="${NEW:-out/utools}"
export INI_TAG=hic_uniform SCORE_MAPS="$HALVES/b" TOOLS_RUN=ours
for C in GM12878 H1ESC HFFC6; do
  [ -f "data/_hic_half/a/$C.mcool" ] || { echo "data/_hic_half/a/$C.mcool is missing"; exit 1; }
  [ -f "$SCORE_MAPS/$C.mcool" ] || { echo "$SCORE_MAPS/$C.mcool is missing"; exit 1; }
done
R20="chr12:40000000-60000000 chr11:60000000-80000000 chr17:35000000-55000000"
RSIZE="chr1:10000000-11000000 chr1:10000000-12000000 chr1:10000000-15000000 chr1:10000000-20000000 chr1:10000000-30000000 chr1:10000000-60000000 chr1:10000000-110000000"

trial() {  # step of the first measurement
  local step=$1 out="$NEW/$1"
  mkdir -p "$out"
  # The structures of the other tools of the first measurement, to be scored on the same map.
  for d in "$OLD/$step"/*_cudammc "$OLD/$step"/*_multimm_min "$OLD/$step"/*_multimm_md; do
    [ -d "$d" ] && [ ! -e "$out/$(basename "$d")" ] && ln -s "$(cd "$d" && pwd)" "$out/$(basename "$d")"
  done
  OUT="$out" bash slurm/ensemble/tools_trial.sh
}

CELLS=GM12878 N=1 REGIONS="chr1:11500000-12600000" trial toolpage1; echo EXIT >> "$NEW/toolpage1/redo.done"
CELLS=GM12878 N_CHR=1 REGIONS=chr1 trial toolpagechr; echo EXIT >> "$NEW/toolpagechr/redo.done"
N=10 REGIONS="chr1:1-60000000" trial toolpage60; echo EXIT >> "$NEW/toolpage60/redo.done"
CELLS=GM12878 N=1 REGIONS="$RSIZE" trial toolsize; echo EXIT >> "$NEW/toolsize/redo.done"
trial tools; echo EXIT >> "$NEW/tools/redo.done"
N=5 REGIONS="$R20" trial tools20; echo EXIT >> "$NEW/tools20/redo.done"

G="$NEW/toolgenome"
mkdir -p "$G"
if [ "$(ls "$G/GM12878_ours"/*.cif 2>/dev/null | wc -l)" -lt 22 ]; then
  t0=$(date +%s.%N)
  .venv/bin/python -m gnome3d.cli --config slurm/ensemble/gm12878_hic_uniform.ini --data-dir "$DATA/GM12878" \
    --region chr1-chr22 -n 1 --out "$G/GM12878_ours" > "$G/GM12878_ours.log" 2>&1
  rc=$?
  printf "ours\tuniform\t%s\t%.1f\texit %s\n" "$(cat "$G/GM12878_ours"/*.cif 2>/dev/null | grep -c "^ATOM")" "$(echo "$(date +%s.%N) - $t0" | bc -l)" "$rc" >> "$G/timing.tsv"
fi
echo EXIT >> "$G/redo.done"
echo "redo tools DONE $(date)"
