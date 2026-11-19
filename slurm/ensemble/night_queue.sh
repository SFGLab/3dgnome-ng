#!/bin/bash
# The tool runs of the night of 2026-11-16, one after another. design/tools-comparison.md.
#
#   DATA=~/gapfill-data bash slurm/ensemble/night_queue.sh
#
#   out/toolpage1    GM12878 chr1:11.5-12.6 Mb, one structure per tool, the site's first run time chart
#   out/toolpagechr  GM12878 whole chr1, one structure per tool, its third
#   out/toolpage60   chr1:1-60 Mb, ten structures per tool, GM12878 first, its second chart and its Hi-C chart
#   out/toolsize     GM12878 chr1 from 10 Mb, one structure per tool at 1, 2, 5, 10, 20, 50 and 100 Mb
#   out/tools20      three 20 Mb regions dense in loops, five structures, three cells, with the bootstrap
#   out/toolpage60   the same for H1ESC and HFFC6
#   out/toolgenome   one whole genome per tool
# Every step skips what is already there. Each writes EXIT to OUT/night.done.
set -u
cd "$(dirname "$0")/../.."
export DATA="${DATA:-data}" HIC="${HIC:-${DATA:-data}/_hic}"
R20="chr12:40000000-60000000 chr11:60000000-80000000 chr17:35000000-55000000"
trial() { OUT=$1 bash slurm/ensemble/tools_trial.sh; }
boot() {  # out cells regions
  [ -f "$1/bootstrap.txt" ] && return
  for C in $2; do
    M=$(ls -S "$HIC/$C"/*.mcool | head -1)
    for R in $3; do
      T=$(echo "$R" | tr ':-' '__')
      .venv/bin/python playground/ensemble_bootstrap.py "$M" "$R" "$1/${C}_${T}_ours" "$1/${C}_${T}_multimm_md" "$1/${C}_${T}_cudammc" 2>&1 | grep -vE "INFO|WARN" >> "$1/bootstrap.part"
    done
  done
  mv -n "$1/bootstrap.part" "$1/bootstrap.txt"
}
CELLS=GM12878 N=1 REGIONS="chr1:11500000-12600000" trial out/toolpage1; echo EXIT >> out/toolpage1/night.done
CELLS=GM12878 N_CHR=1 REGIONS=chr1 trial out/toolpagechr; echo EXIT >> out/toolpagechr/night.done
CELLS=GM12878 N=10 REGIONS="chr1:1-60000000" trial out/toolpage60; echo EXIT >> out/toolpage60/night.done
RSIZE="chr1:10000000-11000000 chr1:10000000-12000000 chr1:10000000-15000000 chr1:10000000-20000000 chr1:10000000-30000000 chr1:10000000-60000000 chr1:10000000-110000000"
CELLS=GM12878 N=1 REGIONS="$RSIZE" trial out/toolsize; echo EXIT >> out/toolsize/night.done
N=5 REGIONS="$R20" trial out/tools20
boot out/tools20 "GM12878 H1ESC HFFC6" "$R20"; echo EXIT >> out/tools20/night.done
CELLS="H1ESC HFFC6" N=10 REGIONS="chr1:1-60000000" trial out/toolpage60
boot out/toolpage60 "GM12878 H1ESC HFFC6" "chr1:1-60000000"; echo EXIT >> out/toolpage60/night.done
TOOLS=out/tools OUT=out/toolgenome bash slurm/ensemble/tools_genome.sh
