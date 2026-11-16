#!/bin/bash
# The smooth and arcs stages on the CPU kernels against the JAX device by region size, GM12878
# chr1 from 10 Mb, one machine, one job at a time. design/tools-comparison.md.
#
#   DATA=~/gapfill-data OUT=out/crossover SPANS="5 10 20 40 chr1" EXECS="serial threaded batch" \
#     bash slurm/ensemble/executor_crossover.sh
#
# Each span runs N structures under production's executors, `batch`, and with the arcs, estimate
# and smooth stages on `serial` or `threaded`, or as `hybrid` with the arcs solve on the device and
# the smooth stage threaded. A span written chr1 is the whole chromosome. Wall
# clock, bead count and the stage lines of each log go to OUT/crossover.tsv and OUT/stages.txt.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/crossover}"
SPANS="${SPANS:-5 10 20 40}"
N="${N:-2}"
EXECS="${EXECS:-serial batch}"
mkdir -p "$OUT"
TSV="$OUT/crossover.tsv"
[ -f "$TSV" ] || printf "span_mb\texecutor\tstructures\tbeads\tseconds\n" > "$TSV"
git log --format="executor crossover on %h %s" -1 >> "$OUT/crossover.log"
.venv/bin/python -c "
import configparser, sys
c = configparser.ConfigParser(); c.optionxform = str; c.read('slurm/ensemble/gm12878_hic_arcs.ini')
c.write(open(sys.argv[1] + '/batch.ini', 'w'))
for e in ('serial', 'threaded'):
    for k in ('mc_executor_arcs', 'mc_executor_estimate_dist', 'mc_executor_smooth'):
        c['simulation_backend'][k] = e
    c.write(open(sys.argv[1] + '/' + e + '.ini', 'w'))
c['simulation_backend']['mc_executor_arcs'] = 'batch'
c.write(open(sys.argv[1] + '/hybrid.ini', 'w'))" "$OUT"
for S in $SPANS; do
  if [ "$S" = chr1 ]; then R=chr1; else R="chr1:10000000-$((10000000 + S * 1000000))"; fi
  for E in $EXECS; do
    D=$OUT/${S}mb_$E
    [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$N" ] && continue
    t0=$(date +%s.%N)
    .venv/bin/python -m gnome3d.cli --config "$OUT/$E.ini" --data-dir "$DATA/GM12878" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    printf "%s\t%s\t%s\t%s\t%.1f\n" "$S" "$E" "$N" "$(grep -c "^ATOM" "$(ls "$D"/*.cif | head -1)")" "$(echo "$(date +%s.%N) - $t0" | bc -l)" >> "$TSV"
    { echo "== $S Mb $E"; grep -E "coarse\[|smooth\[(serial|batch|threaded)\]|estimate" "$D.log"; } >> "$OUT/stages.txt"
  done
done
echo EXIT >> "$OUT/crossover.log"
