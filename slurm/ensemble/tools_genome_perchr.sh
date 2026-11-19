#!/bin/bash
# cudaMMC on the genome one chromosome at a time, chr1 to chr22, one structure each, timed by wall
# clock. design/tools-comparison.md.
#
#   DATA=~/gapfill-data OUT=out/toolgenome bash slurm/ensemble/tools_genome_perchr.sh
#
# The genome mode of cudaMMC needs contacts between chromosomes, which the Hi-C singletons here do
# not hold. Each chromosome is its own run. OUT/timing_perchr.tsv has a row per chromosome and a
# last row with the sums.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
TOOLS="${TOOLS:-out/tools}"
OUT="${OUT:-out/toolgenome}"
C="${CELL:-GM12878}"
CUDAMMC="${CUDAMMC:-$HOME/tools/cudaMMC/build/cudaMMC}"
mkdir -p "$OUT"
LOG="$OUT/perchr.log"
TIMING="$OUT/timing_perchr.tsv"
[ -f "$TIMING" ] || printf "tool\tsetting\tbeads\tseconds\n" > "$TIMING"
git log --format="tools genome per chromosome on %h %s" -1 >> "$LOG"
c=$(echo "$C" | tr 'A-Z' 'a-z')
INI="slurm/ensemble/${c}_hic_arcs.ini"
key() { .venv/bin/python -c "import configparser,sys; c=configparser.ConfigParser(); c.read(sys.argv[1]); print(c['data'][sys.argv[2]])" "$1" "$2"; }
atoms() { cat "$1"/*.cif 2>/dev/null | grep -c "^ATOM"; }
since() { echo "$(date +%s.%N) - $1" | bc -l; }

total=0
beads=0
failed=0
for n in $(seq 1 22); do
  K=$OUT/${C}_cudammc_perchr_chr${n}_hcm
  [ -d "$K" ] && continue
  mkdir -p "$K"
  t0=$(date +%s.%N)
  "$CUDAMMC" -s "$TOOLS/${C}_cudammc.ini" -c "chr$n" -o "$K/" -n cudammc -m 1 > "$K.log" 2>&1
  rc=$?; el=$(since "$t0")
  .venv/bin/python playground/reference_arm.py "$DATA/$C/$(key "$INI" anchors)" "$K" "$OUT/${C}_cudammc_perchr_chr$n" "chr$n" >> "$OUT/${C}_cudammc_perchr.convert.log" 2>&1
  nb=$(atoms "$OUT/${C}_cudammc_perchr_chr$n")
  if [ $rc != 0 ] || [ "$nb" = 0 ]; then
    failed=$((failed + 1))
    printf "cudammc\tchr%s\texit %s\t%.1f\n" "$n" "$rc" "$el" >> "$TIMING"
  else
    printf "cudammc\tchr%s\t%s\t%.1f\n" "$n" "$nb" "$el" >> "$TIMING"
  fi
  echo "[cudammc chr$n] $(date +%H:%M) $nb beads exit $rc" >> "$LOG"
  total=$(echo "$total + $el" | bc -l)
  beads=$((beads + nb))
done
printf "cudammc\tchr1 to chr22, %s failed\t%s\t%.1f\n" "$failed" "$beads" "$total" >> "$TIMING"
echo "perchr DONE $(date)" >> "$LOG"
echo EXIT >> "$LOG"
