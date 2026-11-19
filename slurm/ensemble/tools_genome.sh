#!/bin/bash
# One whole genome, chr1 to chr22, one structure, on each of 3dgnome, cudaMMC and MultiMM, timed by
# wall clock. design/tools-comparison.md.
#
#   DATA=~/gapfill-data OUT=out/toolgenome WAIT=out/toolscale2/scale.log bash slurm/ensemble/tools_genome.sh
#
# MultiMM runs at our total bead count and on failure at half and a quarter of it. A MultiMM run
# stops after MM_TIMEOUT seconds. OUT/timing.tsv has a row per tool with its setting, bead count and
# seconds, or its exit code for a failed run.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
TOOLS="${TOOLS:-out/tools}"
OUT="${OUT:-out/toolgenome}"
C="${CELL:-GM12878}"
CHROMS="${CHROMS:-chr1-chr22}"
CUDAMMC="${CUDAMMC:-$HOME/tools/cudaMMC/build/cudaMMC}"
MULTIMM="${MULTIMM:-$HOME/tools/mmenv/bin/MultiMM}"
WAIT="${WAIT:-}"
MM_TIMEOUT="${MM_TIMEOUT:-5400}"
mkdir -p "$OUT"
LOG="$OUT/genome.log"
TIMING="$OUT/timing.tsv"
[ -n "$WAIT" ] && while ! grep -q "^EXIT" "$WAIT" 2>/dev/null; do sleep 120; done
[ -f "$TIMING" ] || printf "tool\tsetting\tbeads\tseconds\n" > "$TIMING"
git log --format="tools genome on %h %s" -1 >> "$LOG"
c=$(echo "$C" | tr 'A-Z' 'a-z')
INI="slurm/ensemble/${c}_hic_arcs.ini"
key() { .venv/bin/python -c "import configparser,sys; c=configparser.ConfigParser(); c.read(sys.argv[1]); print(c['data'][sys.argv[2]])" "$1" "$2"; }
atoms() { cat "$1"/*.cif 2>/dev/null | grep -c "^ATOM"; }
since() { echo "$(date +%s.%N) - $1" | bc -l; }
row() {  # tool setting beads seconds
  printf "%s\t%s\t%s\t%.1f\n" "$1" "$2" "$3" "$4" >> "$TIMING"
  echo "[$1 $2] $(date +%H:%M) $3" >> "$LOG"
}

O=$OUT/${C}_ours
if [ "$(ls "$O"/*.cif 2>/dev/null | wc -l)" -lt 22 ]; then
  t0=$(date +%s.%N)
  .venv/bin/python -m gnome3d.cli --config "$INI" --data-dir "$DATA/$C" --region "$CHROMS" -n 1 --out "$O" > "$O.log" 2>&1
  rc=$?; el=$(since "$t0")
  if [ $rc = 0 ]; then row ours production "$(atoms "$O")" "$el"; else row ours production "exit $rc" "$el"; fi
fi
NB=$(atoms "$O")

K=$OUT/${C}_cudammc_hcm
if [ ! -d "$K" ]; then
  mkdir -p "$K"
  t0=$(date +%s.%N)
  "$CUDAMMC" -s "$TOOLS/${C}_cudammc.ini" -c "$CHROMS" -o "$K/" -n cudammc -m 1 > "$K.log" 2>&1
  rc=$?; el=$(since "$t0")
  nb=0
  for n in $(seq 1 22); do
    .venv/bin/python playground/reference_arm.py "$DATA/$C/$(key "$INI" anchors)" "$K" "$OUT/${C}_cudammc_chr$n" "chr$n" >> "$K.convert.log" 2>&1
    nb=$((nb + $(atoms "$OUT/${C}_cudammc_chr$n")))
  done
  if [ $rc = 0 ]; then row cudammc "loop_density 5" "$nb" "$el"; else row cudammc "loop_density 5" "exit $rc" "$el"; fi
fi

LOOPS=$(realpath "$DATA/$C/$(key "$INI" clusters)")
for DIV in 1 2 4; do
  N=$((NB / DIV))
  MM=$OUT/${C}_multimm_$N
  [ -d "$MM" ] && continue
  printf "[Main]\nPLATFORM = CUDA\nLOOPS_PATH = %s\nOUT_PATH = %s\nN_BEADS = %s\nSIM_RUN_MD = True\nSIM_N_STEPS = 10000\nSAVE_PLOTS = False\nCOB_USE_COMPARTMENT_BLOCKS = False\n" \
    "$LOOPS" "$(realpath -m "$MM")" "$N" > "$MM.ini"
  t0=$(date +%s.%N)
  timeout "$MM_TIMEOUT" "$MULTIMM" -c "$MM.ini" > "$MM.log" 2>&1
  rc=$?; el=$(since "$t0")
  if [ $rc = 0 ] && [ -n "$(find "$MM" -name "*.cif" 2>/dev/null | head -1)" ]; then row multimm "$N beads" "$N" "$el"; break; fi
  if [ $rc = 124 ]; then row multimm "$N beads" "stopped at ${MM_TIMEOUT} s" "$el"; else row multimm "$N beads" "exit $rc" "$el"; fi
done
echo "genome DONE $(date)" >> "$LOG"
echo EXIT >> "$LOG"
