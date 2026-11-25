#!/bin/bash
# pyLoopSage and FLAMINGO on the runs of the tools comparison, one machine, one job at a time.
# design/tools-comparison.md.
#
#   DATA=~/gapfill-data HALVES=~/gapfill-data/_hic_halves OUT=out/utools/toolpage60 N=10 \
#   CELLS="GM12878 H1ESC HFFC6" REGIONS="chr1:1-60000000" GENOME=~/data/ref/hg38.fa \
#   LOOPSAGE=~/tools/lsenv/bin/loopsage R_LIBS_USER=~/R/library bash slurm/ensemble/tools_more.sh
#
# OUT is a folder of tools_trial.sh with the structures of the three other tools. pyLoopSage reads
# the clusters of slurm/ensemble/<cell>_hic_uniform.ini, as the other tools do, through
# playground/loopsage_arm.py. FLAMINGO reads half a of the Hi-C map, HALVES/a/<cell>.mcool, through
# playground/flamingo_arm.py. Every run is timed into OUT/timing_more.tsv. Every tool with
# structures in OUT is scored against half b: the pooled scores into <cell>_<region>_pooled_more.txt,
# the bands on the bins of every tool, by the contacts and by the distance, into
# <cell>_<region>_bands_more.{json,txt} and the bootstrap of our tool against each other tool into
# bootstrap_more.txt. TOOLS_RUN names the tools to run. SCORE=no runs them without the scores.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
HALVES="${HALVES:-$DATA/_hic_halves}"
OUT="${OUT:?the folder of a step of the tools comparison}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
REGIONS="${REGIONS:-chr1:1-60000000}"
N="${N:-10}"
GENOME="${GENOME:?the FASTA of the genome, for the CTCF motifs of pyLoopSage}"
LOOPSAGE="${LOOPSAGE:-loopsage}"
RSCRIPT="${RSCRIPT:-Rscript}"
THREADS="${THREADS:-20}"
TOOLS_RUN="${TOOLS_RUN:-loopsage flamingo}"
SCORE="${SCORE:-yes}"
PY=.venv/bin/python
want() { case " $TOOLS_RUN " in *" $1 "*) return 0 ;; esac; return 1; }
LOG="$OUT/tools_more.log"
TIMING="$OUT/timing_more.tsv"
[ -f "$TIMING" ] || printf "cell\tregion\ttool\tstructures\tbeads\tseconds\n" > "$TIMING"
key() { $PY -c "import configparser,sys; c=configparser.ConfigParser(); c.read(sys.argv[1]); print(c['data'][sys.argv[2]])" "$1" "$2"; }

timed() {  # cell region tool dir command...
  C=$1; R=$2; TOOL=$3; D=$4; shift 4
  local t0 t1
  t0=$(date +%s.%N)
  "$@"
  local rc=$?
  t1=$(date +%s.%N)
  local nb=0
  ls "$D"/*.cif > /dev/null 2>&1 && nb=$(grep -c "^ATOM" "$(ls "$D"/*.cif | head -1)")
  printf "%s\t%s\t%s\t%s\t%s\t%.1f\n" "$C" "$R" "$TOOL" "$N" "$nb" "$(echo "$t1 - $t0" | bc -l)" >> "$TIMING"
  echo "[$C $R $TOOL] exit $rc $(date +%H:%M) beads $nb" >> "$LOG"
}

for C in $CELLS; do
  c=$(echo "$C" | tr 'A-Z' 'a-z')
  LOOPS="$DATA/$C/$(key "slurm/ensemble/${c}_hic_uniform.ini" clusters)"
  M="$HALVES/b/$C.mcool"
  for R in $REGIONS; do
    T=$(echo "$R" | tr ':-' '__')
    L=$OUT/${C}_${T}_loopsage
    if want loopsage && [ "$(ls "$L"/*.cif 2>/dev/null | wc -l)" -lt "$N" ]; then
      timed "$C" "$R" loopsage "$L" $PY playground/loopsage_arm.py "$LOOPS" "$R" "$L" --genome "$GENOME" \
        --loopsage "$LOOPSAGE" --n "$N" > "$L.log" 2>&1
    fi
    F=$OUT/${C}_${T}_flamingo
    if want flamingo && [ "$(ls "$F"/*.cif 2>/dev/null | wc -l)" -lt "$N" ]; then
      timed "$C" "$R" flamingo "$F" $PY playground/flamingo_arm.py "$HALVES/a/$C.mcool" "$R" "$F" \
        --rscript "$RSCRIPT" --threads "$THREADS" --n "$N" > "$F.log" 2>&1
    fi
    [ "$SCORE" = yes ] || continue
    dirs=""; named=""; others=""
    for A in ours cudammc multimm_md loopsage flamingo; do
      D=$OUT/${C}_${T}_$A
      ls "$D"/*.cif > /dev/null 2>&1 || continue
      dirs="$dirs $D"; named="$named $A=$D"
      [ "$A" = ours ] || others="$others $D"
    done
    # Every frame of pyLoopSage after its burn in, beside its N frames, in the bands alone.
    ls "$OUT/${C}_${T}_loopsage_all"/*.cif > /dev/null 2>&1 && named="$named loopsage_all=$OUT/${C}_${T}_loopsage_all"
    $PY playground/ensemble_score.py "$M" "$R" $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${T}_pooled_more.txt"
    $PY playground/rnapii/band_trace.py "$M" "${R%%:*}" "$DATA/$C/${C}_xc_ctcf_clusters_3+.bedpe" "$DATA/$C/${C}_xc_rnapii_clusters_3+.bedpe" \
      "$OUT/${C}_${T}_bands_more.json" $named > "$OUT/${C}_${T}_bands_more.txt" 2>&1
    $PY playground/ensemble_bootstrap.py "$M" "$R" "$OUT/${C}_${T}_ours" $others 2>&1 | grep -vE "INFO|WARN" >> "$OUT/bootstrap_more.txt"
    echo "[$C $R] scored $(date +%H:%M)" >> "$LOG"
  done
done
echo "tools more DONE $(date)" >> "$LOG"
