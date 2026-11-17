#!/bin/bash
# 3dgnome against cudaMMC and MultiMM on the same inputs, one machine, one job at a time.
# design/plans/2026-11-16-genomes-and-comparison.md.
#
#   DATA=~/gapfill-data HIC=~/gapfill-data/_hic OUT=out/tools N=5 \
#   CELLS="GM12878 H1ESC HFFC6" REGIONS="chr1:10000000-15000000 chr1" \
#   CUDAMMC=~/tools/cudaMMC/build/cudaMMC CUDAMMC_TEMPLATE=~/tools/cudaMMC/config.ini \
#   MULTIMM=~/tools/mmenv/bin/MultiMM bash slurm/ensemble/tools_trial.sh
#
# Inputs are the production config's, `slurm/ensemble/<cell>_hic_arcs.ini`: the CTCF anchors and
# clusters and the Hi-C singletons at 25 kb. cudaMMC reads them through its own template config
# with only the data section changed and builds its ensemble with `-m`; its beads come from its
# loop density, its own default. MultiMM reads the clusters as loops at our arm's bead count over
# our arm's span, through `playground/multimm_arm.py`, and both its minimised and its dynamics
# ensembles are kept. Every run is timed by wall clock into OUT/timing.tsv with its bead count. A
# region written chr1 runs the whole chromosome and is scored on chr1:1-60000000. Each region is
# scored pooled, `playground/ensemble_score.py`, per structure by the battery on raw and balanced
# counts, and by the contact spread.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
HIC="${HIC:-$DATA/_hic}"
OUT="${OUT:-out/tools}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
REGIONS="${REGIONS:-chr1:10000000-15000000 chr1:20000000-25000000 chr1:35000000-40000000 chr1}"
N="${N:-5}"
N_CHR="${N_CHR:-3}"
CUDAMMC="${CUDAMMC:-$HOME/tools/cudaMMC/build/cudaMMC}"
CUDAMMC_TEMPLATE="${CUDAMMC_TEMPLATE:-$HOME/tools/cudaMMC/config.ini}"
MULTIMM="${MULTIMM:-$HOME/tools/mmenv/bin/MultiMM}"
PLATFORM="${PLATFORM:-CUDA}"
# The tools to run. A tool left out is scored where its output exists.
TOOLS_RUN="${TOOLS_RUN:-ours cudammc multimm}"
want() { case " $TOOLS_RUN " in *" $1 "*) return 0 ;; esac; return 1; }
mkdir -p "$OUT"
LOG="$OUT/tools.log"
TIMING="$OUT/timing.tsv"
[ -f "$TIMING" ] || printf "cell\tregion\ttool\tstructures\tbeads\tseconds\n" > "$TIMING"
git log --format="tools trial on %h %s" -1 >> "$LOG"

mcool() { ls -S "$HIC/$1"/*.mcool | head -1; }
key() { .venv/bin/python -c "import configparser,sys; c=configparser.ConfigParser(); c.read(sys.argv[1]); print(c['data'][sys.argv[2]])" "$1" "$2"; }
beads() { grep -c "^ATOM" "$(ls "$1"/*.cif | head -1)"; }

timed() {  # cell region tool n dir command...
  C=$1; R=$2; TOOL=$3; NN=$4; D=$5; shift 5
  local t0 t1
  t0=$(date +%s.%N)
  "$@"
  local rc=$?
  t1=$(date +%s.%N)
  local nb=0
  ls "$D"/*.cif > /dev/null 2>&1 && nb=$(beads "$D")
  printf "%s\t%s\t%s\t%s\t%s\t%.1f\n" "$C" "$R" "$TOOL" "$NN" "$nb" "$(echo "$t1 - $t0" | bc -l)" >> "$TIMING"
  echo "[$C $R $TOOL] exit $rc $(date +%H:%M) beads $nb" >> "$LOG"
}

cudammc_config() {  # cell ini out
  .venv/bin/python - "$CUDAMMC_TEMPLATE" "$2" "$3" "$DATA/$1" <<'PY'
import configparser, sys
template, ini, out, data = sys.argv[1:5]
ours = configparser.ConfigParser()
ours.read(ini)
c = configparser.ConfigParser()
c.optionxform = str
c.read(template)
d = ours["data"]
c["data"]["data_dir"] = data.rstrip("/") + "/"
c["data"]["anchors"] = d["anchors"]
c["data"]["clusters"] = d["clusters"]
c["data"]["factors"] = "CTCF"
c["data"]["singletons"] = d["singletons"]
c["data"]["split_singleton_files_by_chr"] = "no"
# Read only for more than one chromosome; the cis file stands in for it.
c["data"]["singletons_inter"] = d["singletons"]
c["data"]["segment_split"] = f"{data}/{d['segment_split']}"
c["data"]["centromeres"] = f"{data}/{d['centromeres']}"
with open(out, "w") as fh:
    c.write(fh)
PY
}

run_cudammc() {  # cell region n hcmdir cif dir
  C=$1; R=$2; NN=$3; H=$4; D=$5
  # cudaMMC parses a region as chr:start-end, whatever its README says.
  local cfg="$OUT/${C}_cudammc.ini" spec="$R"
  [ -f "$cfg" ] || cudammc_config "$C" "slurm/ensemble/$(echo "$C" | tr 'A-Z' 'a-z')_hic_arcs.ini" "$cfg"
  mkdir -p "$H" "$D"
  "$CUDAMMC" -s "$cfg" -c "$spec" -o "$H/" -n cudammc -m "$NN" > "$H.log" 2>&1 || return $?
  .venv/bin/python playground/reference_arm.py "$DATA/$C/$(key "$cfg" anchors)" "$H" "$D" "${R%%:*}" >> "$H.log" 2>&1
}

for C in $CELLS; do
  c=$(echo "$C" | tr 'A-Z' 'a-z')
  INI="slurm/ensemble/${c}_hic_arcs.ini"
  M=$(mcool "$C")
  LOOPS="$DATA/$C/$(key "$INI" clusters)"
  for R in $REGIONS; do
    T=$(echo "$R" | tr ':-' '__')
    NN=$N; SR=$R; ST=$T
    [ "$R" = chr1 ] && NN=$N_CHR && SR=chr1:1-60000000 && ST=chr1_60mb
    O=$OUT/${C}_${T}_ours
    if want ours && [ "$(ls "$O"/*.cif 2>/dev/null | wc -l)" -lt "$NN" ]; then
      echo "[$C $R ours] start $(date +%H:%M)" >> "$LOG"
      timed "$C" "$R" ours "$NN" "$O" .venv/bin/python -m gnome3d.cli --config "$INI" --data-dir "$DATA/$C" \
        --region "$R" -n "$NN" --out "$O" > "$O.log" 2>&1
    fi
    K=$OUT/${C}_${T}_cudammc
    if want cudammc && [ "$(ls "$K"/*.cif 2>/dev/null | wc -l)" -lt "$NN" ]; then
      echo "[$C $R cudammc] start $(date +%H:%M)" >> "$LOG"
      timed "$C" "$R" cudammc "$NN" "$K" run_cudammc "$C" "$R" "$NN" "${K}_hcm" "$K"
    fi
    MR=$R
    if [ "$R" = chr1 ]; then
      MR=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'playground')
from pathlib import Path
from law_spring_readout import read_models
from hole_readout import read_spans
st, en = read_spans(Path('$O'))
print(f'chr1:{st.min()}-{en.max()}')")
    fi
    MM=$OUT/${C}_${T}_multimm
    if want multimm && [ "$(ls "${MM}_md"/*.cif 2>/dev/null | wc -l)" -lt "$NN" ]; then
      echo "[$C $R multimm] start $(date +%H:%M)" >> "$LOG"
      timed "$C" "$R" multimm "$NN" "${MM}_md" .venv/bin/python playground/multimm_arm.py "$LOOPS" "$MR" "$O" "$MM" \
        --platform "$PLATFORM" --multimm "$MULTIMM" --n "$NN" > "$MM.log" 2>&1
    fi
    dirs=""
    for A in ours cudammc multimm_min multimm_md; do
      D=$OUT/${C}_${T}_$A
      if [ "$R" = chr1 ]; then
        S=$OUT/${C}_${ST}_$A
        [ -d "$S" ] || .venv/bin/python playground/slice_arm.py "$D" "$S" "$SR" > /dev/null 2>&1
        [ -f "$D.log" ] && [ ! -f "$S.log" ] && cp "$D.log" "$S.log"
        D=$S
      fi
      ls "$D"/*.cif > /dev/null 2>&1 && dirs="$dirs $D"
    done
    .venv/bin/python playground/ensemble_score.py "$M" "$SR" $dirs 2>&1 | grep -vE "INFO|WARN" > "$OUT/${C}_${ST}_pooled.txt"
    for B in no yes; do
      .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 \
        --singletons "$DATA/$C/${C}_hic_25kb_singletons.bedpe" "$M" "$SR" 25000 $dirs 2>&1 \
        | grep -vE "INFO|WARN" > "$OUT/${C}_${ST}_battery_balance_$B.txt"
    done
    for d in $dirs; do
      .venv/bin/python playground/contact_spread.py "$OUT/${C}_${ST}_ours" "$d" 2>&1 | grep -vE "INFO|WARN"
    done > "$OUT/${C}_${ST}_spread.txt"
    echo "[$C $R] scored $(date +%H:%M)" >> "$LOG"
  done
done
echo "tools DONE $(date)" >> "$LOG"
