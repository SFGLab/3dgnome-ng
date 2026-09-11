#!/bin/bash
# The three cell validation gate on the production settings, with MultiMM run on the same inputs.
#
#   bash slurm/ensemble/validation_gate.sh                 # all three cells
#   CELLS="GM12878" bash slurm/ensemble/validation_gate.sh # one cell
#
# Per cell, in order: our pipeline on the production ini regenerated from CANONICAL, five
# structures of chr1:1-60 Mb on the GPU; MultiMM at our bead count, loops only and then loops
# with the phased compartment bed, five members each; the battery over every arm with the
# exponent yardstick fitted on the cell's own singletons; and the compartment saddle.
#
# Runs on the workstation from a project checkout with the project venv and a separate MultiMM
# venv at $MMVENV (python3 -m venv ~/mmvenv && ~/mmvenv/bin/pip install MultiMM). Sequential,
# one GPU job at a time. A cell whose arm directory already holds its cifs is not rerun, so the
# script can be restarted to fill in what is missing.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT="${OUT:-out/validation}"
R="${REGION:-chr1:1-60000000}"
N="${N:-5}"
MMVENV="${MMVENV:-$HOME/mmvenv}"
HIC_ROOT="${HIC_ROOT:-/mnt/storagelinux/_hic}"
CELLS="${CELLS:-GM12878 H1ESC HFFC6}"
mkdir -p "$OUT"
git log --format="gate on %h %s" -1

for C in $CELLS; do
  L=$(echo "$C" | tr '[:upper:]' '[:lower:]')
  INI=slurm/ensemble/${L}_hic_arcs.ini
  MCOOL=$(ls "$HIC_ROOT/$C"/*.mcool | head -1)
  SING=data/$C/${C}_hic_25kb_singletons.bedpe
  LOOPS=data/$C/${C}_clusters_3+.bedpe
  TRACK=data/$C/${C}_compartments.bedGraph
  D=$OUT/${C}_prod
  mkdir -p "$D"
  if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -lt "$N" ]; then
    echo "[$C ours] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$INI" --data-dir "data/$C" --region "$R" -n "$N" --out "$D" > "$D.log" 2>&1
    echo "[$C ours] done $(date +%H:%M) exit $? structures $(ls "$D"/*.cif 2>/dev/null | wc -l)"
  fi
  for ARM in loops comps; do
    if [ "$(ls "$OUT/${C}_mm_${ARM}_md"/*.cif 2>/dev/null | wc -l)" -ge "$N" ]; then continue; fi
    EXTRA=""
    [ "$ARM" = comps ] && EXTRA="--compartments $TRACK"
    echo "[$C multimm $ARM] start $(date +%H:%M)"
    # shellcheck disable=SC2086
    .venv/bin/python playground/multimm_arm.py "$LOOPS" "$R" "$D" "$OUT/${C}_mm_${ARM}" $EXTRA \
      --platform "${MM_PLATFORM:-OpenCL}" --multimm "$MMVENV/bin/MultiMM" --n "$N" 2>&1 | tail -3
    echo "[$C multimm $ARM] done $(date +%H:%M)"
  done
  ARMS="$D $OUT/${C}_mm_loops_min $OUT/${C}_mm_loops_md $OUT/${C}_mm_comps_min $OUT/${C}_mm_comps_md"
  # shellcheck disable=SC2086
  .venv/bin/python playground/validation_battery.py --balance no --singletons "$SING" "$MCOOL" "$R" 25000 $ARMS 2>&1 \
    | grep -vE "INFO|WARN" | tee "$OUT/${C}_battery.txt"
  # shellcheck disable=SC2086
  .venv/bin/python playground/saddle_offline.py "$MCOOL" "$R" 100000 "$TRACK" $ARMS 2>&1 \
    | grep -vE "INFO|WARN" | tee "$OUT/${C}_saddle.txt"
done
echo "VALIDATION GATE DONE $(date)"
