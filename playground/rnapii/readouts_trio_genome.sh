#!/bin/bash -l
# Structure readouts of the two whole genome trio arms against each person's own Hi-C, one
# array task per person and chromosome, CPU only.
#
#   sbatch -A <account> --array=0-197%24 playground/rnapii/readouts_trio_genome.sh
#   ROOT=~/trio_lr REPO=~/trio_lr/repo PY=~/3dgnome-ng/.venv/bin/python bash readouts_trio_genome.sh 7
#
# Under Slurm the task is the array index. On a workstation it is the first argument, ROOT
# holds out/ and data/, REPO the code and PY the interpreter. A task whose two outputs are there
# does nothing, so a list of tasks can be run again.
#
# Task k is person k mod 9 on chromosome k div 9 plus one, so the first nine tasks are chr1 of
# everyone and tasks 189 to 197 are chr22. The arms are out/lr_trio_ctcf and
# out/lr_trio_rnapol2, ten conformations a person and chromosome. Both write
# <OUT>/<SAMPLE>/<chr>, so each is linked under a name of its own,
# out/xc/trio_lr/<SAMPLE>_<ARM>_<chr>, C for CTCF loops and R for CTCF and RNAPII loops.
#
# Two readouts a task. ensemble_score.py sums the contacts of the ten conformations into one
# map and scores it against the person's map at 25 kb. pileup.py piles the map and both arms at
# the person's RNAPII loops at 10 kb and keeps the piles, so trio_genome_summary.py can add the
# chromosomes up loop by loop. Maps are read raw for every person so the nine are on one footing.
#
# The arms on the uniform chain are those of crosscell_arms.sh with CELLS, out/xt/<SAMPLE>_UC and
# out/xt/<SAMPLE>_UR. They read half a of the map of a person and are scored against half b:
#
#   C_ARM='out/xt/{S}_UC' R_ARM='out/xt/{S}_UR' MAP='data/_hic_half/b/{S}.mcool' TAG=trio_xt \
#     ROOT=~/trio_xt REPO=~/trio_xt/repo PY=~/3dgnome-ng/.venv/bin/python bash readouts_trio_genome.sh 7
#
# C_ARM, R_ARM and MAP are paths under ROOT with {S} for the person. TAG is the folder of the
# readouts under out/readouts and of the links under out/xc.

#SBATCH --job-name=trio_read
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --output=slurm/ensemble/logs/trio_read_%A_%a.out

set -euo pipefail
ROOT="${ROOT:-$SLURM_SUBMIT_DIR}"
REPO="${REPO:-$ROOT}"
cd "$ROOT"
if [ -z "${PY:-}" ]; then source .venv/bin/activate; PY=python; fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PEOPLE=(HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240)
K="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
S="${PEOPLE[$((K % 9))]}"
CHR="chr$((K / 9 + 1))"
TAG="${TAG:-trio_lr}"
DEFAULTS=('out/lr_trio_ctcf/{S}' 'out/lr_trio_rnapol2/{S}' 'data/_hic/{S}/{S}.mcool')
C_ARM="${C_ARM:-${DEFAULTS[0]}}"
R_ARM="${R_ARM:-${DEFAULTS[1]}}"
MAP="${MAP:-${DEFAULTS[2]}}"
OUT="$ROOT/out/readouts/$TAG"
LINKS="$ROOT/out/xc/$TAG"
mkdir -p "$OUT" "$LINKS"

if [ -s "$OUT/${S}_${CHR}_pooled.txt" ] && [ -s "$OUT/${S}_${CHR}_pileup.npz" ]; then echo "[$S $CHR] already read"; exit 0; fi

ARMS=()
for pair in "C:$C_ARM" "R:$R_ARM"; do
  A="${pair%%:*}"; D="${pair#*:}"; D="$ROOT/${D//\{S\}/$S}/$CHR"; DEST="$LINKS/${S}_${A}_${CHR}"
  n=$(ls "$D"/*.cif 2>/dev/null | wc -l)
  [ "$n" -eq 10 ] || { echo "[$S $CHR] arm $A has $n conformations, not 10"; exit 1; }
  [ -e "$DEST" ] || ln -s "$D" "$DEST"
  ARMS+=("$DEST")
done
echo "[$S $CHR] arms: ${ARMS[*]##*/}"
M="${MAP//\{S\}/$S}"
date +"%H:%M:%S pooled score"
"$PY" "$REPO/playground/ensemble_score.py" "$M" "$CHR" "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN" > "$OUT/${S}_${CHR}_pooled.txt.part"
mv "$OUT/${S}_${CHR}_pooled.txt.part" "$OUT/${S}_${CHR}_pooled.txt"
cat "$OUT/${S}_${CHR}_pooled.txt"
date +"%H:%M:%S pile-up"
"$PY" "$REPO/playground/rnapii/pileup.py" --json "$OUT/${S}_${CHR}_pileup.json" "$M" "$CHR" 10000 \
  "data/$S/${S}_rnapol2_clusters_3+.bedpe" "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN"
date +"%H:%M:%S done"
