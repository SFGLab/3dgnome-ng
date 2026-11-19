#!/bin/bash -l

# Cross line RNAPII comparison, design/2026-11-11-rnapii-release.md. Nine array tasks, one per line
# and arm, ten conformations each. Arm C is CTCF only, R is CTCF and RNAPII and M is R with RNAPII
# ends near a CTCF anchor moved onto it.
#
#   sbatch -A <account> --array=0-8 slurm/ensemble/crosscell_arms.sh
#   CHROMS="chr2 chr3" sbatch -A <account> --array=0-8 slurm/ensemble/crosscell_arms.sh
#   DRY=1 SLURM_ARRAY_TASK_ID=4 bash slurm/ensemble/crosscell_arms.sh
#   SPLIT=1 CHROMS="chr2 ... chr22" sbatch -A <account> --array=0-188%24 slurm/ensemble/crosscell_arms.sh
#
# Task i runs line i / 3 of GM12878 H1ESC HFFC6 and arm i % 3 of C R M. With SPLIT=1 it runs line
# and arm i % 9 on chromosome i / 9 of CHROMS. A member whose .cif exists is skipped. After chr1 the
# arm is sliced to chr1:1-60000000. The inputs come from playground/rnapii/crosscell_prep.py and the
# configs from slurm/ensemble/crosscell_configs.py.

#SBATCH --job-name=xc_rnapii
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=24:00:00
# The account is given with sbatch -A. --output and --error are relative to the checkout, the
# directory of the submission. Create slurm/ensemble/logs before the submission.
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/xc_%A_%a.out
#SBATCH --error=slurm/ensemble/logs/xc_%A_%a.out

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${OUT:-$ROOT/out/xc}"
CHROMS="${CHROMS:-chr1}"
N="${N:-10}"
DRY="${DRY:-0}"
SPLIT="${SPLIT:-0}"

LINES=(GM12878 H1ESC HFFC6)
ARMS=(C R M)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
COMBO="$TASK"
if [ "$SPLIT" = "1" ]; then
  read -r -a CHROM_LIST <<< "$CHROMS"
  COMBO=$((TASK % 9))
  [ $((TASK / 9)) -lt ${#CHROM_LIST[@]} ] || { echo "task $TASK beyond ${#CHROM_LIST[@]} chromosomes"; exit 1; }
  CHROMS="${CHROM_LIST[$((TASK / 9))]}"
fi
LINE="${LINES[$((COMBO / 3))]}"
ARM="${ARMS[$((COMBO % 3))]}"
LOWER="$(echo "$LINE" | tr '[:upper:]' '[:lower:]')"
CONFIG="$ROOT/slurm/ensemble/${LOWER}_xc_${ARM}.ini"
DEST="$OUT/${LINE}_${ARM}"

echo "[map] task=$TASK line=$LINE arm=$ARM config=$CONFIG out=$DEST chroms=$CHROMS"
if [ "$DRY" = "1" ]; then
  [ -f "$CONFIG" ] && echo "[dry] config present" || echo "[dry] config MISSING"
  for C in $CHROMS; do
    echo "[dry] gnome3d-ng --config $CONFIG --region $C --members 0-$((N - 1)) --out $DEST/$C"
  done
  exit 0
fi

cd "$ROOT"
mkdir -p slurm/ensemble/logs "$DEST"
source .venv/bin/activate
# The venv may hold an editable install of another checkout. This checkout goes first.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Every file of the config must exist. Without the second cluster file an RNAPII arm would run as a
# CTCF arm.
python - "$CONFIG" "$ROOT/data/$LINE" <<'PYCHECK'
import sys
from pathlib import Path

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
d = Path(sys.argv[2])
names = [s.data_anchors, s.data_singletons, *s.data_pet_clusters.split(",")]
missing = [n for n in names if not (d / n).is_file()]
assert not missing, f"missing under {d}: {missing}"
print(f"[guard] config ok: clusters={s.data_pet_clusters} factors={s.data_factors or 'CTCF'}")
PYCHECK

for C in $CHROMS; do
  echo "[launch] node=$(hostname) $LINE $ARM $C members=0-$((N - 1))"
  srun python -m gnome3d.cli --config "$CONFIG" --data-dir "data/$LINE" --region "$C" \
    --members "0-$((N - 1))" --out "$DEST/$C" \
    --log-file "$ROOT/slurm/ensemble/logs/xc_${SLURM_ARRAY_JOB_ID:-0}_${TASK}_${C}.detail.log"
  if [ "$C" = "chr1" ]; then
    python playground/slice_arm.py "$DEST/chr1" "$OUT/${LINE}_${ARM}_60mb" chr1:1-60000000 > /dev/null
  fi
done
echo "[done] $LINE $ARM $(date)"
