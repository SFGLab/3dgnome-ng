#!/bin/bash -l

# The cross line RNAPII comparison, design/2026-11-11-rnapii-release.md. Nine array tasks, one
# per line and arm, ten conformations each, arms C (CTCF only), R (CTCF and RNAPII) and M (R
# with RNAPII ends near a CTCF anchor moved onto it).
#
#   sbatch --array=0-8 slurm/ensemble/crosscell_arms.sh
#   CHROMS="chr2 chr3" sbatch --array=0-8 slurm/ensemble/crosscell_arms.sh
#   DRY=1 SLURM_ARRAY_TASK_ID=4 bash slurm/ensemble/crosscell_arms.sh   # print, run nothing
#
# The autosomes, one task per line, arm and chromosome, 9 times the chromosome count:
#
#   SPLIT=1 CHROMS="chr2 chr3 ... chr22" sbatch --array=0-188%24 slurm/ensemble/crosscell_arms.sh
#
# Task i runs line i / 3 of GM12878 H1ESC HFFC6 and arm i % 3 of C R M. With SPLIT=1 it runs
# line and arm i % 9 on chromosome i / 9 of CHROMS. A member whose .cif
# exists is skipped by gnome3d-ng, so resubmitting the array fills the gaps. After chr1 the arm
# is sliced to chr1:1-60000000, the battery's region.
#
# The inputs under data/<line>/ come from playground/rnapii/crosscell_prep.py and the configs
# from slurm/ensemble/crosscell_configs.py. data/ is gitignored, so they are copied to the
# cluster checkout by hand before the first submission.

#SBATCH --job-name=xc_rnapii
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --account=sfglab
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/xc_%A_%a.out
#SBATCH --error=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/xc_%A_%a.out

set -euo pipefail

ROOT="${ROOT:-/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng}"
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
# The venv may hold an editable install of another checkout; this checkout's engine goes first.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Every file the config names must exist; a missing second cluster file would otherwise run a
# CTCF only arm under an RNAPII arm's name.
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
