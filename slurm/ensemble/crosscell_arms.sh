#!/bin/bash -l

# Cross line RNAPII comparison, design/2026-11-11-rnapii-release.md. Nine array tasks, one per line
# and arm, ten conformations each. Arm C is CTCF only, R is CTCF and RNAPII and M is R with RNAPII
# ends near a CTCF anchor moved onto it.
#
#   sbatch -A <account> --array=0-8 slurm/ensemble/crosscell_arms.sh
#   CHROMS="chr2 chr3" sbatch -A <account> --array=0-8 slurm/ensemble/crosscell_arms.sh
#   DRY=1 SLURM_ARRAY_TASK_ID=4 bash slurm/ensemble/crosscell_arms.sh
#   SPLIT=1 CHROMS="chr2 ... chr22" sbatch -A <account> --array=0-188%24 slurm/ensemble/crosscell_arms.sh
#   ARMS="C UC UR" OUT=out/xu SPLIT=1 CHROMS="chr1 ... chr22" sbatch -A <account> --array=0-197%24 \
#     --mem=96G --cpus-per-task=16 slurm/ensemble/crosscell_arms.sh
#   ARMS="UCN URN" OUT=out/xu SPLIT=1 CHROMS="chr1 ... chr22" sbatch -A <account> --array=0-131%24 \
#     --mem=96G --cpus-per-task=16 slurm/ensemble/crosscell_arms.sh
#
# ARMS is the list of arms, C R M by default. With n arms, task i runs line i / n of GM12878 H1ESC
# HFFC6 and arm i % n. With SPLIT=1 it runs line and arm i % 3n on chromosome i / 3n of CHROMS. A
# member whose .cif exists is skipped. After chr1 the arm is sliced to chr1:1-60000000. The inputs
# come from playground/rnapii/crosscell_prep.py and the configs from
# slurm/ensemble/crosscell_configs.py. UC and UR are C and R on the uniform chain. They read half a
# of the deep map of the line, data/_hic_half/a/<LINE>.mcool of playground/thin_cooler.py. UCN and
# URN are UC and UR without the loop dropout.

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
read -r -a ARMS <<< "${ARMS:-C R M}"
NARMS=${#ARMS[@]}
NCOMBO=$((3 * NARMS))
TASK="${SLURM_ARRAY_TASK_ID:-0}"
COMBO="$TASK"
if [ "$SPLIT" = "1" ]; then
  read -r -a CHROM_LIST <<< "$CHROMS"
  COMBO=$((TASK % NCOMBO))
  [ $((TASK / NCOMBO)) -lt ${#CHROM_LIST[@]} ] || { echo "task $TASK beyond ${#CHROM_LIST[@]} chromosomes"; exit 1; }
  CHROMS="${CHROM_LIST[$((TASK / NCOMBO))]}"
fi
[ "$COMBO" -lt "$NCOMBO" ] || { echo "task $TASK beyond $NCOMBO lines and arms"; exit 1; }
LINE="${LINES[$((COMBO / NARMS))]}"
ARM="${ARMS[$((COMBO % NARMS))]}"
LOWER="$(echo "$LINE" | tr '[:upper:]' '[:lower:]')"
CONFIG="$ROOT/slurm/ensemble/${LOWER}_xc_${ARM}.ini"
DEST="$OUT/${LINE}_${ARM}"

echo "[map] task=$TASK line=$LINE arm=$ARM config=$CONFIG out=$DEST chroms=$CHROMS"
if [ "$DRY" = "1" ]; then
  [ -f "$CONFIG" ] && echo "[dry] config present" || echo "[dry] config MISSING"
  for C in $CHROMS; do
    echo "[dry] python -m gnome3d.cli --config $CONFIG --region $C --members 0-$((N - 1)) --out $DEST/$C"
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
contact_map = str(s.data_contact_map).partition("::")[0]
assert not contact_map or Path(contact_map).is_file(), f"missing contact map {contact_map}"
print(
    f"[guard] config ok: clusters={s.data_pet_clusters} factors={s.data_factors or 'CTCF'}"
    f" uniform_chain={s.uniform_chain} contact_map={s.data_contact_map or 'none'}"
)
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
