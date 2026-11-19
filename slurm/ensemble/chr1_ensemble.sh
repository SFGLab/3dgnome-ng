#!/bin/bash -l

# GM12878 chr1 ensemble, one conformation per array task.
#
#   sbatch -A <account> --array=0-99%20 slurm/ensemble/chr1_ensemble.sh
#   PER_TASK=4 sbatch -A <account> --array=0-24 slurm/ensemble/chr1_ensemble.sh
#
# --time is per task. Raise it with PER_TASK. A member whose .cif exists is skipped.

#SBATCH --job-name=gm_chr1_ens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --time=08:00:00
# The account is given with sbatch -A. --output and --error are relative to the checkout, the
# directory of the submission. Create slurm/ensemble/logs before the submission.
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/chr1_%A_%a.out
#SBATCH --error=slurm/ensemble/logs/chr1_%A_%a.out

set -euo pipefail

# The checkout. The default is the directory of the submission.
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-$ROOT/slurm/ensemble/gm12878_chr1_hic_tads.ini}"
OUT="${OUT:-$ROOT/out/chr1_ensemble}"
PER_TASK="${PER_TASK:-1}"
MCOOL="${MCOOL:-$ROOT/data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool}"
SINGLETONS="$ROOT/data/GM12878/GM12878_chr1_hic_25kb_singletons.bedpe"

cd "$ROOT"
mkdir -p slurm/ensemble/logs "$OUT"
source .venv/bin/activate

# One XLA cache for the whole array.
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

TASK="${SLURM_ARRAY_TASK_ID:-0}"
FIRST=$((TASK * PER_TASK))
LAST=$((FIRST + PER_TASK - 1))
MEMBERS="${FIRST}-${LAST}"

# Every input must exist. data/ is not in the repository.
missing=0
for f in "$SINGLETONS" \
         "$ROOT/data/GM12878/GM12878_tads.bed" \
         "$ROOT/data/GM12878/GM12878_anchors_3+_oriented.bed" \
         "$ROOT/data/GM12878/GM12878_clusters_3+.bedpe"; do
  [ -s "$f" ] || { echo "[guard] missing $f" >&2; missing=1; }
done
if [ "$missing" = "1" ]; then
  echo "[guard] run the one-time setup first: sbatch slurm/ensemble/setup.sh" >&2
  exit 1
fi

python - "$CONFIG" <<'PYCHECK'
import sys

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"

# Each of these must hold. A wrong value gives a plausible structure of another configuration.
assert "hic" in s.data_singletons, f"singletons={s.data_singletons!r} does not look Hi-C derived"
assert s.use_ctcf_motif, "CTCF orientation term is off"
assert s.use_excluded_volume, "excluded volume is off"
assert s.use_dynamic_loop_density, "dynamic subanchors are off"
assert s.use_anchor_heatmap, "the anchor distance map is off"
assert not s.use_compartments, "use_compartments is on; this ensemble runs without epigenome terms"
print(f"[guard] config ok: singletons={s.data_singletons}")
PYCHECK

echo "[launch] node=$(hostname) task=${TASK} members=${MEMBERS} gpus=${SLURM_GPUS_ON_NODE:-?}"
srun python -m gnome3d.cli --config "$CONFIG" --region chr1 --members "$MEMBERS" --out "$OUT" \
  --log-file "$ROOT/slurm/ensemble/logs/chr1_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.detail.log"
