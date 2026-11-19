#!/bin/bash -l

# Whole genome ensemble of one cell line, one array task per chromosome and block of PER_TASK
# conformations.
#
#   CELL=GM12878 sbatch -A <account> --array=0-45%12 slurm/ensemble/genome_ensemble.sh
#   CELL=H1ESC N_MODELS=40 PER_TASK=10 sbatch -A <account> --array=0-91%12 slurm/ensemble/genome_ensemble.sh
#
# chrom_index = TASK / CHUNKS, chunk = TASK % CHUNKS, CHUNKS = N_MODELS / PER_TASK. The array length
# is the number of chromosomes times CHUNKS, which the script prints at start. Chromosomes are
# placed independently. A member whose .cif exists is skipped.

#SBATCH --job-name=e3d_ens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=long
# 24 h for PER_TASK=10 on the slowest chromosome, about 16 h. A conformation is written when it
# finishes.
#SBATCH --time=24:00:00
# The account is given with sbatch -A. --output and --error are relative to the checkout, the
# directory of the submission. Create slurm/ensemble/logs before the submission.
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/ens_%x_%A_%a.out
#SBATCH --error=slurm/ensemble/logs/ens_%x_%A_%a.out

set -euo pipefail

CELL="${CELL:-${1:-}}"
[ -n "$CELL" ] || { echo "set CELL=GM12878|H1ESC|HFFC6" >&2; exit 1; }

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
# Blocks come from arc gaps. `out/<cell>_genome` holds older TAD block ensembles.
CONFIG="${CONFIG:-$ROOT/slurm/ensemble/${LOWER}_hic_arcs.ini}"
OUT="${OUT:-$ROOT/out/${LOWER}_genome_arcs}"
N_MODELS="${N_MODELS:-20}"
PER_TASK="${PER_TASK:-10}"
CHROMS="${CHROMS:-chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX}"

read -r -a CHROM_ARR <<< "$CHROMS"
CHUNKS=$(( (N_MODELS + PER_TASK - 1) / PER_TASK ))
NEEDED=$(( ${#CHROM_ARR[@]} * CHUNKS ))

TASK="${SLURM_ARRAY_TASK_ID:-0}"
CHROM_IDX=$(( TASK / CHUNKS ))
CHUNK=$(( TASK % CHUNKS ))

echo "[ens:$CELL] array length needed = $NEEDED (${#CHROM_ARR[@]} chroms x $CHUNKS chunks of $PER_TASK)"
if [ "$CHROM_IDX" -ge "${#CHROM_ARR[@]}" ]; then
  echo "[ens:$CELL] task $TASK is past the end; submit --array=0-$((NEEDED - 1))" >&2
  exit 1
fi

CHROM="${CHROM_ARR[$CHROM_IDX]}"
FIRST=$(( CHUNK * PER_TASK ))
LAST=$(( FIRST + PER_TASK - 1 ))
[ "$LAST" -ge "$N_MODELS" ] && LAST=$(( N_MODELS - 1 ))
MEMBERS="${FIRST}-${LAST}"

cd "$ROOT"
mkdir -p slurm/ensemble/logs "$OUT/$CHROM"
source .venv/bin/activate

# One XLA cache for the whole array.
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

python - "$CONFIG" "$CELL" <<'PYCHECK'
import sys

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
cell = sys.argv[2]

# Each of these must hold. A wrong value gives a plausible structure of another configuration.
assert cell in s.data_anchors, f"config is not for {cell}: anchors={s.data_anchors!r}"
assert s.use_ctcf_motif and s.use_excluded_volume and s.use_dynamic_loop_density
assert s.use_anchor_heatmap, "the anchor distance map is off"
assert s.mc_executor_jax_bucket_shapes, "shape bucketing is off; this run would be ~5x slower"
assert not s.use_compartments, "use_compartments is on; these runs exclude epigenome terms"
from pathlib import Path  # noqa: E402

required = ["data_anchors", "data_pet_clusters", "data_singletons", "data_segment_split",
            "data_centromeres"]
for field in required:
    name = getattr(s, field)
    assert name, f"{field} is unset"
    path = Path(s.data_path(name))
    assert path.is_file() and path.stat().st_size > 0, (
        f"{field}: missing or empty {path}; run sbatch slurm/ensemble/setup_cell.sh {cell}")

print(f"[guard] {cell} ok, singletons={s.data_singletons}, bucketing on, "
      f"heat_min_reduction={s.subanchor_heat_min_reduction}")
PYCHECK

# DRY_RUN prints the resolved shard and stops.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry] task=$TASK chrom=$CHROM members=$MEMBERS out=$OUT/$CHROM"
  exit 0
fi

echo "[launch] node=$(hostname) cell=$CELL chrom=$CHROM members=$MEMBERS task=$TASK"
srun gnome3d-ng --config "$CONFIG" --region "$CHROM" --members "$MEMBERS" --out "$OUT/$CHROM" \
  --log-file "$ROOT/slurm/ensemble/logs/ens_${CELL}_${CHROM}_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.detail.log"
