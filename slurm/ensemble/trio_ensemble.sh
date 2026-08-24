#!/bin/bash -l

# Conformational ensembles for the nine trio samples, sharded chromosome first.
#
#   sbatch --array=0-89%6   slurm/ensemble/trio_ensemble.sh          # chr1, all nine samples
#   CHROMS=chr1 sbatch --array=0-89%6 slurm/ensemble/trio_ensemble.sh
#   sbatch --array=0-2069%6 slurm/ensemble/trio_ensemble.sh          # whole genome, if allowed
#
# One array task is one chromosome by one sample by a block of PER_TASK conformations. The
# chromosome is the slowest varying dimension, so the first 9*CHUNKS tasks cover one chromosome
# across every sample. That ordering is deliberate: a trio comparison on that chromosome becomes
# possible while the rest of the genome is still queued, rather than after the last sample.
#
# The mapping lives in playground/trio/trio_samples.py::shard rather than here, so this script and
# playground/trio/trio_status.py cannot disagree about which index means what.
#
# Array size. Nine samples by 23 chromosomes by 10 chunks is 2070 tasks, and Slurm's default
# MaxArraySize is 1001. Check with `scontrol show config | grep MaxArraySize`. If it is under
# 2070, submit one chromosome at a time with CHROMS=chrN, which is 90 tasks, or raise PER_TASK.
#
# Resuming. A member whose .cif already exists is skipped by gnome3d-ng, and a task whose whole
# block is already present exits before requesting any GPU work. So a cancelled array, a dead
# node or a preempted job all resume by resubmitting. Ask for exactly the gaps with
# `python playground/trio/trio_status.py --resubmit`.
#
# Sizing a task. gnome3d.cli writes each structure as it finishes, so a job killed at the wall
# clock keeps every conformation already done and loses only the one in flight. That makes a
# long task cheap to overrun and PER_TASK safe to raise. The partition allows 5 days, so the
# limit is scheduling rather than capacity: a short job backfills into gaps that a long one
# cannot, so the smallest PER_TASK that keeps the job count reasonable is the right one.
#
# Wall clock is PER_TASK times the per conformation time, which scales with a chromosome's
# anchor count. chr1 holds about 11% of the genome's anchors, so a fixed PER_TASK leaves most of
# the limit unused on the small chromosomes. Since submission is per chromosome anyway, pass a
# different PER_TASK and --time for each:
#
#   CHROMS=chr1  PER_TASK=10 sbatch --array=0-89%6 --time=24:00:00 ...
#   CHROMS=chr21 PER_TASK=50 sbatch --array=0-17%6 --time=24:00:00 ...
#
# --time on the command line overrides the directive above, so the file needs no edit.

#SBATCH --job-name=trio_ens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --account=sfglab
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/trio_%x_%A_%a.out
#SBATCH --error=slurm/ensemble/logs/trio_%x_%A_%a.out

set -euo pipefail

# Derived from this script's own location, so the checkout can live anywhere. Override
# with ROOT=... if the job is launched from a copy outside the tree.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${OUT:-$ROOT/out/trio}"
N_MODELS="${N_MODELS:-100}"
PER_TASK="${PER_TASK:-10}"
CHROMS="${CHROMS:-}"
SAMPLES="${SAMPLES:-}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SHARD=$(python playground/trio/trio_shard.py --task "$TASK" --n-models "$N_MODELS" \
          --per-task "$PER_TASK" --chroms "$CHROMS" --samples "$SAMPLES") || {
  echo "[trio_ens] task $TASK is past the end of the array" >&2
  exit 1
}
read -r CHROM SAMPLE FIRST LAST TOTAL <<< "$SHARD"
MEMBERS="${FIRST}-${LAST}"
LOWER=$(echo "$SAMPLE" | tr '[:upper:]' '[:lower:]')
CONFIG="${CONFIG:-$ROOT/slurm/ensemble/${LOWER}_trio.ini}"
DEST="$OUT/$SAMPLE/$CHROM"

echo "[trio_ens] array length needed = $TOTAL"
# Printed so an overrun is diagnosable from the log rather than only from sacct.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o "%l" 2>/dev/null || echo unknown)
  echo "[trio_ens] time limit $LIMIT for $((LAST - FIRST + 1)) conformations"
fi
echo "[trio_ens] task=$TASK chrom=$CHROM sample=$SAMPLE members=$MEMBERS out=$DEST"

# Skip before asking for a GPU. A resubmitted array is mostly finished tasks, and starting the
# python stack for each of them wastes an allocation slot that another chromosome could use.
HAVE=0
# gnome3d.cli writes member i as <region>_s<i+1>.cif, so member 0 is chr1_s1.cif.
for m in $(seq "$FIRST" "$LAST"); do
  [ -s "$DEST/${CHROM}_s$((m + 1)).cif" ] && HAVE=$((HAVE + 1))
done
WANT=$((LAST - FIRST + 1))
echo "[trio_ens] $HAVE/$WANT members already present"
if [ "$HAVE" -eq "$WANT" ]; then
  echo "[trio_ens] block complete, nothing to do"
  exit 0
fi

# Fail before the allocation is burnt. data/ is gitignored, so a fresh checkout has none of it.
missing=0
for f in "data/$SAMPLE/${SAMPLE}_hic_25kb_singletons.bedpe" \
         "data/$SAMPLE/${SAMPLE}_blocks.bed" \
         "data/$SAMPLE/${SAMPLE}_segments.bed" \
         "data/$SAMPLE/${SAMPLE}_anchors_3+_oriented.bed" \
         "data/$SAMPLE/${SAMPLE}_clusters_3+.bedpe"; do
  [ -s "$f" ] || { echo "[guard] missing $f" >&2; missing=1; }
done
if [ "$missing" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "[guard] run the one-time setup first: sbatch slurm/ensemble/trio_setup.sh $SAMPLE" >&2
  exit 1
fi

python - "$CONFIG" "$SAMPLE" <<'PYCHECK'
import sys

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
sample = sys.argv[2]

# Each of these fails silently rather than loudly if unset, producing a plausible structure that
# answers a different question.
assert s.ib_split_source == "tads", f"ib_split_source={s.ib_split_source!r}"
assert sample in s.data_anchors, f"config is not for {sample}: anchors={s.data_anchors!r}"
assert "hic" in s.data_singletons, f"singletons={s.data_singletons!r} does not look contact derived"
assert s.use_ctcf_motif and s.use_excluded_volume and s.use_dynamic_loop_density
assert s.use_anchor_heatmap and s.use_subanchor_heatmap
assert s.mc_executor_jax_bucket_shapes, "shape bucketing is off; this run would be ~5x slower"
for flag in ("use_compartments", "use_bridging", "use_fibre_compaction", "use_lamina"):
    assert not getattr(s, flag), f"{flag} is on; these runs exclude epigenome terms"
print(f"[guard] {sample} config ok, bucketing on")
PYCHECK

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry] task=$TASK chrom=$CHROM sample=$SAMPLE members=$MEMBERS out=$DEST"
  exit 0
fi

mkdir -p "$DEST"
# One XLA cache for the whole array, so each IB shape compiles once per machine rather than once
# per task. With bucketing that is a handful of shapes.
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1

echo "[launch] node=$(hostname) sample=$SAMPLE chrom=$CHROM members=$MEMBERS"
srun gnome3d-ng --config "$CONFIG" --region "$CHROM" --members "$MEMBERS" --out "$DEST" \
  --log-file "$ROOT/slurm/ensemble/logs/trio_${SAMPLE}_${CHROM}_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.detail.log"
