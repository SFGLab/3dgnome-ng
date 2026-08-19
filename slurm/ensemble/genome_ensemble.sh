#!/bin/bash -l

# Whole-genome conformational ensemble for one cell line, sharded over chromosomes.
#
#   CELL=GM12878 sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh
#   CELL=H1ESC   N_MODELS=20 PER_TASK=5 sbatch --array=0-91%6 slurm/ensemble/genome_ensemble.sh
#
# One array task = one chromosome x a block of PER_TASK conformations. The array index maps to
#
#   chrom_index = TASK / CHUNKS        CHUNKS = N_MODELS / PER_TASK
#   chunk       = TASK % CHUNKS        members = chunk*PER_TASK .. +PER_TASK-1
#
# so the array length is (number of chromosomes) * CHUNKS. With the defaults below that is
# 23 * 10 = 230, printed by the script at startup so a wrong --array is obvious immediately.
#
# Why per chromosome rather than one whole-genome job per conformation. The downstream
# enhancer3D analysis is intra-chromosomal (enhancer-promoter distances within a chromosome) and
# its published models are per-chromosome, so nothing consumes inter-chromosomal placement.
# Sharding this way turns one ~2.6 h GM12878 genome run into 23 short independent tasks that
# backfill into free GPUs, and a failure costs one chromosome rather than a whole genome. The
# cost is that chromosomes are placed independently, with no chromosome-level MC between them;
# if you need a single coherent nucleus, run `--region ""` in one task instead.
#
# Total work is unchanged by the sharding: about 2.6 GPU-hours per GM12878 conformation, 1.6 for
# H1ESC and 1.3 for HFFC6, scaling with each line's anchor count.
#
# Requeue safety: a member whose .cif already exists is skipped, so a requeued or resubmitted
# task resumes rather than redoing finished work.

#SBATCH --job-name=e3d_ens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=08:00:00
#SBATCH --account=sfglab
# Absolute for the same reason as setup_cell.sh: sbatch parses #SBATCH before the shell runs, so
# these cannot use $ROOT, and the directory must already exist.
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/ens_%x_%A_%a.out
#SBATCH --error=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/ens_%x_%A_%a.out

set -euo pipefail

CELL="${CELL:-${1:-}}"
[ -n "$CELL" ] || { echo "set CELL=GM12878|H1ESC|HFFC6" >&2; exit 1; }

ROOT="${ROOT:-/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng}"
LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
CONFIG="${CONFIG:-$ROOT/slurm/ensemble/${LOWER}_hic_tads.ini}"
OUT="${OUT:-$ROOT/out/${LOWER}_genome}"
N_MODELS="${N_MODELS:-100}"
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

# Share one XLA cache across the whole array. Each distinct IB shape compiles once per machine
# instead of once per task; with bucketing that is a handful of shapes rather than hundreds.
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Fail before the allocation is burnt. data/ is gitignored, so a fresh checkout has none of this.
missing=0
for f in "data/$CELL/${CELL}_hic_25kb_singletons.bedpe" \
         "data/$CELL/${CELL}_tads.bed" \
         "data/$CELL/${CELL}_anchors_3+_oriented.bed" \
         "data/$CELL/${CELL}_clusters_3+.bedpe"; do
  [ -s "$f" ] || { echo "[guard] missing $f" >&2; missing=1; }
done
if [ "$missing" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "[guard] run the one-time setup first: sbatch slurm/ensemble/setup_cell.sh $CELL" >&2
  exit 1
fi

python - "$CONFIG" "$CELL" <<'PYCHECK'
import sys

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
cell = sys.argv[2]

# Each of these fails silently rather than loudly if unset: arc-gap blocks instead of TADs,
# ChIA-PET singletons instead of Hi-C, or epigenome terms left on from another run all produce a
# plausible structure that answers a different question.
assert s.ib_split_source == "tads", f"ib_split_source={s.ib_split_source!r}"
assert cell in s.data_anchors, f"config is not for {cell}: anchors={s.data_anchors!r}"
assert "hic" in s.data_singletons, f"singletons={s.data_singletons!r} does not look Hi-C derived"
assert s.use_ctcf_motif and s.use_excluded_volume and s.use_dynamic_loop_density
assert s.use_anchor_heatmap and s.use_subanchor_heatmap
assert s.mc_executor_jax_bucket_shapes, "shape bucketing is off; this run would be ~5x slower"
for flag in ("use_compartments", "use_bridging", "use_fibre_compaction", "use_lamina"):
    assert not getattr(s, flag), f"{flag} is on; these runs exclude epigenome terms"
print(f"[guard] {cell} config ok, bucketing on, heat_min_reduction={s.subanchor_heat_min_reduction}")
PYCHECK

# DRY_RUN prints the resolved shard and stops, so the array mapping can be checked without
# an allocation.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry] task=$TASK chrom=$CHROM members=$MEMBERS out=$OUT/$CHROM"
  exit 0
fi

echo "[launch] node=$(hostname) cell=$CELL chrom=$CHROM members=$MEMBERS task=$TASK"
srun gnome3d-ng --config "$CONFIG" --region "$CHROM" --members "$MEMBERS" --out "$OUT/$CHROM" \
  --log-file "$ROOT/slurm/ensemble/logs/ens_${CELL}_${CHROM}_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.detail.log"
