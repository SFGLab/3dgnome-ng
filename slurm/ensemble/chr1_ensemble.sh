#!/bin/bash -l

# GM12878 chr1 conformational ensemble: TAD blocks, CTCF anchors, Hi-C singletons, distance
# map, dynamic subanchors, excluded volume. One array task produces one conformation.
#
#   sbatch --array=0-99%20 slurm/ensemble/chr1_ensemble.sh
#   PER_TASK=4 sbatch --array=0-24 slurm/ensemble/chr1_ensemble.sh   # 4 conformations per task
#
# Why one GPU per task rather than one big multi-GPU job. Conformations are independent, so a
# fixed pool of GPUs delivers the most conformations per hour when each one gets a whole
# conformation to itself. Splitting a single conformation across N GPUs shortens that
# conformation but does not raise throughput, and it loses a little to load imbalance, so it is
# the wrong shape for producing 100 of them. Ask for several GPUs only when latency on one
# structure matters more than total count; multigpu_mode=groups in the ini then uses them, and
# it is inert on one GPU.
#
# --time is 8 h for a measured 5 h 09 m conformation at these settings, on an RTX 4060 Ti. A
# faster card needs less. Raise PER_TASK and --time together, since the cap applies per task.
#
# Requeue safety: a member whose .cif already exists is skipped, so a task that is requeued
# after NODE_FAIL or a time cap resumes rather than redoing finished work. That also makes it
# safe to resubmit the same array to fill in whatever is missing.

#SBATCH --job-name=gm_chr1_ens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --time=08:00:00
#SBATCH --account=sfglab
# --output/--error are absolute and cannot use $ROOT: sbatch parses #SBATCH lines before
# the shell runs, and a relative path would resolve against whatever directory you
# submitted from. The directory must already exist, because slurm opens these files
# before the script's own mkdir would run. Create it once:
#   mkdir -p $ROOT/slurm/ensemble/logs
# Running from a different checkout means also passing --output/--error to sbatch.
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/chr1_%A_%a.out
#SBATCH --error=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/chr1_%A_%a.out

set -euo pipefail

# Override for a checkout elsewhere, e.g. ROOT=$HOME/3dgnome-ng sbatch ...
ROOT="${ROOT:-/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng}"
CONFIG="${CONFIG:-$ROOT/slurm/ensemble/gm12878_chr1_hic_tads.ini}"
OUT="${OUT:-$ROOT/out/chr1_ensemble}"
PER_TASK="${PER_TASK:-1}"
MCOOL="${MCOOL:-$ROOT/data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool}"
SINGLETONS="$ROOT/data/GM12878/GM12878_chr1_hic_25kb_singletons.bedpe"

cd "$ROOT"
mkdir -p slurm/ensemble/logs "$OUT"
source .venv/bin/activate

# Share one XLA cache across the array. Each distinct IB shape compiles once per machine
# instead of once per task, and with ~250 shapes per conformation that is not a rounding error.
export GNOME3D_JAX_CACHE="${GNOME3D_JAX_CACHE:-$ROOT/.cache/gnome3d-jax}"
mkdir -p "$GNOME3D_JAX_CACHE"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

TASK="${SLURM_ARRAY_TASK_ID:-0}"
FIRST=$((TASK * PER_TASK))
LAST=$((FIRST + PER_TASK - 1))
MEMBERS="${FIRST}-${LAST}"

# Fail before the allocation is burnt rather than minutes in. data/ is gitignored, so a fresh
# checkout has none of this and every one of these is a hard requirement.
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

# Each of these is a requirement of this ensemble, and each fails silently rather than loudly
# if it is not set: arc-gap blocks instead of TADs, ChIA-PET singletons instead of Hi-C, or the
# epigenome terms left on from another run all produce a plausible structure that answers a
# different question.
assert s.ib_split_source == "tads", f"ib_split_source={s.ib_split_source!r}, expected tads"
assert s.data_ib_split, "ib_split is empty; tads mode would raise at load"
assert "hic" in s.data_singletons, f"singletons={s.data_singletons!r} does not look Hi-C derived"
assert s.use_ctcf_motif, "CTCF orientation term is off"
assert s.use_excluded_volume, "excluded volume is off"
assert s.use_dynamic_loop_density, "dynamic subanchors are off"
assert s.use_anchor_heatmap and s.use_subanchor_heatmap, "distance-map terms are off"
for flag in ("use_compartments", "use_bridging", "use_fibre_compaction", "use_lamina"):
    assert not getattr(s, flag), f"{flag} is on; this ensemble runs without epigenome terms"
print(f"[guard] config ok: {s.ib_split_source} blocks, singletons={s.data_singletons}")
PYCHECK

echo "[launch] node=$(hostname) task=${TASK} members=${MEMBERS} gpus=${SLURM_GPUS_ON_NODE:-?}"
srun gnome3d-ng --config "$CONFIG" --region chr1 --members "$MEMBERS" --out "$OUT" \
  --log-file "$ROOT/slurm/ensemble/logs/chr1_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.detail.log"
