#!/bin/bash -l

# One-time setup for one trio sample: mcool, balancing, TAD boundaries, blocks and segments,
# config, contact singletons.
#
#   sbatch slurm/ensemble/trio_setup.sh HG00512
#
# Every stage checks for its own output first, so a killed or requeued job resumes instead of
# redoing work. Rerun after any failure.
#
# What must arrive before this runs. The anchors and clusters are built on the laptop, because
# the motif track they need is gitignored and does not travel, and the raw .hic is copied
# because it is the input to everything here:
#
#   rsync -av data/_trio/<S>/<S>_allres.hic  eden:$ROOT/data/_trio/<S>/
#   rsync -av data/<S>/                      eden:$ROOT/data/<S>/
#
# The singleton stage is the long one, roughly 20-40 minutes, because it reads a dense contact
# matrix per chromosome. The mcool conversion is about 7 minutes.

#SBATCH --job-name=trio_setup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --account=sfglab
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/trio_setup_%x_%j.out
#SBATCH --error=slurm/ensemble/logs/trio_setup_%x_%j.out

set -euo pipefail

S="${1:?usage: sbatch slurm/ensemble/trio_setup.sh <SAMPLE>}"
# Where the checkout is. Slurm copies the batch script into a spool directory before running
# it, so BASH_SOURCE points at that copy rather than at the tree and cannot locate the root.
# SLURM_SUBMIT_DIR is the directory sbatch was run from, which is also what the relative
# --output paths above resolve against, so submitting from the repo root makes both agree.
# Outside Slurm the script's own location is correct. Override with ROOT=... for anything else.
if [ -n "${ROOT:-}" ]; then
  :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
# A wrong root shows up as a permission error on the first mkdir, several lines from the cause.
[ -f "$ROOT/pyproject.toml" ] && [ -d "$ROOT/gnome3d" ] || {
  echo "[error] ROOT=$ROOT is not the 3dgnome checkout." >&2
  echo "[error] submit from the repo root, or pass ROOT=/path/to/3dgnome-ng" >&2
  exit 1
}
BINSIZE="${BINSIZE:-25000}"
CHROMS="${CHROMS:-chr1-chr22,chrX}"
MCOOL="$ROOT/data/_hic/$S/$S.mcool"
SINGLETONS="$ROOT/data/$S/${S}_hic_$((BINSIZE / 1000))kb_singletons.bedpe"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "[trio_setup:$S] node=$(hostname)"

# Checked, never installed. These jobs run one per sample and therefore nine at a time against
# one shared .venv, and concurrent pip installs into a single environment interleave their
# temporary files, so one process deletes what another is still writing. That surfaces as
# "OSError: [Errno 2] No such file or directory" inside site-packages.
#
# hic2cool is imported lazily inside trio_prepare, so without this check a missing dependency
# would fail at the mcool stage rather than here.
python -c "import cooler, cooltools, scipy, hic2cool" 2>/dev/null || {
  echo "[error] missing dependencies. Install them ONCE, from a login shell, before submitting:" >&2
  echo "        source .venv/bin/activate" >&2
  echo "        pip install hic2cool 'scipy>=1.10' 'cooler>=0.9' 'cooltools>=0.7'" >&2
  echo "[error] do not let the array install them, nine jobs racing on one venv is what breaks." >&2
  exit 1
}
python -c "
import cooler, cooltools, hic2cool
print(f'[trio_setup] cooler {cooler.__version__} cooltools {cooltools.__version__} hic2cool ok')
"

# The text inputs are built on the laptop. Without them the mcool would still convert and the
# run would then fail hours later at load time, so fail here instead.
for f in "data/$S/${S}_anchors_3+_oriented.bed" "data/$S/${S}_clusters_3+.bedpe" \
         "data/$S/hg38_centromeres.bed" "data/_trio/$S/${S}_allres.hic"; do
  [ -s "$f" ] || { echo "[trio_setup:$S] MISSING $f, rsync it first" >&2; exit 1; }
done

# Skips the text stages because their outputs are already present, converts the mcool if absent.
python playground/trio/trio_prepare.py --samples "$S"

# hic2cool does not write balancing weights and the singleton reader fetches the balanced
# matrix, so an unbalanced resolution yields "No column 'bins/weight' found" on every
# chromosome and an empty output file.
if python -c "
import sys
import cooler
sys.exit(0 if 'weight' in cooler.Cooler('$MCOOL::/resolutions/$BINSIZE').bins().columns else 1)
" 2>/dev/null; then
  echo "[trio_setup:$S] $BINSIZE already balanced"
else
  echo "[trio_setup:$S] balancing $BINSIZE (one time)"
  cooler balance -p "${SLURM_CPUS_PER_TASK:-8}" "$MCOOL::/resolutions/$BINSIZE"
fi

if [ -s "data/$S/${S}_tads.bed" ]; then
  echo "[trio_setup:$S] have data/$S/${S}_tads.bed"
else
  python -m validation tracks --cell "$S" --skip-compartments --skip-signal
fi

# Cheap and deterministic, so they are simply rebuilt. trio_segments refuses to write a pair of
# files that would leave IB placement skipping most segments.
python playground/trio/trio_segments.py --samples "$S"
python playground/trio/trio_configs.py --samples "$S"

# A previous failed run can leave a zero row file, and prep_singletons treats any existing
# output as done, so it would skip regeneration for ever.
if [ -f "$SINGLETONS" ] && [ ! -s "$SINGLETONS" ]; then
  echo "[trio_setup:$S] removing empty $SINGLETONS from an earlier failure"
  rm -f "$SINGLETONS"
fi
python slurm/ensemble/prep_singletons.py \
  --mcool "$MCOOL" --chroms "$CHROMS" --binsize "$BINSIZE" --out "$SINGLETONS"

python - "$ROOT/slurm/ensemble/$(echo "$S" | tr '[:upper:]' '[:lower:]')_trio.ini" <<'PYCHECK'
import sys
from pathlib import Path

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
for field in ("data_anchors", "data_pet_clusters", "data_singletons",
              "data_segment_split", "data_centromeres"):
    name = getattr(s, field)
    assert name, f"{field} is unset"
    p = Path(s.data_path(name))
    assert p.is_file() and p.stat().st_size > 0, f"{field}: missing or empty {p}"
    print(f"[trio_setup] ok {field}: {p.name} ({p.stat().st_size} bytes)")
PYCHECK

echo "[trio_setup:$S] done"
