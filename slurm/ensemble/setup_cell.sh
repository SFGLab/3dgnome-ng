#!/bin/bash -l

# One time setup of one cell line: environment, Hi-C, TAD calls and genome wide Hi-C singletons.
#
#   sbatch -A <account> slurm/ensemble/setup_cell.sh GM12878
#
# Run once per cell line before its array. Runs as a job, see setup.sh. SKIP_INSTALL=1 skips the
# install. The singleton step takes 20 to 40 minutes for a genome at 25 kb.

#SBATCH --job-name=e3d_setup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=06:00:00
# The account is given with sbatch -A. --output and --error are relative to the checkout, the
# directory of the submission. Create slurm/ensemble/logs before the submission.
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/setup_%x_%j.out
#SBATCH --error=slurm/ensemble/logs/setup_%x_%j.out

set -euo pipefail

CELL="${1:?usage: sbatch slurm/ensemble/setup_cell.sh <GM12878|H1ESC|HFFC6>}"
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BINSIZE="${BINSIZE:-25000}"
CHROMS="${CHROMS:-chr1-chr22,chrX}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SINGLETONS="$ROOT/data/$CELL/${CELL}_hic_$((BINSIZE / 1000))kb_singletons.bedpe"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
# Progress reaches the slurm log at once.
export PYTHONUNBUFFERED=1

echo "[setup:$CELL] node=$(hostname)"
LEVEL=$(awk '/^flags/{if(/avx512f/) print "v4"; else if(/avx2/) print "v3";
                      else if(/sse4_2/) print "v2"; else print "v1"; exit}' /proc/cpuinfo)
echo "[setup:$CELL] cpu x86-64 level = $LEVEL (NumPy wheels need v2 or better)"
[ "$LEVEL" = "v1" ] && { echo "[setup] node below the NumPy baseline" >&2; exit 1; }

# Not the validation extra, whose hicrep and pyBigWig build from source.
if [ "$SKIP_INSTALL" != "1" ]; then
  pip install --upgrade pip
  pip install -e .
  pip install "jax[cuda12]"
  pip install "scipy>=1.10" "cooler>=0.9" "cooltools>=0.7"
fi
python -c "
import jax, numpy
print(f'[setup] numpy {numpy.__version__}  jax {jax.__version__}  devices {jax.devices()}')
"

# The ChIA-PET anchors and loops, the segment breakpoints and the centromeres have no download and
# must be copied in.
missing=0
for f in "${CELL}_anchors_3+_oriented.bed" "${CELL}_clusters_3+.bedpe" \
         "ccds_all_hg38_merged100k_${CELL}.breakpoints.bed" "hg38_centromeres.bed"; do
  [ -s "data/$CELL/$f" ] || { echo "[setup:$CELL] MISSING data/$CELL/$f" >&2; missing=1; }
done
[ "$missing" = "1" ] && { echo "[setup:$CELL] copy the ChIA-PET inputs first" >&2; exit 1; }

MCOOL=$(ls "data/_hic/$CELL"/*.mcool 2>/dev/null | head -1 || true)
if [ -z "$MCOOL" ]; then
  echo "[setup:$CELL] fetching Hi-C"
  python -m legacy.validation fetch --manifest "legacy/validation/manifests/${CELL}_hic.json" --out data/_hic
  MCOOL=$(ls "data/_hic/$CELL"/*.mcool | head -1)
fi
echo "[setup:$CELL] mcool = $MCOOL"

# TAD boundaries only.
if [ -s "data/$CELL/${CELL}_tads.bed" ]; then
  echo "[setup:$CELL] have data/$CELL/${CELL}_tads.bed"
else
  python -m legacy.validation tracks --cell "$CELL" --skip-compartments
fi

# The singleton reader needs balancing weights at its resolution. They are computed here when the
# mcool lacks them.
BINURI="$MCOOL::/resolutions/$BINSIZE"
if python -c "
import sys
import cooler
sys.exit(0 if 'weight' in cooler.Cooler('$BINURI').bins().columns else 1)
" 2>/dev/null; then
  echo "[setup:$CELL] $BINSIZE already balanced"
else
  echo "[setup:$CELL] balancing $BINSIZE (one time)"
  cooler balance -p "${SLURM_CPUS_PER_TASK:-8}" "$BINURI"
fi

# A failed run leaves an empty file, which prep_singletons would take as done.
if [ -f "$SINGLETONS" ] && [ ! -s "$SINGLETONS" ]; then
  echo "[setup:$CELL] removing empty $SINGLETONS from an earlier failure"
  rm -f "$SINGLETONS"
fi

python slurm/ensemble/prep_singletons.py \
  --mcool "$MCOOL" --chroms "$CHROMS" --binsize "$BINSIZE" --out "$SINGLETONS"

python - "$ROOT/slurm/ensemble/$(echo "$CELL" | tr '[:upper:]' '[:lower:]')_hic_tads.ini" <<'PYCHECK'
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
    print(f"[setup] ok {field}: {p.name} ({p.stat().st_size} bytes)")
PYCHECK

echo "[setup:$CELL] done. submit with:"
echo "  CELL=$CELL sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh"
