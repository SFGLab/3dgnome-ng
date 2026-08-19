#!/bin/bash -l

# One-time setup for one cell line: environment, Hi-C, TAD calls, genome-wide Hi-C singletons.
#
#   sbatch slurm/ensemble/setup_cell.sh GM12878
#   sbatch slurm/ensemble/setup_cell.sh H1ESC
#   sbatch slurm/ensemble/setup_cell.sh HFFC6
#
# Run once per cell line and let it finish before submitting that line's array. The three are
# independent, so they can run at the same time; each holds one GPU only to verify CUDA.
#
# Why a job rather than login-node commands. Installing on eden's login node fails with
#
#   RuntimeError: NumPy was built with baseline optimizations: (X86_V2)
#                 but your machine doesn't support: (X86_V2)
#
# because that node's CPU is below the x86-64-v2 baseline current NumPy wheels are built
# against, so NumPy cannot import there and neither the install nor the data preparation can
# run. A dgx node is new enough. The install is idempotent, so the second and third cell lines
# skip it; pass SKIP_INSTALL=1 to skip it explicitly.
#
# The singleton step is the long one: it reads a dense contact matrix per chromosome and writes
# every non-zero intra-chromosomal pair, roughly 20-40 minutes for a genome at 25 kb.

#SBATCH --job-name=e3d_setup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --account=sfglab
# --output/--error are absolute and cannot use $ROOT: sbatch parses #SBATCH lines before the
# shell runs, and a relative path would resolve against whatever directory you submitted from.
# The directory must already exist, because slurm opens these files before the script's own
# mkdir would run. Create it once:  mkdir -p $ROOT/slurm/ensemble/logs
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/setup_%x_%j.out
#SBATCH --error=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/setup_%x_%j.out

set -euo pipefail

CELL="${1:?usage: sbatch slurm/ensemble/setup_cell.sh <GM12878|H1ESC|HFFC6>}"
ROOT="${ROOT:-/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng}"
BINSIZE="${BINSIZE:-25000}"
CHROMS="${CHROMS:-chr1-chr22,chrX}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SINGLETONS="$ROOT/data/$CELL/${CELL}_hic_$((BINSIZE / 1000))kb_singletons.bedpe"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
# Long stages print progress with plain print(); block-buffered into a slurm
# log that makes a working job look hung for many minutes.
export PYTHONUNBUFFERED=1

echo "[setup:$CELL] node=$(hostname)"
LEVEL=$(awk '/^flags/{if(/avx512f/) print "v4"; else if(/avx2/) print "v3";
                      else if(/sse4_2/) print "v2"; else print "v1"; exit}' /proc/cpuinfo)
echo "[setup:$CELL] cpu x86-64 level = $LEVEL (NumPy wheels need v2 or better)"
[ "$LEVEL" = "v1" ] && { echo "[setup] node below the NumPy baseline" >&2; exit 1; }

# Deliberately NOT `pip install -e ".[validation]"`. That extra also pulls hicrep and pyBigWig,
# which build from sdists whose setup.py imports numpy, and that build is what fails. Neither is
# needed: hicrep is declared but never imported anywhere in validation/, and pyBigWig is used
# only by the ATAC signal path, which these runs skip.
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

# Inputs with no download path. The manifests cover Hi-C and epigenomic signal only; the
# ChIA-PET anchors and loops, the segment breakpoints and the centromeres must be copied in.
missing=0
for f in "${CELL}_anchors_3+_oriented.bed" "${CELL}_clusters_3+.bedpe" \
         "ccds_all_hg38_merged100k_${CELL}.breakpoints.bed" "hg38_centromeres.bed"; do
  [ -s "data/$CELL/$f" ] || { echo "[setup:$CELL] MISSING data/$CELL/$f" >&2; missing=1; }
done
[ "$missing" = "1" ] && { echo "[setup:$CELL] copy the ChIA-PET inputs first" >&2; exit 1; }

MCOOL=$(ls "data/_hic/$CELL"/*.mcool 2>/dev/null | head -1 || true)
if [ -z "$MCOOL" ]; then
  echo "[setup:$CELL] fetching Hi-C"
  python -m validation fetch --manifest "validation/manifests/${CELL}_hic.json" --out data/_hic
  MCOOL=$(ls "data/_hic/$CELL"/*.mcool | head -1)
fi
echo "[setup:$CELL] mcool = $MCOOL"

# TAD boundaries only. Compartments and the accessibility signal feed the epigenome terms, which
# these runs leave off, and skipping the signal step means the ATAC bigWig is never needed.
if [ -s "data/$CELL/${CELL}_tads.bed" ]; then
  echo "[setup:$CELL] have data/$CELL/${CELL}_tads.bed"
else
  python -m validation tracks --cell "$CELL" --skip-compartments --skip-signal
fi

# The singleton reader fetches the balanced matrix as well as the raw counts, so the resolution
# it reads must carry balancing weights. GM12878's mcool ships with them at 25 kb; H1ESC's does
# not, and every chromosome then fails with "No column 'bins/weight' found" leaving an empty
# file. `validation tracks` already does this for the 10 kb level it calls TADs from, so this is
# the same one-time step at the resolution the singletons use.
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

# A previous failed run leaves a zero-row file, and prep_singletons treats any existing output as
# done, so it would skip regeneration for ever.
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
for field in ("data_anchors", "data_pet_clusters", "data_singletons", "data_ib_split",
              "data_segment_split", "data_centromeres"):
    name = getattr(s, field)
    assert name, f"{field} is unset"
    p = Path(s.data_path(name))
    assert p.is_file() and p.stat().st_size > 0, f"{field}: missing or empty {p}"
    print(f"[setup] ok {field}: {p.name} ({p.stat().st_size} bytes)")
PYCHECK

echo "[setup:$CELL] done. submit with:"
echo "  CELL=$CELL sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh"
