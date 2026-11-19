#!/bin/bash -l

# One time setup of the chr1 ensemble: installs the environment, fetches the Hi-C and derives the
# inputs.
#
#   sbatch -A <account> slurm/ensemble/setup.sh
#
# Runs as a job. NumPy does not import on a login node with a CPU under x86-64-v2. The array does
# not install. Concurrent pip installs corrupt a shared venv. Needs network on the compute node.

#SBATCH --job-name=gm_chr1_setup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --time=04:00:00
# The account is given with sbatch -A. --output and --error are relative to the checkout, the
# directory of the submission. Create slurm/ensemble/logs before the submission.
#SBATCH --open-mode=append
#SBATCH --output=slurm/ensemble/logs/setup_%j.out
#SBATCH --error=slurm/ensemble/logs/setup_%j.out

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CELL=GM12878
MCOOL="$ROOT/data/_hic/$CELL/hic.4DNFIQ32RWCQ.mcool"
SINGLETONS="$ROOT/data/$CELL/${CELL}_chr1_hic_25kb_singletons.bedpe"
BINSIZE="${BINSIZE:-25000}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
# Progress reaches the slurm log at once.
export PYTHONUNBUFFERED=1

echo "[setup] node=$(hostname) python=$(python -V 2>&1)"
# The x86-64 level of the node.
LEVEL=$(awk '/^flags/{if(/avx512f/) print "v4"; else if(/avx2/) print "v3";
                      else if(/sse4_2/) print "v2"; else print "v1"; exit}' /proc/cpuinfo)
echo "[setup] cpu x86-64 level = $LEVEL (NumPy wheels need v2 or better)"
if [ "$LEVEL" = "v1" ]; then
  echo "[setup] this node is below the NumPy baseline too; pick a newer partition" >&2
  exit 1
fi

# Not the validation extra, whose hicrep and pyBigWig build from source. scipy, cooler and cooltools
# are needed.
if [ "$SKIP_INSTALL" != "1" ]; then
  echo "[setup] installing"
  pip install --upgrade pip
  pip install -e .
  pip install "jax[cuda12]"
  pip install "scipy>=1.10" "cooler>=0.9" "cooltools>=0.7"
else
  echo "[setup] SKIP_INSTALL=1, leaving the environment alone"
fi

python -c "
import jax, numpy, gnome3d
print(f'[setup] numpy {numpy.__version__}  jax {jax.__version__}')
print(f'[setup] jax devices: {jax.devices()}')
"

# The ChIA-PET anchors and loops, the segment breakpoints and the centromeres have no download and
# must be copied in.
missing=0
for f in "${CELL}_anchors_3+_oriented.bed" "${CELL}_clusters_3+.bedpe" \
         "ccds_all_hg38_merged100k_${CELL}.breakpoints.bed" "hg38_centromeres.bed"; do
  if [ -s "data/$CELL/$f" ]; then
    echo "[setup] have data/$CELL/$f"
  else
    echo "[setup] MISSING data/$CELL/$f" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo "[setup] copy the ChIA-PET inputs first, they are about 12 MB in total:" >&2
  echo "  rsync -av <source>:<checkout>/data/$CELL/{\"${CELL}_anchors_3+_oriented.bed\"," >&2
  echo "    \"${CELL}_clusters_3+.bedpe\",ccds_all_hg38_merged100k_${CELL}.breakpoints.bed," >&2
  echo "    hg38_centromeres.bed} $ROOT/data/$CELL/" >&2
  exit 1
fi

# Each step skips work that is done.
if [ -s "$MCOOL" ]; then
  echo "[setup] have $MCOOL"
else
  echo "[setup] fetching Hi-C"
  python -m legacy.validation fetch --manifest legacy/validation/manifests/${CELL}_hic.json --out data/_hic
fi

# TAD boundaries only.
echo "[setup] calling TADs"
python -m legacy.validation tracks --cell "$CELL" --skip-compartments

echo "[setup] building Hi-C singletons at ${BINSIZE}bp"
python slurm/ensemble/prep_singletons.py \
  --mcool "$MCOOL" --region chr1 --binsize "$BINSIZE" --out "$SINGLETONS"

# Checks what the array job reads.
python - "$ROOT/slurm/ensemble/gm12878_chr1_hic_tads.ini" <<'PYCHECK'
import sys
from pathlib import Path

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
for field in ("data_anchors", "data_pet_clusters", "data_singletons",
              "data_segment_split", "data_centromeres"):
    name = getattr(s, field)          # attribute, not .get - a typo here must fail loudly
    assert name, f"{field} is unset in the config"
    p = Path(s.data_path(name))
    assert p.is_file() and p.stat().st_size > 0, f"{field}: missing or empty {p}"
    print(f"[setup] ok {field}: {p.name} ({p.stat().st_size} bytes)")
PYCHECK

echo "[setup] done. submit with: sbatch --array=0-99%20 slurm/ensemble/chr1_ensemble.sh"
