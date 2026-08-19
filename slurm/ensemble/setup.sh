#!/bin/bash -l

# One-time setup for the chr1 ensemble: install the environment, then fetch and derive every
# input the array job needs. Run this once and watch it finish before submitting the array.
#
#   sbatch slurm/ensemble/setup.sh
#
# Why this is a job rather than login-node commands. Installing on the login node failed with
#
#   RuntimeError: NumPy was built with baseline optimizations: (X86_V2)
#                 but your machine doesn't support: (X86_V2)
#
# which says the login node's CPU is older than the x86-64-v2 baseline current NumPy wheels are
# built against. NumPy therefore cannot even import there, so neither the build nor the data
# preparation can run on that node. A GPU compute node is new enough. The check below prints the
# node's level so the next failure of this kind is legible rather than cryptic.
#
# Why setup is not folded into the array job. The array runs up to 100 tasks against one venv,
# and concurrent pip installs into a shared prefix corrupt it. Installing once, here, keeps the
# array read-only with respect to the environment.
#
# Both steps need outbound network from the compute node. If the cluster blocks that, run the
# install with a pip cache or mirror the lab already provides, and copy the mcool in by hand.

#SBATCH --job-name=gm_chr1_setup
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --account=sfglab
# --output/--error are absolute and cannot use $ROOT: sbatch parses #SBATCH lines before
# the shell runs, and a relative path would resolve against whatever directory you
# submitted from. The directory must already exist, because slurm opens these files
# before the script's own mkdir would run. Create it once:
#   mkdir -p $ROOT/slurm/ensemble/logs
# Running from a different checkout means also passing --output/--error to sbatch.
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/setup_%j.out
#SBATCH --error=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/slurm/ensemble/logs/setup_%j.out

set -euo pipefail

ROOT="${ROOT:-/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng}"
CELL=GM12878
MCOOL="$ROOT/data/_hic/$CELL/hic.4DNFIQ32RWCQ.mcool"
SINGLETONS="$ROOT/data/$CELL/${CELL}_chr1_hic_25kb_singletons.bedpe"
BINSIZE="${BINSIZE:-25000}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

cd "$ROOT"
mkdir -p slurm/ensemble/logs
source .venv/bin/activate
# Long stages print progress with plain print(); block-buffered into a slurm
# log that makes a working job look hung for many minutes.
export PYTHONUNBUFFERED=1

echo "[setup] node=$(hostname) python=$(python -V 2>&1)"
# x86-64 microarchitecture level, the thing the NumPy failure above is really about.
LEVEL=$(awk '/^flags/{if(/avx512f/) print "v4"; else if(/avx2/) print "v3";
                      else if(/sse4_2/) print "v2"; else print "v1"; exit}' /proc/cpuinfo)
echo "[setup] cpu x86-64 level = $LEVEL (NumPy wheels need v2 or better)"
if [ "$LEVEL" = "v1" ]; then
  echo "[setup] this node is below the NumPy baseline too; pick a newer partition" >&2
  exit 1
fi

# ---- environment ---------------------------------------------------------------------------
# Deliberately NOT `pip install -e ".[validation]"`. That extra also pulls hicrep and pyBigWig,
# both of which build from an sdist whose setup.py imports numpy, which is where the login-node
# install died. Neither is needed here: hicrep is declared but never imported anywhere in
# validation/, and pyBigWig is imported lazily by the ATAC signal path, which this ensemble
# skips. scipy, cooler and cooltools are the real requirements - scipy because
# validation/studies/synthetic.py imports it at module level and the CLI loads every study,
# cooler and cooltools for the mcool and the insulation call.
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

# ---- inputs that cannot be downloaded -------------------------------------------------------
# The manifests under validation/manifests/ cover Hi-C and epigenomic signal only. The ChIA-PET
# anchors and loops, the segment breakpoints and the centromeres have no fetch path and must be
# copied in. Checked first, because everything below is pointless without them.
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

# ---- fetch and derive ------------------------------------------------------------------------
# All three are idempotent, so a rerun after a partial failure is free.
if [ -s "$MCOOL" ]; then
  echo "[setup] have $MCOOL"
else
  echo "[setup] fetching Hi-C"
  python -m validation fetch --manifest validation/manifests/${CELL}_hic.json --out data/_hic
fi

# Only the TAD boundaries are wanted. Compartments and the accessibility signal feed the
# epigenome terms, which this ensemble runs without, and skipping the signal step means the ATAC
# bigWig never has to be fetched and pyBigWig never has to be installed.
echo "[setup] calling TADs"
python -m validation tracks --cell "$CELL" --skip-compartments --skip-signal

echo "[setup] building Hi-C singletons at ${BINSIZE}bp"
python slurm/ensemble/prep_singletons.py \
  --mcool "$MCOOL" --region chr1 --binsize "$BINSIZE" --out "$SINGLETONS"

# ---- verify what the array job will look for -------------------------------------------------
python - "$ROOT/slurm/ensemble/gm12878_chr1_hic_tads.ini" <<'PYCHECK'
import sys
from pathlib import Path

from gnome3d.settings import Settings

s = Settings()
assert s.load_ini(sys.argv[1]), f"cannot load {sys.argv[1]}"
for field in ("data_anchors", "data_pet_clusters", "data_singletons", "data_ib_split",
              "data_segment_split", "data_centromeres"):
    name = getattr(s, field)          # attribute, not .get - a typo here must fail loudly
    assert name, f"{field} is unset in the config"
    p = Path(s.data_path(name))
    assert p.is_file() and p.stat().st_size > 0, f"{field}: missing or empty {p}"
    print(f"[setup] ok {field}: {p.name} ({p.stat().st_size} bytes)")
PYCHECK

echo "[setup] done. submit with: sbatch --array=0-99%20 slurm/ensemble/chr1_ensemble.sh"
