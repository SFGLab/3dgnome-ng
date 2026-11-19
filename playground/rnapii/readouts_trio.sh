#!/bin/bash -l
# Structure readouts of the trio arms against each person's own HGSVC Hi-C, one array task per
# person, CPU only. design/2026-11-11-rnapii-release.md.
#
#   sbatch -A <account> --array=0-8 playground/rnapii/readouts_trio.sh
#
# The trio arrays write <OUT>/<SAMPLE>/chr1 for both arms, so each arm is first sliced to a
# uniquely named directory, out/xc/trio/<SAMPLE>_<ARM>_60mb, and the readouts read those.
# Maps are read raw for every person so the nine are on one footing.

#SBATCH --job-name=xc_trio_read
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=short
#SBATCH --time=04:00:00

set -euo pipefail
ROOT="${ROOT:-$SLURM_SUBMIT_DIR}"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PEOPLE=(HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240)
S="${PEOPLE[${SLURM_ARRAY_TASK_ID:-0}]}"
REGION=chr1:1-60000000
OUT="$ROOT/out/readouts/trio"
mkdir -p "$OUT" "$ROOT/out/xc/trio"

ARMS=()
for pair in "C:out/xc_trio_ctcf_hgsvc" "R:out/xc_trio_rnapol2_hgsvc"; do
  A="${pair%%:*}"; D="$ROOT/${pair#*:}/$S/chr1"; DEST="$ROOT/out/xc/trio/${S}_${A}_60mb"
  if [ -d "$D" ]; then
    [ -d "$DEST" ] || python playground/slice_arm.py "$D" "$DEST" $REGION > /dev/null
    ARMS+=("$DEST")
  fi
done
echo "[$S] arms: ${ARMS[*]##*/}"
M="data/_hic/$S/$S.mcool"
python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons "data/$S/${S}_hgsvc_25kb_singletons.bedpe" \
  --json "$OUT/${S}_battery.json" "$M" $REGION 25000 "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN" > "$OUT/${S}_battery.txt"
python playground/saddle_offline.py --balance no --json "$OUT/${S}_saddle.json" "$M" $REGION 100000 \
  "data/$S/${S}_compartments.bedGraph" "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN" > "$OUT/${S}_saddle.txt"
python playground/rnapii/pileup.py --json "$OUT/${S}_pileup.json" "$M" $REGION 10000 \
  "data/$S/${S}_rnapol2_clusters_3+.bedpe" "${ARMS[@]}" > "$OUT/${S}_pileup.txt" 2>&1
echo "[$S] done"
tail -n 4 "$OUT/${S}_battery.txt" "$OUT/${S}_pileup.txt"
