#!/bin/bash
# The readouts of the trio arms on the uniform chain, for every person and chromosome whose two
# arms are in the readout root, on a workstation. A task whose outputs are there is skipped, and a
# second call makes what the first did not.
#
#   ROOT=~/trio_xt bash playground/rnapii/trio_readouts.sh
#
# The root is that of redo_pull.sh with MAP="UC=C UR=R": out/lr/<PERSON>_C and out/lr/<PERSON>_R,
# data/<PERSON>/ with the RNAPII loops of the person and data/_hic_half/b/<PERSON>.mcool, half b of
# the map of the person from slurm/ensemble/trio_halves.sh. The arms read half a. When every task
# is read, trio_genome_summary.py adds the chromosomes up into out/readouts/trio_xt/trio_genome.json.
set -uo pipefail
export ROOT="${ROOT:-$HOME/trio_xt}"
export REPO="${REPO:-$ROOT/repo}"
export PY="${PY:-$HOME/3dgnome-ng/.venv/bin/python}"
export TAG="${TAG:-trio_xt}"
export C_ARM='out/lr/{S}_C' R_ARM='out/lr/{S}_R' MAP='data/_hic_half/b/{S}.mcool'
PEOPLE=(HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240)
cd "$ROOT"
mkdir -p logs
TASKS=""
for k in $(seq 0 197); do
  S="${PEOPLE[$((k % 9))]}"; C="chr$((k / 9 + 1))"
  [ "$(ls "out/lr/${S}_C/$C"/*.cif 2>/dev/null | wc -l)" -eq 10 ] || continue
  [ "$(ls "out/lr/${S}_R/$C"/*.cif 2>/dev/null | wc -l)" -eq 10 ] || continue
  TASKS="$TASKS $k"
done
echo "$(date +%H:%M:%S) tasks with both arms: $(echo $TASKS | wc -w) of 198"
echo $TASKS | tr ' ' '\n' | grep . | xargs -P 6 -I{} bash -c "bash $REPO/playground/rnapii/readouts_trio_genome.sh {} > logs/task_{}.log 2>&1 || echo 'task {} failed'"
n=$(ls "out/readouts/$TAG"/*_pooled.txt 2>/dev/null | wc -l)
if [ "$n" -eq 198 ]; then
  "$PY" "$REPO/playground/rnapii/trio_genome_summary.py" "out/readouts/$TAG" "out/readouts/$TAG/trio_genome.json" > "out/readouts/$TAG/trio_genome.txt" 2>&1
  tail -n 6 "out/readouts/$TAG/trio_genome.txt"
fi
echo "$(date +%H:%M:%S) trio readouts: $n pooled scores of 198"
echo "$(date +%H:%M:%S) end of the pass"
