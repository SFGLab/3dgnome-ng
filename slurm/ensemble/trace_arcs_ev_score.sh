#!/bin/bash
# The scores of the arms of trace_arcs_ev.sh and trace_arcs_arm.sh on chr19 of a cell line against
# half b, beside the two arms of the genome runs, the uniform chain with CTCF loops and with CTCF
# and RNAPII loops. CELL is GM12878 unless given. SCORES is the folder of the tables, OUT/scores.
#
#   DATA=~/gapfill-data OUT=~/3dgnome-gapfill/out/arcs_ev XU=~/xu/out/lr XUN=~/xun/out/lr \
#     ARMS="base noev r133" bash slurm/ensemble/trace_arcs_ev_score.sh
#
# XU holds the genome runs with the loop dropout and XUN, when given, those without it.
# Four tables go to the folder of the scores. bands.txt has the SCC per bead pair by band of
# separation over every bin. trace.txt has the same over the bins with beads in every arm, with
# the classes of pairs 100 to 500 kb apart. ends.txt has the distances of the ends of the loops
# with the overlaps. end_pairs.txt has the contacts of pairs of beads by the loop ends at them.
set -u
cd "$(dirname "$0")/../.."
DATA="${DATA:-data}"
OUT="${OUT:-out/arcs_ev}"
XU="${XU:-out/xu/lr}"
PY=.venv/bin/python
[ -x "$PY" ] || PY="${PYTHON:?no .venv here, set PYTHON}"
CELL="${CELL:-GM12878}"
MAP="$DATA/_hic_halves/b/$CELL.mcool"
CTCF="$DATA/$CELL/${CELL}_xc_ctcf_clusters_3+.bedpe"
RNAPII="$DATA/$CELL/${CELL}_xc_rnapii_clusters_3+.bedpe"
S="${SCORES:-$OUT/scores}"
mkdir -p "$S"
DIRS=("$XU/${CELL}_C/chr19" "$XU/${CELL}_R/chr19")
NAMED=("ctcf=$XU/${CELL}_C/chr19" "rnapii=$XU/${CELL}_R/chr19")
if [ -n "${XUN:-}" ]; then
  NAMED+=("ctcf_nodrop=$XUN/${CELL}_C/chr19" "rnapii_nodrop=$XUN/${CELL}_R/chr19")
fi
for A in ${ARMS:-base noev r133}; do
  DIRS+=("$OUT/lr/${CELL}_$A/chr19")
  NAMED+=("$A=$OUT/lr/${CELL}_$A/chr19")
done
"$PY" playground/rnapii/short_range_scc.py "$MAP" chr19 "$S/bands.json" "${DIRS[@]}" > "$S/bands.txt" 2>&1 &
"$PY" playground/rnapii/loop_end_distances.py chr19 "$CTCF" "$RNAPII" "${DIRS[@]}" > "$S/ends.txt" 2>&1 &
"$PY" playground/rnapii/band_trace.py "$MAP" chr19 "$CTCF" "$RNAPII" "$S/trace.json" "${NAMED[@]}" > "$S/trace.txt" 2>&1 &
"$PY" playground/rnapii/end_pair_contacts.py chr19 "$CTCF" "$RNAPII" "$S/end_pairs.json" "${NAMED[@]}" > "$S/end_pairs.txt" 2>&1 &
wait
echo "trace arcs ev score DONE $(date) ${ARMS:-base noev r133}" >> "$S/done.log"
