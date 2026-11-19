#!/bin/bash
# The smooth stage law spring on GM12878 with CTCF and RNAPII loops, workstation, screen lawspring.
#
#   bash slurm/ensemble/law_spring_trial.sh
#
# Phase 1, two 5 Mb regions, five structures per arm, arms law_weight 0, 0.01 and 0.1. Phase 2,
# whole chr1, three structures per arm, the same arms, scored on the 1-60 Mb slice. Each phase
# ends with the law readout and the battery, raw and balanced. Weights: the heat term sums over
# about two hundred partners per bead at 1 kb beads and a 100 kb range against two chain bonds,
# so 0.01 puts the law on the scale of the bonds and 0.1 above it. CHR1_ARMS replaces the arm
# list for phase 2 and SKIP_CHR1 stops after phase 1.
set -u
: "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
cd "$(dirname "$0")/../.."
C=GM12878
DATA="${DATA:-data}"
OUT="${OUT:-out/lawspring}"
ARMS="${ARMS:-0 0.01 0.1}"
MCOOL=$(ls $HIC_ROOT/$C/*.mcool | head -1)
SING=$DATA/$C/${C}_hic_25kb_singletons.bedpe
mkdir -p "$OUT"
git log --format="law spring trial on %h %s" -1

# An arm is a weight, or a weight and a range in bp joined by @, 0.1@400000 for example.
for A in $ARMS ${CHR1_ARMS:-}; do
  W=${A%@*}; RNG=100000; case $A in *@*) RNG=${A#*@} ;; esac
  sed "s/^anchors = .*/anchors = GM12878_anchors_ctcf_rnapii.bed/; s/^clusters = .*/clusters = GM12878_clusters_3+.bedpe,GM12878_rnapii_clusters_3+.bedpe\nfactors = CTCF,RNAPOL2/" \
    slurm/ensemble/gm12878_hic_arcs.ini > "$OUT/law_$A.ini"
  .venv/bin/python - "$OUT/law_$A.ini" "$W" "$RNG" <<'PY'
import configparser, sys
path, w, rng = sys.argv[1:4]
c = configparser.ConfigParser()
c.read(path)
c["simulation_arcs_smooth"]["law_weight"] = w
c["simulation_arcs_smooth"]["law_range_bp"] = rng
with open(path, "w") as fh:
    c.write(fh)
PY
done
grep -E "^anchors|^clusters|^factors|^law_" "$OUT"/law_*.ini

run() {  # region tag n
  for A in $ARMS; do
    D=$OUT/${2}_law_$A
    mkdir -p "$D"
    if [ "$(ls "$D"/*.cif 2>/dev/null | wc -l)" -ge "$3" ]; then continue; fi
    echo "[$2 law $A] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config "$OUT/law_$A.ini" --data-dir "$DATA/$C" --region "$1" -n "$3" --out "$D" > "$D.log" 2>&1
    echo "[$2 law $A] exit $? $(date +%H:%M) structures $(ls "$D"/*.cif 2>/dev/null | wc -l)"
  done
}

score() {  # region tag dirs...
  R=$1; T=$2; shift 2
  .venv/bin/python playground/law_spring_readout.py "$MCOOL" "$R" 1000 "$@" 2>&1 | grep -vE "INFO|WARN" | tee "$OUT/${T}_readout.txt"
  for B in no yes; do
    .venv/bin/python playground/validation_battery.py --balance $B --ev-factor 0.7 --singletons "$SING" "$MCOOL" "$R" 25000 "$@" 2>&1 \
      | grep -vE "INFO|WARN" > "$OUT/${T}_battery_balance_$B.txt"
  done
}

for R in chr1:20000000-25000000 chr1:10000000-15000000; do
  T=$(echo "$R" | tr ':-' '__')
  run "$R" "$T" 5
  score "$R" "$T" $(for A in $ARMS; do echo "$OUT/${T}_law_$A"; done)
done
echo "PHASE 1 DONE $(date)"

CHR1_ARMS="${CHR1_ARMS:-$ARMS}"
ARMS=$CHR1_ARMS
[ -n "${SKIP_CHR1:-}" ] && { echo "chr1 skipped"; exit 0; }
run chr1 chr1 3
for A in $ARMS; do
  .venv/bin/python playground/slice_arm.py "$OUT/chr1_law_$A" "$OUT/chr1_60mb_law_$A" chr1:1-60000000 > /dev/null
  cp "$OUT/chr1_law_$A.log" "$OUT/chr1_60mb_law_$A.log"
done
score chr1:1-60000000 chr1_60mb $(for A in $ARMS; do echo "$OUT/chr1_60mb_law_$A"; done)
echo "LAW SPRING TRIAL DONE $(date)"
