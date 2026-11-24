#!/bin/bash
# The halves of the Hi-C maps of the nine people of the trios for the arms on the uniform chain, on
# a workstation. design/uniform-chain.md.
#
#   HIC=~/trio_lr/data/_hic OUT=~/gapfill-data/_hic_trio bash slurm/ensemble/trio_halves.sh
#
# HIC holds <PERSON>/<PERSON>.mcool. The maps have balancing weights from 25 kb up and none at 5 and
# 10 kb, which the halves and the contact cells read. Each map is copied to OUT/whole and the copy
# is balanced at 5 and 10 kb with the settings of its weights at 25 kb, the defaults of cooler. The
# copy is split by playground/thin_cooler.py --chromosomes on chr1 to chr22 into OUT/a and OUT/b,
# three people at once. Half a is data/_hic_half/a/<PERSON>.mcool of the checkout that runs the
# arms. Half b is the map of the score. A person whose half b exists is skipped.
set -euo pipefail
HIC="${HIC:-$HOME/trio_lr/data/_hic}"
OUT="${OUT:-$HOME/gapfill-data/_hic_trio}"
BIN="${BIN:-.venv/bin}"
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22"
mkdir -p "$OUT/logs" "$OUT/whole"
one() {
  local S=$1 W="$OUT/whole/$1.mcool"
  [ -s "$OUT/b/$S.mcool" ] && { echo "[skip] $S"; return 0; }
  echo "[start] $S $(date +%H:%M)"
  [ -s "$W" ] || { cp "$HIC/$S/$S.mcool" "$W.part"; mv "$W.part" "$W"; }
  for RES in 5000 10000; do
    "$BIN/python" -c "import cooler, sys; sys.exit(0 if 'weight' in cooler.Cooler(sys.argv[1]).bins().columns else 1)" "$W::/resolutions/$RES" \
      || "$BIN/cooler" balance -p 8 "$W::/resolutions/$RES" >> "$OUT/logs/$S.log" 2>&1
  done
  # shellcheck disable=SC2086
  "$BIN/python" playground/thin_cooler.py --chromosomes "$W" "$OUT" "$S" $CHROMS >> "$OUT/logs/$S.log" 2>&1
  echo "[done] $S $(date +%H:%M) $(tail -n 1 "$OUT/logs/$S.log")"
}
export -f one
export HIC OUT BIN CHROMS
echo HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240 | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'one {} || echo "[failed] {}"'
echo "trio halves DONE $(date)"
