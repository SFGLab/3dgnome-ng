#!/bin/bash
# Readouts of the cross line RNAPII comparison, design/2026-11-11-rnapii-release.md.
#
#   ROOT=~/xc_readout bash playground/rnapii/readouts_crossline.sh structure GM12878
#   ROOT=~/xc_readout bash playground/rnapii/readouts_crossline.sh expression
#
# structure <LINE>: battery, saddle and pile-up on the line's arms E, C, R and M, chr1:1-60 Mb,
# against the deep 4DN map, JSON under $ROOT/out/readouts. H1ESC's map has no weights, so its
# saddle reads raw counts.
# expression: every arm of every line exported to the enhancer3d layout, then the cross line
# CLI once per arm on the three lines, summary under $ROOT/out/readouts/e3d_<arm>.
#
# Expects $ROOT/repo (this repository), $ROOT/out/xc/<LINE>_<ARM>/chr1 and <LINE>_<ARM>_60mb,
# $ROOT/loops/<LINE>_xc_rnapii_clusters_3+.bedpe, the engine venv at $PY, the deep maps under
# $HIC/<LINE>/, the production data under $DATA/<LINE>/, and the enhancer3d inputs under $E3D.
set -euo pipefail
ROOT="${ROOT:-$HOME/xc_readout}"
PY="${PY:-$HOME/3dgnome-ng/.venv/bin/python}"
HIC="${HIC:-/mnt/storagelinux/_hic}"
DATA="${DATA:-$HOME/3dgnome-ng/data}"
E3D="${E3D:-$HOME/enhancer3d/data}"
E3DX="${E3DX:-$ROOT/e3dx}"
OUT="$ROOT/out/readouts"
REGION=chr1:1-60000000
mkdir -p "$OUT"
cd "$ROOT/repo"
export PYTHONPATH="$ROOT/repo"

structure() {
  local L=$1
  local M; M=$(ls "$HIC/$L"/*.mcool | head -1)
  local BAL=(); [ "$L" = "H1ESC" ] && BAL=(--balance no)
  local ARMS=()
  for A in E C R M; do [ -d "$ROOT/out/xc/${L}_${A}_60mb" ] && ARMS+=("$ROOT/out/xc/${L}_${A}_60mb"); done
  echo "[$L] arms: ${ARMS[*]##*/}"
  "$PY" playground/validation_battery.py --balance no --ev-factor 0.7 --singletons "$DATA/$L/${L}_hic_25kb_singletons.bedpe" \
    --json "$OUT/${L}_battery.json" "$M" $REGION 25000 "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN" | tee "$OUT/${L}_battery.txt" | tail -n 8
  "$PY" playground/saddle_offline.py "${BAL[@]}" --json "$OUT/${L}_saddle.json" "$M" $REGION 100000 \
    "$DATA/$L/${L}_compartments.bedGraph" "${ARMS[@]}" 2>&1 | grep -vE "INFO|WARN" | tee "$OUT/${L}_saddle.txt" | tail -n 6
  "$PY" playground/rnapii/pileup.py --json "$OUT/${L}_pileup.json" "$M" $REGION 10000 \
    "$ROOT/loops/${L}_xc_rnapii_clusters_3+.bedpe" "${ARMS[@]}" 2>&1 | tee "$OUT/${L}_pileup.txt" | tail -n 6
}

expression() {
  local GENES="$OUT/genes_chr1.tsv"
  [ -s "$GENES" ] || awk -F'\t' 'BEGIN{OFS="\t"; print "gene_id","chrom","start","end","strand"}
    $3=="gene" && $1=="chr1" {match($9,/gene_id "[^"]+"/); id=substr($9,RSTART+9,RLENGTH-10); print id,$1,$4-1,$5,$7}' \
    "$E3D/gencode.v40.basic.annotation.gtf" > "$GENES"
  for A in E C R M; do
    local MODELS=() ATL=()
    for L in GM12878 H1ESC HFFC6; do
      local SRC="$ROOT/out/xc/${L}_${A}/chr1" DST="$ROOT/out/e3d/${L}_${A}"
      [ -d "$DST" ] || "$PY" playground/rnapii/e3d_export.py "$SRC" "$DST" chr1 | tail -n 2
      MODELS+=(--model "$L=$DST"); ATL+=(--atlas "$L=$E3D/enhancer_atlas/${L}_enhancers_hg38.bed")
    done
    (cd "$E3DX" && "$PY" -m enhancer3d.cli "${MODELS[@]}" --genes "$GENES" --enh-mode atlas "${ATL[@]}" \
      --deseq "GM12878:H1ESC=$E3D/deseq/gm12878_vs_h1esc_results.parquet" \
      --deseq "H1ESC:HFFC6=$E3D/deseq/h1esc_vs_hffc6_results.parquet" \
      --deseq "GM12878:HFFC6=$E3D/deseq/gm12878_vs_hffc6_results.parquet" \
      --chrom chr1 --n-models 10 --out "$OUT/e3d_$A") 2>&1 | tail -n 30 | tee "$OUT/e3d_$A.txt"
  done
}

case "${1:-}" in
  structure) structure "$2" ;;
  expression) expression ;;
  *) echo "usage: $0 structure <LINE> | expression" >&2; exit 2 ;;
esac
