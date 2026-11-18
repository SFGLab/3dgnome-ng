#!/bin/bash
# Whole genome readouts of the cross line RNAPII comparison, one chromosome a call, on a
# workstation. The models are eden's out/lr, arms C, R and M of the three lines, ten
# conformations a chromosome, and arm E, a seeded ten of the hundred cudaMMC models a chromosome
# from team-faramir's models_ccd, written as cifs by `e3d_export.py --hcm-to-cif`.
#
#   ROOT=~/xc_lr bash playground/rnapii/readouts_crossline_genome.sh expression chr7
#   ROOT=~/xc_lr bash playground/rnapii/readouts_crossline_genome.sh structure chr7
#   ROOT=~/xc_lr bash playground/rnapii/readouts_crossline_genome.sh pairs chr7
#
# expression: every arm of every line exported to the enhancer3d layout, the genes covered in
#   all twelve ensembles, then the cross line command once per arm, under out/readouts/e3d_<arm>/<chr>.
# structure: the summed contact map of each arm against the line's deep map at 25 kb, and the
#   pile-up at the line's RNAPII loops at 10 kb, under out/readouts/structure.
# pairs: the enhancer to promoter pairs an RNAPII loop alone joins and their reference pairs,
#   under out/readouts/pairs. It needs the bead tables beside the structures and the expression
#   step's gene list of the chromosome.
#
# Expects $ROOT/out/lr/<LINE>_<ARM>/<chr>, $ROOT/data/<LINE>/ with the run's loop files,
# $ROOT/genes/genes_<chr>.tsv, the code under $REPO and $E3DX, the deep maps under $HIC/<LINE>/
# and the enhancer3d inputs under $E3D. A step whose output is there is skipped.
set -euo pipefail
ROOT="${ROOT:-$HOME/xc_lr}"
REPO="${REPO:-$ROOT/repo}"
PY="${PY:-$HOME/3dgnome-ng/.venv/bin/python}"
HIC="${HIC:-$HOME/gapfill-data/_hic}"
E3D="${E3D:-$HOME/enhancer3d/data}"
E3DX="${E3DX:-$ROOT/e3dx}"
OUT="$ROOT/out/readouts"
STEP=$1; CHR=$2
LINES=(GM12878 H1ESC HFFC6)
ARMS=(E C R M)
export PYTHONPATH="$REPO"
cd "$ROOT"

expression() {
  local GENES="$ROOT/genes/genes_$CHR.tsv" COMMON="$OUT/genes/${CHR}_common.tsv" DIRS=()
  mkdir -p "$OUT/genes"
  for A in "${ARMS[@]}"; do for L in "${LINES[@]}"; do
    local SRC="$ROOT/out/lr/${L}_${A}/$CHR" DST="$ROOT/out/e3d/${L}_${A}/$CHR"
    ls "$DST"/*.txt > /dev/null 2>&1 || "$PY" "$REPO/playground/rnapii/e3d_export.py" "$SRC" "$DST" "$CHR" --min-covered 0.2 | tail -n 1
    DIRS+=("$DST")
  done; done
  [ -s "$COMMON" ] || "$PY" "$REPO/playground/rnapii/e3d_export.py" --common-genes "$GENES" "$COMMON" "${DIRS[@]}"
  for A in "${ARMS[@]}"; do
    [ -s "$OUT/e3d_$A/$CHR/summary.tsv" ] && continue
    local MODELS=() ATL=()
    for L in "${LINES[@]}"; do
      MODELS+=(--model "$L=$ROOT/out/e3d/${L}_${A}/$CHR"); ATL+=(--atlas "$L=$E3D/enhancer_atlas/${L}_enhancers_hg38.bed")
    done
    mkdir -p "$OUT/e3d_$A"
    (cd "$E3DX" && "$PY" -m enhancer3d.cli "${MODELS[@]}" --genes "$COMMON" --enh-mode atlas "${ATL[@]}" \
      --deseq "GM12878:H1ESC=$E3D/deseq/gm12878_vs_h1esc_results.parquet" \
      --deseq "H1ESC:HFFC6=$E3D/deseq/h1esc_vs_hffc6_results.parquet" \
      --deseq "GM12878:HFFC6=$E3D/deseq/gm12878_vs_hffc6_results.parquet" \
      --chrom "$CHR" --n-models 10 --out "$OUT/e3d_$A/$CHR") 2>&1 | tail -n 12 > "$OUT/e3d_$A/$CHR.txt"
    tail -n 3 "$OUT/e3d_$A/$CHR.txt"
  done
}

structure() {
  mkdir -p "$OUT/structure" "$ROOT/out/links"
  for L in "${LINES[@]}"; do
    [ -s "$OUT/structure/${L}_${CHR}_pooled.txt" ] && [ -s "$OUT/structure/${L}_${CHR}_pileup.npz" ] && continue
    local M; M=$(ls "$HIC/$L"/*.mcool | head -1)
    local DIRS=()
    for A in "${ARMS[@]}"; do
      local DEST="$ROOT/out/links/${L}_${A}_${CHR}"
      [ -e "$DEST" ] || ln -s "$ROOT/out/lr/${L}_${A}/$CHR" "$DEST"
      DIRS+=("$DEST")
    done
    "$PY" "$REPO/playground/ensemble_score.py" "$M" "$CHR" "${DIRS[@]}" 2>&1 | grep -vE "INFO|WARN" > "$OUT/structure/${L}_${CHR}_pooled.txt.part"
    mv "$OUT/structure/${L}_${CHR}_pooled.txt.part" "$OUT/structure/${L}_${CHR}_pooled.txt"
    "$PY" "$REPO/playground/rnapii/pileup.py" --json "$OUT/structure/${L}_${CHR}_pileup.json" "$M" "$CHR" 10000 \
      "$ROOT/data/$L/${L}_xc_rnapii_clusters_3+.bedpe" "${DIRS[@]}" 2>&1 | grep -vE "INFO|WARN" | tail -n 4
  done
}

pairs() {
  mkdir -p "$OUT/pairs"
  for L in "${LINES[@]}"; do
    [ -s "$OUT/pairs/ep_${L}_${CHR}.tsv" ] && continue
    EP_CHROM="$CHR" "$PY" "$REPO/playground/rnapii/ep_pairs.py" "$L" "$ROOT/out/lr" "$ROOT/data/$L/${L}_xc_ctcf_clusters_3+.bedpe" \
      "$ROOT/data/$L/${L}_xc_rnapii_clusters_3+.bedpe" "$E3D/enhancer_atlas/${L}_enhancers_hg38.bed" \
      "$ROOT/genes/genes_$CHR.tsv" "$OUT/genes/${CHR}_common.tsv" "$OUT/pairs/ep_${L}_${CHR}.tsv.part"
    mv "$OUT/pairs/ep_${L}_${CHR}.tsv.part" "$OUT/pairs/ep_${L}_${CHR}.tsv"
  done
}

"$STEP"
