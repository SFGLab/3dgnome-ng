#!/bin/bash
# Micro-C loops and the models at them, one line and chromosome a call, for the whole genome.
# design/microc-loops.md has the method. microc_loops.sh is the same on chr1:0-60 Mb.
#
#   ROOT=~/xc_lr bash playground/rnapii/microc_genome.sh H1ESC chr7
#
# Three steps, each skipped when its output is there. The chromosome is cut out of the line's
# full Micro-C map at 5 kb, raw counts. Loops are called on the cut with the dots caller of
# cooltools, at 5 kb and at 10 kb. The four model sets are then scored at the 5 kb loops with
# the signal floor of 15 the chr1 tables use. Each chromosome is balanced and has its expected
# counts taken on its own, as the chr1 region was.
#
# Expects $ROOT/microc/ with the three maps, H1ESC and HFFC6 as Juicer .hic of 4DN and GM12878
# as the 50 bp cooler of Hong et al., $ROOT/out/lr/<LINE>_<ARM>/<chr>, $ROOT/data/<LINE>/ with
# the run's loop files, the code under $REPO, $PY with cooltools and $STRAW with hic-straw.
set -euo pipefail
ROOT="${ROOT:-$HOME/xc_lr}"
REPO="${REPO:-$ROOT/repo}"
PY="${PY:-$HOME/3dgnome-ng/.venv/bin/python}"
STRAW="${STRAW:-$HOME/3dgnome-web-smoke/strawvenv/bin/python}"
L=$1; CHR=$2
OUT="$ROOT/out/readouts/microc"
SLICES="$ROOT/microc/slices"
mkdir -p "$OUT" "$SLICES" "$ROOT/out/links"
export PYTHONPATH="$REPO"
cd "$ROOT"

# hg38 autosome lengths in bp.
declare -A LEN=([chr1]=248956422 [chr2]=242193529 [chr3]=198295559 [chr4]=190214555 [chr5]=181538259 [chr6]=170805979
  [chr7]=159345973 [chr8]=145138636 [chr9]=138394717 [chr10]=133797422 [chr11]=135086622 [chr12]=133275309
  [chr13]=114364328 [chr14]=107043718 [chr15]=101991189 [chr16]=90338345 [chr17]=83257441 [chr18]=80373285
  [chr19]=58617616 [chr20]=64444167 [chr21]=46709983 [chr22]=50818468)
REGION="$CHR:0-${LEN[$CHR]}"
SLICE="$SLICES/microc_${L}_${CHR}_5000.npz"

if [ ! -s "$SLICE" ]; then
  case "$L" in
    H1ESC) "$STRAW" "$REPO/playground/rnapii/hic_slice.py" "$ROOT/microc/H1ESC_4DNFI2TK7L2F.hic" "$REGION" 5000 "$SLICE.part.npz" ;;
    HFFC6) "$STRAW" "$REPO/playground/rnapii/hic_slice.py" "$ROOT/microc/HFFC6_4DNFI18Q799K.hic" "$REGION" 5000 "$SLICE.part.npz" ;;
    GM12878) "$PY" "$REPO/playground/rnapii/cool_slice.py" "$ROOT/microc/GM12878_microc.50.cool" "$REGION" 5000 "$SLICE.part.npz" ;;
    *) echo "no Micro-C map for $L"; exit 1 ;;
  esac
  mv "$SLICE.part.npz" "$SLICE"
fi

[ -s "$OUT/${L}_${CHR}_5000.bedpe" ] || "$PY" "$REPO/playground/rnapii/microc_call_loops.py" "$SLICE" "$CHR" "$OUT/${L}_${CHR}" "${NPROC:-4}" > "$OUT/${L}_${CHR}_call.log" 2>&1
echo "[$L $CHR] loops called: $(wc -l < "$OUT/${L}_${CHR}_5000.bedpe") at 5 kb, $(wc -l < "$OUT/${L}_${CHR}_10000.bedpe") at 10 kb"

if [ ! -s "$OUT/${L}_${CHR}_5000.json" ]; then
  DIRS=()
  for A in E C R M; do
    DEST="$ROOT/out/links/${L}_${A}_${CHR}"
    [ -e "$DEST" ] || ln -s "$ROOT/out/lr/${L}_${A}/$CHR" "$DEST"
    DIRS+=("$DEST")
  done
  # The scoring holds a chromosome's bin by bin distances, 31 GB for chr3 and more for chr1 and
  # chr2. The six largest chromosomes take it one at a time, whatever runs beside this call.
  HOLD=()
  case "$CHR" in chr1|chr2|chr3|chr4|chr5|chr6) HOLD=(flock "$ROOT/out/microc_large.lock") ;; esac
  "${HOLD[@]}" "$PY" "$REPO/playground/rnapii/microc_loop_compare.py" --json "$OUT/${L}_${CHR}_5000.json.part" "$SLICE" "$OUT/${L}_${CHR}_5000.bedpe" \
    "ctcf=$ROOT/data/$L/${L}_xc_ctcf_clusters_3+.bedpe" "rnapii=$ROOT/data/$L/${L}_xc_rnapii_clusters_3+.bedpe" min_z=15 \
    "${DIRS[@]}" > "$OUT/${L}_${CHR}_5000.txt" 2>&1
  mv "$OUT/${L}_${CHR}_5000.json.part" "$OUT/${L}_${CHR}_5000.json"
fi
tail -n 3 "$OUT/${L}_${CHR}_5000.txt"
