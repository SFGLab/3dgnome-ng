#!/bin/bash
# Micro-C loops and the models at them, H1ESC and HFFC6, chr1:0-60 Mb. design/microc-loops.md.
#
#   ROOT=/mnt/storagelinux/xc_readout_lr bash playground/rnapii/microc_loops.sh
#
# ROOT holds repo/, out/xc/<LINE>_<ARM>_60mb and loops/. The slices are the ones hic_slice.py cut
# for microc_readout.py. Loops are called once per line, a line whose bedpe exists is not called
# again. Writes ROOT/out/<TAG>/<LINE>_<bin size>.{bedpe,json,txt}, TAG microc_loops unless set.
set -u
ROOT="${ROOT:?}"
PY="${PY:-$HOME/3dgnome-ng/.venv/bin/python}"
DATA="${DATA:-$HOME/3dgnome-ng/data}"
SLICES="${SLICES:-$HOME/xc_readout/microc}"
OUT="$ROOT/out/${TAG:-microc_loops}"
mkdir -p "$OUT"
cd "$ROOT/repo"
for L in H1ESC HFFC6; do
  [ -s "$OUT/${L}_5000.bedpe" ] || PYTHONPATH="$ROOT/repo" "$PY" playground/rnapii/microc_call_loops.py \
    "$SLICES/microc_${L}_5000.npz" chr1 "$OUT/$L" 8 > "$OUT/${L}_call.log" 2>&1
  for B in 5000 10000; do
    PYTHONPATH="$ROOT/repo" "$PY" playground/rnapii/microc_loop_compare.py --json "$OUT/${L}_$B.json" \
      "$SLICES/microc_${L}_5000.npz" "$OUT/${L}_$B.bedpe" \
      "ctcf=$DATA/$L/${L}_xc_ctcf_clusters_3+.bedpe" "rnapii=$ROOT/loops/${L}_xc_rnapii_clusters_3+.bedpe" \
      "$ROOT/out/xc/${L}_E_60mb" "$ROOT/out/xc/${L}_C_60mb" "$ROOT/out/xc/${L}_R_60mb" "$ROOT/out/xc/${L}_M_60mb" \
      > "$OUT/${L}_$B.txt" 2>&1
  done
done
echo EXIT > "$OUT/done.txt"
