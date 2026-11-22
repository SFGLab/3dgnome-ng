#!/bin/bash
# Pulls the finished chromosomes of the redo arms from the cluster into the readout root of a
# workstation. Run on the workstation in a session that can reach the cluster.
#
#   EDEN=user@host:/path/to/checkout/out/xu ROOT=~/xu bash playground/rnapii/redo_pull.sh
#
# The readout scripts know the arms E, C, R and M. The arms of slurm/ensemble/crosscell_arms.sh
# get these names in the root, written to $ROOT/out/lr/ARMS.txt:
#   UC, the uniform chain with CTCF loops             -> C
#   UR, the uniform chain with CTCF and RNAPII loops  -> R
#   C, the chain of loop anchors with CTCF loops      -> M
# E, the cudaMMC models, is linked from the root of the pass before, OLD. A chromosome of an arm is
# pulled when it holds its ten structures with their bead tables and no file changed in the last
# ten minutes. A chromosome that is there is not pulled again.
set -euo pipefail
EDEN="${EDEN:?set EDEN to user@host:/path/out/xu}"
ROOT="${ROOT:-$HOME/xu}"
OLD="${OLD:-$HOME/xc_lr}"
HOST="${EDEN%%:*}"; DIR="${EDEN#*:}"
mkdir -p "$ROOT/out/lr"
printf "E\tcudaMMC models of %s\nC\tUC of crosscell_arms.sh, the uniform chain with CTCF loops\nR\tUR, the uniform chain with CTCF and RNAPII loops\nM\tC, the chain of loop anchors with CTCF loops\n" "$OLD" > "$ROOT/out/lr/ARMS.txt"
for L in GM12878 H1ESC HFFC6; do [ -e "$ROOT/out/lr/${L}_E" ] || ln -s "$OLD/out/lr/${L}_E" "$ROOT/out/lr/${L}_E"; done
READY=$(ssh "$HOST" "cd '$DIR' && for d in *_C/chr* *_UC/chr* *_UR/chr*; do [ -d \"\$d\" ] || continue; [ \$(ls \$d/*.cif 2>/dev/null | wc -l) -ge 10 ] && [ \$(ls \$d/*.beads.tsv 2>/dev/null | wc -l) -ge 10 ] && [ -z \"\$(find \$d -mmin -10 -print -quit)\" ] && echo \$d; done")
n=0
for d in $READY; do
  arm=${d%%/*}; chr=${d#*/}; L=${arm%%_*}; A=${arm#*_}
  case "$A" in UC) T=C ;; UR) T=R ;; C) T=M ;; *) continue ;; esac
  dest="$ROOT/out/lr/${L}_${T}/$chr"
  [ "$(ls "$dest"/*.cif 2>/dev/null | wc -l)" -ge 10 ] && continue
  mkdir -p "$dest"
  rsync -a "$HOST:$DIR/$d/" "$dest/"
  n=$((n + 1))
done
echo "pulled $n chromosomes of arms, $(ls -d "$ROOT"/out/lr/*_[CRM]/chr* 2>/dev/null | wc -l) of 198 in the root"
