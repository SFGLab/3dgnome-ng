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
#
# MAP gives other arms their names. The root of the runs without the loop dropout has UCN as C and
# URN as R, with M linked from the root BESIDE:
#
#   MAP="UCN=C URN=R" BESIDE=~/xu ROOT=~/xun EDEN=... bash playground/rnapii/redo_pull.sh
#
# The arms of the nine people of the trios have no arm E, OLD=none:
#
#   MAP="UC=C UR=R" OLD=none ROOT=~/trio_xt EDEN=user@host:/path/out/xt bash playground/rnapii/redo_pull.sh
set -euo pipefail
EDEN="${EDEN:?set EDEN to user@host:/path/out/xu}"
ROOT="${ROOT:-$HOME/xu}"
OLD="${OLD:-$HOME/xc_lr}"
MAP="${MAP:-UC=C UR=R C=M}"
BESIDE="${BESIDE:-}"
HOST="${EDEN%%:*}"; DIR="${EDEN#*:}"
mkdir -p "$ROOT/out/lr"
{
  [ "$OLD" = none ] || printf "E\tcudaMMC models of %s\n" "$OLD"
  for m in $MAP; do printf "%s\t%s of crosscell_arms.sh\n" "${m#*=}" "${m%%=*}"; done
  [ -z "$BESIDE" ] || printf "M\tM of %s\n" "$BESIDE"
} > "$ROOT/out/lr/ARMS.txt"
GLOB=""
for m in $MAP; do GLOB="$GLOB *_${m%%=*}/chr*"; done
for L in GM12878 H1ESC HFFC6; do
  [ "$OLD" = none ] || [ -e "$ROOT/out/lr/${L}_E" ] || ln -s "$OLD/out/lr/${L}_E" "$ROOT/out/lr/${L}_E"
  [ -z "$BESIDE" ] || [ -e "$ROOT/out/lr/${L}_M" ] || ln -s "$BESIDE/out/lr/${L}_M" "$ROOT/out/lr/${L}_M"
done
READY=$(ssh "$HOST" "cd '$DIR' && for d in $GLOB; do [ -d \"\$d\" ] || continue; [ \$(ls \$d/*.cif 2>/dev/null | wc -l) -ge 10 ] && [ \$(ls \$d/*.beads.tsv 2>/dev/null | wc -l) -ge 10 ] && [ -z \"\$(find \$d -mmin -10 -print -quit)\" ] && echo \$d; done; true")
n=0
for d in $READY; do
  arm=${d%%/*}; chr=${d#*/}; L=${arm%%_*}; A=${arm#*_}
  T=""
  for m in $MAP; do if [ "${m%%=*}" = "$A" ]; then T="${m#*=}"; fi; done
  [ -n "$T" ] || continue
  dest="$ROOT/out/lr/${L}_${T}/$chr"
  [ "$(ls "$dest"/*.cif 2>/dev/null | wc -l)" -ge 10 ] && continue
  mkdir -p "$dest"
  rsync -a "$HOST:$DIR/$d/" "$dest/"
  n=$((n + 1))
done
echo "pulled $n chromosomes of arms, $(ls -d "$ROOT"/out/lr/*_[CRM]/chr* 2>/dev/null | wc -l) in the root"
