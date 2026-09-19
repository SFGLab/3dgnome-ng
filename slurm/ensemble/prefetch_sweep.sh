#!/bin/bash
# Prefetching on the GPU: byte parity at 1 against the previous commit, then a K sweep on
# chr1:1-8 Mb and two large runs on chr1:1-60 Mb. Workstation, 3dgnome screen.
set -u
cd ~/3dgnome-ng
C=GM12878; INI=slurm/ensemble/gm12878_hic_arcs.ini; OUT=out/prefetch
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
# parity at prefetch 1: the previous commit in a worktree against the working tree
if [ ! -d /tmp/g3d_prev ]; then git worktree add --detach /tmp/g3d_prev HEAD~1 >/dev/null 2>&1; ln -sfn $PWD/data /tmp/g3d_prev/data; ln -sfn $PWD/.venv /tmp/g3d_prev/.venv; fi
for T in prev head; do
  R=$([ $T = prev ] && echo /tmp/g3d_prev || echo $PWD)
  D=$OUT/parity_$T; mkdir -p $D
  (cd $R && .venv/bin/python -m gnome3d.cli --config $PWD/$INI --data-dir $PWD/data/$C --region chr1:1-8000000 -n 1 --out $PWD/$D > $PWD/$D.log 2>&1)
done
.venv/bin/python - <<'PY'
import numpy as np
def load(p):
    rows = [l.split() for l in open(p) if l.startswith("ATOM")]
    return np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows])
a, b = load("out/prefetch/parity_prev/chr1_1_8000000_s1.cif"), load("out/prefetch/parity_head/chr1_1_8000000_s1.cif")
print("PARITY prefetch=1 vs previous commit:", "byte identical" if a.shape == b.shape and np.array_equal(a, b) else f"DIFFERS max {np.abs(a-b).max() if a.shape == b.shape else 'shape'}")
PY
# the sweep on 8 Mb
for K in 1 8 16 32 64; do
  I=$OUT/gm_k$K.ini; sed "s/^anchor_cap = .*/&\nprefetch = $K/" $INI > $I
  D=$OUT/k${K}_8mb; mkdir -p $D
  .venv/bin/python -m gnome3d.cli --config $I --data-dir data/$C --region chr1:1-8000000 -n 1 --out $D > $D.log 2>&1
  echo "[8 Mb K=$K] $(grep -E 'smooth\[mc\]: .* IBs in' $D.log | tail -n 1)"
done
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" chr1:1-8000000 25000 $OUT/k1_8mb $OUT/k8_8mb $OUT/k16_8mb $OUT/k32_8mb $OUT/k64_8mb 2>&1 | grep -E "^\s+(arm|k)"
# the large case
for K in 32 64; do
  D=$OUT/k${K}_60mb; mkdir -p $D
  .venv/bin/python -m gnome3d.cli --config $OUT/gm_k$K.ini --data-dir data/$C --region chr1:1-60000000 -n 1 --out $D > $D.log 2>&1
  echo "[60 Mb K=$K] $(grep -E 'smooth\[mc\]: .* IBs in' $D.log | tail -n 1)"
done
echo "[60 Mb K=1, production arm] $(grep -E 'smooth\[mc\]: .* IBs in' out/assemble/GM12878_prod.log | head -n 1)"
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" chr1:1-60000000 25000 out/assemble/GM12878_neither $OUT/k32_60mb $OUT/k64_60mb 2>&1 | grep -E "^\s+(arm|GM12878_neither|k)"
echo "PREFETCH SWEEP DONE $(date)"
