#!/bin/bash
# The cell grid in the JAX kernel on the GPU: prefetch K with the grid on, against the
# prefetch sweep without it. Workstation, 3dgnome screen.
set -u
cd ~/3dgnome-ng
C=GM12878; INI=slurm/ensemble/gm12878_hic_arcs.ini; OUT=out/grid
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
for K in 32 64 128; do
  I=$OUT/gm_grid_k$K.ini; sed "s/^prefetch = .*/prefetch = $K\njax_grid = yes/" $INI > $I
  D=$OUT/grid_k${K}_60mb; mkdir -p $D
  .venv/bin/python -m gnome3d.cli --config $I --data-dir data/$C --region chr1:1-60000000 -n 1 --out $D > $D.log 2>&1
  echo "[60 Mb grid K=$K] $(grep -E 'smooth\[mc\]: .* IBs in' $D.log | tail -n 1)"
done
I=$OUT/gm_grid_k64_small.ini; sed "s/^prefetch = .*/prefetch = 64\njax_grid = yes\njax_grid_min_beads = 1024/" $INI > $I
D=$OUT/grid_k64_8mb; mkdir -p $D
.venv/bin/python -m gnome3d.cli --config $I --data-dir data/$C --region chr1:1-8000000 -n 1 --out $D > $D.log 2>&1
echo "[8 Mb grid K=64 min 1024] $(grep -E 'smooth\[mc\]: .* IBs in' $D.log | tail -n 1)"
echo "GRID SWEEP DONE $(date)"
