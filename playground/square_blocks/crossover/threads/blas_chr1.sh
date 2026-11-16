#!/bin/bash
# The maxent fit on a whole chromosome against the linear algebra thread count.
cd ~/3dgnome-gapfill
for T in 1 8; do
  t0=$(date +%s.%N)
  OPENBLAS_NUM_THREADS=$T OMP_NUM_THREADS=$T .venv/bin/python -m gnome3d.cli --config slurm/ensemble/gm12878_hic_arcs.ini --data-dir ~/gapfill-data/GM12878 --region chr1 -n 1 --out out/cpu5mb/chr1_blas$T > out/cpu5mb/chr1_blas$T.log 2>&1
  echo "threads $T seconds $(echo "$(date +%s.%N) - $t0" | bc -l) $(grep -E "coarse\[" out/cpu5mb/chr1_blas$T.log)" >> out/cpu5mb/blas_chr1.txt
done
echo EXIT >> out/cpu5mb/blas_chr1.txt
