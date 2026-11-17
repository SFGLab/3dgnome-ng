#!/bin/bash
# The three tools of out/toolpage60 at the Micro-C loops with the signal floor, three lines.
cd ~/3dgnome-gapfill
O=out/microc_loops_tools_z15; D=~/gapfill-data; T=out/toolpage60
loops() { case $1 in GM12878) echo out/microc_gm12878/GM12878_$2.bedpe;; *) echo playground/rnapii/microc_loops/$1_$2.bedpe;; esac; }
for L in GM12878 H1ESC HFFC6; do for B in 5000 10000; do
  PYTHONPATH=$PWD .venv/bin/python playground/rnapii/microc_loop_compare.py --json $O/${L}_$B.json \
    $D/microc/microc_${L}_5000.npz $(loops $L $B) \
    "ctcf=$D/$L/${L}_clusters_3+.bedpe" "rnapii=$D/$L/${L}_xc_rnapii_clusters_3+.bedpe" min_z=15 \
    $T/${L}_chr1_1_60000000_cudammc $T/${L}_chr1_1_60000000_multimm_md $T/${L}_chr1_1_60000000_ours > $O/${L}_$B.txt 2>&1
done; done
echo EXIT > $O/done.txt
