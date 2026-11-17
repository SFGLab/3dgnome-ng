#!/bin/bash
# The three tools of out/toolpage60 at the Micro-C loops, H1ESC and HFFC6.
cd ~/3dgnome-gapfill
O=out/microc_loops_tools2; D=~/gapfill-data; T=out/toolpage60
for L in H1ESC HFFC6; do for B in 5000 10000; do
  PYTHONPATH=$PWD .venv/bin/python playground/rnapii/microc_loop_compare.py --json $O/${L}_$B.json \
    $D/microc/microc_${L}_5000.npz playground/rnapii/microc_loops/${L}_$B.bedpe \
    "ctcf=$D/$L/${L}_clusters_3+.bedpe" "rnapii=$D/$L/${L}_xc_rnapii_clusters_3+.bedpe" \
    $T/${L}_chr1_1_60000000_cudammc $T/${L}_chr1_1_60000000_multimm_md $T/${L}_chr1_1_60000000_ours > $O/${L}_$B.txt 2>&1
done; done
echo EXIT > $O/done.txt
