#!/bin/bash
# GM12878 Micro-C of Hong et al., Zenodo 15303879: slice, loops, and the three tools at them.
cd ~/3dgnome-gapfill
D=~/gapfill-data; M=$D/microc; O=out/microc_gm12878; T=out/toolpage60; PY=.venv/bin/python
until grep -q "^EXIT" $D/microc_gm12878/wget.log; do sleep 60; done
cat $D/microc_gm12878/wget.log > $O/download.log
[ -s $M/microc_GM12878_5000.npz ] || PYTHONPATH=$PWD $PY playground/rnapii/cool_slice.py $D/microc_gm12878/GM12878_microc.50.cool chr1:0-60000000 5000 $M/microc_GM12878_5000.npz > $O/slice.log 2>&1
PYTHONPATH=$PWD $PY playground/rnapii/microc_call_loops.py $M/microc_GM12878_5000.npz chr1 $O/GM12878 8 > $O/GM12878_call.log 2>&1
for B in 5000 10000; do
  PYTHONPATH=$PWD $PY playground/rnapii/microc_loop_compare.py --json $O/tools_GM12878_$B.json \
    $M/microc_GM12878_5000.npz $O/GM12878_$B.bedpe \
    "ctcf=$D/GM12878/GM12878_clusters_3+.bedpe" "rnapii=$D/GM12878/GM12878_xc_rnapii_clusters_3+.bedpe" \
    $T/GM12878_chr1_1_60000000_cudammc $T/GM12878_chr1_1_60000000_multimm_md $T/GM12878_chr1_1_60000000_ours > $O/tools_GM12878_$B.txt 2>&1
done
echo EXIT > $O/done.txt
