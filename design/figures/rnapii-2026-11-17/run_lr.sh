#!/bin/bash
# Cross line readouts on the production of 2026-11-16, arms C, R and M from the cluster's out/lr, arm E as before.
R="${R:?Set R to the readout directory}"
export ROOT=$R
cd $R
for L in GM12878 H1ESC HFFC6; do bash repo/playground/rnapii/readouts_crossline.sh structure $L > out/readouts_$L.log 2>&1; done
echo STRUCTURE_DONE > out/structure_done.txt
bash repo/playground/rnapii/readouts_crossline.sh expression > out/readouts_expression.log 2>&1
mkdir -p $R/fig2
D=$HOME/3dgnome-ng/data; E=$HOME/enhancer3d/data
for L in GM12878 H1ESC HFFC6; do
  PYTHONPATH=$R/repo $HOME/3dgnome-ng/.venv/bin/python repo/playground/rnapii/ep_pairs.py $L $R/out/xc $D/$L/${L}_xc_ctcf_clusters_3+.bedpe \
    $R/loops/${L}_xc_rnapii_clusters_3+.bedpe $E/enhancer_atlas/${L}_enhancers_hg38.bed \
    $R/out/readouts/genes_chr1.tsv $R/out/readouts/genes_chr1_common.tsv $R/fig2/ep_$L.tsv > out/ep_$L.log 2>&1
done
echo READOUTS DONE > out/readouts_done.txt
