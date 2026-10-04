#!/bin/bash
# The original 3dgnome pipeline's GM12878 models on four chr1 regions, one hundred each, scored
# by the battery on each region's window against our chr1:1-60 Mb arms cut to the same window.
set -u
cd ~/3dgnome-ng
C=GM12878; SRC=out/reference_regions/$C; OUT=out/validation_regions
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
k=0
for spec in "results_GM12878_Deni_chr1_74280_3521135 chr1:850000-3500000" \
            "results_GM12878_Deni_chr1_12398267_14398571 chr1:12600000-13850000" \
            "results_GM12878_Deni_chr1_15415396_17422207 chr1:15775000-17425000" \
            "results_GM12878_Deni_chr1_18268944_20287760 chr1:18275000-20225000"; do
  set -- $spec; DIR=$1; R=$2; k=$((k + 1))
  .venv/bin/python playground/reference_arm.py data/$C/${C}_anchors_3+.bed $SRC/$DIR $OUT/${C}_r${k}_reference chr1 2>&1 | tail -n 2
  for A in prod3 mm_loops_md mm_comps_md; do
    .venv/bin/python playground/slice_arm.py out/validation/${C}_$A $OUT/${C}_r${k}_$A $R 2>&1 | tail -n 1
  done
  DIRS=$(ls -d $OUT/${C}_r${k}_*/ | tr "\n" " ")
  echo "== region $k $R: $DIRS"
  .venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" $R 25000 $DIRS 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_r${k}_battery.txt | grep -E "^\s+(arm|${C}_r${k}_)"
done
echo "REGIONS DONE $(date)"
