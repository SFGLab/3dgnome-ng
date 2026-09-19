#!/bin/bash
# RNAPII loops as a second factor on GM12878 whole chr1, five structures per arm: CTCF only,
# and CTCF with the Tang 2015 RNAPII loops. Battery and saddle on the 1-60 Mb slice, then the
# chr1 enhancer to promoter against expression test of enhancer3d. Workstation, 3dgnome screen.
set -u
cd ~/3dgnome-ng
C=GM12878; INI=slurm/ensemble/gm12878_hic_arcs.ini; OUT=out/rnapii; R=chr1
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
cp $INI $OUT/gm_ctcf.ini
sed "s/^anchors = .*/anchors = GM12878_anchors_ctcf_rnapii.bed/; s/^clusters = .*/clusters = GM12878_clusters_3+.bedpe,GM12878_rnapii_clusters_3+.bedpe\nfactors = CTCF,RNAPOL2/" $INI > $OUT/gm_both.ini
grep -E "^anchors|^clusters|^factors" $OUT/gm_both.ini
for ARM in ctcf both; do
  D=$OUT/${C}_$ARM; mkdir -p $D
  if [ "$(ls $D/*.cif 2>/dev/null | wc -l)" -lt 5 ]; then
    echo "[$ARM] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config $OUT/gm_$ARM.ini --data-dir data/$C --region $R -n 5 --out $D > $D.log 2>&1
    echo "[$ARM] exit $? $(date +%H:%M)  $(grep -E 'joint arcs solve|arcs of factor' $D.log | head -n 3 | tr '\n' ' ')"
  fi
  .venv/bin/python playground/slice_arm.py $D $OUT/${C}_${ARM}_60mb chr1:1-60000000 > /dev/null
done
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" chr1:1-60000000 25000 $OUT/${C}_ctcf_60mb $OUT/${C}_both_60mb 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_battery.txt | grep -E "^\s+(arm|GM12878_)"
.venv/bin/python playground/saddle_offline.py "$MCOOL" chr1:1-60000000 100000 data/$C/${C}_compartments.bedGraph $OUT/${C}_ctcf_60mb $OUT/${C}_both_60mb 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_saddle.txt | grep -E "(arm|GM12878_)\s"
for ARM in ctcf both; do
  echo "== enhancer3d chr1 test, $ARM"
  (cd ~/enhancer3d && ~/3dgnome-ng/.venv/bin/python playground/chr1_ep_distance_expression_go.py --models ~/3dgnome-ng/$OUT/${C}_$ARM --out playground/rnapii_$ARM --skip-enrichment 2>&1 | grep -iE "spearman|pearson|genes|tertile" | tail -n 6)
done
echo "RNAPII ARMS DONE $(date)"
