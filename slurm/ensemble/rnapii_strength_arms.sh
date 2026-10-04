#!/bin/bash
# The RNAPII factor at a fraction of CTCF's loop strength, GM12878 whole chr1, three structures
# per arm, against the CTCF only and full strength arms of rnapii_arms.sh.
set -u
cd ~/3dgnome-ng
C=GM12878; OUT=out/rnapii; R=chr1
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
for F in 0.3 0.1 0; do
  NAME=s$(echo $F | tr -d .)
  sed "s/^background_range_bp = .*/&\nfactor_strength = 1,$F/" $OUT/gm_both.ini > $OUT/gm_$NAME.ini
  D=$OUT/${C}_$NAME; mkdir -p $D
  if [ "$(ls $D/*.cif 2>/dev/null | wc -l)" -lt 3 ]; then
    echo "[$NAME] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config $OUT/gm_$NAME.ini --data-dir data/$C --region $R -n 3 --out $D > $D.log 2>&1
    echo "[$NAME] exit $? $(date +%H:%M)"
  fi
  .venv/bin/python playground/slice_arm.py $D $OUT/${C}_${NAME}_60mb chr1:1-60000000 > /dev/null
done
ARMS="$OUT/${C}_ctcf_60mb $OUT/${C}_both_60mb $OUT/${C}_s03_60mb $OUT/${C}_s01_60mb $OUT/${C}_s0_60mb"
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" chr1:1-60000000 25000 $ARMS 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_battery_strength.txt | grep -E "^\s+(arm|GM12878_)"
.venv/bin/python playground/saddle_offline.py "$MCOOL" chr1:1-60000000 100000 data/$C/${C}_compartments.bedGraph $ARMS 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_saddle_strength.txt | grep -E "(arm|GM12878_)\s"
for NAME in s03 s01 s0; do
  echo "== enhancer3d chr1 test, $NAME"
  (cd ~/enhancer3d && ~/3dgnome-ng/.venv/bin/python playground/chr1_ep_distance_expression_go.py --models ~/3dgnome-ng/$OUT/${C}_$NAME --out playground/rnapii_$NAME --skip-enrichment 2>&1 | grep -E "spearman_rho|pearson_r|genes with" | head -n 3)
done
echo "STRENGTH ARMS DONE $(date)"
