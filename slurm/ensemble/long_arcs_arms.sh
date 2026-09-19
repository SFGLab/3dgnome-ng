#!/bin/bash
# The long loops in the joint solve, alone and with the map source, on GM12878 chr1:1-60 Mb,
# three structures per arm, then the battery and saddle over every arm of the day.
set -u
cd ~/3dgnome-ng
C=GM12878; R=chr1:1-60000000; INI=slurm/ensemble/gm12878_hic_arcs.ini; OUT=out/cmap
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
sed "s/^scope = .*/&\nlong_arcs = yes/" $INI > $OUT/gm_long.ini
sed "s/^scope = .*/&\nlong_arcs = yes/" $OUT/gm_z3p1.ini > $OUT/gm_long_z3p1.ini
for NAME in long long_z3p1; do
  D=$OUT/${C}_$NAME; mkdir -p $D
  if [ "$(ls $D/*.cif 2>/dev/null | wc -l)" -lt 3 ]; then
    .venv/bin/python -m gnome3d.cli --config $OUT/gm_$NAME.ini --data-dir data/$C --region $R -n 3 --out $D > $D.log 2>&1
    echo "[$NAME] exit $? $(date +%H:%M)  $(grep -E 'joint arcs solve' $D.log | head -n 1)"
  fi
done
ARMS="out/grid4/k32_grid_60mb $OUT/${C}_z3p0 $OUT/${C}_z3p1 $OUT/${C}_z2p1 $OUT/${C}_long $OUT/${C}_long_z3p1"
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" $R 25000 $ARMS 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_battery_all.txt | grep -E "^\s+(arm|k32|GM12878_)"
.venv/bin/python playground/saddle_offline.py "$MCOOL" $R 100000 data/$C/${C}_compartments.bedGraph $ARMS 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_saddle_all.txt | grep -E "(arm|k32|GM12878_)\s"
echo "LONG ARMS DONE $(date)"
