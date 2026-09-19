#!/bin/bash
# The map as the contact background's source, against production, on GM12878 chr1:1-60 Mb,
# three structures per arm, battery and saddle. Workstation, 3dgnome screen.
set -u
cd ~/3dgnome-ng
C=GM12878; R=chr1:1-60000000; INI=slurm/ensemble/gm12878_hic_arcs.ini; OUT=out/cmap
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
for ARM in "z3p0 3.0 0" "z3p1 3.0 1" "z2p1 2.0 1"; do
  set -- $ARM; NAME=$1; Z=$2; P=$3
  I=$OUT/gm_$NAME.ini
  sed "s#^singletons = .*#&\ncontact_map = $MCOOL#; s/^background_range_bp = .*/&\ncontact_map_z = $Z\ncontact_map_pool = $P/" $INI > $I
  grep -cE "^contact_map" $I
  D=$OUT/${C}_$NAME; mkdir -p $D
  if [ "$(ls $D/*.cif 2>/dev/null | wc -l)" -lt 3 ]; then
    .venv/bin/python -m gnome3d.cli --config $I --data-dir data/$C --region $R -n 3 --out $D > $D.log 2>&1
    echo "[$NAME] exit $? $(date +%H:%M)"
  fi
done
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" $R 25000 out/grid4/k32_grid_60mb $OUT/${C}_z3p0 $OUT/${C}_z3p1 $OUT/${C}_z2p1 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_battery.txt | grep -E "^\s+(arm|k32|GM12878_)"
.venv/bin/python playground/saddle_offline.py "$MCOOL" $R 100000 data/$C/${C}_compartments.bedGraph out/grid4/k32_grid_60mb $OUT/${C}_z3p0 $OUT/${C}_z3p1 $OUT/${C}_z2p1 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_saddle.txt | grep -E "(arm|k32|GM12878_)\s"
echo "CMAP ARMS DONE $(date)"
