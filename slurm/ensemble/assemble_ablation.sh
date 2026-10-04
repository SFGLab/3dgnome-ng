#!/bin/bash
# Are the boundary stitch and the cross block relaxation still needed at chromosome scope?
# Four arms on GM12878 chr1:1-60 Mb, three structures each, on the workstation GPU one at a
# time, then the battery. Run in the 3dgnome screen:  bash slurm/ensemble/assemble_ablation.sh
set -u
cd ~/3dgnome-ng
C=GM12878; R=chr1:1-60000000; N=3; OUT=out/assemble
MCOOL=$(ls /mnt/storagelinux/_hic/$C/*.mcool | head -1)
mkdir -p $OUT
for ARM in prod nostitch norelax neither; do
  INI=$OUT/${C}_$ARM.ini
  cp slurm/ensemble/gm12878_hic_arcs.ini $INI
  case $ARM in
    nostitch) sed -i 's/^use_boundary_stitch = yes/use_boundary_stitch = no/' $INI ;;
    norelax)  sed -i 's/^use_cross_block_relax = yes/use_cross_block_relax = no/' $INI ;;
    neither)  sed -i 's/^use_boundary_stitch = yes/use_boundary_stitch = no/; s/^use_cross_block_relax = yes/use_cross_block_relax = no/' $INI ;;
  esac
  D=$OUT/${C}_$ARM
  if [ "$(ls $D/*.cif 2>/dev/null | wc -l)" -lt $N ]; then
    echo "[$ARM] start $(date +%H:%M)"
    .venv/bin/python -m gnome3d.cli --config $INI --data-dir data/$C --region $R -n $N --out $D > $D.log 2>&1
    echo "[$ARM] done $(date +%H:%M) exit $?"
  fi
done
.venv/bin/python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$C/${C}_hic_25kb_singletons.bedpe "$MCOOL" $R 25000 $OUT/${C}_prod $OUT/${C}_nostitch $OUT/${C}_norelax $OUT/${C}_neither 2>&1 | grep -vE "INFO|WARN" | tee $OUT/${C}_battery.txt
for ARM in prod nostitch norelax neither; do
  echo "== $ARM boundary and cross block report"
  .venv/bin/python playground/restitch_model.py $OUT/${C}_$ARM/chr1_1_60000000_s1.cif $OUT/${C}_$ARM.ini --no-relax --out /dev/null 2>&1 | grep -iE "boundary|cross|worst" | head -n 6
done
echo "ASSEMBLE ABLATION DONE $(date)"
