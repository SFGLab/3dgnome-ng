#!/bin/bash
# Micro-C loop lists read in the deep Hi-C map of the same cells, and in their own map.
cd ~/3dgnome-gapfill
D=~/gapfill-data; O=out/microc_check; PY=.venv/bin/python
hic() { case $1 in GM12878) echo hic.4DNFIXP4QG5B;; H1ESC) echo hic.4DNFI6HDY7WZ;; HFFC6) echo hic.4DNFIB59T7NN;; esac; }
loops() { case $1 in GM12878) echo out/microc_gm12878/GM12878_$2.bedpe;; *) echo playground/rnapii/microc_loops/$1_$2.bedpe;; esac; }
for L in H1ESC HFFC6 GM12878; do
  [ -s $O/hic_${L}_5000.npz ] || PYTHONPATH=$PWD $PY playground/rnapii/cool_slice.py "$D/_hic/$L/$(hic $L).mcool::/resolutions/5000" chr1:0-60000000 5000 $O/hic_${L}_5000.npz > $O/hic_${L}_slice.log 2>&1
  for B in 5000 10000; do
    PYTHONPATH=$PWD $PY playground/rnapii/loops_in_map.py $D/microc/microc_${L}_5000.npz $(loops $L $B) "$L Micro-C, $B" >> $O/check.txt 2>&1
    PYTHONPATH=$PWD $PY playground/rnapii/loops_in_map.py $O/hic_${L}_5000.npz $(loops $L $B) "$L Hi-C, $B" >> $O/check.txt 2>&1
  done
done
echo EXIT >> $O/check.txt
