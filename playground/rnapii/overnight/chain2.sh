#!/bin/bash
# Wait for the Micro-C slices, run the Micro-C readouts on the readout machine, draw the second
# figure set and bring it back to the checkout. The hosts are ssh names of the caller's config.
S="${S:?Set S to a local scratch directory}"
W="${W:-$PWD}"
SLICER="${SLICER:?Set SLICER to the ssh host that cut the Micro-C slices}"
WORK="${WORK:?Set WORK to the ssh host of the readout machine}"
FIRST_LOG="${FIRST_LOG:?Set FIRST_LOG to the output file of final_chain.sh}"
until ssh $SLICER 'grep -q "SLICE DONE" ~/3dgnome-web-smoke/slice_microc2.log' 2>/dev/null; do echo "$(date +%H:%M) waiting for Micro-C slices"; sleep 300; done
ssh $SLICER 'cat ~/3dgnome-web-smoke/slice_microc2.log' 2>/dev/null
mkdir -p $S/microc
for L in H1ESC HFFC6; do scp -q $SLICER:3dgnome-web-smoke/microc_${L}_5000.npz $S/microc/ 2>/dev/null; done
ls -la $S/microc
scp -q $S/microc/*.npz $WORK:xc_readout/microc/
ssh $WORK 'cd ~/xc_readout && cat > run_microc.sh <<EOS
cd ~/xc_readout
while screen -ls | grep -qE "[.]xcread(2|3)?[[:space:]]"; do sleep 60; done
for L in H1ESC HFFC6; do ROOT=\$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh microc \$L > out/readouts_microc_\$L.log 2>&1; done
echo MICROC DONE > out/microc_done.txt
EOS
screen -dmS xcmicroc bash run_microc.sh'
until ssh $WORK 'test -f ~/xc_readout/out/microc_done.txt' 2>/dev/null; do echo "$(date +%H:%M) waiting for Micro-C readouts"; sleep 300; done
ssh $WORK 'tail -n 6 ~/xc_readout/out/readouts_microc_*.log'
until grep -q "FINAL CHAIN DONE" $FIRST_LOG 2>/dev/null; do echo "$(date +%H:%M) waiting for the first figure chain"; sleep 300; done
ssh $WORK 'cd ~/xc_readout/repo && PYTHONPATH=$PWD MPLBACKEND=Agg ~/3dgnome-ng/.venv/bin/python playground/rnapii/collect_readouts.py ../out/readouts ../rnapii-prep-records ../out/figures_v2 2>&1 | tail -5'
mkdir -p $W/design/figures/rnapii-2026-11-12-v2
ssh $WORK 'cd ~/xc_readout/out && tar --exclude="*.npz" -cf - figures_v2 readouts' | tar -x -C $W/design/figures/rnapii-2026-11-12-v2
ls $W/design/figures/rnapii-2026-11-12-v2/figures_v2
echo "CHAIN2 DONE $(date +%H:%M)"
