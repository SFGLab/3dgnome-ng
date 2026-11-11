#!/bin/bash
S=/private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad
W=~/Desktop/3dgnome-rnapii
FJ="kozlovn@ssh.mini.pw.edu.pl"
until ssh -J $FJ nk2@team-frodo 'grep -q "SLICE DONE" ~/3dgnome-web-smoke/slice_microc2.log' 2>/dev/null; do echo "$(date +%H:%M) waiting for Micro-C slices"; sleep 300; done
ssh -J $FJ nk2@team-frodo 'cat ~/3dgnome-web-smoke/slice_microc2.log' 2>/dev/null
mkdir -p $S/microc
for L in H1ESC HFFC6; do scp -q -J $FJ nk2@team-frodo:3dgnome-web-smoke/microc_${L}_5000.npz $S/microc/ 2>/dev/null; done
ls -la $S/microc
scp -q $S/microc/*.npz home-workstation:xc_readout/microc/
ssh home-workstation 'cd ~/xc_readout && cat > run_microc.sh <<EOS
cd ~/xc_readout
while screen -ls | grep -qE "[.]xcread(2|3)?[[:space:]]"; do sleep 60; done
for L in H1ESC HFFC6; do ROOT=\$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh microc \$L > out/readouts_microc_\$L.log 2>&1; done
echo MICROC DONE > out/microc_done.txt
EOS
screen -dmS xcmicroc bash run_microc.sh'
until ssh home-workstation 'test -f ~/xc_readout/out/microc_done.txt' 2>/dev/null; do echo "$(date +%H:%M) waiting for Micro-C readouts"; sleep 300; done
ssh home-workstation 'tail -n 6 ~/xc_readout/out/readouts_microc_*.log'
until grep -q "FINAL CHAIN DONE" $S/../tasks/b49n72mqa.output 2>/dev/null; do echo "$(date +%H:%M) waiting for the first figure chain"; sleep 300; done
ssh home-workstation 'cd ~/xc_readout/repo && PYTHONPATH=$PWD MPLBACKEND=Agg ~/3dgnome-ng/.venv/bin/python playground/rnapii/collect_readouts.py ../out/readouts ../rnapii-prep-records ../out/figures_v2 2>&1 | tail -5'
mkdir -p $W/design/figures/rnapii-2026-11-12-v2
ssh home-workstation 'cd ~/xc_readout/out && tar --exclude="*.npz" -cf - figures_v2 readouts' | tar -x -C $W/design/figures/rnapii-2026-11-12-v2
ls $W/design/figures/rnapii-2026-11-12-v2/figures_v2
echo "CHAIN2 DONE $(date +%H:%M)"
