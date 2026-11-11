#!/bin/bash
# Wait for the workstation readouts and the eden trio readouts, gather them, draw the figures,
# and bring everything back to the laptop worktree.
X=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-xc-run
W=~/Desktop/3dgnome-rnapii
until ssh home-workstation 'test -f ~/xc_readout/out/readouts_done.txt' 2>/dev/null; do echo "$(date +%H:%M) waiting for workstation readouts"; sleep 600; done
until ssh eden "squeue -u nkozlov -h -j 1940229 2>/dev/null | wc -l" 2>/dev/null | tail -1 | grep -qx 0; do echo "$(date +%H:%M) waiting for trio readouts"; sleep 600; done
ssh eden "cd $X/out/readouts && tar -cf - trio" 2>/dev/null | ssh home-workstation 'tar -x -C ~/xc_readout/out/readouts'
tar -C $W/design/plans -cf - rnapii-prep-records | ssh home-workstation 'tar -x -C ~/xc_readout'
ssh home-workstation 'cd ~/xc_readout/repo && PYTHONPATH=$PWD MPLBACKEND=Agg ~/3dgnome-ng/.venv/bin/python playground/rnapii/collect_readouts.py ../out/readouts ../rnapii-prep-records ../out/figures 2>&1 | tail -5'
mkdir -p $W/design/figures/rnapii-2026-11-12
ssh home-workstation 'cd ~/xc_readout/out && tar --exclude="*.npz" -cf - figures readouts' | tar -x -C $W/design/figures/rnapii-2026-11-12
ls $W/design/figures/rnapii-2026-11-12/figures
echo "FINAL CHAIN DONE $(date +%H:%M)"
