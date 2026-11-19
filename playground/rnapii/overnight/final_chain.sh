#!/bin/bash
# Wait for the readouts of the readout machine and the trio readouts of the cluster, gather them,
# draw the figures and bring everything back to the checkout. The hosts are ssh names of the
# caller's config.
X="${X:?Set X to the staged checkout on the cluster}"
W="${W:-$PWD}"
CLUSTER="${CLUSTER:?Set CLUSTER to the ssh host of the cluster}"
WORK="${WORK:?Set WORK to the ssh host of the readout machine}"
TRIO_JOB="${TRIO_JOB:?Set TRIO_JOB to the job id of the trio readouts}"
until ssh $WORK 'test -f ~/xc_readout/out/readouts_done.txt' 2>/dev/null; do echo "$(date +%H:%M) waiting for the readouts"; sleep 600; done
until ssh $CLUSTER "squeue -h -j $TRIO_JOB 2>/dev/null | wc -l" 2>/dev/null | tail -1 | grep -qx 0; do echo "$(date +%H:%M) waiting for trio readouts"; sleep 600; done
ssh $CLUSTER "cd $X/out/readouts && tar -cf - trio" 2>/dev/null | ssh $WORK 'tar -x -C ~/xc_readout/out/readouts'
tar -C $W/design/plans -cf - rnapii-prep-records | ssh $WORK 'tar -x -C ~/xc_readout'
ssh $WORK 'cd ~/xc_readout/repo && PYTHONPATH=$PWD MPLBACKEND=Agg ~/3dgnome-ng/.venv/bin/python playground/rnapii/collect_readouts.py ../out/readouts ../rnapii-prep-records ../out/figures 2>&1 | tail -5'
mkdir -p $W/design/figures/rnapii-2026-11-12
ssh $WORK 'cd ~/xc_readout/out && tar --exclude="*.npz" -cf - figures readouts' | tar -x -C $W/design/figures/rnapii-2026-11-12
ls $W/design/figures/rnapii-2026-11-12/figures
echo "FINAL CHAIN DONE $(date +%H:%M)"
