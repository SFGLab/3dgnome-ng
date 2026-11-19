#!/bin/bash
# Every person and chromosome of the two trio arms, six tasks at a time, once the pull from
# eden has ended and every structure is here. Then the whole genome table.
cd ~/trio_lr
export ROOT=$HOME/trio_lr REPO=$HOME/trio_lr/repo PY=$HOME/3dgnome-ng/.venv/bin/python
while pgrep -u $USER -x rsync > /dev/null; do sleep 15; done
n=$(find out/lr_trio_ctcf out/lr_trio_rnapol2 -name '*.cif' | wc -l)
echo "$(date +%H:%M:%S) the pull has ended with $n of 3960 structures"
[ "$n" -eq 3960 ] || { echo "not every structure is here, nothing is run"; tail -c 600 pull.log | tr '\r' '\n' | tail -4; exit 1; }
seq 0 197 | xargs -P 6 -I{} bash -c 'bash $REPO/playground/rnapii/readouts_trio_genome.sh {} > logs/task_{}.log 2>&1 || echo "task {} failed"'
echo "$(date +%H:%M:%S) tasks done: $(ls out/readouts/trio_lr/*_pileup.npz | wc -l) pile-ups, $(ls out/readouts/trio_lr/*_pooled.txt | wc -l) pooled scores of 198"
$PY repo/playground/rnapii/trio_genome_summary.py out/readouts/trio_lr out/readouts/trio_lr/trio_genome.json > out/readouts/trio_lr/trio_genome.txt 2>&1
cat out/readouts/trio_lr/trio_genome.txt
echo "$(date +%H:%M:%S) all done"
