#!/bin/bash
# The whole genome cross line readouts: expression, then the pairs, then structure.
cd ~/xc_lr
export ROOT=$HOME/xc_lr
D=repo/playground/rnapii/readouts_crossline_genome.sh
while pgrep -u $USER -x rsync > /dev/null; do sleep 15; done
echo "$(date +%H:%M:%S) bead tables: $(find out/lr -name '*.beads.tsv' | wc -l) of 1320"
seq 22 -1 1 | xargs -P 6 -I{} bash -c "bash $D expression chr{} > logs/expression_chr{}.log 2>&1 || echo 'expression chr{} failed'"
echo "$(date +%H:%M:%S) expression: $(ls out/readouts/e3d_*/chr*/summary.tsv | wc -l) summaries of 88"
seq 22 -1 1 | xargs -P 6 -I{} bash -c "bash $D pairs chr{} > logs/pairs_chr{}.log 2>&1 || echo 'pairs chr{} failed'"
echo "$(date +%H:%M:%S) pairs: $(ls out/readouts/pairs/ep_*.tsv | wc -l) tables of 66"
seq 22 -1 1 | xargs -P 3 -I{} bash -c "bash $D structure chr{} > logs/structure_chr{}.log 2>&1 || echo 'structure chr{} failed'"
echo "$(date +%H:%M:%S) structure: $(ls out/readouts/structure/*_pileup.npz | wc -l) pile-ups, $(ls out/readouts/structure/*_pooled.txt | wc -l) pooled scores of 66"
echo "$(date +%H:%M:%S) all done"
