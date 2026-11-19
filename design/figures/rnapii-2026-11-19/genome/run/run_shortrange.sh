#!/bin/bash
# 2026-11-19. Two things on the cross line models. The short range Hi-C correlation of every
# line and chromosome. And the GM12878 structure readouts again against the deep map: the first
# run took the first mcool by name, the shallow 4DNFIQ32RWCQ for GM12878, and its outputs are
# set aside in out/readouts/structure_gm12878_shallow_map.
cd ~/xc_lr
export ROOT=$HOME/xc_lr
D=repo/playground/rnapii/readouts_crossline_genome.sh
mkdir -p out/readouts/structure_gm12878_shallow_map
for f in out/readouts/structure/GM12878_chr*; do mv -n "$f" out/readouts/structure_gm12878_shallow_map/; done
echo "$(date +%H:%M:%S) set aside: $(ls out/readouts/structure_gm12878_shallow_map | wc -l) files"
seq 22 -1 1 | xargs -P 6 -I{} bash -c "bash $D shortrange chr{} > logs/shortrange_chr{}.log 2>&1 || echo 'shortrange chr{} failed'"
echo "$(date +%H:%M:%S) shortrange: $(ls out/readouts/shortrange/*.json | wc -l) of 66"
seq 22 -1 1 | xargs -P 3 -I{} bash -c "bash $D structure chr{} > logs/structure_deep_chr{}.log 2>&1 || echo 'structure chr{} failed'"
echo "$(date +%H:%M:%S) structure: $(ls out/readouts/structure/GM12878_*_pooled.txt | wc -l) pooled and $(ls out/readouts/structure/GM12878_*_pileup.npz | wc -l) pile-ups of 22 for GM12878"
echo "$(date +%H:%M:%S) all done"
