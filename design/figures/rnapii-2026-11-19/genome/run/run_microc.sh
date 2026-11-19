#!/bin/bash
# Micro-C loops of the whole genome and the models at them, a line once its map is down.
cd ~/xc_lr
export ROOT=$HOME/xc_lr NPROC=4
D=repo/playground/rnapii/microc_genome.sh
for pair in GM12878:GM12878_microc.50.cool H1ESC:H1ESC_4DNFI2TK7L2F.hic HFFC6:HFFC6_4DNFI18Q799K.hic; do
  L=${pair%%:*}; F=microc/${pair#*:}
  while [ ! -s $F ]; do sleep 30; done
  echo "$(date +%H:%M:%S) $L: map is here, $(stat -c %s $F) bytes"
  seq 22 -1 1 | xargs -P 3 -I{} bash -c "bash $D $L chr{} > logs/microc_${L}_chr{}.log 2>&1 || echo 'microc $L chr{} failed'"
  echo "$(date +%H:%M:%S) $L: $(ls out/readouts/microc/${L}_chr*_5000.json 2>/dev/null | wc -l) of 22 chromosomes scored"
done
echo "$(date +%H:%M:%S) all done"
