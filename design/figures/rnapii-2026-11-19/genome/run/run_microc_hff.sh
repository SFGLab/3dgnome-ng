#!/bin/bash
# HFFC6 beside the first Micro-C runner, which is still on H1ESC. A task whose output is there
# is skipped, so that runner finds HFFC6 done when it gets to it.
cd ~/xc_lr
export ROOT=$HOME/xc_lr NPROC=4
D=repo/playground/rnapii/microc_genome.sh
seq 22 -1 1 | xargs -P 3 -I{} bash -c "bash $D HFFC6 chr{} > logs/microc_HFFC6_chr{}.log 2>&1 || echo \"microc HFFC6 chr{} failed\""
echo "$(date +%H:%M:%S) HFFC6: $(ls out/readouts/microc/HFFC6_chr*_5000.json 2>/dev/null | wc -l) of 22 chromosomes scored"
