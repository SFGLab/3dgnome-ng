#!/bin/bash
# Once both runners have ended, each is run again. A step whose output is there is skipped, so
# only what failed the first time is made: the two chromosomes under the old coverage floor and
# the Micro-C calls from before the pandas 3 fix.
cd ~/xc_lr
while pgrep -u $USER -f "run_all.sh|run_microc.sh" > /dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) second pass"
bash run_all.sh > run_all.2.log 2>&1
bash run_microc.sh > run_microc.2.log 2>&1
echo "$(date +%H:%M:%S) second pass done"
