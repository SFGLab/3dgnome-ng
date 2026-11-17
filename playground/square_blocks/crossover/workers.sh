#!/bin/bash
# The threaded smooth stage on a whole chromosome against the worker count, arcs on the device.
cd ~/3dgnome-gapfill; O=out/crossover
for W in 4 8 16; do
  .venv/bin/python -c "
import configparser, sys
c = configparser.ConfigParser(); c.optionxform = str; c.read('$O/hybrid.ini')
[c[s].__setitem__('ib_workers', sys.argv[1]) for s in c.sections() if 'ib_workers' in c[s]]
c.write(open('$O/hybrid_w' + sys.argv[1] + '.ini', 'w'))" $W
  t0=$(date +%s.%N)
  .venv/bin/python -m gnome3d.cli --config $O/hybrid_w$W.ini --data-dir ~/gapfill-data/GM12878 --region chr1 -n 1 --out $O/chr1mb_hybrid_w$W > $O/chr1mb_hybrid_w$W.log 2>&1
  echo "workers $W seconds $(echo "$(date +%s.%N) - $t0" | bc -l) $(grep -E "smooth\[.*nodes in" $O/chr1mb_hybrid_w$W.log)" >> $O/workers.txt
done
echo EXIT >> $O/workers.txt
