#!/bin/bash
# After the tools trial frees the card: one job on the updated site, its contact map numbers.
while ! grep -q "^EXIT" ~/3dgnome-gapfill/out/tools/tools.log; do sleep 120; done
B=http://127.0.0.1:8767
URL=$(curl -s -X POST $B/api/jobs -d cell_type=GM12878 -d region=chr1:12000000-15000000 -d factors=CTCF+RNAPII -d n_models=5 | python3 -c "import sys,json; print(json.load(sys.stdin)['url'])")
echo "job $URL $(date +%H:%M)"
for i in $(seq 1 120); do
  S=$(curl -s $B/api$URL | python3 -c "import sys,json; print(json.load(sys.stdin).get('state'))")
  [ "$S" = done ] || [ "$S" = failed ] && break
  sleep 15
done
echo "state $S $(date +%H:%M)"
curl -s "$B/api$URL/contacts?region=chr1:12000000-15000000&res=25000&upper=hic&lower=model_count&norm=raw" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('rank', r.get('rank'), '|', r.get('rank_about'))
print('scc', r.get('scc'))
import numpy as np
m = np.array(r['lower']); print('bins with model contacts', int((m.sum(0) > 0).sum()), 'of', len(m))"
grep -rh "hole beads\|maxent anchors: [0-9]* anchors\|cross block relax: [0-9]* beads" ~/3dgnome-web-lr/results/*/ 2>/dev/null | head -6
echo WEBCHECK_DONE
