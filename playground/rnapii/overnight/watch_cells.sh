#!/bin/bash
# Wait for all nine cell-line arms on eden, copy the remaining ones to the workstation, queue readouts.
X=/mnt/evafs/groups/sfglab/nkozlov/3dgnome-xc-run/out/xc
while true; do
  n=$(ssh eden "for d in $X/{GM12878,H1ESC,HFFC6}_{C,R,M}/chr1; do ls \$d/*.cif 2>/dev/null | wc -l; done | awk '\$1>=10' | wc -l; ls -d $X/*_60mb | wc -l" 2>/dev/null | tail -2 | tr '\n' ' ')
  set -- $n
  echo "$(date +%H:%M) complete arms: ${1:-?} slices: ${2:-?}"
  [ "${1:-0}" -ge 9 ] && [ "${2:-0}" -ge 9 ] && break
  sleep 300
done
ssh eden "cd $X && tar -cf - H1ESC_R H1ESC_R_60mb H1ESC_M H1ESC_M_60mb HFFC6_R HFFC6_R_60mb HFFC6_M HFFC6_M_60mb" 2>/dev/null | ssh home-workstation 'tar -x -C ~/xc_readout/out/xc'
ssh home-workstation 'cd ~/xc_readout && cat > run_rest.sh <<EOS
cd ~/xc_readout
while screen -ls | grep -q "[.]xcread[[:space:]]"; do sleep 60; done
for L in H1ESC HFFC6; do ROOT=\$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh structure \$L > out/readouts_\$L.log 2>&1; done
ROOT=\$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh expression > out/readouts_expression.log 2>&1
echo READOUTS DONE > out/readouts_done.txt
EOS
screen -dmS xcread2 bash run_rest.sh; screen -ls | grep xcread'
echo "queued H1ESC, HFFC6 and expression readouts $(date +%H:%M)"
