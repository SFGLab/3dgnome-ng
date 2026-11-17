cd ~/xc_readout
while screen -ls | grep -q "[.]xcread[[:space:]]"; do sleep 60; done
for L in H1ESC HFFC6; do ROOT=$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh structure $L > out/readouts_$L.log 2>&1; done
ROOT=$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh expression > out/readouts_expression.log 2>&1
echo READOUTS DONE > out/readouts_done.txt
