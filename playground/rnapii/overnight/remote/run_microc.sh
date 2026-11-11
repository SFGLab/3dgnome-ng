cd ~/xc_readout
while screen -ls | grep -qE "[.]xcread(2|3)?[[:space:]]"; do sleep 60; done
for L in H1ESC HFFC6; do ROOT=$HOME/xc_readout bash repo/playground/rnapii/readouts_crossline.sh microc $L > out/readouts_microc_$L.log 2>&1; done
echo MICROC DONE > out/microc_done.txt
