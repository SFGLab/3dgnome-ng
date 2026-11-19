#!/bin/bash
# Fetches the deep contact map of HFFc6 from the public 4DN bucket. A long connection to the bucket
# slows down. Each pass ends after 25 seconds and the next resumes the partial file. The md5 of the
# whole file is 59581e9ec46ead9619906e4d1fd4c57d.
#
#   D=/srv/3dgnome/data/maps/HFFC6 bash web/deploy/maps_hffc6.sh
B=https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput
D="${D:?Set D to the directory that receives the map}"
N=22837705555
for i in $(seq 1 20000); do
  [ "$(stat -c %s $D/4DNFIB59T7NN.mcool.part 2>/dev/null)" = "$N" ] && break
  curl -sS -C - --max-time 25 --speed-limit 1500000 --speed-time 15 -o $D/4DNFIB59T7NN.mcool.part $B/bd06e6c5-765f-4660-8b1c-23b681bcfff0/4DNFIB59T7NN.mcool
  echo "pass $i rc=$? $(stat -c %s $D/4DNFIB59T7NN.mcool.part) of $N"
  sleep 1
done
md5sum $D/4DNFIB59T7NN.mcool.part | tee $D/4DNFIB59T7NN.md5
[ "$(stat -c %s $D/4DNFIB59T7NN.mcool.part)" = "$N" ] && mv -n $D/4DNFIB59T7NN.mcool.part $D/4DNFIB59T7NN.mcool
echo MAP_DONE
