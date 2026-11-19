#!/bin/bash
# The rest of the deep contact map of HFFc6, in eight ranges at once. A connection to the bucket
# is fast for its first seconds and slows after, so each range is fetched in passes of 25
# seconds on new connections. The ranges are then appended to the partial file in order, and
# the whole is checked by its size and its md5, 59581e9ec46ead9619906e4d1fd4c57d.
#
#   D=/srv/3dgnome/data/maps/HFFC6 bash web/deploy/maps_hffc6_parallel.sh
U=https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/bd06e6c5-765f-4660-8b1c-23b681bcfff0/4DNFIB59T7NN.mcool
D="${D:?Set D to the directory that holds the partial map}"
N=22837705555
K=8
P=$D/4DNFIB59T7NN.mcool.part
S=$(stat -c %s $P)
C=$(( (N - S + K - 1) / K ))
echo "have $S of $N, $K ranges of $C"
fetch() {
  i=$1; a=$(( S + i * C )); b=$(( a + C - 1 )); [ $b -ge $N ] && b=$(( N - 1 ))
  want=$(( b - a + 1 )); seg=$D/range.$i
  for pass in $(seq 1 5000); do
    have=$(stat -c %s $seg 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && break
    curl -sS --max-time 25 --speed-limit 1000000 --speed-time 15 -r $(( a + have ))-$b $U >> $seg 2>/dev/null
    sleep 1
  done
  echo "range $i done, $(stat -c %s $seg) of $want"
}
for i in $(seq 0 $(( K - 1 ))); do fetch $i & done
wait
for i in $(seq 0 $(( K - 1 ))); do cat $D/range.$i >> $P; done
echo "joined: $(stat -c %s $P) of $N"
md5sum $P | tee $D/4DNFIB59T7NN.md5
[ "$(stat -c %s $P)" = "$N" ] && [ "$(cut -d' ' -f1 $D/4DNFIB59T7NN.md5)" = 59581e9ec46ead9619906e4d1fd4c57d ] && mv -n $P $D/4DNFIB59T7NN.mcool && echo MAP_DONE
