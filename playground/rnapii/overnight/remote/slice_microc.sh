cd ~/3dgnome-web-smoke
U=http://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/hic
for pair in H1ESC:4DNFI2TK7L2F HFFC6:4DNFI18Q799K; do L=${pair%%:*}; A=${pair#*:}
  for R in 5000 10000; do strawvenv/bin/python hic_slice.py $U/$A.hic chr1:0-60000000 $R microc_${L}_${R}.npz; done
done
echo SLICE DONE
