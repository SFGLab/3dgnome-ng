cd "${HIC_ROOT:?Set HIC_ROOT to the directory that holds the Hi-C maps of each cell line}"
curl -s -C - -o H1ESC/microc/4DNFI2TK7L2F.hic http://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/hic/4DNFI2TK7L2F.hic && echo H1 OK
curl -s -C - -o HFFC6/microc/4DNFI18Q799K.hic http://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/hic/4DNFI18Q799K.hic && echo HFF OK
ls -la H1ESC/microc HFFC6/microc
echo FETCH DONE
