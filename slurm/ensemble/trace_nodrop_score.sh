#!/bin/bash
# The scores of the runs of trace_nodrop.sh beside the two chains of the redo, against half b of the
# deep maps, on a workstation. design/uniform-chain.md.
#
#   REPO=~/3dgnome-gapfill/out/_wt_redo G=~/3dgnome-gapfill/out XU=~/xu \
#     HALVES=~/gapfill-data/_hic_halves bash slurm/ensemble/trace_nodrop_score.sh
#
# 1. The nine runs of 20 Mb, three arms: uniform is G/utools/tools20/<row>_ours, anchors is
#    G/tools20/<row>_ours, nodrop is G/nodrop/tools20/<row>_nodrop. The pooled scores, the scores by
#    the separation of two bins at 25 and 100 kb and the paired statistics are in G/nodrop/scores.
# 2. chr19 of GM12878: the pooled scores of the arms C and R with the dropout, XU/out/lr, and
#    without it, G/nodrop/lr, and the distance of the enhancer to promoter pairs of ep_pairs.py in
#    the runs without the dropout beside the table of the redo.
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
G="${G:-$HOME/3dgnome-gapfill/out}"
XU="${XU:-$HOME/xu}"
B="${HALVES:-$HOME/gapfill-data/_hic_halves}/b"
E3D="${E3D:-$HOME/enhancer3d/data}"
PY="$REPO/.venv/bin/python"
L="$G/trace_links"; S="$G/nodrop/scores"
mkdir -p "$L" "$S"
cd "$REPO"
for C in GM12878 H1ESC HFFC6; do
  for R in chr12:40000000-60000000 chr11:60000000-80000000 chr17:35000000-55000000; do
    T=$(echo "$R" | tr ':-' '__'); ROW="${C}_$T"
    [ -e "$L/${ROW}_uniform" ] || ln -s "$G/utools/tools20/${ROW}_ours" "$L/${ROW}_uniform"
    [ -e "$L/${ROW}_anchors" ] || ln -s "$G/tools20/${ROW}_ours" "$L/${ROW}_anchors"
    [ -e "$L/${ROW}_nodrop" ] || ln -s "$G/nodrop/tools20/${ROW}_nodrop" "$L/${ROW}_nodrop"
    [ -s "$S/${ROW}_pooled.txt" ] || "$PY" playground/ensemble_score.py "$B/$C.mcool" "$R" \
      "$L/${ROW}_uniform" "$L/${ROW}_nodrop" "$L/${ROW}_anchors" 2>&1 | grep -vE "INFO|WARN" > "$S/${ROW}_pooled.txt"
    [ -s "$S/${ROW}_bands.txt" ] || for RES in 25000 100000; do
      "$PY" playground/band_scc.py --res "$RES" "$B/$C.mcool" "$R" "$L" "$ROW" uniform nodrop anchors 2>&1 | grep "^BAND"
    done > "$S/${ROW}_bands.txt"
  done
done
"$PY" playground/arm_stats.py "$S" uniform nodrop anchors | tee "$S/stats.txt"
cat "$S"/*_bands.txt | cut -c1-260

CHR=chr19; C=GM12878
for A in C R; do
  [ "$(ls "$G/nodrop/lr/${C}_$A/$CHR"/*.cif 2>/dev/null | wc -l)" -ge 10 ] || { echo "$C $A $CHR without the dropout has under ten structures, the chromosome is left out"; exit 0; }
done
for A in C R; do
  [ -e "$L/${C}_${CHR}_${A}_drop" ] || ln -s "$XU/out/lr/${C}_$A/$CHR" "$L/${C}_${CHR}_${A}_drop"
  [ -e "$L/${C}_${CHR}_${A}_nodrop" ] || ln -s "$G/nodrop/lr/${C}_$A/$CHR" "$L/${C}_${CHR}_${A}_nodrop"
done
[ -s "$S/${C}_${CHR}_pooled.txt" ] || "$PY" playground/ensemble_score.py "$B/$C.mcool" "$CHR" \
  "$L/${C}_${CHR}_C_drop" "$L/${C}_${CHR}_C_nodrop" "$L/${C}_${CHR}_R_drop" "$L/${C}_${CHR}_R_nodrop" 2>&1 | grep -vE "INFO|WARN" > "$S/${C}_${CHR}_pooled.txt"
cat "$S/${C}_${CHR}_pooled.txt"
[ -s "$S/ep_${C}_${CHR}_nodrop.tsv" ] || EP_CHROM="$CHR" PYTHONPATH="$REPO" "$PY" playground/rnapii/ep_pairs.py "$C" "$G/nodrop/lr" \
  "$XU/data/$C/${C}_xc_ctcf_clusters_3+.bedpe" "$XU/data/$C/${C}_xc_rnapii_clusters_3+.bedpe" \
  "$E3D/enhancer_atlas/${C}_enhancers_hg38.bed" "$XU/genes/genes_$CHR.tsv" \
  "$XU/out/readouts/genes/${CHR}_common.tsv" "$S/ep_${C}_${CHR}_nodrop.tsv"
"$PY" - "$XU/out/readouts/pairs/ep_${C}_${CHR}.tsv" "$S/ep_${C}_${CHR}_nodrop.tsv" <<'PY' | tee "$S/ep_${C}_${CHR}.txt"
import sys
import pandas as pd
from scipy.stats import wilcoxon

for name, path in (("with the dropout", sys.argv[1]), ("without the dropout", sys.argv[2])):
    t = pd.read_csv(path, sep="\t")
    for s in ("rnapii_only", "control"):
        w = t[t.set == s].assign(k=lambda d: d.groupby("arm").cumcount()).pivot(index="k", columns="arm", values="distance").dropna()
        p = wilcoxon(w.C, w.R).pvalue if len(w) > 5 else float("nan")
        print(f"{name}, {s}: {len(w)} pairs, CTCF {w.C.mean():.2f}, CTCF and RNAPII {w.R.mean():.2f}, "
              f"change {100 * (w.R.mean() / w.C.mean() - 1):+.0f} %, closer in {(w.R < w.C).mean():.2f} of the pairs, p {p:.1g}")
PY
