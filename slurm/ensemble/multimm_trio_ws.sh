#!/bin/bash
# Waits for a MultiMM run left over from an earlier launch to finish before starting.
until ! pgrep -f "playground/multimm_arm[.]py" >/dev/null; do sleep 60; done
# Idea 21 of design/expression-from-structure.md: MultiMM on each person's CTCF and RNAPOL2
# loops, chr1 at 50,000 beads, ten members, loops only and then loops with the person's own
# compartment track; each member's minimised and after dynamics structures collected into the
# enhancer3d layout, then the same pipeline as the other arms on every ensemble.
set -o pipefail
cd ~/3dgnome-ng
P=.venv/bin/python
CHR=chr1:1-248956422
SAMPLES="HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240"
layout () {   # $1 arm root, $2 kind (min|md), $3 sample
  local s dst
  s=$(echo $3 | tr '[:upper:]' '[:lower:]')
  dst=/mnt/storagelinux/models_multimm_$4_$2/${s}_genome/chr1
  mkdir -p $dst
  for i in $(seq 1 10); do ln -sf $(readlink -f out/multimm_trio/$1_$2/member_$i.cif) $dst/chr1_s$i.cif; done
}
for ARM in loops comp; do
  for S in $SAMPLES; do
    if [ -s out/multimm_trio/${S}_${ARM}_md/member_10.cif ]; then echo "[mm] $S $ARM present"; layout ${S}_${ARM} min $S $ARM; layout ${S}_${ARM} md $S $ARM; continue; fi
    echo "[mm] $S $ARM start $(date)"
    extra=""; [ $ARM = comp ] && extra="--compartments data/$S/${S}_compartments.bedGraph"
    $P playground/multimm_arm.py ~/trio_loops_chr1/${S}_both_chr1.bedpe $CHR /mnt/storagelinux/models_trio_rnapol2/$(echo $S | tr '[:upper:]' '[:lower:]')_genome/chr1 \
       out/multimm_trio/${S}_${ARM} --n 10 --n-beads 50000 --multimm ~/mmvenv/bin/MultiMM --platform OpenCL $extra 2>&1 | grep -E "MultiMM|compartment bed" | cut -c1-140
    layout ${S}_${ARM} min $S $ARM; layout ${S}_${ARM} md $S $ARM
    echo "[mm] $S $ARM done $(date)"
  done
done
echo "[mm] MODELS DONE $(date)"
# the pipeline on the four ensembles
cd ~/enhancer3d
P=~/3dgnome-ng/.venv/bin/python
CELLS=HG00512,HG00513,HG00514,HG00731,HG00732,HG00733,GM19238,GM19239,GM19240
CNT=data/trio_gene_counts.csv
export TRIO_ELEMENTS=own_elements/{s}_own_elements_hg38.bed
for E in loops_md loops_min comp_md comp_min; do
  MODELS=/mnt/storagelinux/models_multimm_$E; OUT=playground/trio_multimm_$E; mkdir -p $OUT
  export TRIO_SCOPE="chr1, MultiMM $E, own elements"
  echo "[e3d] ===== $E start $(date)"
  $P playground/genome_ep_distances.py --models $MODELS --out $OUT/dist --enhancers own --cells $CELLS --chroms chr1 --no-states --no-ccd --workers 1 --max-linear-bp 3000000 2>&1 | grep -E "FAILED|wrote"
  T=$OUT/dist/genes_all_cell_lines_hic_tads.parquet
  TRIO_ANCHORS=~/trio_anchors_rnapol2 TRIO_ANCHOR_FILE='{s}_anchors_ctcf_rnapol2.bed' $P playground/trio_control.py $T $OUT/trio_control.csv 2>&1 | grep -E "rho_3d|separation"
  $P playground/trio_expression.py $T $CNT $OUT/trio_expression_ownlinear.csv 2>&1 | grep -E "mean partial"
  $P playground/table_deviation.py ours=playground/trio_rnapol2_own/dist/genes_all_cell_lines_hic_tads.parquet multimm=$T 2>&1 | grep -v Warning
  $P playground/expression_model.py $MODELS $CNT $OUT/model --chrom chr1 2>&1 | grep -E "gain of D|G\+L\+I \(|  D \("
  echo "[e3d] ===== $E done $(date)"
done
echo "ALL DONE $(date)"
