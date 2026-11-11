#!/bin/bash
# The hosted data the site was smoke tested with on 2026-11-11: three cell types and the HGSVC
# freeze 4 insertions and deletions. Run from the repository root.
#
#   SITE=/srv/gnome3d/data E3D=~/enhancer3d/data HIC_SINGLETONS=data HGSVC=data/_hgsvc \
#     bash web/scripts/build_site_data.sh
#
# Inputs: data/<LINE>/ with the production CTCF anchors and the cross line prep outputs of
# playground/rnapii/crosscell_prep.py, <LINE>_hic_25kb_singletons.bedpe under HIC_SINGLETONS,
# GENCODE v40 basic and the EnhancerAtlas beds under E3D, the HGSVC VCF under HGSVC.
set -euo pipefail
SITE="${SITE:?set SITE to the new data directory}"
E3D="${E3D:-$HOME/enhancer3d/data}"
HIC_SINGLETONS="${HIC_SINGLETONS:-data}"
HGSVC="${HGSVC:-data/_hgsvc}"
PY="${PY:-.venv/bin/python}"
GTF="$E3D/gencode.v40.basic.annotation.gtf"

"$PY" web/scripts/make_dataset.py data "$SITE" GM12878 --gtf "$GTF" \
  --enhancers "$E3D/enhancer_atlas/GM12878_enhancers_hg38.bed" --label "GM12878, lymphoblastoid" --lymphoblastoid \
  --singletons "$HIC_SINGLETONS/GM12878/GM12878_hic_25kb_singletons.bedpe"
"$PY" web/scripts/make_dataset.py data "$SITE" H1ESC --gtf "$GTF" \
  --enhancers "$E3D/enhancer_atlas/H1ESC_enhancers_hg38.bed" --label "H1, embryonic stem cell" \
  --singletons "$HIC_SINGLETONS/H1ESC/H1ESC_hic_25kb_singletons.bedpe"
"$PY" web/scripts/make_dataset.py data "$SITE" HFFC6 --gtf "$GTF" \
  --enhancers "$E3D/enhancer_atlas/HFFC6_enhancers_hg38.bed" --label "HFFc6, foreskin fibroblast" \
  --note "RNAPII loops come from BJ, a different foreskin fibroblast line, since HFFc6 RNAPII ChIA-PET was not reachable." \
  --singletons "$HIC_SINGLETONS/HFFC6/HFFC6_hic_25kb_singletons.bedpe"
"$PY" web/scripts/add_sv_set.py "$SITE" HGSVC "$HGSVC/variants_freeze4_sv_insdel_alt.vcf.gz" \
  --label "HGSVC freeze 4 insertions and deletions"
