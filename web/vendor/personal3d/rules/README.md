# Introduction

This repository provides a tool for building individualized genomic interaction
networks based on a reference interaction network and information on structural
variants (SVs). The method is described in **Spatial Chromatin Architecture 
Alteration by Structural Variations in Human Genomes at Population Scale** by 
Sadowski et al. (bioRxiv 266981).

# Usage
The program requires the following python packages to be installed: 
_sortedcontainers_, _PyVCF_, _pandas_, _numpy_.
 
Create a directory with the source code and run _SVintroduce.py_ script. A BED
file with genomic segments, a BEDPE file with genomic interactions, a BED file
with chromosome lengths and a VCF file containing SVs should be provided as an
input. An exemplary data is given in the _data_ folder.
```bash
$ python SVintroduce.py data/GM12878_CCDs.bed data/GM12878_CTCF_PET_clusters_CTCF-cohesin_co-occupancy.bedpe data/human_hg19_chrom_sizes.bed data/example.vcf -m data/MA0139.1.pfm -s TESTID
```

# Result

The program will output modified files containing: (1) genomic segment 
boundaries, (2) genomic interactions, (3) merged interaction anchors and (4) 
chromosome lengths. These files can then be used as an input to the 
[3D-GNOME modeling engine](https://bitbucket.org/3dome/3dome_mmc) to generate 
chromatin 3D models. 
