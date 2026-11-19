# README #

Example arguments:
-s /d/chiapet/stg.ini -c chr14 -o /d/chiapet/chr14/ -n chr14_test


# Installation #
In the following description it is assumed that git directory is ~/git/ and C++ projects directory is ~/projects/. True directories should be adjusted accordingly.

* Download git repository to ~/git/3dnome-mmc/.

## Create two projects: ##
* 3dnome-mmc (shared library, created in ~/projects/3dnome-mmc)
* 3dnome-mmc-tools (C++ application serving as an interface for the library, created in ~/projects/3dnome-mmc-tools)

## To set-up the shared library project: ##
1. Add a link to C++ files from git:
```
#!bash
~/projects/3dnome-mmc$ ln -s ~/git/3dnome-mmc/ src
```

2. Exclude tools/main.cpp from build (Right click -> Resource Configuration -> Exclude from Build)

3. Set C++ language standard to C++0x (-std=c++0x)

4. Project should compile right now

5. For simplicity we can link the resulting library to system-wide libraries directory:
```
#!bash
/usr/local/lib$ sudo ln -s ~/projects/3dnome-mmc/Release/lib3dnome-mmc.so lib3dnome-mmc.so
/usr/local/lib$ sudo ln -s ~/projects/3dnome-mmc/Debug/lib3dnome-mmc.so lib3dnome-mmc_d.so
/usr/local/lib$ sudo ldconfig
```

## To set-up the interface project: ##
1. Link to main.cpp from git:
```
#!bash
~/projects/3dnome-mmc-tools$ ln -s ~/git/3dnome_mmc/tools/main.cpp main.cpp
```
2. Set C++ language standard to C++0x (-std=c++0x)
3. Add library source directory to Includes (-I). In Eclipse this can be achieved by adding "${workspace_loc:/3dnome-mmc/src}" in the "Project -> Properties -> C/C++ Build -> Settings -> <Your compiler> -> Includes" field.  
4. Add "3dnome-mmc" as a required library (-l) and "/usr/local/lib" as a library search path (-L). In Eclipse these options can be found in "Project -> Properties -> C/C++ Build -> Settings -> <Your compiler> Linker -> Libraries" panel.


# Input and output files #
Input of the algorithm consists of several files. First of all, there are files corresponding to different types of ChIA-PET data: anchors, PET clusters and singletons. As singleton files tend to be large it is possible to provide inter- and intrachromosomal singletons files separately (using **singletons_inter** and **singletons** options, respectively). This allows the program to skip the interchromosomal files reading if they are not needed (e.g. when a single chromosome is reconstructed). When the subanchor heatmaps are to be generated it is beneficial to create the intrachromosomal singletons files for every chromosome separately (one should denote this by setting the flag **split_singleton_files_by_chr**), and to use these per-chromosome files rather than the bigger, aggregated files. These files can be easily created using a following commands:
```
#!bash
mkdir chr;
for i in `seq 1 22` X; 
   do 
      ( cat data.txt | awk '{if ($1 == "chr'$i'") print $0}' > chr/data.txt.chr$i ) 
   done
```
, where data.txt is the original data file. The resulting files are created in the *chr* subdirectory of the data directory, and they have the same name as the original file but with a chromosome id as a suffix.

Additionally to the ChIA-PET files mentioned one can provide a BED file with centromeres' locations and a file with definition of split of the chromosomes into segments. 

Paths of the files are provided via the setting file in the **[data]** section. There are following options available:

* data_dir - path to the data directory (anchors, clusters, singletons and singletons_inter are all relative to this directory)
* anchors - name of the anchors file
* clusters - names of the clusters files (comma separated). Each file should contain clusters   
* singletons - names of the singleton files (comma separated)
* singletons_inter - names of interchromosomal singleton files (comma separated)
* split_singleton_files_by_chr - if *yes* then the chromosome-splitted intrachromosmal files are used 
* factors - names of protein factors used in the experiment (e.g. "CTCF,RNAPII")
* segment_split - path to the BED file with segments split definition
* centromeres - path to the BED file with centromeres' locations

Following is the format description of the corresponding input files.

## Anchor file ##
This file contain info about the anchors. For every anchor the chromosome and genomic position of start and end is provided. Optionally the CTCF motif orientation may be provided as 'L' (leftward) or 'R' (rightward). 'N' may be used if the orientation is unknown. File format is as follows:

```
#!text
chromosome_id anchor_start anchor_end [orientation]
```
A sample file (with motif orientation provided) may look like this:

```
#!text
chr1	838908	841157	R
chr1	911279	912011	L
chr1	918286	922335	R
chr1	967152	969271	R
chr1	997732	1000167	L
```

## Cluster files ##
These files contain info about the PET clusters. Each file corresponds to a single transcription factor used in the experiment. Every line of a file describes a single PET cluster and consists of the chromosome, start and end position for the anchors overlapping with the PET cluster. Additionally, a PET count is provided. File format is as follows:

```
#!text
chromosome_id_1 anchor_1_start anchor_1_end chromosome_id_2 anchor_2_start anchor_2_end PET_count
```
A sample file may look like this:

```
#!text
chr1	838908	841157	chr1	911279	912011	11
chr1	838908	841157	chr1	997732	1000167	7
chr1	918286	922335	chr1	997732	1000167	75
chr1	967152	969271	chr1	997732	1000167	52
chr1	967152	969271	chr1	1306262	1308124	7

```

## Singleton files ##
Singleton files possess the same format as the clusters files, but the genomic coordinates correspond to the paired-end read mapping locations rather than the overlapping anchors:
```
#!text
chromosome_id_1 read_1_start read_1_end chromosome_id_2 read_2_start read_2_end PET_count
```

```
#!text
chr9	77361535	77361615	chr9	78624184	78624253	1
chr2	74756819	74756889	chr2	201675832	201675878	1
chr12	51818625	51818729	chr12	70877421	70877572	1
chr4	169237541	169237655	chr4	181760704	181760799	1
chr6	129055926	129055968	chr6	129170407	129170485	1
```