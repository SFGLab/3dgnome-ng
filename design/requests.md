# Requests and their state

The maintainer moves between the website and tbooo. This file lists each request, its state and the
place of its record. It is updated in the same change as the work. Last update 2026-11-18.

State words: done, open, waiting, held. A waiting item states the person or the job it waits for.

## Website, 3D-GNOME NG on team-aragorn

The site is at `https://3dgnome.mini.pw.edu.pl/new/` and outside access works since 2026-11-18. It runs
the image at `d8db2ca`. Record: `web/deploy/aragorn.md`.

| Request | State | Notes |
|---|---|---|
| Serve the site under `/new` | done | `BASE_PATH` |
| Share a job list between computers | done | Sync codes |
| One container for site and engine, a registry on the raid | done | |
| HiCDiffusion as a job step | done | |
| Accessions for every data source, privacy notice | done | Nothing is left to confirm |
| Packages without published advisories | done | Checked against osv.dev |
| 1000 Genomes call set, 3,202 samples | done | `data/svs/1KGP.vcf.gz`, typed sample field |
| Duplications applied | done | In the vendored `personal3d`, not yet in triosformer |
| Three faults of the model page's look, nicer variants button | done | |
| TBOOO on the Cite page | done | Commit `a01b0ed`, live since `72c821d` |
| Loading state of the contact map | done | Commit `72c821d`. A card with a spinner inside the map frame. |
| Layout of the model page: example tags, the own data card | done | Commit `71a8239`, live. A long tag goes below its title. The own data card takes a whole row where it would start one. |
| Accessions of the CTCF loops, checked against enhancer3D | done | Commit `71a8239`, live. The files are enhancer3D's, the 4DN in situ ChIA-PET sets 4DNES7IB5LY9, 4DNESR9S8R38 and 4DNESCQ7ZD21, read from the portal's records. The GM12878 row had named Tang 2015. |
| Example descriptions checked against the data | done | Commit `aa464ad`, live. HOXA has 8 CTCF loops and no longer says dense. The MYC window moved to chr8:127.0-128.5 Mb to hold the enhancers its loops reach. GSTM1 states the deletion's change. TNFRSF8 was true and stays. |
| HFFc6 as a hosted cell type | done | Live since 2026-11-18. The build of 2026-11-11 had made two cell types. The deep map 4DNFIB59T7NN is on the host, its md5 equal to the copy on team-frodo. A test job of chr1:20.0-21.5 Mb finished with 68 CTCF and 112 RNAPII loops, and its contact map reads the deep map. |
| Plain texts in the settings selector | done | Commit `f9d38aa`, live. Each of the 32 settings has its own short text in `settings_schema.HELP`. `SETTINGS.md` is unchanged. The hosted cell types had no compartment track then and have one since the row on compartment tracks below. |
| Choice box of a setting past its pane, loading sheen of the map | done | Commit `6240daa`, live. |
| Hover and press on every clickable element, one arrow for every select, the contact map's toolbar | done | Commits `ae97841` and `6ed7f40`, live. Set once in `base.css`. A link is underlined under the pointer. |
| Sync panel, own data dialog and job heading in the site's design | done | Commit `6719582`, live. Two text styles in the panel, the code as high as its buttons, link buttons underlined under the pointer, the dialog's steps and droppers as on the form, the heading with the region in megabases. |
| Readable 3D model with a focus | done | Commits `cf529f7` and `8b38a72`, live. Chromatin toned along the region, weaker fog. The chain at the pointer and at each selected bead is solid and the rest is see through. |
| Compare view with both sides and a dropdown of jobs | done | Commit `8b38a72`, live. The head of the other side opens the jobs under itself. |
| Pager over tables with rows per page | done | Commit `cf529f7`, live. 25, 50, 100 or 250, kept in the browser. |
| Hover for job tabs, table rows and sorting column headers | done | Commit `cf529f7`, live. |
| Light theme by default, compare rows that select a gene, every structure file listed, no header tagline | done | Commits `18eafd6` and `299a34f`, live. |
| Crash of the job page on any structure button | done | Shipped in `299a34f`, fixed in `575dad6`. An effect returned the promise recent Chrome gives from scrollIntoView. A test covers it. |
| See through part without speckle, clearing of a compare selection | done | Commit `575dad6`, live. |
| Own data dialog: readable rules, one choice height, three text styles, one vertical gap | done | Commits `0fb2ee2` and `dfc6e74`, live. |
| Variants dialog keeps the hosted choice over a look at the upload, its button waits for a region | done | Commit `c777674`, live. |
| Gene name sent to the form on choice, plain background choices, the job name label | done | Commit `380e6d0`, live. |
| Running job: the log over the whole screen under a header with the bar | done | Commit `5e79213`, live. The log holds the last 200 lines and follows its end. |
| Front page previews that tell the examples apart, A, B and the distance in the hero | done | Commit `65f688a`, live. Drawn from the jobs by `web/scripts/make_previews.py`. With RNAPII loops 6.0 +- 1.6 bead units over 10 structures, with CTCF loops alone 9.9 +- 2.9. |
| Front page: the first picture back with A, B and the distance, one card for TNFRSF8 with and without its RNAPII loop, no line on private jobs | done | Commit `a736297`, live. The rendered picture is the hero again with its marks drawn over it. The two TNFRSF8 cards are one card of two drawings on one scale, A at one place in both. The chain is smooth, nearer strands cross over farther ones, the stretch between A and B is darker, and the view is the turn about the line from A to B that leaves the fewest beads in front of the two. |
| Agreement card of the contact map in two short lines | done | Commit `a736297`, live. The number is a Spearman correlation among pairs of bins the same distance apart. The full account of each number is its tooltip. The value itself is the model's and is not changed. |
| Compartment tracks for the hosted cell types | done | Commit `a736297`, live. `web/scripts/make_compartments.py`, `web/deploy/aragorn.md`. A and B at 100 kb from each deep map, per arm, the gene rich side A. The term stays off by default and a request can turn it on. The setting's text no longer says what the site lacks. Test job `d9957acdd5291b1e`, chr1:20-25 Mb with the term on, three structures in 827 s: the track is read and the smooth stage takes 275 s a structure against 4 s without, since the term visits every bead at every step. The setting's text says a job takes many times longer. A cell grid for the term is not built. |
| Contact map: no line of the two halves over the map, a box for the part a zoom in will show | done | Commit `aef4f8c`, live. The halves are in the two choices beside the map. The box is the square of the map on screen that the next map fills, drawn until that map arrives. A zoom out draws none. |
| Jobs keep their state over a restart of the site | done | Commit `9037156`, live. The engine already skipped a structure whose file exists and the site already queued an interrupted job again, so finished structures were kept. Closed now: a structure file takes its name only once whole, the table beside it is written first, and the log of the first run is kept with a line on where the job goes on. The structure in progress at the restart is computed again. A HiCDiffusion prediction is made again. |
| Method page: each energy term drawn on a small chain, tandem duplications among the variant rules | done | Commit `90cd894`, live. Each row of the two terms tables has a drawing of a few beads before the term acts and after, with the term's effect as the drawing's description. The variants section has a Duplication card and says how the Hi-C contacts follow. |
| Method page: the orientation term drawn after the 3D-GNOME figure, room inside Figure 3b | done | Commit `4f8247e`, live. The drawing shows the two motif directions along the genome over the loop they give, a hairpin for motifs that face each other and a full turn for motifs of one direction, the two motifs pointing the same way in space. The page's equation for the term had the wrong sign. The engine scores `((1 - o_i . o_j) / 2)^2` with `symmetric_motifs = yes`, least for motifs that point the same way, and the page had a plus. The setting's text said the motifs turn toward each other and is corrected. |
| Gene picker in the region field | done | Commit `cc7cb74`, live. Typed letters list up to eight matching genes with their positions. A click or the arrow keys and Enter take one. |
| No comma before "and" | done | The maintainer's rule of 2026-11-18, in `design/writing-rules.md`. Commit `0fc99d9`, live. Gone from the text under `web`, and the page tests refuse it. The other documents of the repository are not swept. |
| A restart of the site stranded a running job | done | Commits `ce6425b` and `2e30505`, live. The site queues again every job a restart left running or waiting, and job files are written whole. Before a deployment the host is still checked for a running engine. |
| HFFc6's own RNAPII ChIA-PET in place of BJ | open | The 4DN portal answers again since 2026-11-18, and the set 4DNESI1WZ5HT was out of reach on 2026-11-11. A new loop set means the cross line prep is run again. |
| Sample table for the picker: population, sex, trios | done | Commit `729afe3`, live. `data/svs/1KGP.samples.tsv` from the panel's pedigree file: 3,202 samples, 602 with both parents in the set, 2 with one. A typed population code lists its samples, and a chosen sample states its parents and children as buttons. |
| Small variants in CTCF motifs | open | `design/tbooo-data.md`, step 2. Needs the 2022 panel on aragorn. |
| Measured expression beside a structure | open | `design/tbooo-data.md`, step 3. MAGE first, then GEUVADIS. |
| The last two accessions | done | Commit `ae97841`, live. Expression is the ENCODE total RNA-seq that enhancer3D reads, ENCSR820PHH, ENCSR895ZTB and ENCSR420ZKB. HGSVC freeze 4 is HGSVC2 release v2.0, Ebert et al., Science 2021. |
| Privacy notice: legal basis and log retention | waiting, data protection officer | |
| Side panel text on 14 days against shared jobs kept without limit | open | Wording |
| Removal of a job from a shared list | open | No way exists today |
| Front page line on the old server's 1000 Genomes structures | open | True of the old site, and no longer a difference between the two |
| Duplication patch in triosformer | open | `web/vendor/VENDORED.md` lists the changes |
| New site at `/`, old site at `/old` | open | The front page already links to `/old`. The old site has not been served under a prefix. |
| Trio card on the Results page and figure 2g | done | Commit `89412be`, live. Whole genome, 2026-11-18. Both arms are 198 of 198 blocks on eden, `out/lr_trio_ctcf` and `out/lr_trio_rnapol2`, after the six blocks that failed on a filesystem fault and on dgx-1's `CUDA_ERROR_LAUNCH_FAILED` were sent again. The structures, maps and loop files, 42 GB, were pulled to `~/trio_lr` on team-frodo and read there, 198 tasks of person by chromosome, `playground/rnapii/readouts_trio_genome.sh` and `trio_genome_summary.py`. Against each person's own Hi-C on chromosomes 1 to 22, with RNAPII loops: pile-up at RNAPII loops relative to the map's higher in 9 of 9, +44 to +67 percent, median +53. Stratum adjusted correlation of the summed map higher in 5 of 9, -4.3 to +6.8 percent, median +0.1. The chr1:1-60 Mb card had 2 of 9 and a median of -2 percent, on the earlier arms and the per structure score. Readouts and the redrawn figures are in `design/figures/rnapii-2026-11-18/`. The trio expression statistics are not redone on the whole genome. |
| Significance marks on the RNAPII figures | done | `playground/rnapii/fig_stats.py`, figures in `design/figures/rnapii-2026-11-19/figures/`. Marked where a panel compares two sets of many values. 1d, enhancer to promoter distance of the pairs an RNAPII loop alone joins, with against without RNAPII loops, Wilcoxon signed rank over 2,206 to 3,636 pairs a line, and against the reference pairs, Mann-Whitney U: p under 0.001 on every line. 2e, the link to expression, block bootstrap of the difference on the genes of both model sets: +0.019, +0.017 and +0.044, p 0.63, 0.62 and 0.29, no pair significant on chr1. 2g, nine people, exact signed rank: pile-up p 0.004, correlation with the whole map p 1.0. 2h, the TNFRSF8 locus over ten structures a set, Mann-Whitney U, p 0.001. The dumbbell panels of the three cell lines carry no test, since three paired values cannot reach 0.05. Pairs and genes of one chromosome are not independent and the nine people are three families, which the module says. Both results the maintainer doubted were checked on 2026-11-19 and stand. 2e: about 620 genes a pair on chr1, each correlation near -0.26 with a bootstrap spread of 0.04, and the per gene values of the two model sets agree at a Spearman of 0.44 to 0.55 only, so the spread of the difference is 0.035 to 0.041 against differences of 0.019 to 0.044. The models with RNAPII loops have the stronger link in 69, 70 and 86 percent of draws. The three pairs together, genes resampled once a draw: mean difference +0.027, spread 0.026, p 0.30. The smallest difference needs about 11,000 genes to be twice its spread, which the whole genome has and chr1 does not. 2g: the rises have a rank sum of 23 and the falls of 22 of 45, the most even split nine values allow, and its exact p is 1. |
| Three cell lines on chromosomes 1 to 22, figures and the Results page | done | 2026-11-19. The structures of the four model sets, E the cudaMMC models, C CTCF loops, R CTCF and RNAPII loops, M RNAPII ends merged into CTCF anchors, were pulled to `~/xc_lr` on team-frodo and read there. The cudaMMC models cover the whole genome on team-faramir, 100 a chromosome, and a seeded ten a chromosome were converted; an earlier note here that they cover chr1 alone was wrong. Micro-C loops were called on every autosome at 5 kb, `playground/rnapii/microc_genome.sh`, each chromosome balanced on its own, 25,783, 10,046 and 19,192 loops for GM12878, H1ESC and HFFc6. Drivers `readouts_crossline_genome.sh` and `crossline_genome_collect.py`, readouts, figure inputs and figures in `design/figures/rnapii-2026-11-19/genome/`. Hi-C stratum adjusted correlation of the summed map, E, C, R, M: 0.10, 0.29, 0.29, 0.34 for GM12878, 0.13, 0.45, 0.42, 0.46 for H1ESC, 0.15, 0.47, 0.47, 0.51 for HFFc6. Signed rank over the 22 chromosomes: E against C p under 0.001 on every line, C against R p 0.90, 0.028 and 0.50. Enhancer to promoter distance of the pairs an RNAPII loop alone joins, median, C to R: 4.25 to 3.46 over 22,586 pairs, 4.04 to 3.09 over 31,537, 4.60 to 3.82 over 15,878, every test under 0.001. Link to expression, C to R: 0.194 to 0.238, 0.230 to 0.269 and 0.173 to 0.215, differences +0.043, +0.039 and +0.044 with p 0.002, under 0.001 and 0.001 over 10,000 draws; chr1 alone had no pair significant. Micro-C loops placed above moved, E, C, R: 0.12, 0.23, 0.31, then 0.12, 0.35, 0.40, then 0.10, 0.39, 0.47, C against R under 0.001 on every line. Loops with an RNAPII loop and no CTCF loop: 0.13 to 0.66 of 2,130, 0.18 to 0.58 of 586, 0.23 to 0.64 of 1,318. Every comparison panel of figures 1 and 2 and of the supplement carries its mark. The cell line charts of the Results page hold these numbers and a note gives the tests. The charts under Other tools stay on chr1:1-60 Mb, MultiMM has no whole genome run. The check of the Micro-C loops against Hi-C was made on chr1:0-60 Mb alone. |
| Results kept in the repo, and no comparison shown unless significant | done | 2026-11-19. Everything the whole genome readouts wrote on team-frodo is under `design/figures/rnapii-2026-11-19/genome/`: `by_chromosome/` the readouts one line and chromosome a file, the expression tables of each model set in `e3d_<ARM>/`, the enhancer to promoter pairs in `pairs/`, the pooled scores and pile-ups in `structure/`, the Micro-C loop calls at 5 and 10 kb with their scores and call logs in `microc/`; `genes/` the gene tables; `inputs/` the loop and anchor files of each line the scoring read; `run/` the runner scripts and every log; `readouts/` and `fig_inputs/` the collected tables the figures read. The trio runner scripts and logs are in `design/figures/rnapii-2026-11-18/run/`. Left on the workstation: the structures, 23 GB, the cudaMMC models, the expression intermediates, 36 GB, and the Micro-C slices at 5 kb, 3.2 GB, all of which the scripts make again. The maintainer's rule: a figure or a paper shows no result that is not significant. `fig_stats.SHOWN` is 0.05 and `figures_two.py` draws a comparison only under it. Figure 2g loses the trio correlation with the whole Hi-C map, p over 0.99, and keeps the enrichment at RNAPII loops, p 0.004. The panel was nine dots of a change and did not say against what. It is one row a person now, from the models with CTCF loops to the models with RNAPII loops, on the share of the person's own Hi-C enrichment the models reach, 0.34 to 0.41 and 0.51 to 0.63, with the Hi-C map as the line at one. Figure 2f loses its Hi-C half, C against R at p 0.90, 0.028 and 0.50, and keeps the Micro-C loops. The supplement keeps every value and marks a model set only when it differs from C; the H1ESC fall in Hi-C correlation with RNAPII loops, 0.45 to 0.42, is marked there. The Results page loses the 5 of 9 statistic and the sentence on unchanged agreement. The chr1 figure sets of 2026-11-18 and of `rnapii-2026-11-19/figures/` are as they were drawn. |
| GM12878 was scored against the shallow Hi-C map on the whole genome | done | Found 2026-11-19. The structure step of `readouts_crossline_genome.sh` took the first mcool of a line's directory by name, the shallow 4DNFIQ32RWCQ for GM12878 and the deep map for H1ESC and HFFc6. It takes the largest file now. GM12878 was read again against 4DNFIXP4QG5B: stratum adjusted correlation E, C, R, M 0.19, 0.53, 0.52, 0.58 where the shallow map gave 0.10, 0.29, 0.29, 0.34; E against C p under 0.001, C against R p 0.35, C against M p under 0.001. The shallow readouts are kept in `by_chromosome/structure_gm12878_shallow_map/` and `readouts_gm12878_shallow_map/`. Figures 1c and supplement a and the Results page carried the shallow numbers for some hours and are corrected. Micro-C, the pairs and expression never read the Hi-C map. |
| Hi-C correlation at short range, figure 2g | done | 2026-11-19, `playground/rnapii/short_range_scc.py`, readouts in `by_chromosome/shortrange/`. The measure fixed before the run: raw stratum adjusted correlation at 10 kb on pairs up to 2 Mb, C against R, signed rank over 22 chromosomes, medians. GM12878 0.452 to 0.470, p 0.14. H1ESC 0.390 to 0.357, p 0.054. HFFc6 0.406 to 0.436, p 0.008. One line of three, so it does not make a panel. By separation at 10 kb the picture is consistent on every line: under 100 kb R is higher on GM12878, 0.477 to 0.505, p 0.042, and HFFc6, 0.408 to 0.472, p 0.002, and not on H1ESC; from 100 to 500 kb R is lower on all three, 0.412 to 0.367, 0.345 to 0.302 and 0.395 to 0.296, each p under 0.001; from 0.5 to 2 Mb lower on H1ESC and HFFc6. The set with RNAPII ends merged into CTCF anchors, M, is higher than C to 2 Mb on GM12878, 0.452 to 0.501, and HFFc6, 0.406 to 0.449, both p under 0.001, and level on H1ESC. Figure 2f stays the Micro-C loops. The wording there is the share of Micro-C loops whose two ends the models bring together, above the same loops shifted away. The maintainer asked for the result by separation in the figure and written down. It is figure 2g, one panel a band, a line drawn only in the bands where the two model sets differ, and the trio and locus panels are 2h and 2i. `crossline_genome_collect.py` writes the bands into `by_chromosome.tsv`. The result is written in `design/2026-11-12-rnapii-figures.md`, section Results on chromosomes 1 to 22, and in a note on the Results page. The mechanism of the loss from 100 to 500 kb is not traced. |

## Other sites on team-aragorn

Record: `web/deploy/aragorn.md` and `web/deploy/hosted/`. Files on the host: `/mnt/raid/hosted-sites/`.

| Request | State | Notes |
|---|---|---|
| Stack of every site for the faculty admin | done | Sent by the maintainer. Outside access was reopened. |
| Backup of all images | done | `/mnt/raid/aragorn-backup-2026-11-18/` |
| spring-model, lab page and Bertrand without critical records | done | Lab page is on Django 5.2. Bertrand's server is compiled again from source. |
| Old 3D-GNOME on a supported stack | done | Switched by the maintainer. enhancer3D stays on it. |
| Reboot of the host for the new kernel | waiting, maintainer | |
| Old 3D-GNOME: enhancer3D search and file uploads | open | Not tested on the port |
| Old 3D-GNOME: database on SQLite | held | Offered. It removes the MariaDB container and its records. |
| Lab page: five logos return 403, file browser setting, four pending migrations | open | All three predate the switch. `sudo chgrp -R 101 /var/www/labwebpage/uploads` fixes the logos. |
| Bertrand: Docker socket mount, ports 8080 and 8081 open | waiting, its owner | |

Left for the maintainer to delete. Nothing was removed.

- aragorn: the trial copies `sm-stage`, `bertrand-trial`, `labwebpage-trial` and `gnome3d-old-trial`, the
  stopped `webservice_container` and `sm_webservice-web-1`, `/mnt/raid/3dgnome-new/stage/`, the request
  21 made as a test on the old site.
- frodo: `~/3dgnome-web-bump*`, the containers `gnome3d-bump` and `gnome3d-base`, the test registry.
- eden: `~/.conda`, `~/.mamba` and `~/.cache/conda`, bookkeeping files of the tool install.

## tbooo

Held by the maintainer on 2026-11-18 and taken up again the same day. Plan and findings:
`design/tbooo-data.md`. Code: branch `sv-panel` in `~/Desktop/biobank/tbooo`, not pushed.

| Request | State | Notes |
|---|---|---|
| Inventory of the three copies | done | `design/tbooo-data.md` |
| Citation of the repository | done | See the website table |
| Downloader on the 2022 panel with structural variants | done | Commit `0fe656c` |
| 2022 panel on eden | done | In `data/raw/1kg/`, 46 files, 31 GB, each with the md5 of the release's manifest, checked by job 1945284. The first job, 1945120, was stopped with chromosome 7 at 786 MB of 1,752 MB on a connection of 46 to 77 KB/s. `playground/sgdp_grch38/panel_file_ranges.sbatch` fetched chromosome 7 in eight byte ranges in eleven minutes, job 1945276. The first job's file of chromosome 8 had the right size and another md5, and the same script replaced it. The partial of 7 and the bad 8 are kept beside the files as `.partial-1945120` and `.replaced-1945284`. |
| SGDP sample table and id map of the instance, keyed on the name inside the VCF | done | The maintainer's decision of 2026-11-18. `playground/sgdp_grch38/sgdp_ids.sbatch`, trial 1945270 and the instance 1945271, tbooo commit `c382e3d`. 277 samples, every one of the 277 names inside the pVCF and none besides, where the map of June had 265 keyed on an ENA accession no VCF carries. 238 people are in both maps and the eid stays for 2 of them. The old table, map and rename file are beside the new ones as `.before-illumina-key`. No genotype file of the instance carries an SGDP id. `metadata/ukb_sqc_v2.txt` has 265 rows of the old ids and `Showcase/participant.parquet` is of June, so the QC and phenotype stages have to run again. The 1000 Genomes map is untouched. |
| SGDP on GRCh38 for chromosomes 1 to 22 | done | Array 1945285, 22 of 22 tasks, 18 minutes for chr21 to 2 h 13 min for chr2, from the branch at `57fe5be`. In `data/raw/sgdp/grch38` of the instance, 2.9 GB: a phased and an unphased file and a count table a chromosome. 29,981,279 variant records on GRCh37, 7,216 not lifted, 29,784,371 phased for 277 samples under the new ids, none with an unphased or missing genotype. 187,159 records, 0.62 percent, are at a position the panel has with other alleles and stay in the unphased file alone. X is not built. Script `playground/sgdp_grch38/sgdp38_genome.sbatch`. |
| SGDP on the site | done | Commit `978de9b`, live. The maintainer's request of 2026-11-19. Test job `64d80f9d00847d22`, S_Mbuti-1 on chr1:100.5-104.9 Mb: 8 variants applied, none skipped, the region 4,559 bp longer. The SGDP calls hold no insertion or deletion of any length, so the route of long indels gives nothing and the set is imputed from the 30x panel, arrays 1945920 and 1945925 on eden. 35,434 symbolic records at DR2 of 0.8 or more for 277 samples, named by the project's ids. On the 23 people of both projects, held out: 98.45 percent of genotypes agree, 95.3 percent of the panel's alternate alleles imputed, 94.6 percent of the imputed in the panel. The cut of 0.8 and the choice of imputation over HGDP are the assistant's, made when the maintainer asked for the work to be finished overnight, and the catalog entry is one key to remove. The 2022 panel and tbooo's phased SGDP files are on aragorn under `data/variants/`. `web/deploy/aragorn.md` has the record. |
| QC and phenotype tables of the instance on the new SGDP ids | done | Job 1945274, five minutes. `ukb_sqc_v2.txt` has 3,479 samples, the 3,202 of 1000 Genomes and the 277 of SGDP, each SGDP id in the id map. `ukb_rel.txt` has 50 related pairs, as before. `participant.parquet` has 3,479 rows and 39 columns, 277 SGDP rows, GEUVADIS scores on 449 rows. The branch's phenotype stage writes the column `p24051` where the table of June had `p23149`. The three old files are beside the new ones as `.before-illumina-key`. plink2 2.0.0-a.7.13 and KING 2.3.2 are in `tools/env-qc`. Script `playground/sgdp_grch38/instance_tables.sbatch`. |
| 2022 panel on aragorn | done | Pulled from eden on 2026-11-19, 32.4 GB, `data/variants/1KGP/`, with the release's manifest. |
| Sweep of the code for mistakes | open | Fixed so far: SGDP sample table and ids `1cc0ff0`, genetic positions of the array files `c9e3f7a`, sample files of the imputed and exome outputs `fe74093`, size of the per sample files `897ea07`, flags of the participant table and the QC file, and the run targets that did nothing `b24ae1c`, expression scores lost from a rebuilt participant table `5a2b26b`, the cohort file of two builds `a71dfef`. Read and left as it is: restricted cohort ids are numbered over the sorted sample list, so a new sample renumbers the later ones. |
| The 100 TB | done | Cause found and fixed in `897ea07`. Checked on chromosome 22 of the panel for 20 samples: no file holds a record its sample does not carry, and a file is about 0.5 MB, 1.6 GB for 3,202 samples. |
| SGDP on GRCh38, phased | done for the autosomes | `tbooo map sgdp38`, commit `a71dfef` on `sv-panel`. The trials and its first run on chromosome 22 are below. The X chromosome is not built. The full run is the maintainer's. |
| Structural variants for SGDP | held | Imputation, long indels, or HGDP in its place. HGDP is unverified. |
| Full run on eden | waiting, maintainer | After the branch is complete. Then its data goes to the site. |

Trial on chromosome 22, job 1945118, 17 minutes on 16 cores in all.

- The read of the 13.8 GB file of every base took under 7 minutes and left 388,268 variant records.
- The lift to GRCh38 lost 575 of 392,261 records.
- 23 SGDP samples are also in the panel. For one of them the two call sets agree at 99.98 percent of
  236,870 shared sites, and at 99.88 percent of the sites where either carries the variant.
- Beagle phased in 6 minutes. Its output holds the panel's 1,066,557 sites alone. The 125,931 SGDP
  records the panel does not have, a third of them, are dropped. The stage needs a phasing step that
  keeps them.
- Of the panel's 1,462 structural variants on the chromosome, 696 are imputed into one SGDP sample or
  more. The quality is spread wide: the median dosage r squared is 0.3, and four in ten are at 0.73
  or over.

Phasing trial on chromosome 22, jobs 1945205 and 1945229. The reference leaves out the 23 people both
projects sequenced and their relatives, and the panel's phase of those people is the truth.

- With the panel as the reference the switch rate over the shared sites is 1.01 percent. Beagle
  writes the panel's sites alone.
- With the SGDP records the panel lacks added to the reference, as sites where every panel sample has
  the reference allele, all 120,883 of them are kept and the switch rate is 1.15 percent. The stage
  does this.
- A fifth of the shared sites, kept out of the reference, stood in for such records. Their phase
  agrees with the panel's at 90 percent for an allele count of 2 to 5 among SGDP, 95 percent for 6
  to 20 and 97 percent above. For a variant one sample carries it is chance, and 81 percent of the
  records the panel lacks are of that kind.
- With no reference the switch rate is 4.2 percent.

First run of the stage, job 1945242, chromosome 22, 19 minutes on 16 cores: 388,268 variant records,
575 not lifted, 384,636 phased for 277 samples, none with an unphased or missing genotype. 3,019
records at a position the panel has with other alleles stay in the unphased file alone.

On eden the tools are in `tbooo/tools/`: bcftools 1.25, Picard 3.5.0, Beagle 5.5. The batch scripts are in
`playground/sgdp_grch38/`. A copy of the branch for the stage's run is in `tbooo/tools/sv-panel-src/`, and its outputs are in
`data/tmp/sgdp38_smoke/`. On frodo the raid is full, and `data/tmp/nygc_split_chr1` to `chr3`, 6.7 TB,
are of no use to the fixed stage.

## Standing answers

- Compose commands for the maintainer carry full `-f` paths and no `cd`.
- No scripted requests to the public address from the laptop.
- Jobs on eden are submitted by the maintainer, or by the assistant when told to for a named piece of work.
- The restricted cohorts of tbooo are private and stay out of the site.
