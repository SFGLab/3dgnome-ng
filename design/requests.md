# Requests and their state

The maintainer moves between the website and tbooo. This file lists each request, its state and the
place of its record. It is updated in the same change as the work. Last update 2026-11-18.

State words: done, open, waiting, held. A waiting item states the person or the job it waits for.

## Website, 3D-GNOME NG on team-aragorn

The site is at `https://3dgnome.mini.pw.edu.pl/new/` and outside access works since 2026-11-18. It runs
the image at `6ed7f40`. Record: `web/deploy/aragorn.md`.

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
| Plain texts in the settings selector | done | Commit `f9d38aa`, live. Each of the 32 settings has its own short text in `settings_schema.HELP`. `SETTINGS.md` is unchanged. The hosted cell types have no compartment track, and the Compartments group does nothing on the site. |
| Choice box of a setting past its pane, loading sheen of the map | done | Commit `6240daa`, live. |
| Hover and press on every clickable element, one arrow for every select, the contact map's toolbar | done | Commits `ae97841` and `6ed7f40`, live. Set once in `base.css`. A link is underlined under the pointer. |
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
| Trio card on the Results page and figure 2g | waiting, eden jobs | Seven of 396 runs are missing. Poly arm tasks 194 and 195 were resubmitted on 2026-11-18. Tasks 11, 75, 84, 101 and 106 of array 1943270 failed, memory is the likely cause. |

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
| 2022 panel on eden | waiting, job 1945120 | Into `data/raw/1kg/`, checked against the release's manifest |
| 2022 panel on aragorn | held | After tbooo is consistent |
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
