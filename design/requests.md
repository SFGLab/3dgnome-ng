# Requests and their state

The maintainer moves between the website and tbooo. This file lists each request, its state and the
place of its record. It is updated in the same change as the work. Last update 2026-11-18.

State words: done, open, waiting, held. A waiting item states the person or the job it waits for.

## Website, 3D-GNOME NG on team-aragorn

The site is at `https://3dgnome.mini.pw.edu.pl/new/` and outside access works since 2026-11-18. It runs
the image at `72c821d`. Record: `web/deploy/aragorn.md`.

| Request | State | Notes |
|---|---|---|
| Serve the site under `/new` | done | `BASE_PATH` |
| Share a job list between computers | done | Sync codes |
| One container for site and engine, a registry on the raid | done | |
| HiCDiffusion as a job step | done | |
| Accessions for every data source, privacy notice | done | Four accessions are still marked to confirm, see below |
| Packages without published advisories | done | Checked against osv.dev |
| 1000 Genomes call set, 3,202 samples | done | `data/svs/1KGP.vcf.gz`, typed sample field |
| Duplications applied | done | In the vendored `personal3d`, not yet in triosformer |
| Three faults of the model page's look, nicer variants button | done | |
| TBOOO on the Cite page | done | Commit `a01b0ed`, live since `72c821d` |
| Loading state of the contact map | done | Commit `72c821d`. A card with a spinner inside the map frame. |
| Sample table for the picker: population, sex, trios | open | `design/tbooo-data.md`, step 1. Agreed 2026-11-18. |
| Small variants in CTCF motifs | open | `design/tbooo-data.md`, step 2. Needs the 2022 panel on aragorn. |
| Measured expression beside a structure | open | `design/tbooo-data.md`, step 3. MAGE first, then GEUVADIS. |
| Four accessions to confirm | waiting, maintainer | CTCF ChIA-PET of H1ESC and HFFc6, expression of the three lines, HGSVC freeze 4 |
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

Held by the maintainer on 2026-11-18 while the trial and the download run. Plan and findings:
`design/tbooo-data.md`. Code: branch `sv-panel` in `~/Desktop/biobank/tbooo`, not pushed.

| Request | State | Notes |
|---|---|---|
| Inventory of the three copies | done | `design/tbooo-data.md` |
| Citation of the repository | done | See the website table |
| Downloader on the 2022 panel with structural variants | done | Commit `0fe656c` |
| 2022 panel on eden | waiting, job 1945120 | Into `data/raw/1kg/`, checked against the release's manifest |
| 2022 panel on aragorn | held | After tbooo is consistent |
| Sweep of the code for mistakes | open | Fixed so far: SGDP sample table and ids `1cc0ff0`, genetic positions of the array files `c9e3f7a`, sample files of the imputed and exome outputs `fe74093`, size of the per sample files `897ea07`. Not read yet: the rest of QC, phenotypes, expression, restricted cohorts. |
| The 100 TB | done | Cause found and fixed in `897ea07`. The fixed stage is not run yet. |
| SGDP on GRCh38, phased | waiting, job 1945118 | The trial on chromosome 22 decides the tools. The stage itself is not written. |
| Structural variants for SGDP | held | Imputation, long indels, or HGDP in its place. HGDP is unverified. |
| Full run on eden | waiting, maintainer | After the branch is complete. Then its data goes to the site. |

On eden the tools are in `tbooo/tools/`: bcftools 1.25, Picard 3.5.0, Beagle 5.5. The batch scripts are in
`playground/sgdp_grch38/`. On frodo the raid is full, and `data/tmp/nygc_split_chr1` to `chr3`, 6.7 TB,
are of no use to the fixed stage.

## Standing answers

- Compose commands for the maintainer carry full `-f` paths and no `cd`.
- No scripted requests to the public address from the laptop.
- Jobs on eden are submitted by the maintainer, or by the assistant when told to for a named piece of work.
- The restricted cohorts of tbooo are private and stay out of the site.
