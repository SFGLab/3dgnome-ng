# The site on team-aragorn

The new site runs beside the old one at `https://3dgnome.mini.pw.edu.pl/new/`. The faculty
proxy ends TLS and passes the host to nginx on team-aragorn, port 80. nginx passes `/new/` to
the site's container on the loopback, port 8770, and everything else to the old site.

## Layout

```
/mnt/raid/3dgnome-new/
  app/repo/         a checkout of the deploy branch, from a git bundle
  app/site.env      the variables below, a copy of aragorn.env
  app/maps.sh       fetches the deep contact maps of GM12878 and H1ESC from the public 4DN bucket
  app/maps_hffc6_v3.sh, app/maps_hffc6_parallel.sh  the same for HFFc6, `web/deploy/maps_hffc6.sh` and `maps_hffc6_parallel.sh` here
  data/             DATA_DIR, read only in the container
    catalog.json
    cell_types/     GM12878, H1ESC, HFFC6, each with compartments.bed
    maps/<CELL>/    the deep map of each cell type, the catalog's contact_map
    svs/            HGSVC, 1KGP and SGDP, one indexed VCF each, and the sample tables of 1KGP and SGDP
    variants/       1KGP, the 30x panel of 2022-04-22. SGDP, its GRCh38 phased files. SGDP_sv, its imputed structural variants
    hicdiffusion/   the two checkpoints
    reference/      hg38.fa and its index
  results/          RESULTS_DIR, written by the container as nk2
  registry/         the image registry's storage
  logs/             build and download logs
```

The raid is XFS on `/dev/md0`, RAID6 over eight disks, mounted at `/mnt/raid`. `/mnt/raid`
belongs to root, so the registry is inside the site's directory.

## Compartments

Each cell type has `compartments.bed`, A and B labels at 100 kb from its deep map, the catalog's
`compartments` key. `web/scripts/make_compartments.py` writes it inside the site's container,
which has cooler and the maps:

```
docker exec -i gnome3d-site-site-1 python - /data/maps/GM12878/4DNFIXP4QG5B.mcool::/resolutions/100000 \
    /data/cell_types/GM12878/genes.tsv /data/cell_types/GM12878/centromeres.bed \
    < make_compartments.py > GM12878.bed 2> GM12878.stats.jsonl
```

Each arm is taken alone and the side with more gene starts is A. The reports of the build of
2026-11-18 are in `web/deploy/compartments/`. GM12878 has 43 arms, H1ESC 40 and HFFC6 41. The
arms left out are the short arms of the acrocentric chromosomes, chrY, and in H1ESC chr18p and
chr19q, where the signal does not follow gene density. On the autosomes of GM12878 the labels
agree with the validation study's 100 kb track at 86 to 94 percent of bins. A job writes its
chromosome's rows into its inputs and the term stays off unless the request turns it on. The
catalog before the key is `data/catalog.json.before-compartments`.

## The 1000 Genomes call set

`data/svs/1KGP.vcf.gz` holds the phased structural variants of the 3,202 samples of the 1000
Genomes 30x panel, 97,862 sites on chromosomes 1 to 22. The 3D-GNOME 3.0 server keeps them as one
VCF per sample in `/home/gnome/3dgnome_data/data_vcf_modified/data`, each with the records that
sample carries. `web/scripts/merge_sample_vcfs.py` made the one file from those, run in a
container of the site's image with that directory mounted read only. Two samples were compared
record for record with their source files. The site applies its 51,380 deletions, 31,538
insertions, 572 inversions and 14,372 duplications, the last read as tandem. The catalog before the set was
added is `data/catalog.json.before-1kgp`.

## The SGDP call set

`data/svs/SGDP.vcf.gz` holds imputed structural variants of the 277 samples of the Simons Genome
Diversity Project, 35,434 sites on chromosomes 1 to 22: 17,395 deletions, 13,915 insertions, 3,962
duplications and 162 inversions, 3,382 to 5,424 a sample. The project's calls hold single base
variants alone. tbooo lifts them to GRCh38 and phases them against the 30x panel, and
`playground/sgdp_grch38/sgdp_sv_impute.sbatch` there imputes the panel's 97,862 symbolic records
onto the samples with Beagle 5.5. `site_sv_set.sh` keeps the records with DR2 of 0.8 or more
that a sample carries and names the samples by the project's ids. The cut is a choice.

The check. Twenty three people are in both projects. Imputed from a panel without them and their
relatives on chromosomes 18 to 22, the records the set would keep agree with the panel's own
genotypes at 98.45 percent, 95.3 percent of the panel's alternate alleles are imputed and 94.6
percent of the imputed ones are in the panel. Under 1 percent in the panel the two shares are
84.5 and 76.7 percent. Those people are of the panel's populations, so a sample of a population
the panel lacks does worse. Table: `playground/sgdp_grch38/results/sv_eval/` in tbooo.

`data/variants/` holds what the set was made from and what the small variant work needs: the
panel, tbooo's phased SGDP files and the imputed files with dosage and genotype probability.
The catalog before the set was added is `data/catalog.json.before-sgdp`.

## Feature flags

`data/config.json`, beside the catalog, turns parts of the site on and off. The copy of the
site's file is `web/deploy/config.json`.

    {"features": {"results": true, "method": true, "trios": false}}

`results` and `method` are the two pages. Off, the page leaves the top bar, its address leads to
the front page and other pages stop pointing at it. `trios` is the nine people on the Results
page, off on this site. A feature the file leaves out is on. The server reads the file again
when it changes, so an edit needs no restart and no build. A site does not start on a file with
an unknown key or feature, and a broken edit of a running site is logged and the configuration
it had stays.

## Run

All commands run from `app/repo/web` as nk2, who is in the docker group.

```bash
E=/mnt/raid/3dgnome-new/app/site.env
docker compose --env-file $E -f docker-compose.registry.yml up -d      # once
docker compose --env-file $E -f docker-compose.single.yml build
docker push localhost:5000/gnome3d-web:<tag>
docker compose --env-file $E -f docker-compose.single.yml up -d --no-build
```

The image is built on team-aragorn itself and pushed to its own registry, which keeps the
earlier tags. A new deployment fetches the new commit into `app/repo`, sets `IMAGE` to the new
tag in `site.env`, and runs the last three commands. A step back sets `IMAGE` to an earlier tag
and runs `pull` and `up -d --no-build`.

## nginx

In `/etc/nginx/sites-available/3dgnome.mini.pw.edu.pl`, inside the `server` block, then
`sudo nginx -t && sudo systemctl reload nginx`:

```nginx
location = /new { return 301 /new/; }
location /new/ {
    proxy_pass http://127.0.0.1:8770;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto https;
    client_max_body_size 200m;
    proxy_read_timeout 300s;
}
```

`client_max_body_size` matches `UPLOAD_MAX_MB`. The server strips `/new` itself, so the
`proxy_pass` has no path.

## Versions and advisories

The faculty asks each hosted site for its container stack with exact versions. An image is
checked against osv.dev by name and version at three levels. The operating system's packages
are read with `dpkg-query` or `apk list` and scored by the CVSS 3 base score of each record.
The Python packages of both interpreters come from `pip freeze`. The frontend is checked with
`npm audit --omit=dev`. No scanner runs on the host.

On 2026-11-18 the image at `5891e3b`, and the one at `2ee8c6b` built from it, has one record at 9.0 or over, CVE-2019-1010022 in glibc,
which its authors dispute and which every Debian image carries. Both Python environments are
clear but for one record of moderate severity on biopython 1.81, which HiCDiffusion's
requirements pin and which has no fixed release. The shipped frontend is clear. The image holds
no compiler, and its libcurl 8.23.0 and htslib 1.25 are built from their releases in the build
stage. `site.env.<tag>` beside `site.env` keeps the variables of each deployed tag.

## The other sites on the host

team-aragorn serves six more names from other people's containers, each behind the host nginx
at an address its compose file fixes. `hosted/` holds what was built for them on 2026-11-18,
and `/mnt/raid/hosted-sites/` on the host holds the same files with a copy of each site's
source, the build logs and the package lists. Each site runs through its own compose file with
`compose.bump.yml` laid over it, and the step back is the same command without the overlay.
The commands name both files by their full path, since the sites' directories are closed to
other users and a failed `cd` leaves the shell in the last site's directory.

```bash
sudo docker compose -f <the site's compose file> -f /mnt/raid/hosted-sites/<site>/compose.bump.yml up -d --no-build
```

| Site | Compose file | What the overlay runs |
|---|---|---|
| spring-model | `/home/mkadlof/sm_webservice/compose.yaml` | a rebuild on python:3.12-slim with Django 5.2.19 and OpenMM 8.6.1, nginx 1.30.6, Redis 7.2.17, PostgreSQL 16.16 |
| lab page | `/var/www/labwebpage/compose.yaml` | nginx 1.30.6, and the Python environment on python:3.11-alpine with each package at the patch release of its own series, Django 4.2.30 |
| Bertrand | `/home/vyatsenko/bertrand-interface/docker-compose.yml` | the documentation on nginx 1.30.6, and the server compiled again with Go 1.27.4 on Alpine 3.24, PocketBase kept at 0.18.2 |

spring-model's database took three migrations of django_celery_results at the switch, which
the old code ignores. The lab page stays on Django 4.2, which left support in April 2026, and
its move to 5.2 needs new major releases of its admin plugins. The old 3D-GNOME site in
`/home/gnome/3dgnome_webserver` runs Flask 1.1 on Python 3.8 and Ubuntu 20.04 and cannot be
cleared in place. `hosted/3dgnome-old/` is its port, two containers in place of one, and
`compose.port.yml` there carries the switch and the step back. Its analysis libraries stay at
pandas 1.5.3 and numpy 1.26.4 on Python 3.11, since the site's code was written for pandas 0.25
and later releases refuse or change calls it makes. The site's own tests are left out of the
image. Run in a container they empty its database. Image backups of 2026-11-18 and the dumps taken before each switch are in
`/mnt/raid/aragorn-backup-2026-11-18/`.
