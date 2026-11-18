# The site on team-aragorn

The new site runs beside the old one at `https://3dgnome.mini.pw.edu.pl/new/`. The faculty
proxy ends TLS and passes the host to nginx on team-aragorn, port 80. nginx passes `/new/` to
the site's container on the loopback, port 8770, and everything else to the old site.

## Layout

```
/mnt/raid/3dgnome-new/
  app/repo/         a checkout of the deploy branch, from a git bundle
  app/site.env      the variables below, a copy of aragorn.env
  app/maps.sh       fetches the two deep contact maps from the public 4DN bucket
  data/             DATA_DIR, read only in the container
    catalog.json
    cell_types/     GM12878, H1ESC
    maps/<CELL>/    the deep map of each cell type, the catalog's contact_map
    svs/            HGSVC and 1KGP, one indexed VCF each
    hicdiffusion/   the two checkpoints
    reference/      hg38.fa and its index
  results/          RESULTS_DIR, written by the container as nk2
  registry/         the image registry's storage
  logs/             build and download logs
```

The raid is XFS on `/dev/md0`, RAID6 over eight disks, mounted at `/mnt/raid`. `/mnt/raid`
belongs to root, so the registry is inside the site's directory.

## The 1000 Genomes call set

`data/svs/1KGP.vcf.gz` holds the phased structural variants of the 3,202 samples of the 1000
Genomes 30x panel, 97,862 sites on chromosomes 1 to 22. The 3D-GNOME 3.0 server keeps them as one
VCF per sample in `/home/gnome/3dgnome_data/data_vcf_modified/data`, each with the records that
sample carries. `web/scripts/merge_sample_vcfs.py` made the one file from those, run in a
container of the site's image with that directory mounted read only. Two samples were compared
record for record with their source files. The site applies its 51,380 deletions, 31,538
insertions, 572 inversions and 14,372 duplications, the last read as tandem. The catalog before the set was
added is `data/catalog.json.before-1kgp`.

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
