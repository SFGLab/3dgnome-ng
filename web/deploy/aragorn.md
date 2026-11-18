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
    svs/            HGSVC
    hicdiffusion/   the two checkpoints
    reference/      hg38.fa and its index
  results/          RESULTS_DIR, written by the container as nk2
  registry/         the image registry's storage
  logs/             build and download logs
```

The raid is XFS on `/dev/md0`, RAID6 over eight disks, mounted at `/mnt/raid`. `/mnt/raid`
belongs to root, so the registry is inside the site's directory.

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

The faculty asks each hosted site for its stack with exact versions. The image's packages are
checked against osv.dev by name and version, both interpreters, and the frontend with
`npm audit --omit=dev`. On 2026-11-18 the image at `4a106e9` has no advisory in either Python
environment but one of moderate severity on biopython 1.81, which HiCDiffusion's requirements
pin and which has no fixed release, and none in the shipped frontend. `site.env.<tag>` beside
`site.env` keeps the variables of each deployed tag.

## The other sites on the host

team-aragorn serves six more names from other people's containers, each behind the host nginx
at an address its compose file fixes. `hosted/spring-model/` holds a rebuild of spring-model on
a current base, tried on 2026-11-18 beside the running site: `Dockerfile.bump` and
`requirements.bump.txt` build `sm_webservice-django:2026-11-18` from a copy of the source,
`compose.stage.yml` runs the trial copy on the loopback at port 8772 with an empty database,
and `compose.bump.yml` is the overlay that moves the running site onto the new images. The
trial copy applied its migrations and finished two modelling tasks. The copy of the source and
the build log are in `/mnt/raid/3dgnome-new/stage/spring-model/`.
