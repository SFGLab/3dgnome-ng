# Jobs of the site

## How long a job takes

Measured 2026-11-11 on an RTX 3090 with a warm JAX cache: GM12878 chr1:1-3 Mb, CTCF and
RNAPII loops, HG00512's first haplotype from HGSVC (69 variants), ten structures, 58 s from
submission to the finished page. Jobs run on the queue and the page refreshes itself, so a user
waits about a minute. The first job after a restart also pays the JAX compile, which the cache
under `RESULTS_DIR/.jax-cache` keeps for the next.

## Caching and privacy

A request on hosted inputs is keyed by a hash of the normalised request. The first job of a
request has that key as its address. An example posts its request plain and opens the job of
the key, the same job for every visitor. The form posts with `fresh=1` and starts a job of its
own each time, under a random key when the site holds the job of the request. A request with an uploaded VCF or any of your
own data gets a random link under `/p/`, never enters the shared cache, and is removed by the
retention sweep the app runs at start. Uploads waiting for a job sit under `RESULTS_DIR/uploads`
and are swept after the same retention period. Shared jobs are never removed by the site.

The key takes the expression of a request by the id of its source, its columns and its kind. The
label of a table and its place on the server are not in the key. A job reads a hosted table from
the place the catalog gives its source.

A job has its request and the status `queued` on disk from the moment it waits, shared or
private. A restart of the site finds the waiting and the running jobs by that status and queues
them again.

The downloads of a job state no path of the server. `request.json` is the request as the job
page states it, without the places of uploads. `status.json` and `engine.log` have their paths
cut to the job's directory. The zip has the same three files. The files on disk keep the paths
the job runs with.

The variants dialog and the job read the variants of a sample with one reader and rewire the
same loops, `svs.sample_variants` and `inputs.loop_network`.

## Sync codes

The My jobs page keeps its list in the browser. A sync code puts one list on several browsers
without an account. `POST /api/lists` stores the browser's jobs and answers a code of 16
characters, 80 random bits, shown as `K7M2-9QXD-4THP-W3RZ`. `GET /api/lists/{code}` reads the
list and `PUT /api/lists/{code}` merges a browser's jobs into it and answers the merged list. A
job on both sides keeps the entry edited last, and a write never removes a job. A browser with
a code sends its jobs on every page load, every 30 seconds, on return to the tab and after each
job it starts or renames.

The person chooses on making the code whether the list takes private jobs. A list made without
them refuses a `/p/` link from any browser, and those jobs stay in the browser that started
them. A list made with them gives every holder of the code the private links on it.

The lists are files under `RESULTS_DIR/lists`, named by a digest of the code, so the directory
does not hold the codes. A list holds at most 50 jobs, the newest. The site never removes a
list.

## Uploads

The Your own data dialog takes loops, one seven column BEDPE per factor, anchors in BED with an
optional orientation column, contacts, enhancers in BED and an expression table. The structural variants dialog takes
a VCF compressed with bgzip. `POST /api/uploads` stores each file in chunks under the
`UPLOAD_MAX_MB` cap and checks it on arrival; the dialog shows the check and holds the job until
every file passes. A job copies the files it uses into its own directory.

The Expression step of the dialog takes a table of genes by samples, uploaded with the kind
`expression`. The Your own table card of the Gene expression step of the form opens the dialog
on that step.
The check lists the samples and states that the values are whole numbers or not. The request
names the upload in `expression_upload`, the chosen samples in `expression_columns` and counts
or values in `expression_kind`. The job is private, as a job with any other upload is.

Contacts come by one of three routes, and each ends as the region's bin pairs at 25 kb without
the diagonal, the way the hosted maps hold them.

| route | where the window is read |
|---|---|
| link to a .hic or .mcool | on the server, `POST /api/uploads/link`, over HTTP range requests: hic-straw for .hic, h5py on an fsspec file for .mcool, each in a child process that reads through a range proxy on the loopback interface. Every connection resolves the host once and goes only to the public address it checked, redirects included. A read is stopped after 2 minutes, and at most `LINK_READS` run at once; one more is refused at once. |
| local .hic file | in the browser, with the hic-straw JavaScript reader, uploaded as a BEDPE |
| BEDPE of the region | on the user's computer; `gnome3d extract --mcool FILE --region chr:start-end --out region.bedpe` writes it from a local .mcool, with `pip install gnome3d[extract]` for cooler |

In production the worker's network should also refuse private ranges, as a second guard
behind the address check: with Docker, a network whose egress rule drops 10.0.0.0/8,
172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16 and the host's own addresses.

At job build the pairs are thinned to the contacts per Mb of the cell type's hosted map on the
region's chromosome, by drawing contacts without replacement. A deep map and a shallow one then
reach the engine at the density its maps are read at. The whole remote file never comes to the
server.

Checked 2026-11-12 through the page: an own data job with CTCF and RNAPII loops
and contacts read from a local .hic, a job with an uploaded HGSVC VCF, and a job with contacts by
link to ENCODE ENCFF718AWL.hic, 37 GB, whose 2 Mb window read in about 9 s.
