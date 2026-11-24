# Jobs of the site

## Duration of a job

Measured 2026-11-11 on an RTX 3090 with a warm JAX cache: GM12878 chr1:1-3 Mb, CTCF and RNAPII
loops, the first haplotype of HG00512 from HGSVC with 69 variants, ten structures, 58 s from
submission to the finished page. Jobs run on the queue and the page refreshes itself. A user
waits about a minute. The first job after a restart also runs the JAX compile, which the cache
under `RESULTS_DIR/.jax-cache` keeps for the next.

## Caching and privacy

A request on hosted inputs is keyed by a hash of the normalised request. The first job of a
request has that key as its address. An example posts its request plain and opens the job of
the key, the same job for every visitor. The form posts with `fresh=1` and starts a job of its
own each time, under a random key when the site holds the job of the request. A request with an
uploaded VCF or any of your own data gets a random link under `/p/` and does not enter the shared
cache. The retention sweep that the app runs at start removes it. Uploads waiting for a job are
under `RESULTS_DIR/uploads` and are swept after the same retention period. The site does not
remove shared jobs.

The key takes the expression of a request by the id of its source, its columns and its kind. The
label of a table and its place on the server are not in the key. A job reads a hosted table from
the place the catalog gives its source.

A job has its request and the status `queued` on disk from the moment it waits, shared or
private. A restart of the site finds the waiting and the running jobs by that status and queues
them again.

The downloads of a job state no path of the server. `request.json` is the request as the job
page states it, without the places of uploads. `status.json` and `engine.log` have their paths
cut to the directory of the job. The zip has the same three files. The files on disk keep the
paths the job runs with. `jobs.store.public_files` is the one rule of the files a visitor may
have. The summary lists them, a download serves one of them and the zip holds them all.

The variants dialog and the job read the variants of a sample with one reader and rewire the
same loops, `variants.vcf.sample_variants` and `engine.inputs.loop_network`.

## Sync codes

The My jobs page keeps its list in the browser. A sync code puts one list on several browsers
without an account. `POST /api/lists` stores the browser's jobs and answers a code of 16
characters, 80 random bits, shown as `K7M2-9QXD-4THP-W3RZ`. `GET /api/lists/{code}` reads the
list and `PUT /api/lists/{code}` merges a browser's jobs into it and answers the merged list. A
job on both sides keeps the entry edited last. A write does not remove a job. A browser with
a code sends its jobs on every page load, every 30 seconds, on return to the tab and after each
job it starts or renames.

The person chooses on making the code whether the list takes private jobs. A list made without
them refuses a `/p/` link from any browser. Those jobs stay in the browser that started them. A
list made with them gives every holder of the code the private links on it.

The lists are files under `RESULTS_DIR/lists`. A file has a digest of the code as its name and
the directory does not hold the codes. A list holds at most 50 jobs, the newest. The site does
not remove a list.

## Uploads

The Your own data dialog takes loops, one seven column BEDPE per factor, anchors in BED with an
optional orientation column, contacts, enhancers in BED and an expression table. The structural
variants dialog takes a VCF compressed with bgzip. `POST /api/uploads` stores each file in chunks
under the `UPLOAD_MAX_MB` cap and checks it on arrival. The dialog shows the check and holds the
job until every file passes. A job copies the files it uses into its own directory.

The Expression step of the dialog takes a table of genes by samples, uploaded with the kind
`expression`. The Your own table card of the Gene expression step of the form opens the dialog
on that step. The check lists the samples and states that the values are whole numbers or not.
The request has the upload in `expression_upload`, the chosen samples in `expression_columns` and
counts or values in `expression_kind`. The job is private, as a job with any other upload is.

Contacts come by one of three routes. Each ends as the bin pairs of the region at 25 kb without
the diagonal, the form the hosted maps have.

| route | where the window is read |
|---|---|
| link to a .hic or .mcool | on the server, `POST /api/uploads/link`, over HTTP range requests: hic-straw for .hic, h5py on an fsspec file for .mcool, each in a child process that reads through a range proxy on the loopback interface. Every connection resolves the host once and goes only to the public address it checked, redirects included. A read is stopped after 2 minutes. At most `LINK_READS` run at once and one more is refused at once. |
| local .hic file | in the browser, with the hic-straw JavaScript reader, uploaded as a BEDPE |
| BEDPE of the region | on the computer of the user. `gnome3d extract --mcool FILE --region chr:start-end --out region.bedpe` writes it from a local .mcool, with `pip install gnome3d[extract]` for cooler |

In production the network of the worker should also refuse private ranges, as a second guard
behind the address check. With Docker this is a network whose egress rule drops 10.0.0.0/8,
172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16 and the addresses of the host.

At job build the pairs are thinned to the contacts per Mb of the hosted map of the cell type on
the chromosome of the region, by drawing contacts without replacement. A deep map and a shallow
one then reach the engine at the density of its hosted maps. The whole remote file does not come
to the server.

Checked 2026-11-12 through the page: an own data job with CTCF and RNAPII loops and contacts
read from a local .hic, a job with an uploaded HGSVC VCF and a job with contacts by link to
ENCODE ENCFF718AWL.hic, 37 GB, whose 2 Mb window read in about 9 s.

## The chain of a job

A request has the field `chain`, `anchors` or `uniform`. `anchors` is the chain of loop anchors with
the chromatin between them and the value of a request without the field. `uniform` is one bead
per kilobase, the config `CANONICAL_UNIFORM` of the engine.

- The key of a request on `anchors` is the key it had before the field existed. `uniform` is in
  the key.
- `build_inputs` gives a job on `uniform` the keys `UNIFORM_KEYS` of
  `web/gnome3d_web/engine/settings.py`: those of `UNIFORM_CHAIN` of `gnome3d/production.py` and
  `holes.use_hole_beads = no`. The changes of the form are written after them.
- The contact cells of the chain are read from the map of the cell type under the Hi-C background
  alone. With structural variants, with own contacts, with a HiCDiffusion background and with no
  background the job has no `contact_map` and the chain has no contact cells.
- `GET /api/settings?cell=..&chain=uniform` answers the settings with the values a job on that
  chain gets, without the group of the beads for stretches without loops and with the solver
  `lbfgs` alone. A request that still asks for one of them is refused at its settings with the
  reason, `uniform_clash`.
- The site offers the chain when `features.uniform_chain` of `config.json` is on,
  `design/website/settings.md`.
- An example asks for the uniform chain on a site that offers it and for loop anchors on a site that does
  not, `model/request.ts::fromExample`. The drawings of the title page are made from the jobs of the examples.
- A job pairs a gene with the enhancer nearest to it in 3D within 1 Mb, which for about a third of the genes is
  under 5 kb from the TSS, a bead or two away. An example about one enhancer names it, `enhancer` of
  `model/examples.ts`. The job page and the comparison then open with `gene` and `enhancer` in the address,
  mark that enhancer as B and measure it from the TSS in each job. Another selected gene ends this.

## The states of a list of jobs

`POST /api/jobs/states` takes `{"jobs": [...]}` with the page path of each job, `/jobs/<key>` or `/p/<token>`,
and answers the state of each with the stage of a running job. A path that is no job of the site has null. The
list of jobs of a browser asks it once. It asks again every 15 seconds only while one of its jobs waits or runs.
Before 2026-11-24 the list asked the status route of every job every 15 seconds.

## The check of the variants

`POST /api/variants/check` applies each variant of the sample to the loops of the chromosome on its own and
words the change. One application builds the network of the chromosome, 0.2 s for chr1 of GM12878 since the
change to the vendored rules of 2026-11-24, `web/vendor/VENDORED.md`. Before it one application took 2.2 s on
the host of the site. The network of the region alone is not used: it gave other words in 3 of 178 applications.

