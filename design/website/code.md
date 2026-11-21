# The code of the server

`web/gnome3d_web` is a FastAPI app. `create_app` in `app.py` builds one `Site`, adds the routes of
each part of the site and adds the frontend last.

## Layout

```
app.py            create_app
context.py        Site: the settings, the catalog, config.json, the lists, the queue,
                  the limits of a region and the lookup of a job
answers.py        the answers every route shares: the two refusals, a packed JSON body,
                  the base path and the job of an address
errors.py         Refused and UnknownCell
frontend.py       the built frontend, for every address that is no route
settings.py       the environment variables
siteconfig.py     config.json
regions.py        a region, its parsing and its text
contacts.py       the maths of contact maps
density.py        the depth of a hosted contact map and the thinning to it
expression.py     expression tables and the contrast of two jobs
remote.py         the reading of a linked contact map
catalog/          the hosted data
  model.py        cell types, call sets and expression sources
  genes.py        the search of genes by name
  preview.py      the loops and genes of a region for the form
engine/           the engine side of a job
  settings.py     the settings a request may change
  inputs.py       the input files of a job
  hicdiffusion.py the predicted contact background
jobs/
  request.py      the request of a job and its key
  store.py        the files of a job directory
  submit.py       the check of a request and its queueing
  run.py          one job, from its request to its structures and marks
  workers.py      the queue: inline, threads or rq, and the slots
  marks.py        the beads of promoters and enhancers
uploads/
  checks.py       the check of an upload, by its kind
  store.py        the stored file, its record and the uploads a request names
variants/
  vcf.py          the reader of a VCF
  effect.py       the effect of variants on loops, for the dialog
results/
  views.py        the answers of a job page
  contact_view.py the contact maps of a job page
  compare.py      two jobs compared
lists/
  store.py        the shared job lists
```

Each package with routes has them in its `routes.py`.

## Rules

A part of the site with routes, rules and files of its own is a package. A module that several
packages use and that has no routes is a flat module beside them.

A route reads the request, calls a function and shapes the answer. A rule of the site is a
function that takes plain values and can be tested without a request.

A route takes the site as the parameter `site: Here`. No module keeps state of the site. A cache
is keyed by the path of the file it read.

`jobs/store.py` alone states the names of the files of a job directory and what a download
gives of them. The other modules call it.

A request the site refuses raises `Refused` with the form field it concerns, and the answer is
400 with the message and the field. A cell type the catalog lacks raises `UnknownCell`, and the
answer is 404.

No module has the name of a module of Python. The engine runs scripts from inside the package,
and a module named `site`, `http` or `queue` there would be found in place of Python's own.

The packages import in one direction: `catalog`, then `engine`, then `jobs`. `uploads` is used
by `jobs/submit.py` and by `variants`. `results` reads through `jobs/store.py`.

## What a change must not move by accident

Three tests state what running sites and links in the wild depend on. A change of one of them is
a decision and is made in the test.

- `web/tests/test_routes.py` lists every method and path of the interface.
- `web/tests/test_jobs.py` pins the keys of ten kinds of request. A site finds its jobs by them.
- `web/tests/test_job_on_disk.py` reads two job directories written before the refactor of
  2026-11-21 and compares the answers with the ones of that time.
