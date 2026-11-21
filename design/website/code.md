# The code of the site

The server is first, the frontend after it.

## The server

`web/gnome3d_web` is a FastAPI app. `create_app` in `app.py` builds one `Site`, adds the routes of
each part of the site and adds the frontend last.

### Layout

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

Each package with routes has them in its `routes.py` and the shapes of its answers and of its
files in its `shapes.py`.

### Rules

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

### Shapes

The shape of every JSON answer and of every file the site keeps is declared once, as a
`TypedDict` in the `shapes.py` of its package. The function that builds an answer has the shape
as its return type and pyright checks the two against each other. The route names the shape as
its `response_model`, and the app describes the whole interface at `/openapi.json`. The types of
the frontend are generated from that description: `web/scripts/write_openapi.py` writes it to
`web/frontend/src/api/openapi.json` and `npm run api` writes `src/api/schema.d.ts` from it. Both
files are in the repository. A test of the server and a step of CI fail while one is stale.

A route still answers with the value its function returned. The shape is checked by pyright and
by `web/tests/test_contract.py`, not at each request. The contact map of 800 bins is megabytes
of JSON, and a check of every number of it at each request would cost more than the map.

`pyright -p web/pyrightconfig.json` checks the package in basic mode. Five modules that work on
tables turn off the rules that pandas and h5py trip: `remote.py`, `expression.py`,
`engine/inputs.py`, `contacts.py` and `density.py`.

### What a change must not move by accident

Four tests state what running sites and links in the wild depend on. A change of one of them is
a decision and is made in the test.

- `web/tests/test_routes.py` lists every method and path of the interface.
- `web/tests/test_jobs.py` pins the keys of ten kinds of request. A site finds its jobs by them.
- `web/tests/test_contract.py` asks every route and compares the answer with its declared
  shape: every key declared, no key undeclared, every value of its type. It fails for a route
  it has not asked.
- `web/tests/test_job_on_disk.py` reads two job directories written before the refactor of
  2026-11-21 and compares the answers with the ones of that time.

## The frontend

`web/frontend/src` is a React app. A page puts the parts of the site together.

### Layout

```
App.tsx  main.tsx  routes.ts  base.ts  flags.ts
api/            every request to the server and the shapes of its answers
  http.ts       the one way to ask: JSON, text or an ApiError
  site.ts  jobs.ts  uploads.ts  lists.ts   the routes
  queries.ts    one definition for each question, its key in the cache and its call
  types.ts      the shapes of schema.d.ts by the names the pages use
  schema.d.ts   generated from openapi.json, which the server writes
lib/            pure functions two parts use: formats, regions, the saving of a file, CSV
components/     components two parts use: the layout of the site, the dialog frame,
                the shell of a document page, the pager, an answer that is not there
content/        the references of the Cite and the Method page
myjobs/         the job list of the browser, its sync and its panel
model/          the form
  form.ts       the state of the form and its rules
  request.ts    the request of a job: to the server, from an example, from an earlier job
  own/          the own data dialog, one file for each step
job/            the page of a job
  compare/      the comparison of two jobs
  export/       the saving of pictures
pages/          one component for each address
styles/         the tokens and the sheets every page shares: base, doc and dialog
test/           the stand in for the server and the check of the text of a page
```

### Rules

A request and the shape of its answer are in `api/`. A page does not call fetch.

A pure function that two parts use is in `lib/`, and a component that two parts use is in
`components/`. Everything else is in the folder of its part, with its test beside it.

`job/`, `model/` and `myjobs/` do not import each other or a page, and the shared folders import
none of them. `eslint.config.js` has the rule.

A component file exports components alone. A helper beside a component is in a module of its
own.

The stylesheet of a part is a CSS module beside its components, `job/job.module.css` for the job
page. Its rules apply to the components of that part alone. Two parts can have a class of one
name with two looks. The sheets of `styles/` are shared and plain.

A component takes its class names through `scoped` of `lib/css.ts`: `cn('factor-row bad')` gives
each name its scoped name and keeps the plain name beside it. A rule of a module applies by the
scoped name. The plain name is for the shared sheets, for a rule of another sheet that refers to
the component, written `:global(.name)` there, for a selector in code and for a test. A class
name is not checked by TypeScript.

A question to the server has one definition in `api/queries.ts`. A view that waits for an answer
shows `Unanswered`: a busy mark, or the message of the question that failed.

### What a change must not move by accident

The addresses of the pages and their query parameters, the keys of the storage of the browser
and the text of the pages are in use by links in the wild and by browsers that hold a job list.
The tests of the pages check the text against `design/writing-rules.md`.

A change of markup or of a stylesheet that is meant to be inert is checked by pictures: two
builds are served side by side on the data of the tests, every page and dialog is drawn at a
desktop width, in the dark theme and at the width of a phone, and the pictures are compared
pixel by pixel.
