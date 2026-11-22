# Frontend of the site

The pages are a React app in `web/frontend/`, built with Vite to `web/frontend/dist/`. FastAPI
serves the build. Every client route returns `dist/index.html`, while the API and the files of a
job keep their own handlers. Without a build the app answers 503 with the way to build it.
`FRONTEND_DIST` points the app at another build. The Docker image builds the frontend in a Node
stage.

Development needs Node 18 or newer:

```bash
cd web/frontend
npm ci
npm test            # Vitest, the pages and their text against design/writing-rules.md
npm run dev         # Vite on port 5173, passing /api, /jobs, /p and /model to FastAPI on 8000
npm run build       # writes dist/ for FastAPI
```

A site under a path of a host, `https://host/new/`, needs `BASE_PATH=/new` twice, at the build
and on the server:

```bash
BASE_PATH=/new npm run build                 # in web/frontend
BASE_PATH=/new uvicorn gnome3d_web.app:create_app --factory
```

The build writes the path into its asset links, its router and its API requests. The server
answers every route under the path and at the root. The proxy in front may strip the path or
pass it on. The server answers 503 with both paths when the build and the server disagree. With
Docker Compose, `BASE_PATH=/new` in the environment sets both. A proxy rule for `/new/` does not
match `/new`. A second rule redirects `/new` to `/new/`.

The Model a region page at `/model` reads the JSON API: `GET /api/catalog`, `GET /api/region`,
`GET /api/preview` for the loop arcs of a region, `GET /api/settings` for the engine settings a
user may change, `POST /api/settings/ini` to read an uploaded config and `POST /api/jobs`, which
answers with the url of the job or the field a request fails on. The settings a user may change,
with their ranges, are listed in `web/gnome3d_web/engine/settings.py`. Every other setting keeps
the value of the cell type.

The page of a job at `/jobs/<key>`, or `/p/<token>` for a private job, is the explorer. It has
the Structure tab in 3D with hover, two point measurement and a gene card, the Genes, Contact
map, Loops and Log tabs and a compare view of two jobs. It reads `GET /api/jobs/<key>/` followed
by `summary`, `structure/<i>`, `annotations`, `picks`, `distance?a=&b=`, `genes`, `contacts`,
`loops`, `log` and `files.zip`, the same under `/api/p/<token>/`. The compare view reads
`GET /api/compare?a=&b=`. Results that read every structure, the picks and the gene statistics,
are written to the job directory as `explore_<name>.json`.

Step 5 of the form, Gene expression, is optional. It lists the hosted tables that
`GET /api/expression?cell=&sv_set=&sample=` returns for the cell type and the sample. A card for
a table of the user opens the Your own data dialog on its Expression step. A job with expression
has an Expression column in the Genes tab. The compare view of two such jobs reads
`GET /api/compare/expression?a=&b=`.

Under its two structures the compare view has the results, `job/compare/CompareResults.tsx`. The tables are
on the left, one tab for the 3D distance and one for the expression, each with every gene of
both jobs a page at a time. Above them a card has the main values of the selected gene and a
card the rank correlation of the two changes. The figures are an accordion on the right, under
the tables on a narrow screen: the box plot of the distance of the selected gene in the
structures of the two jobs, from `GET /api/compare/gene?a=&b=&gene=`, the plot of the change of
expression against the change of distance and the volcano plot. A row of `GET /api/compare` has
the p of a Mann-Whitney U test of the distances. The two jobs have the name of the first thing
they differ in, the cell type, the sample, the loops or the contact background. Every change
states its job. A job measures a gene to its own nearest enhancer, as enhancer3D did with the
enhancers of each cell line. Two cell types have two sets of enhancers. A change of distance
between them can come from a nearer enhancer on the genome as well as from the folding. The table
states both separations of a gene where they differ and a line counts those genes. For GM12878
against H1ESC on chr1:11.5-12.6 Mb all 37 genes have two enhancers. The option Enhancers of both
jobs only, off by default and offered to two cell types without structural variants, measures
every gene in both jobs to the nearest enhancer that overlaps in the two sets, `shared=1` on the
three compare routes. Each plot has a title and ticks on its two axes and a key of its marks. The
volcano plot has every gene of the test in the background, on a grid of 0.2 by 0.5 with each
cell once.

The Method page has a tab for each chain of the form where the site offers the uniform chain: sections 3 and
4 under Loop anchors and segments, section 5 under One bead per kilobase. A link to a section of the other tab
opens that tab. Without the feature the page has no tabs and no section 5.

The title page has two rows of examples. The first is the TNFRSF8 comparison over two columns
with the contact map. The second is the GSTM1 deletion with the comparison of VPS13D in GM12878
and H1ESC over two columns. `web/scripts/make_previews.py` draws each from its finished job, the
two cell types each at its own enhancer.

An address the site does not have gets the page of a missing address, `pages/NotFound.tsx`, with a
random chain of 404 steps. That holds for the address of a job the site lacks too. The server
answers such an address with the app and the status 404.

The Help page has a tutorial on the TNFRSF8 example with links to the finished jobs of every
example, the sample output of the site. It has sample input files for every kind of upload under
`src/assets/samples/`, cut from the hosted GM12878 data of the example region. It has a guide to
every part of a job page.

An open figure is saved as a PNG with its title, `job/export/savePng.ts`. The tables of a job,
the genes, the loops and the two tables of the comparison, are saved whole as CSV, `lib/csv.ts`.
`design/website/data.md` has the tables and the comparison.
