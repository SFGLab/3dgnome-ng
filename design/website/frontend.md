# Frontend of the site

The pages are a React app in `web/frontend/`, built with Vite to `web/frontend/dist/`. FastAPI
serves the build: every client route returns `dist/index.html`, and the API and the files of a
job keep their own handlers. Without a build the app
answers 503 and says how to build. `FRONTEND_DIST` points the app at another build. The Docker
image builds the frontend in a Node stage.

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
answers every route under the path and at the root, and the proxy in front may strip the path or
pass it on. The server answers 503 with both paths named when the build and the server disagree.
With Docker Compose, `BASE_PATH=/new` in the environment sets both. A proxy rule for `/new/`
does not match `/new`, and a second rule redirects `/new` to `/new/`.

The Model a region page at `/model` reads the JSON API: `GET /api/catalog`, `GET /api/region`,
`GET /api/preview` for the loop arcs of a region, `GET /api/settings` for the engine settings a
user may change, `POST /api/settings/ini` to read an uploaded config, and `POST /api/jobs`, which
answers with the job's url or the field a request fails on. The settings a user may change, with
their ranges, are listed in `web/gnome3d_web/settings_schema.py`; every other setting keeps the cell
type's value.

A job's page at `/jobs/<key>`, or `/p/<token>` for a private job, is the explorer: the Structure
tab in 3D with hover, two point measurement and a gene card, and the Genes, Contact map, Loops
and Log tabs, and a compare view of two jobs. It reads `GET /api/jobs/<key>/` followed by
`summary`, `structure/<i>`, `annotations`, `picks`, `distance?a=&b=`, `genes`, `contacts`,
`loops`, `log` and `files.zip`, the same under `/api/p/<token>/`, and `GET /api/compare?a=&b=`.
Results that read every structure, the picks and the gene statistics, are written to the job
directory as `explore_<name>.json`.

Step 5 of the form, Gene expression, is optional. It lists the hosted tables that
`GET /api/expression?cell=&sv_set=&sample=` returns for the cell type and the sample, and a card
for a table of the user, which opens the Your own data dialog on its Expression step. A job with expression has an Expression column in the Genes tab. The
compare view of two such jobs reads `GET /api/compare/expression?a=&b=` and shows the change of
expression of each gene against the change of its distance to the nearest enhancer, as a plot and
a table. `design/website/data.md` has the tables and the comparison.
