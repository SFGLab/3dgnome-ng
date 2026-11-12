# Website frontend, part 1: foundation and static pages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A React app that carries the approved design's look and its five static pages, Title, Method, Results, Help and Cite, served by the existing FastAPI app, with the old form moved to `/model` until part 2 replaces it.

**Architecture:** A Vite project in `web/frontend/` builds to `web/frontend/dist/`. FastAPI serves `dist/` as static files and returns `index.html` for every client route, while `/api/*`, `POST /jobs`, `/jobs/{key}` and `/p/*` keep their current handlers. The Docker image builds the frontend in a Node stage and copies `dist/` in.

**Tech Stack:** React 18, TypeScript 5, Vite 5, react-router-dom 6, Vitest with @testing-library/react and jsdom, Node 22 LTS for the build only.

**Spec:** web/design/site.pen (approved 2026-11-12), read through the Pencil MCP with the file open in the editor. Frames: Title page `RjlvI`, Method page `n6RrWu`, Help page `n7ivTu`, Results page `g7Kqf`, Cite page `M9iFy`, top bar of every screen (for example `lumhZ`), cite strip (for example `KnIbb`). Figures and equations already rendered in web/design/images/ by web/design/render_mock.py and web/design/render_equations.py. The writing rules: design/writing-rules.md.

The later parts, each its own plan: part 2 Model a region with the engine settings, part 3 the job explorer, part 4 the own data uploads in design/plans/2026-11-12-web-own-data.md.

## Global Constraints

- Text follows design/writing-rules.md: no heading or sentence opens with What, Where, How, Who, Whether, When, Which or Why; no ", so" and no "because" clauses; headings are noun phrases; no caps for emphasis; no marketing terms.
- Page copy is the design's text verbatim. A change of wording goes back to the design first.
- Branding: "3D-GNOME" with a superscript "NG", rendered as `3D-GNOME<sup>NG</sup>`, in plain text `3D-GNOMEᴺᴳ`. The Python package is gnome3d-ng.
- Colours come from the Pencil variables only, as CSS custom properties with the same names, light and dark. Factor colours: CTCF #2A78D6, RNAPII #7A3FB0, promoter #EB6834, enhancer #1BAF7A.
- Links to the previous services: https://3dgnome.mini.pw.edu.pl/old and https://3dgnome.mini.pw.edu.pl/old/enhancer3D/.
- No runtime Node: production serves files from `dist/`. No external CDN at runtime; fonts are self hosted or system fonts.
- Dependencies pinned by `package-lock.json`, committed.

## Review Focus

- A browser reload on a deep link, `/method` or `/cite`, returns the app, not a 404. Test in Task 3.
- The SPA fallback never answers an API or job route: `GET /api/catalog` stays JSON, `POST /jobs` still creates a job, `/jobs/{key}` still returns the job page. Test in Task 3.
- The server starts when `dist/` has not been built and says so on `/`, instead of failing at import. Test in Task 3.
- Dark mode: every colour used in a page resolves to a token defined for both themes. Test in Task 1.
- Page text keeps to the writing rules after later edits. Test helper in Task 2, called by every page test.

---

### Task 1: Vite project, tokens and test runner

**Files:**
- Create: `web/frontend/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/styles/tokens.css`, `src/styles/base.css`, `src/test/setup.ts`
- Test: `web/frontend/src/styles/tokens.test.ts`

**Interfaces:**
- Produces: `npm run build` writes `web/frontend/dist/index.html` and `dist/assets/`; `npm test` runs Vitest once; `tokens.css` defines every Pencil variable as `--<name>` under `:root` and again under `@media (prefers-color-scheme: dark)` and `:root[data-theme="dark"]`.

- [ ] **Step 1: Write the failing test.** `tokens.test.ts` reads `tokens.css` as text and asserts that the set of custom property names under `:root` equals the set under `:root[data-theme="dark"]`, and that it contains `--accent`, `--ink`, `--muted`, `--faint`, `--line`, `--surface`, `--panel`, `--bg`, `--ctcf`, `--rnapii`, `--promoter`, `--enhancer`, `--font`, `--mono`.
- [ ] **Step 2: Run it.** `cd web/frontend && npm test` — Expected: FAIL, tokens.css missing.
- [ ] **Step 3: Scaffold and write the tokens.** `npm create vite@5` with the react-ts template, then add vitest, jsdom, @testing-library/react, @testing-library/jest-dom, react-router-dom. Read the variables with the Pencil MCP, `Print(GetVariables())` in an execute call on site.pen, and write each as a custom property with its light and dark value. Fonts: the family named by `$font` and `$mono`, self hosted under `src/fonts/` when its licence allows, else the system stack.
- [ ] **Step 4: Run it.** `npm test` — Expected: PASS. `npm run build` — Expected: `dist/index.html` exists.
- [ ] **Step 5: Commit.** `git add web/frontend && git commit -m "web frontend: Vite project, design tokens, test runner"`

### Task 2: Layout, routes and the copy check

**Files:**
- Create: `web/frontend/src/routes.ts`, `src/App.tsx`, `src/components/Layout.tsx`, `src/components/Wordmark.tsx`, `src/components/CiteStrip.tsx`, `src/test/copy.ts`
- Test: `src/components/Layout.test.tsx`, `src/test/copy.test.ts`

**Interfaces:**
- Produces: `ROUTES = { title: "/", model: "/model", method: "/method", results: "/results", help: "/help", cite: "/cite" } as const` in `routes.ts`; `<Layout>{children}</Layout>` with the top bar, navigation in the design's order Model a region, My jobs, Method, Results, Help, Cite, and the cite strip; `checkCopy(root: HTMLElement): string[]` in `src/test/copy.ts`, one message per violation.

- [ ] **Step 1: Write the failing tests.**
  - `Layout.test.tsx`: the wordmark renders `3D-GNOME` followed by a `sup` holding `NG`; the navigation links have the hrefs of `ROUTES` in the design's order; the cite strip links to `/cite`.
  - `copy.test.ts`: `checkCopy` on `<h2>How it works</h2>` returns one message; on `<p>It runs, so it is fast.</p>` one; on `<p>The model has loops.</p>` none.
- [ ] **Step 2: Run them.** `npm test` — Expected: FAIL, modules missing.
- [ ] **Step 3: Implement.** Top bar and cite strip copy verbatim from frames `lumhZ` and `KnIbb`. `checkCopy` walks headings and text nodes: a heading or a sentence whose first word is one of the eight wh-words, the substring ", so ", and the word "because". Capitals are left to review, since the site's text is full of acronyms.
- [ ] **Step 4: Run them.** `npm test` — Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: layout, wordmark, routes, copy check"`

### Task 3: FastAPI serves the app

**Files:**
- Modify: `web/gnome3d_web/app.py`, `web/gnome3d_web/settings.py` (`FRONTEND_DIST`, default `web/frontend/dist` relative to the repository)
- Modify: `web/gnome3d_web/templates/index.html` is served at `/model` instead of `/`
- Test: `web/tests/test_frontend.py`

**Interfaces:**
- Consumes: `dist/index.html` and `dist/assets/*` from Task 1.
- Produces: `GET /`, `/method`, `/results`, `/help`, `/cite` and any path not matched by another route return `dist/index.html`; `GET /assets/<file>` returns the file; `GET /model` returns the old form; `Settings.frontend_dist: Path`.

- [ ] **Step 1: Write the failing test.** `test_frontend.py` builds a temporary dist with `index.html` holding the marker `FRONTEND-OK` and `assets/app.js`, sets `FRONTEND_DIST`, and checks: `/` and `/method` and `/cite` return 200 with the marker; `/assets/app.js` returns the file; `/model` returns the old form; `/api/catalog` returns JSON; `POST /jobs` with the form of test_app.py redirects to `/jobs/`; `/jobs/<that key>` returns the job page, not the marker. A second app with `FRONTEND_DIST` pointing at a missing directory starts and answers `/` with 503 and the text "The frontend is not built".
- [ ] **Step 2: Run it.** `python web/tests/test_frontend.py` — Expected: FAIL.
- [ ] **Step 3: Implement.** Mount `StaticFiles` at `/assets`, move the form route to `/model`, register the client route fallback after every other route.
- [ ] **Step 4: Run it and the other web tests.** Every `web/tests/test_*.py` — Expected: all pass. The form posts and redirects of test_app.py move to `/model` where they read the page.
- [ ] **Step 5: Commit.** `git commit -m "web: FastAPI serves the frontend build, old form at /model"`

### Task 4: Title page

**Files:**
- Create: `web/frontend/src/pages/Title.tsx`
- Test: `src/pages/Title.test.tsx`

- [ ] **Step 1: Write the failing test.** Renders inside `Layout`; the main heading is the design's; the "Previous services" section links to `https://3dgnome.mini.pw.edu.pl/old` and `https://3dgnome.mini.pw.edu.pl/old/enhancer3D/`; a link to `/model` is present; `checkCopy` returns no messages.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from frame `RjlvI`, text verbatim, structure image `web/design/images/structure.png` copied to `src/assets/`.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: title page"`

### Task 5: Cite page and the reference list

**Files:**
- Create: `web/frontend/src/content/references.ts`, `src/pages/Cite.tsx`
- Test: `src/content/references.test.ts`, `src/pages/Cite.test.tsx`

**Interfaces:**
- Produces: `type Reference = { key: string; authors: string; title: string; venue: string; year: number; doi?: string; url?: string; unconfirmed?: boolean }` and `REFERENCES: Reference[]` in the design's order, used again by the Method page's reference list in Task 6.

- [ ] **Step 1: Write the failing tests.** `references.test.ts`: every key is unique; every entry has a `doi` or a `url`; DOIs match `^10\.\d{4,9}/\S+$`; the keys include the design's 3D-GNOME 2016, 2.0, 3.0, cudaMMC, enhancer3D, LoopSage, MultiMM and Sadowski 2019 entries; the HGSVC freeze 4 entry has `unconfirmed: true`. `Cite.test.tsx`: one list item per reference, each DOI rendered as a link to `https://doi.org/<doi>`, the unconfirmed entry marked as the design marks it, `checkCopy` clean.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** from frame `M9iFy`, entries verbatim.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: cite page and reference list"`

### Task 6: Method page

**Files:**
- Create: `web/frontend/src/pages/Method.tsx`, `src/components/Equation.tsx`, `src/components/Figure.tsx`
- Copy: `web/design/images/eq/*.png` and `sizes.json`, `method_*.png` into `src/assets/`
- Test: `src/pages/Method.test.tsx`

**Interfaces:**
- Consumes: `REFERENCES` from Task 5; citations render as `[n]` links to the page's reference list, numbered by first use as the design numbers them.
- Produces: `<Equation name="anchor_energy" />` renders the PNG at the display size of `sizes.json` with the TeX source as `alt`.

- [ ] **Step 1: Write the failing test.** Section headings appear in the order Inputs, Distance law, Anchor phase, Chain phase, Ensemble, Engine, Structural variants, Validation, Limitations, References; every `img` has a non empty `alt`; the anchor and chain term tables cite excluded volume as [5, 14], repulsion as [1, 5] and confinement as [5, 15] as the design does; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from frame `n6RrWu` with its story frame `nles2`, the equation frames and the engine figure `XrdIq`.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: method page"`

### Task 7: Results and Help pages

**Files:**
- Create: `web/frontend/src/pages/Results.tsx`, `src/pages/Help.tsx`
- Test: `src/pages/Results.test.tsx`, `src/pages/Help.test.tsx`

- [ ] **Step 1: Write the failing tests.** Results: the run time table against cudaMMC shows the design's placeholder marking on every value; the SCC, distance and expression numbers equal the design's (SCC E→C 0.22→0.37, 0.15→0.27, 0.15→0.33). Help: the input formats table lists anchors BED, loops BEDPE with at least seven columns, singletons, breakpoints, VCF, .hic and .mcool with the design's wording; no section of admin content; `checkCopy` clean on both.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** from frames `g7Kqf` and `n7ivTu`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: results and help pages"`

### Task 8: Image build and documentation

**Files:**
- Modify: `web/Dockerfile` (a `node:22-slim` stage runs `npm ci && npm run build` in `web/frontend`, the Python stage copies `dist/`), `web/README.md` (frontend development: `npm run dev` with the Vite proxy to FastAPI on port 8000, `npm test`, `npm run build`), `web/frontend/vite.config.ts` (dev proxy for `/api`, `/jobs`, `/p`, `/model`)
- Test: build the image on team-frodo and run `web/tests/test_frontend.py` against it

- [ ] **Step 1: Build.** On team-frodo, `docker build -f web/Dockerfile --build-arg GPU=1 -t gnome3d-web .` — Expected: success, `/app/web/frontend/dist/index.html` in the image.
- [ ] **Step 2: Run.** `docker run --rm -p 127.0.0.1:8768:8000 gnome3d-web` and `curl -s 127.0.0.1:8768/method` — Expected: the app's `index.html`.
- [ ] **Step 3: Commit.** `git commit -m "web: image builds the frontend, README for frontend development"`
