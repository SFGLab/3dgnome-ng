# Website frontend, part 3: the job explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, the user's standing choice). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The design's job explorer at `/jobs/<key>` and `/p/<key>`: the job's side panel, the Structure tab with a 3D view, hover and two point measurement, the Genes, Contact map, Loops and Log tabs, and the compare view of two ensembles.

**Architecture:** FastAPI gains read only JSON endpoints over a finished job's files, computed once and cached in the job directory: the summary, one structure with its annotations, the ensemble picks, distances between two positions over the ensemble, the gene table with per structure statistics, contact maps and a comparison of two jobs. The React explorer reads them with TanStack Query. The 3D view is three.js through @react-three/fiber. The old Jinja job pages move to `/jobs/<key>/classic` and `/p/<key>/classic`.

**Tech Stack:** as parts 1 and 2, plus three, @react-three/fiber 8 and @react-three/drei 9.

**Spec:** the screens in web/design/exports: B1HMV (Structure tab), uXzd2 (Genes), I9eDE4 (Contact map), giKuB (compare), PNG for the layout and `.txt` for the text, and web/design/site.pen when open. The writing rules of design/writing-rules.md. Parts 1 and 2 for the conventions.

## Global Constraints

- The Global Constraints of parts 1 and 2 hold: writing rules, verbatim design text, tokens, `checkCopy` in every page test, no runtime Node.
- Distances are in bead units. Every ensemble statistic says over how many structures it is taken.
- A job page answers for a job that is queued, running, failed or done. Only a done job shows the tabs.
- The endpoints read files the job wrote; none runs the engine. A computed result is written beside the job's files and read back on the next request.
- A private job's endpoints live under `/api/p/<token>/` and are reachable only with the token.

## Review Focus

- A job with one structure: the strip of per structure values, the range and the standard deviation still render, the deviation as 0. Test in Task 1.
- Two positions that fall in a stretch without beads: the distance endpoint names the nearest bead of each and how far it is, and refuses beyond 5 kb. Test in Task 1.
- A contact map request over the whole job region at 5 kb, about 400 bins on a side for 2 Mb and 4,000 for 20 Mb: the server refuses more than 800 bins on a side and says which resolution fits. Test in Task 2.
- A job still running when the page opens: the page polls and shows the tabs when it is done, without a reload. Test in Task 4.
- Compare of two jobs on different regions: "Same region" is off and the table lists only genes both jobs have. Test in Task 9.

---

### Task 1: Job data endpoints

**Files:**
- Create: `web/gnome3d_web/explore.py`
- Modify: `web/gnome3d_web/app.py`, `web/gnome3d_web/queue.py` (queue position)
- Test: `web/tests/test_explore.py`

**Interfaces:**
- Produces, each under `/api/jobs/<key>` and `/api/p/<token>`:
  - `GET .../summary` → `{state, stage, ahead, started, finished, request, loops: {CTCF, RNAPII}, structures, genes, loop_count, files: [{name, size}], models: [names]}`; `ahead` is the jobs in front in the queue, null when not queued.
  - `GET .../structure/<i>` → `{chain: [[x, y, z, bp], ...], beads: [[start, end, x, y, z, kind]], marks: {promoters: [{bead, gene}], enhancers: [bead]}, anchors: [{bead, start, end, loops: [{factor, other, pets}]}]}`, `i` from 1.
  - `GET .../annotations` → `{genes: [{name, tss, strand, start, end}], enhancers: [[start, end]], loops: [{factor, a: [s, e], b: [s, e], pets}]}` in the job's region.
  - `GET .../picks` → `{medoid, compact, random}`, structure numbers from 1; medoid by RMSD after Kabsch superposition, compact by radius of gyration.
  - `GET .../distance?a=<bp>&b=<bp>` → `{a: {bead, bp, offset}, b: {...}, values: [...], mean, sd, n}`; 400 when a position is more than 5 kb from every bead.
  - `GET .../genes` → rows of ep.tsv with `min, q1, median, q3, max` over structures and `joined_by`: `CTCF`, `RNAPII`, `CTCF and RNAPII` or empty, from the loops whose ends hold the promoter and the enhancer bead.
  - `GET .../loops` → the loops in the region with factor, ends, PETs and span.
  - `GET .../log` → the engine log as text.
  - `GET .../files.zip` → every file of the job.
- `explore.py` holds the computations as plain functions over a job directory, `summary(job: Path) -> dict` and so on; results that read every structure are cached as `explore_<name>.json` in the job directory.

- [ ] **Step 1: Write the failing test** `test_explore.py`: run two fixture jobs with the fake engine (structures 3 and 1); summary of the 3 structure job lists its files and loop counts; structure 2 has a chain with genomic positions and the fixture's anchors with their loops; picks are structure numbers in 1..3; the distance between the TSS of ALPHA and of BETA has 3 values and their mean and SD; on the 1 structure job the SD is 0; a position 50 kb from any bead is a 400 naming the nearest bead's distance; the genes rows carry the five statistics and `joined_by`; the log endpoint returns the log; the zip holds ep.tsv; a queued job reports `ahead`; a private job's endpoints answer under `/api/p/<token>` and not under `/api/jobs/`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: job data endpoints for the explorer"`

### Task 2: Contact maps and their agreement

**Files:**
- Create: `web/gnome3d_web/contacts.py`
- Modify: `web/gnome3d_web/app.py`
- Test: `web/tests/test_contacts.py`

**Interfaces:**
- Produces: `GET .../contacts?region=chr:s-e&res=25000&upper=hic&lower=model&norm=ice` → `{bins: [start...], upper: [[...]], lower: [[...]], upper_label, lower_label, scc}`; `upper` and `lower` are each one of `hic` (the job's singletons, binned), `model` (bead pairs closer than 1.5 bead units, summed over structures, binned by bead midpoints), and `norm` is `ice` or `raw`; `scc` is the stratum adjusted correlation of the two maps at that resolution over distances up to 2 Mb, as in HiCRep, null when a map is empty. More than 800 bins on a side is a 400 naming the coarsest resolution that fits.
- `contacts.py`: `model_map(beads_per_structure, region, res) -> np.ndarray`, `hic_map(singletons: Path, region, res) -> np.ndarray`, `ice(m) -> np.ndarray`, `scc(a, b, res, max_bp=2_000_000) -> float | None`.

- [ ] **Step 1: Write the failing test.** `model_map` of two beads 1 unit apart in one bin counts 1 per structure; `ice` of a matrix gives rows of equal sum; `scc` of a matrix with itself is 1 and of a matrix with its transpose shuffled by diagonal is below 0.5; the endpoint on the 3 structure fixture job returns square maps of equal size with bins on the requested resolution; 5 kb over the job's whole 4 Mb fixture region is refused naming 10 kb.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.** The stratum adjusted correlation follows Yang et al. 2017: per diagonal the Pearson correlation, weighted by the diagonal's length and the product of the two variances after a rank transform, without smoothing.
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: contact maps of a job and their stratum adjusted correlation"`

### Task 3: Compare endpoint

**Files:**
- Modify: `web/gnome3d_web/explore.py`, `web/gnome3d_web/app.py`
- Test: `web/tests/test_explore.py`

**Interfaces:**
- Produces: `GET /api/compare?a=/jobs/<key>&b=/jobs/<key>` → `{same_region: bool, rows: [{gene, linear_bp, a: {mean, sd}, b: {mean, sd}, ratio}]}`, rows of the genes both jobs have, sorted by the larger of ratio and its inverse, largest first, at most 20.

- [ ] **Step 1: Write the failing test.** Two fixture jobs on one region: rows of shared genes, ratio of each, sorted; two jobs on different regions: `same_region` false and only shared genes listed; a url that is not a job is a 404.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: comparison of two jobs' enhancer to promoter distances"`

### Task 4: Explorer shell, side panel and job states

**Files:**
- Create: `web/frontend/src/pages/Job.tsx`, `src/pages/job.css`, `src/job/SidePanel.tsx`, `src/job/Tabs.tsx`, `src/job/api.ts`
- Modify: `src/App.tsx`, `src/api.ts`, `web/gnome3d_web/app.py` (old job pages at `.../classic`, the client routes `/jobs/<key>` and `/p/<token>`)
- Test: `src/pages/Job.test.tsx`, `web/tests/test_frontend.py`

**Interfaces:**
- Produces: `jobApi(base: string)` with the calls of Tasks 1 to 3, `base` being `/api/jobs/<key>` or `/api/p/<token>`; `<Job />` reads the key from the route; tabs Structure, Genes with the gene count, Contact map, Loops with the loop count, Log; header actions Compare with another job, Copy link, Download all (`files.zip`).
- Side panel from the design: state and run time ("Done in 6 min 12 s"), the job name from this browser's recent jobs or the region, Rename job (in this browser), cell type and region, the facts table, the structure selector with previous, next, a select of 1 to N and the Medoid, Most compact and Random picks, the files list with the structures folder, the selected structure's file, "N more", ep.tsv, marks.json, engine.log and "all files, zip".
- A queued job shows "Queued, N jobs ahead", a running job its stage, a failed job its error and log tail; the page polls every 5 s until done.

- [ ] **Step 1: Write the failing tests.** With the API mocked: a done job shows the five tabs with the counts, the side panel facts and files; next and previous change "3 of 100"; Medoid selects the medoid's number; a queued job shows "Queued, 2 jobs ahead" and, once the mock reports done, the tabs without a reload; a failed job shows its error; Rename job stores the name and the title shows it; `checkCopy` clean. `test_frontend.py`: `/jobs/<key>` returns the app, `/jobs/<key>/classic` the old page, `/jobs/<key>/files/ep.tsv` the file.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** from B1HMV, text from B1HMV.txt.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: job explorer shell, side panel and job states"`

### Task 5: Structure view

**Files:**
- Create: `web/frontend/src/job/StructureView.tsx`, `src/job/colors.ts`
- Test: `src/job/colors.test.ts`, `src/job/StructureView.test.tsx`

**Interfaces:**
- Produces: `<StructureView structure annotations mode highlight onHover onPick measure />`, a tube along the chain coloured by `mode`: `elements` (promoters, enhancers, chromatin grey), `position` (a sequential ramp along the region), `factor` (CTCF and RNAPII anchors in their colours); OrbitControls; reset view, screenshot as PNG, fullscreen; the legend of the mode; the caption from the design.
- `colorAt(mode, bp, annotations) -> string` in `colors.ts`, the colour of a point of the chain.
- Hover shows the box of the design: coordinates, bead number, the gene the bead is in, and for an anchor its factor and loop count.

- [ ] **Step 1: Write the failing tests.** `colorAt` gives the promoter colour inside a promoter's span, the enhancer colour inside an enhancer, grey elsewhere; the position ramp is monotone in lightness; the factor mode colours a CTCF anchor with `--ctcf`. `StructureView.test.tsx` renders with the three.js canvas mocked and checks the legend per mode, the toolbar buttons and the caption with the structure number; `checkCopy` clean.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** with @react-three/fiber and drei's OrbitControls and Line; a ray against the tube picks the nearest chain point and its bead.
- [ ] **Step 4: Run them and look at it:** a headless Chrome screenshot of a frodo job's Structure tab at 1440 px. Expected: PASS, the tube visible and coloured.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: structure view"`

### Task 6: Measure, find a gene and the side cards

**Files:**
- Create: `web/frontend/src/job/Measure.tsx`, `src/job/GeneCard.tsx`, `src/job/FindGene.tsx`
- Test: `src/job/measure.test.tsx`

- [ ] **Step 1: Write the failing test.** With Measure on, two picks call the distance endpoint with their positions and the card shows "6.7 ± 1.9" from the mock, the sentence "Mean and standard deviation over 100 structures. The value in this structure is 5.9.", a dot per structure with the current one marked, and A and B with their positions; a third pick starts over; Find a gene with "tnfrsf8" selects TNFRSF8 and the Selected gene card shows its TSS, mean distance, nearest enhancer, separation, range and joined by; "All enhancers of this gene" opens the Genes tab filtered to it; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from B1HMV.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: measure, find a gene and the gene card"`

### Task 7: Genes tab

**Files:**
- Create: `web/frontend/src/job/GenesTab.tsx`
- Test: `src/job/GenesTab.test.tsx`

- [ ] **Step 1: Write the failing test.** The table has the design's columns; it sorts by 3D distance ascending by default and by any column on its header; the filter matches gene names; the chips All genes, Joined by a loop and Joined by RNAPII only filter on `joined_by`; each row has a range strip from min to max with the quartiles and the median; the histogram card shows the median sentence of the design computed from the rows; clicking a row selects the gene for the Structure tab; ep.tsv links to the file; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from uXzd2.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: genes tab"`

### Task 8: Contact map, Loops and Log tabs

**Files:**
- Create: `web/frontend/src/job/ContactMapTab.tsx`, `src/job/heatmap.ts`, `src/job/LoopsTab.tsx`, `src/job/LogTab.tsx`
- Test: `src/job/heatmap.test.ts`, `src/job/ContactMapTab.test.tsx`, `src/job/tabs.test.tsx`

**Interfaces:**
- `heatmap.ts`: `transform(m: number[][], show: 'observed' | 'expected' | 'oe' | 'pearson') -> number[][]`, `colour(v: number, lo: number, hi: number, log: boolean) -> [r, g, b]` on the design's white to red ramp.
- The tab draws one canvas, the upper half from `upper` and the lower from `lower`, a loop track above, the hover bar of the design, the colour range with min, max and log scale, the upper and lower half selects, the agreement card with `scc`, locus field with Go, Show, Normalization and Resolution, zoom in, zoom out and reset; scroll zooms and drag pans by changing the locus.

- [ ] **Step 1: Write the failing tests.** `transform` expected of a matrix is its diagonal means, O/E of a matrix equal to its expectation is all ones, Pearson of O/E is symmetric with ones on the diagonal; `colour` at lo is white and at hi the darkest red; the tab requests the contacts endpoint with the locus, resolution and halves, shows "0.37" from the mock in the agreement card and "Computed on 25 kb bins up to 2 Mb apart in this view."; a refused resolution shows the server's message; Loops lists the loops with factor, ends, span and PETs; Log shows the log text; `checkCopy` clean.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** from I9eDE4.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: contact map, loops and log tabs"`

### Task 9: Compare view

**Files:**
- Create: `web/frontend/src/job/Compare.tsx`
- Test: `src/job/Compare.test.tsx`

- [ ] **Step 1: Write the failing test.** Compare with another job lists this browser's recent jobs and takes a pasted job link; with a second job chosen the view shows two structure panels with their job titles, "Link rotation and zoom" and "Same region" ticked when the regions match and "Same region" off otherwise, an "A to B" label on each panel when a measurement exists, and the table "Largest changes in enhancer to promoter distance" from the compare endpoint with "2.4 times closer"; Comparing closes it; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from giKuB. Linked views share one camera state.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: compare view"`

### Task 10: End to end

**Files:**
- Modify: `web/README.md`

- [ ] **Step 1: Deploy to the site of serve_local.sh on team-frodo** and open the TNFRSF8 job of part 2 and a CTCF only job of the same region. Expected: every tab loads; a measurement between two beads gives a mean and SD; the contact map shows both halves and an agreement value; compare of the two jobs fills the table.
- [ ] **Step 2: Screenshots** of each tab at 1440 px and the Structure tab at 390 px through puppeteer. Expected: no overflow, the 3D view drawn.
- [ ] **Step 3: Commit** the README section on the explorer and its endpoints. `git commit -m "web: job explorer checked on team-frodo"`
