# Website frontend, part 2: Model a region — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, the user's standing choice) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The design's Model a region screen as the `/model` page of the React app, with the server work it needs: RNAPII-only loops, a choice of contact background, up to 100 structures, the engine settings as a tree with sliders, a loop preview and region lookup.

**Architecture:** FastAPI gains JSON endpoints for the catalog, region lookup, the loop preview, the settings schema and job submission. The engine settings a user may change are a curated list over SETTINGS.md, validated on the server and written into the job's config.ini. The React page follows the design's five steps and posts the job as multipart form data. The old Jinja form moves to `/model/classic`.

**Tech Stack:** as part 1, plus @tanstack/react-query 5 and @radix-ui/react-slider and react-collapsible for the settings tree.

**Spec:** web/design/site.pen, frame `l1JkG5` (Model a region), with the own data dialogs `LgRHG` and `N3hxD` out of scope (part 4). SETTINGS.md for every setting. design/writing-rules.md for the text. Part 1's plan for the frontend conventions: design/plans/2026-11-12-web-frontend-1-foundation.md.

## Global Constraints

- Every Global Constraint of part 1 holds: the writing rules, verbatim design text, tokens, no runtime Node, `checkCopy` in every page test.
- A user changes only the settings on the curated list, each inside its range. The server refuses anything else with the key and the allowed range.
- A job's cache key covers every input that changes its structures: cell type, region, factors, contact background, structural variants, structures and the setting changes.
- `MAX_REGION_MB` (20) and the 2 Mb HiCDiffusion limit hold as today. `MAX_STRUCTURES`, default 100, caps the structures.
- The own data card and the own loops option are shown and disabled, with the text "Uploads open in a later version." Part 4 enables them.

## Review Focus

- A region typed as a gene name that two genes share, or in lower case, resolves to one region and says which. Test in Task 3.
- A setting change that puts the run outside the engine's tested range, such as a huge step count, is refused before the job starts. Test in Task 1.
- The same request with the same settings in a different order finds the same job; with one setting changed, a new one. Test in Task 1.
- RNAPII-only loops run without the CTCF orientation term reading RNAPII loops as CTCF loops. Test in Task 2.
- The page works with JavaScript state lost: a reload of `/model?example=TNFRSF8` fills the form again. Test in Task 6.

---

### Task 1: Settings schema and overrides on the server

**Files:**
- Create: `web/gnome3d_web/settings_schema.py`
- Modify: `web/gnome3d_web/jobs.py` (`JobRequest.overrides: dict[str, str]`, in `request_key`), `web/gnome3d_web/inputs.py` (write the overrides into config.ini), `web/gnome3d_web/app.py` (`GET /api/settings`, `POST /api/settings/ini`), `.dockerignore` and `web/Dockerfile` (SETTINGS.md into the image)
- Test: `web/tests/test_settings_schema.py`

**Interfaces:**
- Produces: `SettingRow(section: str, key: str, type: str, default: str, production: str, meaning: str)`; `read_settings_md(path: Path) -> list[SettingRow]`; `EDITABLE: dict[str, list[Editable]]`, group name to `Editable(section, key, lo, hi, step)` for numbers or `Editable(section, key, choices=(...))` for choices and booleans; `schema(cell_config: Path) -> list[dict]`, one group per entry with its settings, each with `section, key, type, default, server, lo, hi, step, choices, meaning`; `validate_overrides(raw: dict[str, str]) -> dict[str, str]` keyed `section.key`, raising `SettingError` with the key and the allowed range; `apply_overrides(config: Path, overrides: dict[str, str]) -> None`; `parse_ini(text: str) -> tuple[dict[str, str], list[str]]`, the editable overrides and one message per refused line.
- The groups, in the design's order, as `section.key` with its range: Distance law (`distance.polymer_exponent` 0-0.6, `distance.contact_half_saturation` 0.1-5), Loops (`springs.stretch_constant` 0.01-1, `springs.squeeze_constant` 0.01-1, `springs.arc_weight_exponent` 0-2, `motif_orientation.weight` 0-200, `springs.loop_dropout`, `springs.loop_contact`), Pairs without a loop (`springs.background_weight` 0-1, `springs.background_range_bp` 0-1,000,000, `springs.use_contact_background`, `excluded_volume.arcs_repulsion_cutoff_factor` 0-3), Anchor fit (`simulation_arcs.solver` mc or lbfgs, `simulation_arcs.solver_tol` 0-1e-3, `simulation_arcs.start` centroid, walk or hilbert), Chains (`main.noise_smooth` 0.1-5, `simulation_arcs_smooth.hard_wall`, `simulation_arcs_smooth.start` line or coil), Excluded volume (`excluded_volume.use_excluded_volume`, `excluded_volume.weight` 0-10, `excluded_volume.skip_neighbors` 0-5), Confinement (`confinement.use_confinement`, `confinement.weight` 0-10), Compartments (`compartments.use_compartments`, `compartments.weight` 0-2). The design's Ensemble node is left out: SETTINGS.md has no seed key, and the structures field sets the ensemble size. Sections, types and meanings come from SETTINGS.md; a key on this list that SETTINGS.md lacks fails the test.

- [ ] **Step 1: Write the failing test.** `test_settings_schema.py`: `read_settings_md` on the repository's SETTINGS.md returns 177 rows with `polymer_exponent` in `[distance]`; every editable key exists in SETTINGS.md with its section; `validate_overrides({"distance.polymer_exponent": "0.3"})` passes and `{"distance.polymer_exponent": "0.9"}` raises naming `0..0.6`; `{"simulation_backend.mc_executor_smooth": "serial"}` raises "not a setting users may change"; `apply_overrides` on a copy of a config changes only that key; `parse_ini` of an ini with one editable key and one data key returns the first and a message for the second; `request_key` of two requests with the same overrides in a different dict order is equal, with one value changed it differs.
- [ ] **Step 2: Run it.** `python web/tests/test_settings_schema.py` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement** the module and the two endpoints. `GET /api/settings?cell=GM12878` returns `schema(...)` for that cell type's config. `POST /api/settings/ini` takes the file and returns `{overrides, refused}`.
- [ ] **Step 4: Run it and every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: settings schema from SETTINGS.md, curated overrides written into the job config"`

### Task 2: RNAPII-only loops, contact background and structures

**Files:**
- Modify: `web/gnome3d_web/catalog.py` (`FACTOR_CHOICES = ("CTCF", "RNAPII", "CTCF+RNAPII")`, `clusters_for("RNAPII")`), `web/gnome3d_web/inputs.py`, `web/gnome3d_web/jobs.py` (`JobRequest.background: str = "hic"`, `n_models` cap from `MAX_STRUCTURES`), `web/gnome3d_web/settings.py` (`MAX_STRUCTURES`)
- Test: `web/tests/test_inputs.py`, `web/tests/test_jobs.py`

**Interfaces:**
- Produces: `background` is one of `hic`, `hicdiffusion`, `none`; `hicdiffusion` replaces today's boolean `hicdiffusion` field, which stays readable in old request.json files. RNAPII-only inputs: the anchors are the RNAPII loop ends found in the cell type's CTCF+RNAPII anchors file, orientation N, the config has `[motif_orientation] use_motif_orientation = no` and `[data] factors = RNAPOL2`. `none` writes `[springs] use_contact_background = no` and `[anchor_heatmap] use_anchor_heatmap = no`; the singletons stay for the exponent fit.

- [ ] **Step 1: Write the failing tests.** In `test_inputs.py`: RNAPII-only inputs on the fixture have one cluster file, anchors only at RNAPII loop ends, every orientation N, and motif orientation off in the config; `none` sets the two keys off; `hic` leaves the config as the template has it. In `test_jobs.py`: `n_models=100` is kept, 101 becomes 100; the background is in the cache key.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test, then one job of each new kind on team-frodo** with the site of `web/scripts/serve_local.sh` and the GM12878 site data: RNAPII-only on chr1:11,500,000-12,600,000 and background none on the same region, 3 structures each. Expected: both done, the RNAPII job's engine log has no orientation term line.
- [ ] **Step 5: Commit.** `git commit -m "web: RNAPII-only loops, contact background choice, up to 100 structures"`

### Task 3: Catalog, region and preview endpoints, job submission as JSON

**Files:**
- Modify: `web/gnome3d_web/app.py`, `web/gnome3d_web/catalog.py` (`CellType.description`, `CellType.data: list[str]`, read from catalog.json with defaults)
- Test: `web/tests/test_api.py`

**Interfaces:**
- Produces:
  - `GET /api/catalog` → `{cell_types: {NAME: {label, description, data, lymphoblastoid, note}}, sv_sets: {...}, factors: [...], max_region_mb, max_structures, hicdiffusion: bool}`.
  - `GET /api/region?cell=GM12878&q=TNFRSF8` → `{chrom, start, end, label, gene?}` or 400 `{error}`; a gene name matches case insensitively; when two genes share a name the first by position is taken and `gene` says which.
  - `GET /api/preview?cell=GM12878&region=chr1:11500000-12600000` → `{loops: {CTCF: [[a, b, count], ...], RNAPII: [...]}, genes: [{name, tss}], counts: {CTCF, RNAPII}}`, loop ends as anchor midpoints, only loops inside the region.
  - `POST /api/jobs` multipart: the fields of `JobRequest` plus `overrides` as JSON text and an optional `vcf` file → 201 `{url}` or 400 `{error, field}`.

- [ ] **Step 1: Write the failing test** `test_api.py` against the fixture: the catalog lists GMX with its data chips and `max_structures` 100; `/api/region?q=alpha` resolves to ALPHA's 2 Mb window; `q=chr1:0-25000000` is 400 naming the 20 Mb limit; the preview of chr1:0-3000000 lists the fixture's 14 CTCF loops and its one RNAPII loop and both genes; `POST /api/jobs` with overrides returns a url that ends in the job's key, and the same post again returns the same url; a refused override is 400 with `field: "overrides"`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: JSON endpoints for the catalog, regions, the loop preview and job submission"`

### Task 4: Model page, side panel and step 1

**Files:**
- Create: `web/frontend/src/api.ts`, `src/pages/Model.tsx`, `src/pages/model.css`, `src/model/examples.ts`, `src/model/recent.ts`, `src/model/CellTypes.tsx`, `src/model/RegionField.tsx`
- Modify: `src/main.tsx` (QueryClientProvider), `src/App.tsx`, `web/gnome3d_web/app.py` (the old form at `/model/classic`, `/model` to the frontend)
- Test: `src/pages/Model.test.tsx`, `src/model/recent.test.ts`

**Interfaces:**
- Consumes: the endpoints of Task 3.
- Produces: `api.catalog()`, `api.region(cell, q)`, `api.preview(cell, region)`, `api.settings(cell)`, `api.parseIni(file)`, `api.submit(form: FormData)`; `EXAMPLES: Example[]` with `{id, title, cell, description, region, factors, svSet?, sample?}`; `recentJobs(): RecentJob[]` and `rememberJob(job: RecentJob)` over localStorage, at most 10, newest first, surviving a storage that throws.
- Examples: TNFRSF8 (GM12878, chr1:11,500,000-12,600,000, CTCF+RNAPII), MYC (GM12878, chr8:127,500,000-129,000,000, CTCF+RNAPII), HOXA cluster (H1ESC, chr7:27,000,000-27,400,000, CTCF). Deletion at TAL1 is added only once a hosted sample with a deletion removing a TAL1 loop anchor is found in the site data; the task records the sample or the ruling that leaves it out.

- [ ] **Step 1: Write the failing tests.** The page shows the five step headings in the design's order; the cell type cards come from the catalog with their chips; the own data card is disabled; typing TNFRSF8 shows "chr1:11,500,000-12,600,000, 1.1 Mb" from the region API; clicking the MYC example fills cell type and region; `/model?example=TNFRSF8` opens filled; recent jobs render from localStorage with their state; `checkCopy` clean. `recent.test.ts`: order, cap of 10, a throwing storage returns an empty list.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** from frame `l1JkG5`, text verbatim. API calls are mocked in tests with `vi.fn` on `fetch`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: model page, examples, recent jobs, cell type and region"`

### Task 5: Loop preview

**Files:**
- Create: `web/frontend/src/model/LoopPreview.tsx`
- Test: `src/model/LoopPreview.test.tsx`

**Interfaces:**
- Consumes: `api.preview`.
- Produces: `<LoopPreview cell region factors />`, an SVG of the region as an axis with CTCF arcs above and RNAPII arcs below in their factor colours, the genes as ticks with the region's gene named, and the key "118 CTCF loops" and "64 RNAPII loops" from the counts; factors not chosen are drawn faint.

- [ ] **Step 1: Write the failing test.** With a mocked preview of 2 CTCF and 1 RNAPII loop, the SVG has 2 paths above the axis and 1 below, the key reads "2 CTCF loops" and "1 RNAPII loops" as the counts give, and every arc has a `title` with its ends and count.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.** Arc height grows with span; widths scale the region to the SVG.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: loop preview"`

### Task 6: Steps 2 to 4

**Files:**
- Create: `web/frontend/src/model/LoopOptions.tsx`, `src/model/Variants.tsx`, `src/model/Background.tsx`
- Test: `src/model/steps.test.tsx`

- [ ] **Step 1: Write the failing test.** Loop options CTCF, RNAPII, CTCF and RNAPII, Your own loops, the last disabled; CTCF and RNAPII selected by default. Variants: choosing a hosted set lists its samples; haplotype first or second; a VCF file can be chosen. Background: Hi-C selected by default; HiCDiffusion disabled with its reason when the region is over 2 Mb or the server has no HiCDiffusion; the HiCDiffusion notice shows for a cell type that is not lymphoblastoid; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from the design.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: loops, structural variants and contact background"`

### Task 7: Step 5, structures, job name and the settings tree

**Files:**
- Create: `web/frontend/src/model/SettingsTree.tsx`, `src/model/settings.ts`
- Test: `src/model/SettingsTree.test.tsx`, `src/model/settings.test.ts`

**Interfaces:**
- Consumes: `api.settings`, `api.parseIni`.
- Produces: `changedSettings(schema, values) -> Record<string, string>` keyed `section.key`, only values that differ from the server's; `toIni(changes) -> string`, one section block per section; `<SettingsTree schema values onChange />` with the filter, groups as collapsible nodes with a badge of changes, a pane of the selected group with a slider and a number box per numeric setting and a select per choice, "default <x>" with a reset per changed setting, Reset all, Upload config.ini and Download.

- [ ] **Step 1: Write the failing tests.** `settings.test.ts`: `changedSettings` drops unchanged values; `toIni` of two changes in two sections writes two sections. `SettingsTree.test.tsx` with a mocked schema of two groups: the filter "expon" leaves `polymer_exponent`; moving the value of `polymer_exponent` to 0.3 shows the badge "1" on Distance law and "1 changed" in the head; reset returns it; a value typed outside the range is clamped to it; an uploaded ini with one refused key shows the refusal message; the structures field accepts 1 to `max_structures`; the job name field has the design's placeholder.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** with @radix-ui/react-slider and @radix-ui/react-collapsible, styled from the tokens.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: structures, job name and the settings tree"`

### Task 8: Submit, routing and the full check

**Files:**
- Modify: `web/frontend/src/pages/Model.tsx`, `web/README.md`
- Test: `src/pages/Model.test.tsx`, `web/tests/test_frontend.py`

- [ ] **Step 1: Write the failing tests.** The submit button reads "Build 100 structures" for 100 structures; submitting posts one multipart request with the chosen fields and the overrides as JSON, remembers the job with its name in recent jobs and moves to the returned url; a 400 shows its message at the field it names. `test_frontend.py`: `/model` returns the app and `/model/classic` the old form.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every frontend and web test, build, then on team-frodo submit the TNFRSF8 example from the page through the site of `serve_local.sh`.** Expected: all pass; the job finishes and its page opens. Screenshots of the page at 1440 and 390 px wide in headless Chrome show no overflow.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: job submission from the model page"`
