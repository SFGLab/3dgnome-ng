# Website frontend part 5: follow ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close what part 4 left: the VCF sent once, your own cell type choosing listed loops and
enhancers, a light and dark switch, every deferred minor of part 4, and 4DN files fetched from
their bucket.

**Architecture:** Server changes in `web/gnome3d_web/` (app, jobs, inputs, remote), frontend
changes in `web/frontend/src/` (model, components), one fetch change in `validation/studies/`.
Each task carries its own failing test first.

**Tech Stack:** FastAPI, React 18 with TypeScript and Vitest, numpy and pandas.

**Spec:** the user's review of part 4 on 2026-11-12, design/plans/2026-11-12-web-own-data.md,
design/4dn-data-access.md, web/design/site.pen.

## Global Constraints

- Writing rules of design/writing-rules.md in every visible string; every page test runs checkCopy.
- Nothing is deleted; upload directories stay until the retention sweep.
- Frontend code follows web/frontend/.prettierrc.json.
- Never touch main; production on team-aragorn stays untouched.

## Review Focus

- A job form whose VCF upload id is unknown, expired, of another kind or failed its check: a 400 on the variants step.
- Your own cell type with listed loops and no contacts upload: refused, as today, naming contacts.
- The theme stored in a browser where storage throws: the page renders on the system theme.
- A burst of link reads beyond the limit: refused at once with a message, never queued for minutes.
- A 4DN accession absent from the bucket: fetch names the accession and stops.

---

### Task 1: The VCF sent once

**Files:** Modify `web/gnome3d_web/app.py` (`create_job`), `web/frontend/src/model/form.ts`, `VariantsDialog.tsx`, `src/pages/Model.tsx`. Test `web/tests/test_owndata.py`, `src/model/VariantsDialog.test.tsx`, `src/pages/Model.test.tsx`.

**Interfaces:** POST /api/jobs takes `vcf_upload`, an upload id of kind vcf that passed its check; the job copies that file as `upload.vcf.gz`. The form holds `vcfUpload: UploadResult | null` in place of `vcf: File`. The dialog reuses a passed upload rather than uploading again.

- [ ] Failing tests: a job with `vcf_upload` runs and is private; an unknown, failed or non vcf id is a 400 on `sample`'s step; the page posts `vcf_upload` and no file; reopening the dialog with an upload makes no second upload.
- [ ] Implement, run, commit `web: the VCF of a job is its upload`.

### Task 2: Your own cell type takes listed loops and enhancers

**Files:** Modify `web/gnome3d_web/app.py`, `jobs.py`, `inputs.py`, `owndata.py` (`variant_check`); `web/frontend/src/model/OwnData.tsx`, `LoopOptions.tsx`, `form.ts`, `src/pages/Model.tsx`. Test `web/tests/test_owndata.py`, `test_inputs.py`, `src/model/OwnData.test.tsx`, `src/model/steps.test.tsx`.

**Interfaces:** the `own` JSON gains `loops_from` (a listed cell type, used with `factors` when no loops are uploaded) and `enhancers_from` (a listed cell type or `none`, used when no enhancers are uploaded; default `none`). On your own cell type, step 2 offers the three listed loop sets with a select "Loops of" naming the cell type, and the own loops option. The Enhancers tab offers Upload a BED, Enhancers of a listed cell type with its select, and None.

- [ ] Failing tests: an own cell type job with `loops_from` H1ESC and CTCF writes H1ESC's CTCF loops and anchors; `enhancers_from` marks that cell type's enhancers and `none` marks none; an unknown cell type in either is a 400; the variant check reads the borrowed loops; step 2 on your own cell type enables the listed sets with the select; the Enhancers tab's three choices put the right fields in the form; checkCopy clean.
- [ ] Implement, run, commit `web: your own cell type takes listed loops and enhancers`.

### Task 3: Light and dark

**Files:** Create `web/frontend/src/components/ThemeSwitch.tsx`; modify `Layout.tsx`, `layout.css`, `index.html` (the stored theme applied before first paint). Test `src/components/Layout.test.tsx`.

**Interfaces:** a three way switch, System, Light and Dark, in the header after the navigation; it sets `data-theme` on `<html>` (absent for System) and stores the choice in localStorage, every read and write in try/catch.

- [ ] Failing tests: the switch sets and clears `data-theme`; a stored choice is applied on load; a throwing storage leaves the page on System; checkCopy clean.
- [ ] Implement, screenshot both themes at 1440 and 390 px, commit `web frontend: a light and dark switch`.

### Task 4: The deferred minors of part 4

**Files:** Modify `web/gnome3d_web/app.py`, `remote.py`, `inputs.py`, `web/README.md`, `web/docker-compose.yml`. Tests in `web/tests/test_remote.py`, `test_inputs.py`, `test_owndata.py`.

- [ ] Link reads at most `LINK_READS` at once (default 4); one more is a 503 with "The server is reading other links. Try again in a minute." at once.
- [ ] The thinning target is at least one contact per pair window, never zero.
- [ ] A non vcf upload id at /api/variants/check and an `own` entry that is not an object are 400s naming the field.
- [ ] DNS rebinding: the address resolved at the check is the one read. The .mcool reader connects through an aiohttp resolver pinned to the checked addresses; hic-straw reads through a local range proxy in the same process bound to the pinned address. README states the worker's network rule for production as a second line.
- [ ] Each with its failing test; commit `web: the deferred minors of part 4`.

### Task 5: 4DN files from their bucket

**Files:** Modify `validation/studies/fetch.py`; create `validation/manifests/4dn_keys.tsv`. Test `harness/test_fetch_4dn.py`.

**Interfaces:** a 4DN accession resolves to its key in `4dn-open-data-public` through the table, filled by a listing of `fourfront-webprod/wfoutput/` when an accession is missing; an accession absent from the bucket stops with its name.

- [ ] Failing test with the listing mocked: a known accession gives its bucket url; a missing one fills the table; an absent one stops naming it.
- [ ] Implement; resolve every 4DN accession of the manifests into the table; commit `validation: 4DN files from their bucket`.

### Task 6: End to end on team-frodo

- [ ] Through the page: an own cell type job with H1ESC CTCF loops, contacts by link to 4DNFI1UEG1HD.hic in the bucket and GM12878 enhancers; a GM12878 job with an uploaded VCF; the switch in both themes. Each job done and its explorer open.
- [ ] Ledger copied to design/plans; commit `web: part 5 checked on team-frodo`.
