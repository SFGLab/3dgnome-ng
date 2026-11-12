# Website frontend, part 4: your own data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, the user's standing choice). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The own data uploads of the design: loops per factor, anchors, contacts and enhancers in the Your own data dialog, the structural variants dialog with a VCF's samples and the change each variant makes to the loops, contact maps by link or from a local .hic, and the `gnome3d-ng extract` command for a local .mcool.

**Architecture:** An upload is checked when it arrives and kept under `RESULTS_DIR/uploads/<id>/`, private, swept with the retention period. The check returns the design's one line summary or the line, what was found and the fix. A job that names uploads is private; its inputs take the uploads in place of the cell type's files. The own data cell card uses the first hosted cell type for genes, segments, centromeres and the config. Contacts by link read the window's bin pairs over HTTP range requests and become an upload like any other.

**Tech Stack:** as parts 1 to 3, plus hic-straw (Python) for .hic links, h5py with fsspec for .mcool links, and the hic-straw JavaScript library for a local .hic in the browser.

**Spec:** the dialogs in web/design/exports: LgRHG (your own loops), jjmfO (your own contacts), N3hxD (structural variants), PNG for the layout and `.txt` for the text. design/plans/2026-11-12-web-own-data.md for the formats, checks and privacy. gnome3d/io.py for what the engine reads.

## Global Constraints

- The Global Constraints of parts 1 to 3 hold.
- Every upload is streamed to disk and refused past `UPLOAD_MAX_MB`, default 200, with the size and the limit. Nothing is read whole into memory.
- A job with any upload is private, under `/p/<token>`, and deleted with its uploads after `RETENTION_DAYS`.
- Formats as gnome3d/io.py reads them: loops are BEDPE with seven columns and an integer count in column 7; anchors are BED with an optional orientation in column 4 on every line or on none; contacts are BEDPE with an integer count, each end placed by its start column; enhancers are BED.
- An error names the line, what was found and the fix, as the design does.

## Review Focus

- A loops file with Windows line ends, a header line or a trailing empty line: read, the header line skipped with a note. Test in Task 1.
- A file over the size cap: refused while streaming, with no partial file left in uploads. Test in Task 1.
- An own data job whose loops have no anchors file and whose loop ends overlap: anchors built from the ends, overlapping ends merged, orientation N. Test in Task 2.
- A VCF with no sample columns, or a sample that is not in it: a clear error at the sample field. Test in Task 3.
- A link that answers without range requests or with HTML: refused naming the reason, no download of the whole file. Test in Task 7.

---

### Task 1: Upload store and checks

**Files:**
- Create: `web/gnome3d_web/owndata.py`
- Modify: `web/gnome3d_web/app.py`, `web/gnome3d_web/settings.py` (`UPLOAD_MAX_MB`), `web/gnome3d_web/queue.py` (sweep uploads)
- Test: `web/tests/test_owndata.py`

**Interfaces:**
- Produces: `POST /api/uploads` multipart `kind` (loops, anchors, contacts, enhancers, vcf), `file`, optional `cell` and `region` → 201 `{id, kind, name, size, ok, summary, error: {line, found, fix} | null}`; `GET /api/uploads/<id>/preview?region=` → loops of a loops upload inside the region as `[[a, b, pets]]`.
- `owndata.py`: `store(stream, name, kind) -> Path` streaming with the cap; `check_loops(path, region) -> Check`, `check_anchors(path) -> Check`, `check_contacts(path, region) -> Check`, `check_enhancers(path) -> Check`, `vcf_samples(path) -> list[str]`; `Check(ok, summary, error)`.
- Loop summary: "12,433 loops read. 118 are in the region. 37 loops between chromosomes are skipped, and 4 loops longer than 1 Mb go to the coarse contact map." with the parts that are zero left out. Ten columns with a letter in column 7: "Line 1 has 10 columns, and column 7 is the letter R where a count is expected. This is the loop format of 3D-GNOME 3.0. Move the orientations to the anchors file and keep one file per factor with seven columns."

- [ ] **Step 1: Write the failing test** `test_owndata.py`: the fixture's ctcf.bedpe as an upload gives "14 loops read." and, with the region chr1:0-1000000, the count in the region; a ten column loop file gives the design's error on line 1; a file with CRLF line ends and a `#` header reads with the header skipped; a loop file with a non integer count gives the line and the fix; anchors with orientation on some lines only give the line where it changes; a contacts BEDPE gives its pair count; an enhancers BED its count; a VCF upload lists its samples; a file over a 1 MB cap set for the test is a 413 with the size, and the uploads directory holds no part of it; the preview of a loops upload lists the loops in a region; uploads older than the retention period are swept.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: uploads of loops, anchors, contacts, enhancers and VCFs, checked on arrival"`

### Task 2: Jobs with own data

**Files:**
- Modify: `web/gnome3d_web/jobs.py` (`JobRequest.own: dict[str, object]`, a private job), `web/gnome3d_web/inputs.py`, `web/gnome3d_web/app.py` (`create_job` reads `own` as JSON, `cell_type` may be `own`)
- Test: `web/tests/test_owndata.py`, `web/tests/test_inputs.py`

**Interfaces:**
- Consumes: upload ids of Task 1.
- Produces: `own = {"loops": [{"factor": "CTCF", "upload": id}, ...], "anchors": id?, "contacts": id?, "enhancers": id?}`; the cell type `own` takes genes, segments, centromeres and the config from the first hosted cell type and needs loops and contacts; a hosted cell type with own loops keeps its contacts unless contacts are uploaded. Without an anchors upload the anchors are the loop ends, overlapping ends merged, orientation N. The engine's factor names are the uploads' factor names, the first one taking the orientation term only when it is named CTCF.

- [ ] **Step 1: Write the failing tests.** Inputs with own loops of two factors write two cluster files with the factors named; without an anchors upload the anchors are the merged loop ends with N; own contacts replace the singletons; own enhancers replace the cell type's in the marks; a job with own data posted to `/api/jobs` returns a `/p/` url and runs to done with the fake engine; an own data job without contacts on cell type `own` is a 400 on the contacts field.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test, then one own data job on team-frodo:** GM12878's CTCF and RNAPII loop files uploaded as two factors on cell type `own` with GM12878's singletons as contacts, chr1:11,500,000-12,600,000, 3 structures. Expected: done.
- [ ] **Step 5: Commit.** `git commit -m "web: jobs with own loops, anchors, contacts and enhancers"`

### Task 3: Structural variant check

**Files:**
- Modify: `web/gnome3d_web/owndata.py`, `web/gnome3d_web/app.py`
- Test: `web/tests/test_owndata.py`

**Interfaces:**
- Produces: `POST /api/variants/check` with `cell`, `region`, `factors`, `sample`, `haplotype` and `upload` or `sv_set` → `{summary, applied: {DEL, INS, INV}, skipped: {type: n}, variants: [{type, pos, end, length, genotype, change}]}`; `change` is what the rules do to the loops for that variant alone, worded as in the design: "removes 1 CTCF anchor and its 3 loops", "adds 1 CTCF anchor, motif score 14.2, 1 new loop", "flips 2 anchors, replaces 4 loops", "shortens 2 loops", "no CTCF motif above the cutoff", "no anchor inside".

- [ ] **Step 1: Write the failing test.** On the fixture's DEMO set and P1 haplotype 1 over chr1:0-3,000,000: the summary counts the applied deletion and insertion and the skipped CNV; the deletion's change names the anchors and loops it removes; a sample not in the VCF is a 400 on the sample field; a VCF without sample columns is a 400 naming that.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** by running the rewrite rules on the region's loops with each variant alone and comparing anchors and loops before and after.
- [ ] **Step 4: Run every web test.** Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "web: the change each structural variant makes to the loops"`

### Task 4: Your own data dialog, loops, anchors and enhancers

**Files:**
- Create: `web/frontend/src/model/OwnData.tsx`, `src/model/upload.ts`
- Modify: `src/model/form.ts` (`own`), `src/model/CellTypes.tsx`, `src/model/LoopOptions.tsx` (both enabled), `src/pages/Model.tsx`, `src/api.ts`
- Test: `src/model/OwnData.test.tsx`

**Interfaces:**
- Produces: `uploadFile(kind, file, {cell, region, factor}) -> Promise<UploadResult>`; `<OwnData open initialTab onClose onUse />` with the design's tabs Loops (file count), Anchors, Contacts (.hic, .mcool or BEDPE), Enhancers; a factor row per loops file with its name, colour, file and size, a remove button and the check line; "Add a factor, or drop a BEDPE file here"; the preview of the region's loops; the footer naming the file to fix while any check fails, and "Use these loops" disabled until all pass.

- [ ] **Step 1: Write the failing test.** Adding two loops files shows two rows with the mocked check lines; a third file with the ten column error shows the error box and "Fix or remove the H3K27ac file to continue." with the button disabled; removing it enables the button; the factor name of a row can be edited and the first row is CTCF by default; Anchors and Enhancers take one file each with their check lines; the own data card and the own loops option open the dialog; "Use these loops" puts the uploads in the form; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from LgRHG.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: your own data dialog, loops, anchors and enhancers"`

### Task 5: Contacts tab of the dialog

**Files:**
- Modify: `web/frontend/src/model/OwnData.tsx`
- Test: `src/model/OwnData.test.tsx`

- [ ] **Step 1: Write the failing test.** The tab offers Link to a file, Local .hic file and BEDPE of the region; a BEDPE upload shows its check line; the link field and its Read button call the link endpoint of Task 7 and show its summary or its error with the design's text; the local .hic route reads the window through the reader of Task 8; the hint text and the .mcool card of the design are present; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from jjmfO; the link and local routes call interfaces stubbed until Tasks 7 and 8.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: contacts tab of the own data dialog"`

### Task 6: Structural variants dialog

**Files:**
- Create: `web/frontend/src/model/VariantsDialog.tsx`
- Modify: `src/model/Variants.tsx`, `src/pages/Model.tsx`
- Test: `src/model/VariantsDialog.test.tsx`

- [ ] **Step 1: Write the failing test.** Dropping a VCF in step 3 opens the dialog on Upload a VCF with the file, its samples in the sample select and the haplotype; the check shows the summary of the mock in the green box, the variant track and the table with Type, Position, Length, Genotype and Change to the loops; "Apply 6 variants" names the count and puts the VCF, sample and haplotype in the form; Hosted call set shows the same for a hosted set; the footer note of the design; `checkCopy` clean.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** from N3hxD.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: structural variants dialog"`

### Task 7: Contacts by link

**Files:**
- Create: `web/gnome3d_web/remote.py`
- Modify: `web/gnome3d_web/app.py`, `web/requirements.txt` (hic-straw, h5py, fsspec, aiohttp)
- Test: `web/tests/test_remote.py`

**Interfaces:**
- Produces: `POST /api/uploads/link` with `url`, `cell`, `region` → the upload result of Task 1 for a contacts upload made from the window's bin pairs at 25 kb, its summary naming the file, its size and resolutions; `remote.check_range(url) -> int` the file size, raising when the server answers without `Accept-Ranges: bytes` or with HTML.

- [ ] **Step 1: Write the failing test** against a local HTTP server in a thread: a small .mcool written with cooler on the fixture's singletons, served with range requests, reads the window's pairs into an upload whose pair count matches the fixture's; the same file served without range requests is a 400 with "does not answer range requests"; an HTML answer is a 400 naming it. A .hic link is checked by hand on team-frodo against a public 4DN .hic file, there being no writer for a .hic fixture.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every web test, then a public .hic link on team-frodo.** Expected: all pass; the .hic window read in under a minute.
- [ ] **Step 5: Commit.** `git commit -m "web: contact maps by link with range requests"`

### Task 8: A local .hic read in the browser

**Files:**
- Create: `web/frontend/src/model/hicLocal.ts`
- Modify: `src/model/OwnData.tsx`, `web/frontend/package.json` (hic-straw)
- Test: `src/model/hicLocal.test.ts`

**Interfaces:**
- Produces: `readHicWindow(file: File, region, res) -> Promise<string>` the window's bin pairs as BEDPE text, uploaded as a contacts BEDPE.

- [ ] **Step 1: Write the failing test** with the hic-straw reader mocked: the records become BEDPE lines with integer counts; a chromosome name without chr is mapped; a resolution the file lacks is an error naming the ones it has.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, then a real .hic file on team-frodo through the page.** Expected: PASS; the window uploaded and checked.
- [ ] **Step 5: Commit.** `git commit -m "web frontend: a local .hic file read in the browser"`

### Task 9: gnome3d-ng extract

**Files:**
- Create: `gnome3d/extract.py`
- Modify: `gnome3d/cli.py` (a first argument `extract` runs it)
- Test: `harness/test_extract.py`

**Interfaces:**
- Produces: `gnome3d-ng extract --mcool FILE --region chr:s-e [--res 25000] [--contacts-per-mb N] --out FILE` writing the window's intra chromosomal bin pairs as seven column BEDPE, the code of slurm/ensemble/prep_singletons.py for one region.

- [ ] **Step 1: Write the failing test.** On a small .mcool written with cooler, the output's pair count and positions match the matrix; `--contacts-per-mb` thins to the count; an unknown chromosome names the ones the file has.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, pyright and ruff on the file.** Expected: PASS, no new pyright errors.
- [ ] **Step 5: Commit.** `git commit -m "gnome3d-ng extract: the bin pairs of a region of a .mcool as BEDPE"`

### Task 10: End to end

- [ ] **Step 1: On team-frodo, through the page,** an own data job with two loop factors and uploaded contacts, a job with an uploaded VCF and its variants table, and a contacts link. Expected: each job done and its explorer open.
- [ ] **Step 2: Screenshots** of the three dialogs at 1440 px. Expected: as the design.
- [ ] **Step 3: Commit** the README section on uploads. `git commit -m "web: own data checked on team-frodo"`
