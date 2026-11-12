# Website frontend part 6: review fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** The user's review of 2026-11-12: the own data dialog as a wizard with a side panel and
consistent step subtitles, examples that open finished jobs, the comparison link dropped, a
My jobs page with an empty state, a progress bar with a status log, depth and hover in the 3D view, and every table paginated.

**Architecture:** Mostly `web/frontend/src`; the progress task adds stages to the job's status. Examples reuse the server's request cache:
submitting an example's request returns the shared job, finished or running, so no server change.

**Spec:** the user's review of part 5, 2026-11-12, with a screenshot of the dialog.

## Global Constraints

- Writing rules of design/writing-rules.md; every page test runs checkCopy.
- Frontend code follows web/frontend/.prettierrc.json. Nothing is deleted outside the code changed.

## Review Focus

- A wizard step the user skips: Use stays available once every chosen file passes; skipping is allowed.
- An example whose job is still running: the job page shows its progress, never an empty page.
- My jobs in a browser that blocks storage: the empty state, not an error.

---

### Task 1: The own data dialog as a wizard

**Files:** `src/model/OwnData.tsx`, `owndata.css`; tests `src/model/OwnData.test.tsx`, `src/pages/Model.test.tsx`.

- A side panel lists the steps 1 Loops, 2 Anchors, 3 Contacts, 4 Enhancers as a vertical tablist. Each step's subtitle names what the job will use, in one form for all four: Loops "2 files" or "the listed sets"; Anchors the file name or "the loop ends"; Contacts the file name, "the cell type's Hi-C", or "needed" on your own cell type; Enhancers the file name, "the cell type's", a listed cell type's name, or "none".
- The footer has Back, Next and one action, "Use this data", on every step. A failing check names its file in the footer as before.
- Tests: the four steps with their subtitles; Next and Back move between steps; the action is "Use this data" on every step; checkCopy clean.

### Task 2: Examples that open finished jobs

**Files:** create `src/pages/Examples.tsx`, `src/model/openExample.ts`; modify `src/model/examples.ts`, `src/pages/Title.tsx`, `src/pages/Job.tsx` (a `compare` query opens the comparison with that job), `src/App.tsx`, `src/routes.ts`. Tests `src/pages/Examples.test.tsx`, `src/pages/Title.test.tsx`, `src/pages/Job.test.tsx`.

- Every example is a request plus where its job opens: a tab, or a second request to compare with. Opening one submits its request, ten structures, which the server's cache turns into the shared job, and goes to its page.
- The title page cards and an Examples page at /examples, linked as All examples, open examples. The Model page's side panel keeps filling the form.
- The job page keeps its tab and its comparison in the URL as history entries, so Back returns to the previous tab or closes the comparison instead of leaving the job.
- Tests: a card posts its request and opens the job url with its tab; Back after two tab clicks returns to the first tab; the compare example posts both and opens the first with `compare`; the Job page opens the comparison from the query; checkCopy clean.

### Task 3: The comparison link dropped, My jobs with an empty state

**Files:** `src/pages/Title.tsx`; create `src/pages/MyJobs.tsx`; modify `src/App.tsx`. Tests `src/pages/Title.test.tsx`, `src/pages/MyJobs.test.tsx`.

- The link "Comparison with Hi-C and expression data" goes; the navigation keeps Results.
- /jobs lists this browser's jobs with their state, name and date, and with none shows "You have no jobs in this browser yet." and a link to Model a region; a note says jobs are kept per browser.
- Tests: the empty state; a list of two jobs; storage that throws gives the empty state; checkCopy clean.

### Task 4: A progress bar and status log for a running job

**Files:** `web/gnome3d_web/jobs.py` (`_status` keeps a list of stages with their start times), `explore.py` (`summary` adds `stages` and, while modelling, `structures_done` from the written .cif files); `web/frontend/src/pages/Job.tsx`, create `src/job/Progress.tsx`. Tests `web/tests/test_explore.py`, `src/pages/Job.test.tsx`.

- A queued or running job shows a bar over its stages, queued, inputs, HiCDiffusion when asked, modelling and marks, the modelling share filled by structures written out of those asked, its place in the queue, and below it a status log: each stage with its start time, then the last lines of engine.log, refreshed with the summary.
- Tests: the summary carries the stages and the count of written structures; the page shows the bar at the right share, the stage lines and the log tail; a failed job keeps its log; checkCopy clean.

### Task 5: Depth and hover in the 3D structure

**Files:** `web/frontend/src/job/Scene.tsx`, `StructureView.tsx`. Test `src/job/StructureView.test.tsx` if one exists, else a render check.

- The chain is drawn as a lit tube with a light from the camera, and fog from the middle of the structure to its far side, so near strands are dark and sharp and far strands fade toward the background, the way a molecular viewer shades depth.
- The bead under the pointer is marked plainly: a ring and a larger sphere in a colour no mark uses, the stretch of chain around it brightened, and its position in a label beside the pointer.
- Check: render the TNFRSF8 job on team-frodo before and after, light and dark, with a bead hovered.

### Task 6: Every table paginated

**Files:** create `web/frontend/src/components/Pager.tsx` and a `usePage` hook beside it; modify every table of the site: `src/job/LoopsTab.tsx`, `GenesTab.tsx`, `Compare.tsx`, `src/model/VariantsDialog.tsx`, `src/pages/MyJobs.tsx`, and any other `<table>` found by search. Test `src/components/Pager.test.tsx` and one case per table.

- 50 rows a page by default, a pager under the table with the row range, Previous and Next, and the page reset when a filter or sort changes. A table of 50 rows or fewer shows no pager.
- The loops endpoint, if it returns every loop, gains `offset` and `limit` so a page is fetched, not the whole list.

### Task 7: The contact map at the width of its panel

**Files:** `web/frontend/src/job/ContactMapTab.tsx`, `heatmap.ts`, the job css. Test `src/pages/Job.test.tsx`.

- The maps scale to the panel's width, two side by side on a wide screen and one above the other on a narrow one, the canvas drawn at the device pixel ratio so a bin stays sharp; hover and the axis follow the drawn size.
- Check: 1440, 1024 and 390 px screenshots.

### Task 9: The contact map's Hi-C and model halves, and its agreement, as the battery has them

**Files:** `web/gnome3d_web/contacts.py`, `explore.py`, `catalog.py` (an optional deep map link per cell type), `web/frontend/src/job/ContactMapTab.tsx`. Tests `web/tests/test_contacts.py`, `src/job/ContactMapTab.test.tsx`.

User report 2026-11-13: the model half is mostly white and the agreement reads -0.17 on a job where the two maps agree by eye. Measured on frodo's TNFRSF8 job of 10 structures: per diagonal Pearson 0.43 at one bin and Spearman 0.51 over the map, while the site's score was -0.11. Four differences from the validation battery (`validation/metrics/hic.py`):

- The model half counts pairs within a fixed 1.5 bead units, so with few structures most far cells are 0. It becomes the battery's MultiMM map: per structure the distance between bin centroids scaled by the median step between neighbouring bins, 1/(d+1)^3, averaged over the structures. The counted map stays as a second choice.
- The agreement drops pairs that are zero in both maps before each diagonal's Pearson and weights by rank spread. It becomes the battery's `_scc`: whole diagonals weighted by their length and the two standard deviations, on raw counts, the model counted within the median bond of the structures, diagonals up to 2 Mb.
- The Hi-C half is the job's input, thinned to the hosted density. A cell type whose catalog entry names a deep map in the 4DN bucket gets its window read through `remote.read_window` once per job and resolution and cached in the job; the job's input stays the fallback, and the card names which map it used.
- ICE stays a display choice; the agreement never reads it.

- Tests: the MultiMM map has no zero cell off the diagonal and equals a hand computed value on a three bead chain; the agreement equals the battery's `_scc` on the same pair of maps; a deep map link is read once and cached; the card names the map.

### Task 8: On team-frodo

- Deploy; open each example through the title page and the Examples page; the wizard at 1440 and 390 px in both themes; My jobs empty and filled. Ledger copied to design/plans.
