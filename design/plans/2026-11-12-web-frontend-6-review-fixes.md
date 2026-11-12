# Website frontend part 6: review fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** The user's review of 2026-11-12: the own data dialog as a wizard with a side panel and
consistent step subtitles, examples that open finished jobs, the comparison link dropped, and a
My jobs page with an empty state.

**Architecture:** Frontend only, `web/frontend/src`. Examples reuse the server's request cache:
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
- Tests: a card posts its request and opens the job url with its tab; the compare example posts both and opens the first with `compare`; the Job page opens the comparison from the query; checkCopy clean.

### Task 3: The comparison link dropped, My jobs with an empty state

**Files:** `src/pages/Title.tsx`; create `src/pages/MyJobs.tsx`; modify `src/App.tsx`. Tests `src/pages/Title.test.tsx`, `src/pages/MyJobs.test.tsx`.

- The link "Comparison with Hi-C and expression data" goes; the navigation keeps Results.
- /jobs lists this browser's jobs with their state, name and date, and with none shows "You have no jobs in this browser yet." and a link to Model a region; a note says jobs are kept per browser.
- Tests: the empty state; a list of two jobs; storage that throws gives the empty state; checkCopy clean.

### Task 4: On team-frodo

- Deploy; open each example through the title page and the Examples page; the wizard at 1440 and 390 px in both themes; My jobs empty and filled. Ledger copied to design/plans.
