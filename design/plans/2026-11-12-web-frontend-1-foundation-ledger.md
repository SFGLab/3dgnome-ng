# SDD ledger — plan: design/plans/2026-11-12-web-frontend-1-foundation.md
Setup: Ruling: work on branch rnapii-crosscell in the main checkout, no worktree — the user asked not to use worktrees in this repo (memory project_rnapii_crosscell_release) — cost if wrong: none, the branch is not main.
Setup: Ruling: local Node is 18.20.4, the plan names Node 22 for the build; Vite 5 runs on Node 18, the Docker stage keeps node:22 — cost if wrong: a local build differing from the image build.
Setup: spec web/design/site.pen needs the Pencil editor open; Tasks 1, 2, 4-7 read it. Task 3 does not.
Pre-flight: Task 3 consumes dist/index.html + assets/ from Task 1 build: consistent. Task 6 consumes REFERENCES from Task 5: consistent. Tasks 4-7 consume Layout/ROUTES/checkCopy from Task 2: consistent.
Task 3: Ruling: taken before Tasks 1-2 — the Pencil file is closed and Task 3's test builds its own stand-in dist — cost if wrong: none, Task 3 consumes only the dist layout.
Task 3: Ruling: the fallback also refuses jobs/, p/ and assets/ paths with 404 — test_app's path traversal check under /jobs/<key>/files expects a 404, and those prefixes belong to the server — cost if wrong: part 3 moves job pages to the client and drops jobs/ from the list.
Task 3: complete (commits 928cca6..ba9be36, tests: bash -c 'cd web/tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null || { echo FAIL $t; exit 1; }; done; echo all web tests pass' → all web tests pass)
Task 1: Ruling: dark tokens derived, the design has light values only — neutrals mirrored, factor colours CTCF #4A96E6, RNAPII #8A44C4, promoter #E0622E, enhancer #1FA673 pass every check of the dataviz validator on surface #1C1C1A (the light set fails it in dark: promoter outside the band, RNAPII under 3:1) — cost if wrong: dark mode colours differ from what the user would pick.
Task 1: Ruling: fonts Inter and JetBrains Mono self hosted through @fontsource (OFL) — no CDN at runtime — cost if wrong: about 1 MB of font files in dist.
Task 1: Ruling: the Vite template's demo files (README.md, App.css, App.tsx, index.css, assets/react.svg, public/vite.svg) are left uncommitted and not deleted — the user does not want files deleted — cost if wrong: untracked clutter until the user removes it.
Task 1: complete (commits ba9be36..2532ffc, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 406ms)
Task 2: Ruling: ROUTES gains jobs: "/jobs" — the design's navigation has My jobs between Model a region and Method, the plan's ROUTES lacked it — cost if wrong: one key.
Task 2: Ruling: lucide-react for the design's Lucide icons (quote, arrow-right) — the design names the lucide library — cost if wrong: one dependency.
Task 2: Ruling: test setup calls Testing Library cleanup after each test — Vitest globals are off and auto cleanup needs them — cost if wrong: none.
Task 2: Ruling: the App has a catch-all route rendering nothing until the page tasks add theirs — the Task 2 brief names App.tsx without its routes — cost if wrong: none.
Task 2: complete (commits 2532ffc..d4a8542, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 1.09s)
Task 4: Ruling: the TNFRSF8 example button and the example cards link to /model?example=<name>, and All examples to /model — the job pages they show are part 3 — cost if wrong: links change once part 3 exists.
Task 4: complete (commits d4a8542..cd1bd3a, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 1.01s)
Task 5: Ruling: Reference gains a group field (gnome, algorithms, related, data) and DATA_SOURCES holds the data table rows — the design lists references under four headings and cites data papers in a table by short label — cost if wrong: a field to drop.
Task 5: Ruling: the data table's short citations got full entries checked on Crossref (Tang 2015 10.1016/j.cell.2015.11.024, 4DN 2017 10.1038/nature23884, EnhancerAtlas 2.0 10.1093/nar/gkz980, HiCDiffusion BMC Genomics 25, 964 10.1186/s12864-024-10885-z) — the design gives short labels only — cost if wrong: one wrong DOI link.
Task 5: Ruling: the "a DOI or a URL" test applies to confirmed entries; HGSVC freeze 4 has neither and is marked unconfirmed as the design marks it — cost if wrong: none.
Task 5: Ruling: Source code links to https://github.com/SFGLab/3dgnome-ng, the repository's remote; the primary BibTeX is a placeholder like the design's citation text — cost if wrong: a link to a private repository until it is public.
Task 5: complete (commits cd1bd3a..cc28496, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 1.16s)
Task 6: Ruling: section headings follow the design, numbered, "2. Blocks and target distances" in place of the plan's "Distance law" — the design is the spec — cost if wrong: none.
Task 6: Ruling: the Method reference list renders from REFERENCES by key in the design's order [1]-[17]; small wording differences from the design's list (MultiMM short title, JAX "version 0.4") take the Cite page's form — one source for every reference — cost if wrong: the two pages differ from the design by a few words.
Task 6: Ruling: the engine figure panels are exported from Pencil as SVG (engine_graph.svg, engine_batching.svg); equation alt text is the TeX of render_equations.py in assets/eq/tex.json — cost if wrong: a re-export after a design change.
Task 6: Ruling: layout styles shared by the documentation pages moved from cite.css to styles/doc.css — Cite and Method share the side panel and reading column — cost if wrong: none.
Task 6: complete (commits cc28496..b1c0597, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 1.13s)
Task 7: Ruling: the original models' grey #A9A8A3 becomes token --original (dark #8E8D88) — the design uses it as a raw colour for a series — cost if wrong: none.
Task 7: Ruling: the Results side panel's Limitations link goes to /method#limitations — the design's Results page has no Limitations section — cost if wrong: a dead link otherwise.
Task 7: Ruling: the design's inner "Results" heading repeating the page lead is not rendered — the writing rules forbid repeated facts — cost if wrong: a heading the user wanted.
Task 7: Ruling: the Help settings link says "All 177 settings" — harness/check_settings_doc.py counts 177 keys today, the design's 161 is from 2026-09-11 — cost if wrong: none, the number is measured.
Task 7: complete (commits b1c0597..18f9516, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 1.07s)
Task 8: Ruling: a root .dockerignore allowlists pyproject.toml, README.md, gnome3d and web, without node_modules, dist and web/design — the repository has no .dockerignore and its data/ holds multi GB maps — cost if wrong: a file the image needs left out, caught by the build.
Task 8: Ruling: the dev proxy passes /api, /jobs and /jobs/..., /p/ and /model to FastAPI; /jobs as the My jobs client route is proxied in development until part 3 — cost if wrong: My jobs in npm run dev shows the server's 404 until part 3.
Task 8: verified on team-frodo: image built, dist in /app/web/frontend/dist, /method returns the app, /model, /api/catalog and the JS asset 200.
Task 8: complete (commits 18f9516..7b326b8, tests: bash -c 'cd web/frontend && npm test > /dev/null && npm run build > /dev/null && cd ../tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo frontend and web tests pass' → frontend and web tests pass)
Final: fixed stylesheet split broke the Method layout and hid equations in dark mode — css.test.ts brace balance RED→GREEN, suite 47/47, screenshots of all five pages checked in headless Chrome.
Final review: self-review, no subagent — the user asked for no subagents (cost), memory feedback_native_execution.
Final: Ruling: the plan workspace .superpowers/sdd/2026-11-12-web-frontend-1-foundation is kept, not deleted — the user does not want files deleted — cost if wrong: a gitignored folder of briefs and this ledger.
Final: minor (deferred): My jobs (/jobs) renders an empty page until part 3 builds it.
Final: minor (deferred): the Vite template's demo files (README.md, App.css, index.css, assets/react.svg, public/vite.svg) are untracked in web/frontend and public/vite.svg reaches dist.
Final: minor (deferred): the dark theme was derived, not designed; the user has not seen it.
