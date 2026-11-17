# SDD ledger — plan: design/plans/2026-11-12-web-frontend-3-job-explorer.md
Setup: native execution on rnapii-crosscell, no worktree.
Pre-flight: Tasks 4-9 consume the endpoints of Tasks 1-3 through jobApi(base); Task 6 consumes Task 5's StructureView picks; Task 7 sets the selected gene Task 6 reads. Consistent.
Task 1: Ruling: genes and enhancers come from marks.json, the job's own coordinates; genes.tsv has a TSS and no gene body, so hover can name the gene of a TSS within 5 kb and not "inside a gene" — cost if wrong: the design's "inside VPS13D" needs gene bodies in the site data.
Task 1: Ruling: the SD over structures uses n - 1, 0 for one structure — cost if wrong: none.
Task 1: note: LocalPool.position was written before its check in test_queue.py, which passed on first run; the check holds, its RED was not seen.
Task 1: complete (commits 469d94d..8ff00a3, tests: bash -c 'cd web/tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo all web tests pass' → all web tests pass)
Task 2: Ruling: resolutions offered 1, 2, 5, 10, 25, 50, 100, 250 and 500 kb, at most 800 bins on a side; the plan's 4 Mb fixture example became 2 kb over the 3 Mb job region, refused naming 5 kb — the fixture region is 3 Mb — cost if wrong: none.
Task 2: Ruling: the Hi-C half is the singletons file the job read, labelled "Hi-C of <cell type>", not the full 4DN map the design names — the site holds the thinned singletons, not the mcool — cost if wrong: a sparser upper half than the design shows until the site hosts the maps.
Task 2: Ruling: scipy pinned in web/requirements.txt at the image's 1.17.1 — the web app now imports it — cost if wrong: none.
Task 2: complete (commits 8ff00a3..9cfd49f, tests: bash -c 'cd web/tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo all web tests pass' → all web tests pass)
Task 3: Ruling: ratio is b over a, the second job's mean over the first's; the frontend words a ratio above 1 as "times closer" in the first job — cost if wrong: a flipped sentence, caught in Task 9's test.
Task 3: complete (commits 9cfd49f..0880a2c, tests: bash -c 'cd web/tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo all web tests pass' → all web tests pass)
Task 4: Ruling: the client route fallback serves /jobs/<key> and /p/<token> and refuses every deeper path under them; the private job check of test_app.py now asks the classic page and the summary API for a 404 — /jobs/<anything> is the app's page now and the data is what must stay private — cost if wrong: none.
Task 4: Ruling: the job name and Rename job live in this browser's recent jobs — the server keeps no names, as in part 2 — cost if wrong: a name not shared with the link.
Task 4: Ruling: the contact background fact reads "Hi-C, 4DN" for the hosted Hi-C as the design does, HiCDiffusion or none otherwise — cost if wrong: none.
Task 4: complete (commits 0880a2c..8eb6658, tests: bash -c 'cd web/frontend && npm test > /dev/null && npm run build > /dev/null && cd ../tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo frontend and web tests pass' → frontend and web tests pass)
Task 5: Ruling: the chain is a coloured screen space line of drei's Line, 4 px wide, not a tube mesh — a 20 Mb job is 80,000 chain points and a tube of them is millions of vertices — cost if wrong: a flatter look than the design's tubes.
Task 5: Ruling: hover names the gene whose TSS is within 5 kb of the bead ("at the TSS of X") — genes.tsv has no gene bodies (Task 1) — cost if wrong: the design's "inside VPS13D" needs gene bodies.
Task 5: Ruling: the job page loads lazily with three.js; the side panel's locus is the job's resolved region — cost if wrong: none.
Task 5: Ruling: web/scripts/serve_local.sh reuses an MPS daemon already running in its pipe directory and stops only one it started — a restart raced the old daemon's shutdown on frodo — cost if wrong: none.
Task 5: verified on team-frodo: the TNFRSF8 job's Structure tab in headless Chrome with SwiftShader WebGL, chain drawn and coloured, no console errors.
Task 5: complete (commits 8eb6658..cff8a76, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 2.96s)
Task 6: Ruling: selecting a gene measures its promoter and enhancer beads as A and B, the pair the design shows; a pick in Measure mode clears the gene — cost if wrong: none.
Task 6: Ruling: the gene card leaves out the strand and the PET count of the joining loop — genes.tsv strand is not carried into the job's marks, and the gene table names the factor without the count — cost if wrong: two facts short of the design.
Task 6: complete (commits cff8a76..a6eb9ea, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 3.12s)
Task 7: Ruling: the histogram card reads "3D distance, all genes" where the design writes "3d" — a typo in the design — cost if wrong: none.
Task 7: Ruling: clicking a gene row selects it and opens the Structure tab with its promoter and enhancer measured — cost if wrong: one click more to stay in the table.
Task 7: complete (commits a6eb9ea..dfa0dd8, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 3.57s)
Task 8: Ruling: "separate for each half" scales the lower half to the upper half's 99th percentile, and the colour range applies to the upper half's values; scroll zooms and double click centres, drag to pan is left out — cost if wrong: panning by the zoom buttons and the locus field only.
Task 8: Ruling: the hint reads "Scroll to zoom and double click to centre. Grey bins contain no bead." without the design's "drag to pan" — the build has no drag — cost if wrong: none.
Task 8: complete (commits dfa0dd8..70883bd, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 5.22s)
Task 9: Ruling: compare replaces the tab body while on; the second job shows its structure 1 and the A to B label of the selected gene from each job's gene table; "Same region" is a read only indicator from the compare endpoint — cost if wrong: no picking of the other job's structure or of a free A and B in compare.
Task 9: Ruling: the camera link lives in cameraLink.ts and syncs the OrbitControls of both scenes — cost if wrong: none.
Task 9: complete (commits 70883bd..409d56e, tests: bash -c 'cd web/frontend && npm test && npm run build' → ✓ built in 3.43s)
Task 10: verified on team-frodo through the site of serve_local.sh: the TNFRSF8 job (10 structures, CTCF and RNAPII) and a CTCF only job of the same region; every tab rendered in headless Chrome with no console error; a gene row click opened the Structure tab with the Distance A to B and Selected gene cards; compare filled 10 rows; at 390 px scrollWidth 390; distance endpoint 10 values; contacts 0.21 s at 25 kb over 2 Mb.
Task 10: found and fixed: the A and B labels of a selected gene read the TSS twice when the enhancer lay within 5 kb of the TSS — measure.test.tsx RED→GREEN, suite 128/128.
Task 10: complete (commits 409d56e..ff1c043, tests: bash -c 'cd web/frontend && npm test > /dev/null && npm run build > /dev/null && cd ../tests && for t in test_*.py; do /private/tmp/claude-502/-Users-nk2-Desktop-3dgnome-torch/96f73c5b-3da7-46ed-b9fb-010148ffb399/scratchpad/webvenv/bin/python $t > /dev/null 2>&1 || { echo FAIL $t; exit 1; }; done; echo frontend and web tests pass' → frontend and web tests pass)
Final review: self-review, no subagent — the user asked for no subagents (cost).
Final: minor (deferred): the Structure tab toolbar wraps to two lines at 1440 px beside the side cards.
Final: minor (deferred): the contact map has no drag to pan; zoom buttons, scroll and double click move the view.
Final: minor (deferred): the Hi-C half is the thinned singletons the job read, sparse next to the design's 4DN map; the agreement on a 10 structure job reads near 0.
Final: minor (deferred): "My jobs" in the navigation still has no page of its own.
Final: minor (deferred): the job page bundle is 880 kB with three.js, loaded only on job pages.
