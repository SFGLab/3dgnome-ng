# SDD ledger — plan: design/plans/2026-11-12-web-frontend-6-review-fixes.md
Spec: the user review of part 5 on 2026-11-12
Pre-flight: no shared interfaces beyond App.tsx routes, touched by Tasks 2 and 3 in turn.
Task 1: complete (commits b559a8e..69bc731, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 3: Ruling: the browser's list keeps 50 jobs instead of 10 — My jobs is now a page of its own — cost if wrong: a longer localStorage entry
Task 3: complete (commits ee15bfc..1b8d47e, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 2: Ruling: an example asks for 10 structures — the first visitor runs it, about a minute, and the cache serves everyone after — cost if wrong: a smaller ensemble for the examples
Task 2: Ruling: the Model page side panel keeps filling the form from an example and lists the two new examples too — it is labelled Start from an example — cost if wrong: two more cards there
Task 2: complete (commits 94b67d5..2b5605b, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 7: complete out of order (commit 79ddbe4) — the user hit its 500 on frodo; tests: ContactMapTab 7/7, test_contacts 12/12, suite 185/185; the request that failed now answers 400 on frodo
Task 7: Ruling: one square map split by the diagonal, as before, scaled to the panel, in place of the plan's two maps side by side — the design draws one map — cost if wrong: a layout change
Task 7 follow up: the expected view and the map's alignment, user report 2026-11-13 — heatmap.test 7/7 RED→GREEN for the three new cases; I overwrote the committed heatmap.test.ts once without reading it, recovered it from git and merged its cases back
Task 4: Ruling: bar weights inputs 5, HiCDiffusion 10 when asked, modelling 85 or 75, marks 10 — modelling is most of a job's wall time — cost if wrong: the bar's pace
Task 4: Ruling: logs shown on the site have the job's directory cut from their paths — the frodo run showed server paths in the log — cost if wrong: none
Task 4: verified on team-frodo: a 10 structure job showed 2, 5, 7 and 8 of 10 written over a minute, stages and log tail live
Task 4: complete (commits 79ddbe4..fea01f3, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 5: complete — chain.test 4/4, suite 192/192; checked on team-frodo by rendering with SwiftShader and hovering; the user's report of the hover jumping was the hover mark catching the pointer, fixed by giving the marks no raycast
Task 5: complete (commits fea01f3..af8cc2c, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 9: complete — test_contacts 21/21, ContactMapTab 9/9, heatmap 8/8, suite 195/195; frodo: SCC 0.50 (10 structures) and 0.40 (1 structure) against 4DNFIXP4QG5B, was -0.11 and -0.17
Task 9: Ruling: the deep map is used only where the job read the cell type's hosted map, not with own contacts or a predicted background — it stands for that map — cost if wrong: own data jobs compare with their own contacts
Task 9: Ruling: frodo's catalog was copied to catalog.json.before-deep-maps before add_deep_maps.py rewrote it — never delete — cost if wrong: none
Task 9: note: my busy check before a restart matched the stage name queued in the new stage list; the check now reads the state field
Task 6: Ruling: the loops, genes and variants tables are paged in the browser and the loops endpoint keeps returning every loop — a 20 Mb region holds a few thousand loops, a small answer — cost if wrong: a server page parameter later
Task 6: Ruling: My jobs keeps at most 50 jobs, one page, Compare at most 20 rows, and the Method, Cite and Help tables are fixed text, so they get no pager — cost if wrong: none
Task 6: complete (commits 397e176..ab94a44, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Task 7: complete (commits 2b5605b..ab94a44, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
Final review: self-review (native execution by the user's rule)
Final: fixed every zoom of the contact map re-read the remote deep map — test_contacts 'a deep map is read once per resolution over the job's region' RED→GREEN, suite green
Final: minor (deferred): a failed deep map read is caught broadly and falls back to the job's contacts with a label; the cause is not logged
Final: minor (deferred): the deep map read is not counted against LINK_READS; the first view of each resolution of each job starts one read
Task 10: complete — test_contacts 26/26, test_explore 28/28, frontend 205/205; the old check that the agreement does not follow the halves was the earlier contract, replaced at the user's request
Task 10: user question on CYB561D1 'no loop' and 0 kb: the promoter is a hub of 21 loop ends and none reaches the enhancer 352 bp away; the column now reads none, the promoter anchors N loops, and spans under 1 kb show in bp
Task 8: verified on team-frodo through the page: every example opened its job from the Examples page, two from the cache; My jobs listed them; the wizard, progress view, 3D hover, contact map views, loop halves and resolution switch checked with screenshots; no console errors
Final: Ruling: the agreement stays on raw values for now; on TNFRSF8 raw 0.47 against 0.18 with both halves balanced, shared coverage inflating the raw number — the user parked the choice with the methodology work — cost if wrong: raw agreements read high where bead density follows coverage
Final: minor (deferred): the deep map read of a new resolution shows a note but cannot be cancelled
Task 8: complete (commits ab94a44..6e02171, tests: bash .superpowers/sdd/2026-11-12-web-frontend-6-review-fixes/alltests.sh → all pass)
