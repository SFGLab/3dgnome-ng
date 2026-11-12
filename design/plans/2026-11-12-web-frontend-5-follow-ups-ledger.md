# SDD ledger — plan: design/plans/2026-11-12-web-frontend-5-follow-ups.md
Spec: the user review of part 4 on 2026-11-12, design/plans/2026-11-12-web-own-data.md, design/4dn-data-access.md
Pre-flight: Task 1 and Task 2 both change create_job and the own JSON; Task 2 reads the vcf upload of Task 1 unchanged. Task 4 changes remote.py read paths used by Task 6. Consistent.
Task 1: complete (commits c99ef3a..362cfad, tests: bash .superpowers/sdd/2026-11-12-web-frontend-5-follow-ups/alltests.sh → all pass)
Task 2: Ruling: a listed loop set on your own cell type also takes that cell type's anchors, config, segments and centromeres; uploaded contacts still replace its map — one base cell type per job keeps build_inputs unchanged — cost if wrong: the config of the chosen cell type applies
Task 2: Ruling: the variant check on your own cell type reads the borrowed loops through the dialog's cell, the Loops of choice, with no new server field — the route already takes a cell — cost if wrong: none
Task 2: Ruling: enhancers_from defaults to none on your own cell type — the plan's default — cost if wrong: one click
Task 2: complete (commits 362cfad..e0c19a3, tests: bash .superpowers/sdd/2026-11-12-web-frontend-5-follow-ups/alltests.sh → all pass)
Task 3: Ruling: the 3D view keeps its white canvas in both themes, as the design draws it — cost if wrong: one colour token
Task 3: complete (commits e0c19a3..1bd40aa, tests: bash .superpowers/sdd/2026-11-12-web-frontend-5-follow-ups/alltests.sh → all pass)
Task 4: Ruling: one connect time check replaces the separate address check of the plan's pinned resolver for aiohttp and the curl proxy: every reader goes through one range proxy whose connections resolve once and connect to the checked address — one mechanism for both readers — cost if wrong: none found
Task 4: Ruling: readers run in a spawned child process — hic-straw holds the GIL and deadlocked the in-process proxy; a child also lets the timeout stop the read — cost if wrong: about a second of start up per read
Task 4: Ruling: the proxy answers a redirect from the final address with 502 rather than pass it on — a passed redirect would let the reader connect unchecked; probe follows redirects before the read — cost if wrong: a server that redirects every range request fails
Task 4: complete (commits 1bd40aa..54a4322, tests: bash .superpowers/sdd/2026-11-12-web-frontend-5-follow-ups/alltests.sh → all pass)
Task 5: Ruling: 4DN files are fetched without an md5 check and without the RNA Pol II assay warning — the bucket listing carries neither, the portal that did is gone — cost if wrong: a truncated download goes unnoticed; the size in the table is the remaining check
Task 5: complete (commits 54a4322..f1b10ac, tests: .venv/bin/python harness/test_fetch_4dn.py → 7 passed, 0 failed)
Task 6: verified on team-frodo through the page: your own cell type with H1ESC CTCF loops (54 in region), contacts by link to 4DNFI1UEG1HD.hic in the 4DN bucket (1,240,534 read, 2,358 kept) and GM12878 enhancers, done in 24 s, side panel CTCF of H1ESC and your map, enhancers marked; a GM12878 job with an uploaded HGSVC VCF sent by upload id (job holds upload.vcf.gz, 4 variants), done in 22 s; the explorer in light and dark, no console errors.
Final review: self-review (native execution by the user's rule); no Critical or Important finding
Final: minor (deferred): the link summary's density names the cell type the dialog was opened on; on your own cell type with Loops of changed afterwards the job thins again at build to the chosen cell type's density, so the kept count in the summary can differ from what the job reads
Final: minor (deferred): a 4DN file has no md5 to check after download, only the size in the key table
