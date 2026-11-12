# RNAPII comparison, website and library release, design of 2026-11-11

## Intent

Work on the 3D engine closes with three deliverables, after which the team decides whether
there is a paper.

1. A figure that answers whether adding RNAPII ChIA-PET loops to the CTCF loops brings the
   models closer to reality. Structure and expression are reported side by side with no single
   verdict, and the claim is worded to match whichever way the numbers fall.
2. A website for scientists. A user takes CTCF and RNAPII loops from ChIA-PET, applies
   structural variants, theirs or a hosted set, generates a contact background with
   HiCDiffusion when one is needed, and gets an ensemble of 3D structures of a region with
   enhancers and promoters marked.
3. `gnome3d-ng` on PyPI, the same library the website calls.

The reconciliation plan behind this names four claims to support: closing the engine,
RNAPII improving the structure or the signal, HiCDiffusion for genomes with no measured
contact map, and the engine taking more data modalities. Only the second needs new
measurement, and it is the one most at risk. On GM12878 so far RNAPII lowered agreement with
Hi-C, SCC 0.364 to 0.280 and MultiMM 0.682 to 0.587, and improved the expression test,
Spearman -0.178 to -0.234 (`design/rnapii-loops.md`). Those runs predate the summary arc fix
below.

## What exists

- This repository. The engine, the two factor loader (`[data] clusters` takes several files,
  `[data] factors` names them), the validation battery (`playground/validation_battery.py`,
  `playground/saddle_offline.py`), the GM12878 RNAPII prep (`playground/rnapii/prep_gm12878.py`,
  `anchor_union.py`) and the arm scripts (`slurm/ensemble/rnapii_arms.sh`,
  `rnapii_strength_arms.sh`).
- The summary arc fix, commit `75bedc1`, only on `origin/triosformer-expression`. Before it a
  pair joined by both a CTCF and an RNAPII loop was held at the background, 406 strong pairs
  on a trio chr1. Every two factor run made without it, the GM12878 sweep and the trio RNAPII
  arms included, is superseded.
- `~/Desktop/triosformer`. Nine individuals in three trios, per sample CTCF and RNAPII
  ChIA-PET, HGSVC Hi-C (4DN mcools, independent of the ChIA-PET), HGSVC RNA-seq, and
  `personal3d/`, which reads a person's SVs from HGSVC freeze 4, rewires loops by Sadowski's
  2019 rules, builds a haplotype's 2 Mb sequence window, runs HiCDiffusion on it, lifts the
  map back and writes the engine's inputs. HiCDiffusion weights are archived as
  `hicdiffusion.tar.zst` in `models/manifest.tsv`. Readout scripts
  `analysis/trio_arms_compare.py`, `analysis/trio_arms_jump.py`.
- `~/Desktop/enhancer3d_cross_cell_lines_analysis`. CTCF only cudaMMC models of GM12878, H1ESC
  and HFFc6, 100 per line and chromosome, on ARCH (`/data/mw868/models_fixed`), and the
  statistics E1 to E3 and the cross line jump statistic. No RNAPII anywhere.
- The old site, `~/Desktop/3dgnome_webserver` and `~/Desktop/bio/3dgnome_webserver`, Flask
  calling cudaMMC. Reference for what the user facing flow offered, not a code base to extend.
  The enhancer3D backend in `~/Desktop/enhancer3D` is not reused either.

## Part 1. The RNAPII comparison

### Inputs

Cell lines are the enhancer3d three.

| line | RNAPII ChIA-PET | loops, per replicate | peaks |
|---|---|---|---|
| GM12878 | ENCODE ENCSR905HWW | ENCFF040KUS, ENCFF913VWM | ENCFF109HCZ, ENCFF578JKO |
| H1ESC | ENCODE ENCSR782EKZ | ENCFF324UIT, ENCFF753NSM | ENCFF665IXY, ENCFF116RLS |
| HFFc6 | 4DN 4DNESI1WZ5HT when the portal answers, else ENCODE BJ ENCSR956IXE | BJ: ENCFF601NEC, ENCFF808QOQ, ENCFF775SCY | BJ: ENCFF428VXI, ENCFF084CXY, ENCFF953MBK |

GM12878 and H1 files are already under `data/<CELL>/raw/rnapii_encode/`. They are GRCh38,
seven column bedpe at three PETs or more, the pipeline's format, so no liftover. GM12878 moves
off the Tang 2015 set so that all lines share one assay and pipeline. BJ is a different
foreskin fibroblast line from the HFFc6 whose CTCF and Hi-C we use, and the figure says so
wherever BJ stands in. The 4DN portal failed its TLS handshake from every route on 2026-11-11;
retry before falling back.

CTCF loops are the inputs already under `data/<CELL>/`. The builder confirms they are the same
4DN CTCF ChIA-PET the enhancer3d models were built from and records the accessions. If they
are not, C and R still use the `data/<CELL>/` files, which production runs on, and the
mismatch is stated beside every comparison with E (my call).

### Depth

Replicates are pooled per line and factor. The whole rule of
`~/Desktop/triosformer/pipeline/trio_resample.py` then applies, per factor (my call): span at
most a recorded cap (a parameter of the prep, 1 Mb to start, the trio arms' value, and open to
10 Mb; production's `max_pet_length` of 1 Mb already keeps longer loops out of the springs and
folds them into the segment heatmap, so the cap mainly sets which PETs count toward depth),
anchors on the line's peaks (both ends for CTCF, at least one for RNAPII), and
binomial thinning of every loop's PET count to the lowest line's total for that factor, CTCF
matched across lines and RNAPII matched across lines, never one total for both. Loops under
three PETs after thinning are dropped. One seed per line and factor, recorded. RNAPII peaks
are the files in the table; CTCF peaks come from each line's ENCODE CTCF ChIP-seq, the builder
picks and records the accessions.

The available files hold clusters at three PETs or more, not the libraries' one PET and up
clusters `trio_resample.py` measures depth on. Depth here is the PET total over loops passing
the filters (my call); if the one PET clusters are fetchable from ENCODE or 4DN without much
work, use them instead and say which was used. The gnome3d CTCF arm therefore no longer runs
on exactly the original enhancer3d inputs, which is accepted.

Pooling two replicates' loop calls: loops whose two anchors both overlap are one loop with the
PET counts summed and the anchors' union as its interval, the rest are kept as they are.
Builder's call to refine if a better rule shows up in the files.

### Engine

A branch off `origin/triosformer-expression`, never `main` itself. That branch is `main` plus
the summary arc fix `75bedc1` and `loop_contact` (`e1e11d9`, opt in and left off). Production settings
from `validation/core/config.py`, RNAPII at full strength, `[springs] factor_strength = 1,1`.
Anchors are the union by the rule of `anchor_union.py`.

### Arms

Per cell line.

| arm | what it is |
|---|---|
| E | the existing cudaMMC enhancer3d models, unchanged, the published baseline |
| C | gnome3d, CTCF only |
| R | gnome3d, CTCF and RNAPII |
| M | as R, with every RNAPII loop end that lies within one bead's genomic span (`target_bp_per_subanchor`) of a CTCF anchor absorbed into that anchor, so only RNAPII ends away from CTCF become new beads. Tests whether the Hi-C cost of R is the dense 1 kb anchor set, as the GM12878 sweep suggested |

Trios, on each person's HGSVC Hi-C background: C and R. C on that background is new; the
existing `trio_rnapol2_hgsvc` predates the fix and is rerun as R. M is not run on the trios
(my call, to bound cost; add it if M wins on the cell lines).

Ten conformations per arm. E is subsampled to ten models per line and chromosome with a
recorded seed, so every arm enters the enhancer3d statistics at the same ensemble size; E at
its full 100 is shown as a reference line only. chr1 first for every arm, then the 20
autosomes the enhancer3d report used for the final figure.

What chr1 decides (my call). C and R go to the autosomes whatever chr1 shows, since the figure
needs both. M goes to the autosomes, and to the trios, only if on the cell lines it recovers at
least half of R's structural loss (SCC and MultiMM, measured from C) while keeping at least
half of R's expression gain (partial or E3, measured from C). Otherwise M is reported on chr1
alone.

Jobs on eden and the workstation are launched by the user. The builder writes the configs,
the prep scripts and the commands.

### Readouts

Structure, for C, R and M, and for E where its models convert.

- The validation battery and saddle: Pearson, SCC, MultiMM, distance exponent, Rg, overlaps,
  saddle strength, compartment eigenvector correlation.
- Target maps. Cell lines against their deep 4DN Hi-C (`validation/manifests/<CELL>_hic.json`).
  Production singletons are thinned from the same maps, so this score is shared by every arm
  and fair between them but not independent; the figure states it. Trios against each
  person's HGSVC Hi-C.
- Pile-up at RNAPII loops. Aggregate contact enrichment over the RNAPII loop pixels, in the
  target map and in each arm's simulated contact map, at the loop and against a local
  background. This is the readout that asks directly whether RNAPII loops are realised.

Fine scale, added 2026-11-12. "Closer to reality" depends on the yardstick, and bulk Hi-C at
25 kb rewards the CTCF and domain scaffold while RNAPII acts under 100 kb, through contacts
ligation captures weakly. Micro-C resolves that range. For H1 and HFFc6, 4DN Micro-C
(4DNFI2TK7L2F, 4DNFI18Q799K, mirrored on UCSC) is cut to chr1:0-60 Mb at 5 kb by
`playground/rnapii/hic_slice.py`, and `microc_readout.py` gives each arm the pile-up at RNAPII
loops on that grid and `r_fine`, the agreement of distance normalised contacts 20 kb to 1 Mb
apart in 2 Mb windows. Each claim names its yardstick: the scaffold by bulk Hi-C, the fine
contacts by Micro-C, function by expression.

Expression.

- Cell lines: E1, E2, E3 and the cross line jump statistic, by the enhancer3d analysis code,
  on every arm. That code reads cudaMMC models on a uniform 1 kb chain; gnome3d beads are
  irregular. Genes and enhancers map to their nearest bead and are kept only when that bead is
  within 5 kb (my call, the rule that fixed the earlier coverage artefact where 22 to 25
  percent of genes sat more than 5 kb from any bead). The same rule is applied to E.
- Trios: raw and partial correlation of enhancer distance with expression, family separation,
  the jump statistic per contrast, by `trio_arms_compare.py` and `trio_arms_jump.py`.

## Part 2. The figure

Main figure, story panels.

- a. The inputs and the models: CTCF and RNAPII loops over one locus, a model snapshot per arm.
- b. One locus: target Hi-C, C model contact map, R model contact map, RNAPII loops marked.
  The locus is chosen where RNAPII loops are dense and the Hi-C shows them; the builder
  proposes candidates.
- c. Pile-ups at RNAPII loops, target map against each arm.
- d. Structure battery, paired dots per line and per person, C to R to M.
- e. Expression statistics, paired dots the same way, E included where it applies.
- f. Trade-off: change in structural agreement against change in expression signal, one point
  per line or person per RNAPII arm.

Supplementary, a metric grid: rows are the lines and the nine people, columns are the
metrics, each cell shows the arm comparison and which arm wins.

## Part 3. The website

### Who and what

Scientists. As in the original 3D-GNOME paper, extended.

1. Cell type, always chosen by the user. It selects the base: that cell type's panel of loops
   and its deep contact map. The hosted showcase is the three cell lines. The nine trio
   individuals are not on the site until their release is cleared.
2. Inputs. The cell type's panel loops, or CTCF and RNAPII loops the user uploads.
3. Structural variants. Uploaded VCF or a hosted call set, applied by `personal3d`'s rules to
   the loops of step 2.
4. Contact background. The cell type's deep map, with a HiCDiffusion window on the SV altered
   sequence spliced in where the SVs change it, as `personal3d/engine.py` does. Choosing HiCDiffusion for anything other than a lymphoblastoid line shows a
   visible notice that the model was trained on GM12878 and is out of domain there; the result
   page repeats it. The site never presents HiCDiffusion output as carrying individual specific
   signal, and cites the triosformer and enhancer3d reports for why.
5. Region by gene name or coordinates.
6. Results: an NGL viewer with enhancers and promoters marked (enhancer atlas, GENCODE),
   enhancer to promoter distances, downloads of CIF and bead TSV.

Datasets grow mostly by new SV call sets and their individuals, many of them. Adding one must
be dropping files in and adding a catalogue line.

### Stack, keep it simple

- One Python web app, FastAPI or Flask, serving the form, job status and results.
- One job queue with one GPU worker per card, `rq` on Redis or a Postgres backed queue. A job
  runs plain function calls: SV rewrite, optional HiCDiffusion window, gnome3d.
- SVs stored as bgzipped, tabix indexed VCFs, per individual or per cohort, on disk or any
  object store with the same layout, read by region. A catalogue file lists the call sets. No
  database for SVs.
- A results cache on disk keyed by a hash of individual, region, inputs and engine settings.
  The showcase (three lines, the enhancer3d regions) is precomputed into it. Hosted inputs
  share the cache.
- Uploads are private. An uploaded VCF or loop file and everything computed from it are
  reachable only through the job's secret link, never enter the shared cache, and are deleted
  after a fixed retention period the builder sets and states on the form.
- Built on request, not precomputed for every individual. Whether a request is a live wait or
  a queued job with a notice is decided by one measurement made first: the wall of a 2 to 3 Mb
  region end to end on the workstation GPU, SV rewrite, HiCDiffusion window and a ten
  conformation gnome3d ensemble.

Not reused: the enhancer3D backend (Temporal, MongoDB, Spark), judged overengineered. Temporal
can come back only if retries or long workflows become a real problem.

## Part 4. The library

`gnome3d-ng` published to PyPI with the JAX backend as the optional `[jax]` extra. The site
calls the same public entry point the package exposes, `simulate.run_region`, so the release
and the site cannot drift. Builder's call on versioning and what else enters the public API.

## Order

1. Inputs: RNAPII prep for three lines, depth matching, anchor sets, the M anchor rule.
2. chr1 runs, all arms, cell lines and trios.
3. Readouts on chr1. Read them before spending the autosomes.
4. Autosome runs and readouts.
5. Figure.
6. In parallel from step 2: the timing measurement, then the site and the PyPI release; the
   site goes live on the released package.

## Non goals and anti goals

- Not claiming the models are closer to reality if the structure readouts say otherwise.
- Not presenting HiCDiffusion as individual specific.
- No new engine physics. Engine work is closed; the summary arc fix is the one change taken.
- No work on `main`; the user opens the PR. No jobs launched by the builder on shared machines.

## Open

- The reconciliation plan says 13 cell lines and 4 cell types. The count found is 3 lines and
  9 individuals, 4 lines with BJ. To be settled before any text states a count.
- HFFc6 RNAPII from 4DN against BJ, settled by whether the portal answers.
- Where the site is hosted and whether it needs a login beyond the secret job links.
- Release of the trio data, which decides when the nine individuals join the showcase.

## Runs, the commands

Nothing below has been submitted. The branch is `rnapii-crosscell`; it is not pushed.

Cell lines, on eden.

```bash
# laptop: publish the branch and copy the inputs, which are gitignored
git -C ~/Desktop/3dgnome-rnapii push -u origin rnapii-crosscell
for C in GM12878 H1ESC HFFC6; do
  rsync -av ~/Desktop/3dgnome-torch/data/$C/${C}_xc_* eden:/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/data/$C/
done
# eden, in the checkout
git fetch origin && git switch rnapii-crosscell
for i in $(seq 0 8); do DRY=1 SLURM_ARRAY_TASK_ID=$i bash slurm/ensemble/crosscell_arms.sh; done
sbatch --array=0-8 slurm/ensemble/crosscell_arms.sh
```

The array's guard refuses a task whose config names a file missing under `data/<line>/`,
the Hi-C singletons included.

Trios, from the triosformer checkout for the configs and the engine checkout on this branch for
the runs.

```bash
python pipeline/trio_configs.py --factor CTCF --singletons hgsvc
python pipeline/trio_configs.py --factor RNAPOL2 --singletons hgsvc
CONFIG_TAG=_trio_hgsvc OUT=out/xc_trio_ctcf_hgsvc CHROMS=chr1 \
  sbatch --array=0-8 /path/to/triosformer/slurm/trio_ensemble.sh
CONFIG_TAG=_trio_rnapol2_hgsvc OUT=out/xc_trio_rnapol2_hgsvc CHROMS=chr1 \
  sbatch --array=0-8 /path/to/triosformer/slurm/trio_ensemble.sh
```

Arm E is the existing cudaMMC models of each line's chr1 on ARCH, `/data/mw868/models_fixed/<line>/chr1`.

Arm M at the spec's absorb distance of 1 kb moves few ends (GM12878 66,608 new RNAPII anchors
against 69,867 for R). Decide the distance before submitting tasks 2, 5 and 8; the prep takes
`--absorb-bp`.

## Readouts, the commands

Run after the arrays finish, on the machine holding `out/xc` and the deep maps. `<M>` is the
line's deep mcool, `<T>` its compartment track, `L` the line.

```bash
# arm E: a seeded ten of the cudaMMC .hcm models (faramir models_ccd/<line>/chr1), as gnome3d cifs
python playground/rnapii/e3d_export.py --hcm-to-cif <E hcm dir>/$L/chr1 out/xc/${L}_E/chr1 --n 10
python playground/slice_arm.py out/xc/${L}_E/chr1 out/xc/${L}_E_60mb chr1:1-60000000

# structure and pile-up, per line; H1ESC's mcool has no weights, so its saddle takes --balance no
python playground/validation_battery.py --balance no --ev-factor 0.7 --singletons data/$L/${L}_hic_25kb_singletons.bedpe \
  --json out/xc/${L}_battery.json <M> chr1:1-60000000 25000 out/xc/${L}_{E,C,R,M}_60mb
python playground/saddle_offline.py [--balance no] --json out/xc/${L}_saddle.json <M> chr1:1-60000000 100000 <T> out/xc/${L}_{E,C,R,M}_60mb
python playground/rnapii/pileup.py --json out/xc/${L}_pileup.json <M> chr1:1-60000000 10000 \
  data/$L/${L}_xc_rnapii_clusters_3+.bedpe out/xc/${L}_{E,C,R,M}_60mb
python playground/rnapii/figure_rnapii.py --loci <M> chr1:1-60000000 data/$L/${L}_xc_rnapii_clusters_3+.bedpe \
  out/xc/${L}_C_60mb out/xc/${L}_R_60mb

# expression: export every arm, E included, to the enhancer3d layout, then their CLI with --n-models 10
for A in E C R M; do python playground/rnapii/e3d_export.py out/xc/${L}_$A/chr1 out/xc/e3d/${L}_$A chr1; done
```

The trio structure readouts are the same battery, saddle and pile-up per person, against the
person's HGSVC mcool, with that person's `_rnapol2_clusters_3+.bedpe` as the pile-up loops.
The trio arrays write `<OUT>/<SAMPLE>/chr1`, so both arms' directories are named `chr1`; slice
them to uniquely named directories first, `slice_arm.py <OUT>/<S>/chr1 out/xc/trio/<S>_C_60mb`
and `<S>_R_60mb`, since the figure refuses two rows for one dataset, arm and metric. The trio
expression statistics come from `analysis/trio_arms_compare.py` and `trio_arms_jump.py`.

Each expression table is reduced to a CSV of arm, metric, value and n, metric one of `partial`,
`rho_3d`, `e1`, `e3`, `jump_rho`, `family_sep`, and listed in a manifest with the JSON files
above; then `python playground/rnapii/figure_rnapii.py manifest.json out/xc/figures`. The
manifest format is in that script's docstring.

## Results, 2026-11-12

chr1, the battery and pile-ups on chr1:1-60 Mb, ten conformations per arm. Figures and every
readout table in `design/figures/rnapii-2026-11-12/` (`figures/rnapii_main`, `figures/rnapii_grid`,
`readouts/`). Each claim names its yardstick.

**Large scale structure, bulk Hi-C at 25 kb.** SCC, E C R M:

| line | E | C | R | M | APA at RNAPII loops, map / E / C / R / M |
|---|---|---|---|---|---|
| GM12878 | 0.219 | 0.373 | 0.288 | 0.373 | 1.46 / 7.06 / 1.18 / 1.29 / 1.37 |
| H1 | 0.145 | 0.269 | 0.194 | 0.221 | 1.32 / 3.22 / 1.43 / 1.24 / 1.59 |
| HFFc6, RNAPII from BJ | 0.146 | 0.331 | 0.232 | 0.295 | 1.56 / 6.22 / 1.41 / 1.75 / 2.94 |

The new engine agrees with Hi-C better than the original on every line. RNAPII as is costs
SCC on every line; merging its ends into CTCF anchors within 25 kb recovers all of it on
GM12878 and part on the others. The singletons the arms are built from are thinned from the
same maps, which favours the gnome3d arms over E.

**Fine scale, Micro-C at 5 kb.** `r_fine` E C R M: H1 0.111, 0.121, 0.101, 0.111; HFFc6 0.109,
0.142, 0.083, 0.122. APA at RNAPII loops, map 1.79 and 2.68: H1 3.35, 1.31, 0.96, 1.18; HFFc6
12.84, 1.55, 1.53, 2.30. RNAPII does not improve fine scale agreement; every model's is weak.

**Function, cross line enhancer to promoter distance against expression** (enhancer3d CLI,
1,578 genes covered in all twelve ensembles; more negative is a stronger link). jump statistic
E C R M: GM12878 vs H1 −0.172, −0.260, −0.326, −0.298; H1 vs HFFc6 −0.126, −0.131, −0.242,
−0.217; GM12878 vs HFFc6 −0.289, −0.285, −0.337, −0.293. RNAPII strengthens the link on every
pair and statistic; M keeps most of it. Bootstrap intervals are in `readouts/e3d_*/summary.tsv`
and have not been read against each other yet.

**Trios, each person against their own HGSVC Hi-C** (`playground/rnapii/trio_summary.py`):
R minus C, median over nine and in how many R is higher: SCC −0.007 (2 of 9), Pearson −0.027
(0 of 9), MultiMM −0.026 (0 of 9), saddle −0.096 (2 of 9), APA at RNAPII loops over the map's
+0.169 (9 of 9). RNAPII brings the contacts at its loops closer to the person's own Hi-C in all
nine at a small cost to global agreement.

**Reading.** RNAPII makes the models more predictive of expression and puts more contacts
where its loops are; it does not make them agree better with contact maps globally, at 25 kb or
at 5 kb. The claim that holds is "RNAPII loops carry functional information the CTCF model
lacks, and can be added at a small structural cost, which merged anchors reduce", not
"closer to Hi-C". Open: the bootstrap intervals of C against R, the trio expression statistics
on the new models, chromosomes beyond chr1.
