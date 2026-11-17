# RNAPII figures, redesigned: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two main figures, one per point the work makes, and a supplement, from the chr1 readouts and models of the RNAPII comparison.

**Architecture:** Three compute modules, each a library with a CLI that writes one small table or npz, and one drawing module that reads only those files and the readout JSON. The compute that needs models runs on the workstation, where the models are; the drawing runs anywhere.

**Tech Stack:** Python 3, numpy, pandas, scipy, matplotlib. Tests are plain scripts under `harness/`, run as `.venv/bin/python harness/test_<name>.py`, in the style of `harness/test_figure_rnapii.py`.

**Spec:** `design/2026-11-12-rnapii-figures.md`; numbers and arms from `design/2026-11-11-rnapii-release.md`, Results.

## Global Constraints

- No abbreviations of our own making on any figure text. Arm names: "original models (cudaMMC)", "gnome3d, CTCF loops", "gnome3d, CTCF and RNAPII loops", supplement only "gnome3d, RNAPII ends merged into CTCF anchors".
- Every axis reads up or right as better; the expression link is plotted with its sign flipped and the axis says so.
- Distributions as joy plots; few single values as dumbbells or paired lines, never bars.
- One colour per arm everywhere: original grey, CTCF blue, CTCF and RNAPII orange, merged green. Identity never by colour alone.
- HFFc6 labelled as having its RNAPII loops from BJ.
- Panel d pairs: each element within 2.5 kb of one end of one loop (`annotations.annotate_pairs` rule); promoter is the TSS plus or minus 1 kb; enhancers from the EnhancerAtlas set the expression readout used.
- Panel e: Spearman of `total` against `log2fc`, block bootstrap over `cluster`, 2,000 draws, as `enhancer3d.stats.boot_r`; the point estimate must reproduce `r_total` of `summary.tsv`.
- Code style of AGENTS.md: annotated signatures, short plain doc strings. Nothing deleted, the old `figure_rnapii.py` stays.

## Review Focus

1. A line with no or few RNAPII-only enhancer to promoter pairs on chr1: panel d must show the count and points, not an empty or misleading ridge.
2. A figure text that slips an arm letter, "SCC", "APA", "jump" or a column name onto the page: a test reads every text artist.
3. GM12878 has no Micro-C: its row in c and f says so instead of dropping silently or plotting zero.
4. Bootstrap draws that do not reproduce the published point estimate (wrong column, wrong correlation, NaN log2fc rows): refused, not plotted.
5. A model directory whose beads do not cover the chosen window of b or the locus of h: refused with the window named.

---

### Task 1: Readout tables and bootstrap draws

**Files:**
- Create: `playground/rnapii/fig_data.py`
- Test: `harness/test_fig_data.py`

**Interfaces:**
- Produces:
  - `ARM_NAME: dict[str, str]` letter to spelled out name, `ARM_COLOUR: dict[str, str]`.
  - `agreement(readouts: Path) -> pd.DataFrame` columns `line, arm, measure, value`, measure in `{"hic", "microc"}`, from `<LINE>_battery.json` (`scc`) and `<LINE>_microc.json` (`r_fine`); arm letter from the `arm` field's `<LINE>_<A>_60mb`; the `map` rows dropped. A line with no Micro-C file has no microc rows.
  - `trio_paired(trio_dir: Path) -> pd.DataFrame` columns `person, arm, measure, value`, measure `loop_pileup` (model `apa` over the `map` row's `apa`) and `hic` (`scc`).
  - `boot_draws(genes_tsv: Path, n: int = 2000, seed: int = 1) -> tuple[float, np.ndarray]` Spearman of `total` against `log2fc` over rows with both finite, and the draws, resampling `cluster` blocks.
  - `link_draws(e3d_root: Path, arms: tuple[str, ...]) -> pd.DataFrame` columns `pair, arm, draw, value` with value sign flipped, reading `e3d_<A>/genes_<a>_to_<b>.tsv`; raises `ValueError` naming the file when the point estimate differs from `summary.tsv`'s `r_total` by more than 1e-6.

- [ ] **Step 1: Write the failing tests** `test_agreement_reads_lines`, `test_agreement_no_microc_line` (GM12878 fixture with battery only gives no microc rows), `test_trio_relative_pileup` (map apa 2.0, model 3.0 gives 1.5), `test_boot_reproduces_point` (hand built table of 40 genes in 8 clusters: returned point equals `scipy.stats.spearmanr`), `test_link_draws_refuses_mismatch` (summary with a wrong `r_total` raises ValueError naming the file), `test_link_draws_sign_flipped` (values are minus the draws).
- [ ] **Step 2: Run** `.venv/bin/python harness/test_fig_data.py` — Expected: fails on the import of `fig_data`.
- [ ] **Step 3: Implement** the functions in `playground/rnapii/fig_data.py`.
- [ ] **Step 4: Run** the test — Expected: all pass. Then on the real readouts: `.venv/bin/python -c "import sys; sys.path.insert(0,'playground/rnapii'); from fig_data import link_draws; from pathlib import Path; print(link_draws(Path('design/figures/rnapii-2026-11-12/readouts'), ('C','R')).groupby(['pair','arm']).value.median())"` — Expected: no ValueError; medians near 0.26 to 0.33. If the point estimate does not reproduce, read `enhancer3d/decomposition.py` around line 53 on the workstation for the rows it keeps, and rule.
- [ ] **Step 5: Commit** `rnapii figures: readout tables and bootstrap draws for the redesign`.

### Task 2: Enhancer to promoter distances held by RNAPII loops (panel d)

**Files:**
- Create: `playground/rnapii/ep_pairs.py`
- Test: `harness/test_ep_pairs.py`

**Interfaces:**
- Produces:
  - `loop_pairs(enh: pd.DataFrame, tss: pd.DataFrame, loops: pd.DataFrame, pad: int = 2500) -> pd.DataFrame` rows `gene_id, tss, enh_mid, sep` for every enhancer to promoter pair joined directly by a loop: promoter interval TSS±1 kb and enhancer interval each within `pad` of opposite ends of one loop. `loops` columns `chrom, s1, e1, s2, e2`.
  - `rnapii_only(enh, tss, rnapii, ctcf, pad=2500) -> pd.DataFrame` pairs of `loop_pairs` on RNAPII loops minus those on CTCF loops.
  - `matched_control(pairs: pd.DataFrame, enh, tss, held: pd.DataFrame, rng: np.random.Generator, per_pair: int = 5) -> pd.DataFrame` same columns, pairs joined by no loop of either factor, separation within 10 percent of each RNAPII-only pair's.
  - `pair_distance(model_dir: Path, chrom: str, pairs: pd.DataFrame, max_gap: int = 5000) -> np.ndarray` mean over the `*.beads.tsv` of `model_dir/<chrom>` of the 3D distance between the beads nearest the TSS and the enhancer midpoint; NaN where either is farther than `max_gap` from every bead.
  - CLI: `python playground/rnapii/ep_pairs.py <line> <models_root> <loops_dir> <atlas.bed> <genes.tsv> <out.tsv>` writes `line, set, arm, gene_id, sep, distance` with set in `{"rnapii_only", "control"}` and arm in `{C, R}`; prints the pair count per set.
- [ ] **Step 1: Write the failing tests** `test_loop_pair_found` (one loop, enhancer and promoter on opposite ends, 2 kb off), `test_pad_respected` (3 kb off is not a pair), `test_ctcf_pairs_removed`, `test_control_matches_separation`, `test_pair_distance_mean_and_gap` (two hand built beads files: a straight chain where the distance is known, mean over both; an element 6 kb from every bead gives NaN).
- [ ] **Step 2: Run** `.venv/bin/python harness/test_ep_pairs.py` — Expected: import fails.
- [ ] **Step 3: Implement.** CTCF loops are the run's `data/<LINE>/<LINE>_xc_ctcf_clusters_3+.bedpe`, RNAPII `<LINE>_xc_rnapii_clusters_3+.bedpe`; the genes table is `readouts/genes_chr1.tsv` (find its TSS column), restricted to the 1,578 common genes of `genes_chr1_common.tsv`.
- [ ] **Step 4: Run** the test — Expected: pass. Then on the workstation, for each line, write `~/xc_readout/out/fig2/ep_<LINE>.tsv` and copy to `design/figures/rnapii-2026-11-12/fig_inputs/`. Expected: a printed count per line and set; record the counts in the ledger.
- [ ] **Step 5: Commit** the module, the test and the three tables.

### Task 3: Model contact maps and medoid conformation (panels b and h)

**Files:**
- Create: `playground/rnapii/model_maps.py`
- Test: `harness/test_model_maps.py`

**Interfaces:**
- Consumes: `validation.metrics.hic.simulated_contacts`, `validation.metrics.hic.observed_hic`, `playground/validation_battery.py`'s `load` and `CONTACT_BEADS`.
- Produces:
  - `ensemble_map(cifs: list[Path], start: int, end: int, binsize: int) -> np.ndarray` sum over conformations of `simulated_contacts` with the battery's contact radius; raises `ValueError` naming the window when the beads cover less than 90 percent of it.
  - `medoid(cifs: list[Path]) -> int` index of the conformation with the least summed RMSD to the others after Kabsch alignment on shared beads.
  - CLI: `python playground/rnapii/model_maps.py <line> <models_root> <mcool> <chr1:a-b> <out.npz>` saves `hic`, `orig`, `ctcf`, `rnapii` maps at 25 kb and `medoid_<arm>`.
- [ ] **Step 1: Write the failing tests** `test_map_symmetric_and_sums` (two hand built cifs), `test_window_not_covered_refused`, `test_medoid_picks_centre` (three conformations, one between the other two).
- [ ] **Step 2: Run** — Expected: import fails.
- [ ] **Step 3: Implement.** The window of b and the locus of h are chosen with `figure_rnapii.py --loci` on GM12878; the 5 Mb window contains the 1 Mb locus. Record both in the ledger.
- [ ] **Step 4: Run** the test, then the CLI on the workstation for GM12878, copy the npz to `fig_inputs/`. Expected: four maps of 200 bins.
- [ ] **Step 5: Commit.**

### Task 4: The two figures and the supplement

**Files:**
- Create: `playground/rnapii/figures_two.py`
- Test: `harness/test_figures_two.py`

**Interfaces:**
- Consumes: Task 1 tables, Task 2 `ep_<LINE>.tsv`, Task 3 npz, `gnome3d.geometry.smooth_chain`, `web/gnome3d_web/viewer.py`'s element box rule.
- Produces:
  - `joy(ax, groups: list[tuple[str, dict[str, np.ndarray]]], xlabel: str, min_n: int = 15) -> None` one row per group, one filled ridge per arm (Gaussian kernel density), a group with fewer than `min_n` values per arm drawn as points with its count.
  - `dumbbells(ax, df: pd.DataFrame, a: str, b: str, measure: str) -> None`; a line without the measure gets the text "no Micro-C" in its row.
  - `figure_one(inputs: Path, out: Path) -> Figure`, `figure_two(inputs: Path, out: Path) -> Figure`, `figure_supplement(inputs: Path, out: Path) -> Figure`, each saving png and pdf.
  - CLI: `python playground/rnapii/figures_two.py <inputs_dir> <readouts_dir> <out_dir>`.
- [ ] **Step 1: Write the failing tests** `test_no_own_abbreviations` (every text artist of the three figures drawn from fixture inputs is checked against `\b(E|C|R|M)\b`, `SCC`, `APA`, `jump`, `r_total`, `r_fine`, `_60mb`; the established names Hi-C, Micro-C, CTCF, RNAPII stay allowed), `test_joy_few_points` (a group of 5 values draws points and the text "n = 5"), `test_no_microc_row_says_so`, `test_link_axis_flipped` (axis label says stronger to the right), `test_bj_label`.
- [ ] **Step 2: Run** — Expected: import fails.
- [ ] **Step 3: Implement** the panels of the spec: Figure 1 a schematic drawn with matplotlib patches, b three triangle maps on one colour scale, c dumbbells original to CTCF, d joy plot; Figure 2 e joy plot of the draws, f dumbbells CTCF to CTCF and RNAPII on c's axes, g paired lines per person, h two smooth chain renders with element boxes; supplement merged anchors on c, e, f, and the relabelled grid by calling `figure_rnapii.figure_grid` with the spelled out names.
- [ ] **Step 4: Run** the test, then the CLI into `design/figures/rnapii-2026-11-12/figures_redesign/`. Expected: six files; look at each png and fix overlaps before committing.
- [ ] **Step 5: Commit** the module, the test and the figures; add the figure paths to the spec's Results.
