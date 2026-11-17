# RNAPII Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the inputs, configs, run scripts, readouts and figure that compare CTCF only against CTCF plus RNAPII models on GM12878, H1ESC and HFFc6 and on the nine trio individuals.

**Architecture:** A small prep library under `playground/rnapii/` turns ENCODE loop calls into depth matched loop files and anchor sets per line and arm. A config generator writes one ini per line and arm from `CANONICAL`. Runs are a SLURM array the user submits. Readouts reuse the validation battery and saddle, with a new pile-up script and an exporter into the enhancer3d layout, and one figure script reads their JSON.

**Tech Stack:** Python 3.11, numpy, scipy, cooler, matplotlib, the project's `.venv`. SLURM on eden. Unit checks are plain scripts under `harness/` run with `.venv/bin/python`.

**Spec:** `design/2026-11-11-rnapii-release.md`, Parts 1 and 2.

## Global Constraints

- Work on branch `rnapii-crosscell` created from `origin/triosformer-expression`. Never commit to or push `main`.
- Never submit, cancel or start jobs on eden or the workstation. Write the scripts and hand the user the commands.
- Production settings come from `validation.core.config.CANONICAL`. RNAPII arms set `[springs] factor_strength = 1,1`. `loop_contact` stays off.
- Factor names in configs are `CTCF,RNAPOL2`, the names the trio configs already use.
- Loops are seven column bedpe, `chrA startA endA chrB startB endB pets`, intra chromosomal only, end A before end B.
- Minimum PETs after thinning is 3. Span cap is a prep parameter, default 1,000,000 bp. Absorb distance for arm M is `CANONICAL` `target_bp_per_subanchor`, 1,000 bp.
- Ten conformations per arm. chr1 first. Battery on chr1:1-60000000 at 25 kb, as `slurm/ensemble/rnapii_arms.sh` does.
- Docstrings and comments follow `AGENTS.md`: short plain sentences, no dashes, colons or semicolons as sentence punctuation.
- Before any commit touching `gnome3d/` or `harness/`: `.venv/bin/ruff format`, `.venv/bin/ruff check`, and pyright must not increase its error count.

## Review Focus

1. ENCODE bedpe rows with end B before end A, or a float PET column such as `3.0`. Expect them normalised, never dropped. Test in Task 1.
2. A line and factor whose filtered depth is far below the others, so thinning would cut the others to almost nothing. Expect the prep to print every line's depth and keep fraction and to stop when any keep fraction falls under 0.2. Test in Task 2.
3. HFFc6 running on BJ RNAPII. Expect the record, the config comment and every figure label to say BJ. Test in Task 2 and Task 8.
4. An exported gnome3d model where most 1 kb grid positions are further than 5 kb from a bead. Expect the exporter to report the covered fraction and refuse below 0.5. Test in Task 7.
5. An arm directory holding fewer than ten cif files when the readouts run. Expect every readout to print the count per arm and the figure to mark the arm. Test in Task 8.

---

### Task 1: Loop pooling, filtering, thinning and absorbing

**Files:**
- Create: `playground/rnapii/crosscell_prep.py`
- Test: `harness/test_crosscell_prep.py`

**Interfaces:**
- Consumes: `playground/rnapii/anchor_union.py` `read_anchors`, `union_anchors`, `AnchorRow`, `End`.
- Produces:
  - `Loop = tuple[str, int, int, str, int, int, int]`
  - `read_bedpe(path: Path) -> list[Loop]`, gzip aware, normalises end order and float PETs, drops trans rows.
  - `write_bedpe(path: Path, loops: list[Loop]) -> None`, sorted.
  - `pool_replicates(reps: list[list[Loop]]) -> list[Loop]`
  - `class PeakIndex` with `PeakIndex.from_beds(paths: list[Path])` and `overlaps(chrom: str, start: int, end: int) -> bool`
  - `filter_loops(loops: list[Loop], peaks: PeakIndex, ends_on_peaks: int, max_span: int) -> list[Loop]`
  - `depth(loops: list[Loop]) -> int`, the PET total.
  - `thin(loops: list[Loop], p: float, rng: np.random.Generator, min_pets: int = 3) -> list[Loop]`
  - `absorb_ends(loops: list[Loop], ctcf: list[AnchorRow], absorb_bp: int) -> tuple[list[Loop], int]`, returning the rewritten loops and the number of ends absorbed.

- [ ] **Step 1: Create the branch**

```bash
git fetch origin && git switch -c rnapii-crosscell origin/triosformer-expression
.venv/bin/python harness/test_arc_matrix.py
```
Expected: the test prints its checks and exits 0. This proves the summary arc fix is present.

- [ ] **Step 2: Write the failing tests in `harness/test_crosscell_prep.py`**

Plain script in the style of `harness/test_thinning.py`, one function per check, all called from `main`.

```python
def test_read_normalises(tmp: Path) -> None:
    # one row reversed with a float PET count, one trans row
    p = tmp / "x.bedpe"
    p.write_text("chr1\t500\t600\tchr1\t100\t200\t3.0\nchr1\t1\t2\tchr2\t5\t6\t9\n")
    assert read_bedpe(p) == [("chr1", 100, 200, "chr1", 500, 600, 3)]

def test_pool_sums_overlapping() -> None:
    a = [("chr1", 100, 200, "chr1", 5000, 5100, 4)]
    b = [("chr1", 150, 260, "chr1", 5050, 5200, 2), ("chr1", 9000, 9100, "chr1", 20000, 20100, 3)]
    pooled = pool_replicates([a, b])
    assert ("chr1", 100, 260, "chr1", 5000, 5200, 6) in pooled
    assert ("chr1", 9000, 9100, "chr1", 20000, 20100, 3) in pooled
    assert len(pooled) == 2

def test_filter_rules() -> None:
    peaks = PeakIndex({"chr1": [(90, 210)]})
    one_end = ("chr1", 100, 200, "chr1", 5000, 5100, 5)
    far = ("chr1", 100, 200, "chr1", 2_000_000, 2_000_100, 5)
    assert filter_loops([one_end], peaks, ends_on_peaks=1, max_span=1_000_000) == [one_end]
    assert filter_loops([one_end], peaks, ends_on_peaks=2, max_span=1_000_000) == []
    assert filter_loops([far], peaks, ends_on_peaks=1, max_span=1_000_000) == []

def test_thin_keeps_expectation() -> None:
    loops = [("chr1", i * 1000, i * 1000 + 100, "chr1", i * 1000 + 50_000, i * 1000 + 50_100, 40) for i in range(2000)]
    out = thin(loops, 0.5, np.random.default_rng(1), min_pets=3)
    assert abs(depth(out) / depth(loops) - 0.5) < 0.02
    assert all(l[6] >= 3 for l in out)

def test_absorb() -> None:
    ctcf = [("chr1", 1000, 2000, "R")]
    loops = [("chr1", 2500, 2600, "chr1", 90_000, 90_100, 5)]  # end A 500 bp from the anchor
    out, n = absorb_ends(loops, ctcf, absorb_bp=1000)
    assert out == [("chr1", 1000, 2000, "chr1", 90_000, 90_100, 5)] and n == 1
    out, n = absorb_ends(loops, ctcf, absorb_bp=100)
    assert out == loops and n == 0
```

`PeakIndex` takes the merged per chromosome intervals in its constructor; `from_beds` builds them from files.

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python harness/test_crosscell_prep.py`
Expected: ImportError for `crosscell_prep`.

- [ ] **Step 4: Implement `playground/rnapii/crosscell_prep.py`**

Pooling joins loops whose two ends both overlap, summing PETs and taking the union interval of each end; process loops sorted by end A and compare against the open pooled loops on the same chromosome. Peak lookup and thinning follow `~/Desktop/triosformer/pipeline/trio_resample.py` (`peak_union`, `PeakIndex.overlaps`, `thin` as a binomial draw per loop). Absorbing moves an end onto the nearest CTCF anchor interval when the gap between them is at most `absorb_bp`, zero when they overlap; the loader then maps it by its midpoint. Module docstring states the spec section it implements.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python harness/test_crosscell_prep.py`
Expected: every check prints ok, exit 0.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format harness/test_crosscell_prep.py playground/rnapii/crosscell_prep.py
.venv/bin/ruff check harness/ playground/rnapii/
git add playground/rnapii/crosscell_prep.py harness/test_crosscell_prep.py
git commit -m "rnapii: pooling, peak filter, thinning and end absorption for the cross line inputs"
```

---

### Task 2: Fetch manifests and the prep driver

**Files:**
- Create: `validation/manifests/GM12878_rnapii.json`, `H1ESC_rnapii.json`, `HFFC6_rnapii.json`
- Modify: `playground/rnapii/crosscell_prep.py` (add `main`)
- Test: `harness/test_crosscell_prep.py` (add driver checks)

**Interfaces:**
- Consumes: Task 1 functions. `validation fetch` (`validation/studies/fetch.py`), which resolves ENCFF accessions directly.
- Produces, per line `<C>` under `data/<C>/`:
  - `<C>_xc_ctcf_clusters_3+.bedpe`, the thinned CTCF loops.
  - `<C>_xc_rnapii_clusters_3+.bedpe`, the thinned RNAPII loops.
  - `<C>_xc_rnapii_absorbed_clusters_3+.bedpe`, arm M's RNAPII loops.
  - `<C>_xc_anchors_ctcf_rnapii.bed` and `<C>_xc_anchors_ctcf_rnapii_absorbed.bed`, by `union_anchors` over the production `<C>_anchors_3+_oriented.bed`.
  - `<C>_xc_prep.json` with keys `line, rnapii_source, accessions, seed, max_span, absorb_bp, depth_before, depth_after, keep_fraction, loops_before, loops_after, ends_absorbed, anchors_added` per factor where it applies.
- CLI: `.venv/bin/python playground/rnapii/crosscell_prep.py [--max-span 1000000] [--seed 20261111] [--dry-run]`

- [ ] **Step 1: Write the manifests**

Each lists the RNAPII loops and peaks per replicate as ENCFF entries with `name` `RNAPII_loops_r<k>` and `RNAPII_peaks_r<k>`, and the line's CTCF ChIP-seq peaks as `CTCF_peaks`. GM12878 and H1 accessions are in the spec's table. HFFC6 uses the BJ accessions and carries `"_note": "RNAPII from BJ, ENCSR956IXE, a different foreskin fibroblast line from HFFc6"`. Pick the CTCF ChIP-seq experiment per line on ENCODE, GRCh38, released, `output_type` "conservative IDR thresholded peaks" or the closest, and record the ENCSR in the manifest. The CTCF ChIP for GM12878 is ENCSR000DZN as in `GM12878.json`.

- [ ] **Step 2: Fetch**

Run: `for C in GM12878 H1ESC HFFC6; do .venv/bin/python -m validation fetch --manifest validation/manifests/${C}_rnapii.json --out data/_epigenome; done`
Expected: every file reports downloaded or present with md5 ok, under `data/_epigenome/<C>/`.

- [ ] **Step 3: Add the failing driver checks**

```python
def test_keep_fraction_guard() -> None:
    # depths 100, 1000, 1000 give a keep fraction of 0.1 for two lines
    with pytest_raises(SystemExit):
        check_keep_fractions({"A": 100, "B": 1000, "C": 1000}, floor=0.2)

def test_record_names_bj(tmp: Path) -> None:
    rec = build_record("HFFC6", rnapii_source="BJ ENCSR956IXE", ...)
    assert "BJ" in rec["rnapii_source"]
```
`pytest_raises` is a ten line context manager in the test file. Driver helpers: `check_keep_fractions(depths: dict[str, int], floor: float) -> dict[str, float]` exits with a message naming the line when a fraction is under `floor`; `build_record(line: str, **fields: object) -> dict[str, object]`.

- [ ] **Step 4: Run to verify failure**

Run: `.venv/bin/python harness/test_crosscell_prep.py`
Expected: NameError for `check_keep_fractions`.

- [ ] **Step 5: Implement `main`, `check_keep_fractions`, `build_record`**

Order per factor: read and pool replicates (CTCF is the single production file), filter on the line's own peaks with `ends_on_peaks` 2 for CTCF and 1 for RNAPII, compute every line's depth, check keep fractions at floor 0.2, thin each line to the minimum with `np.random.default_rng([seed, line_index, factor_index])`, write. Then absorb on the thinned RNAPII loops and build both anchor sets. Print one table of line, factor, depth before and after, keep fraction, loops, anchors added, ends absorbed. `--dry-run` prints the table and writes nothing.

- [ ] **Step 6: Run the checks, then the driver dry run**

Run: `.venv/bin/python harness/test_crosscell_prep.py && .venv/bin/python playground/rnapii/crosscell_prep.py --dry-run`
Expected: checks pass; the table shows three lines by two factors and no keep fraction under 0.2. If one is, stop and report the table to the user.

- [ ] **Step 7: Run the driver for real and commit code and manifests**

Run: `.venv/bin/python playground/rnapii/crosscell_prep.py`
Expected: the five files and the record exist for each line. `data/` is gitignored, so copy the three `_xc_prep.json` records into `design/plans/rnapii-prep-records/` and commit them with the code.

```bash
git add validation/manifests/*_rnapii.json playground/rnapii/crosscell_prep.py harness/test_crosscell_prep.py design/plans/rnapii-prep-records/
git commit -m "rnapii: cross line prep driver, ENCODE manifests and the prep records"
```

---

### Task 3: Configs for arms C, R and M

**Files:**
- Create: `slurm/ensemble/crosscell_configs.py`
- Test: `harness/test_crosscell_configs.py`

**Interfaces:**
- Consumes: `validation.core.config.CANONICAL`, `cell_data_section`, and the writer pattern of `slurm/ensemble/make_configs.py`. Task 2 file names.
- Produces: `slurm/ensemble/<cell lower>_xc_<arm>.ini` for arm in `C, R, M`. `build(cell: str, arm: str) -> configparser.ConfigParser`.

Data sections, all else from `CANONICAL`:

| arm | anchors | clusters | factors | factor_strength |
|---|---|---|---|---|
| C | `<C>_anchors_3+_oriented.bed` | `<C>_xc_ctcf_clusters_3+.bedpe` | absent | absent |
| R | `<C>_xc_anchors_ctcf_rnapii.bed` | `<C>_xc_ctcf_clusters_3+.bedpe,<C>_xc_rnapii_clusters_3+.bedpe` | `CTCF,RNAPOL2` | `1,1` |
| M | `<C>_xc_anchors_ctcf_rnapii_absorbed.bed` | `<C>_xc_ctcf_clusters_3+.bedpe,<C>_xc_rnapii_absorbed_clusters_3+.bedpe` | `CTCF,RNAPOL2` | `1,1` |

HFFC6's R and M files start with a comment line naming BJ as the RNAPII source.

- [ ] **Step 1: Write the failing test**

```python
def test_arms() -> None:
    c, r, m = (build("H1ESC", a) for a in ("C", "R", "M"))
    assert c["data"]["clusters"] == "H1ESC_xc_ctcf_clusters_3+.bedpe"
    assert "factors" not in c["data"]
    assert r["data"]["factors"] == "CTCF,RNAPOL2" and r["springs"]["factor_strength"] == "1,1"
    assert m["data"]["anchors"] == "H1ESC_xc_anchors_ctcf_rnapii_absorbed.bed"
    for sec in c.sections():
        if sec != "data":
            assert dict(c[sec]) == {k: v for k, v in r[sec].items() if k != "factor_strength"}
```

Read where `factors` and `factor_strength` live in an existing two factor config, `~/Desktop/triosformer/slurm/configs/gm19238_trio_rnapol2_hgsvc.ini`, and put them in the same sections.

- [ ] **Step 2: Run to verify failure**, `.venv/bin/python harness/test_crosscell_configs.py`, ImportError.

- [ ] **Step 3: Implement `build` and a `main` that writes the nine files.**

- [ ] **Step 4: Run the test, then load each ini through the loader**

Run: `.venv/bin/python harness/test_crosscell_configs.py && for f in slurm/ensemble/*_xc_*.ini; do .venv/bin/python -c "from gnome3d.settings import Settings; Settings.from_ini('$f')" || echo FAIL $f; done`
Expected: test passes, no FAIL lines. Use the loader entry point `gnome3d.cli` uses if it is not `Settings.from_ini`.

- [ ] **Step 5: Commit** the script, the test and the nine ini files.

---

### Task 4: Run scripts and the trio arm commands

**Files:**
- Create: `slurm/ensemble/crosscell_arms.sh`
- Modify: `design/2026-11-11-rnapii-release.md` (append a Runs section with the exact commands)

**Interfaces:**
- Consumes: Task 3 configs. `gnome3d.cli`, `playground/slice_arm.py`.
- Produces: `out/xc/<C>_<arm>/` with ten cif files per chromosome, and `out/xc/<C>_<arm>_60mb/`.

- [ ] **Step 1: Write `crosscell_arms.sh`**

A SLURM array over nine tasks, index `i` to line `i / 3` of `GM12878 H1ESC HFFC6` and arm `i % 3` of `C R M`, header modelled on `slurm/ensemble/chr1_ensemble.sh` with one GPU per task. `CHROMS` defaults to `chr1`. The body is the `gnome3d.cli` call of `rnapii_arms.sh` with `-n 10`, then `slice_arm.py` for chr1. `DRY=1` prints the mapping and the commands without running. Resumes by skipping existing cif files, as `gnome3d.cli` already does.

- [ ] **Step 2: Check it**

Run: `bash -n slurm/ensemble/crosscell_arms.sh && for i in $(seq 0 8); do SLURM_ARRAY_TASK_ID=$i DRY=1 bash slurm/ensemble/crosscell_arms.sh; done`
Expected: nine lines, each line and arm once, each config path existing.

- [ ] **Step 3: Write the commands for the user into the spec's Runs section**

Cell lines: `git pull` on the eden checkout and `git switch rnapii-crosscell`, push of `data/<C>/` inputs by `rsync` from the machine that ran Task 2, then `sbatch --array=0-8 slurm/ensemble/crosscell_arms.sh`. Trios: in the triosformer checkout, `python pipeline/trio_configs.py --factor CTCF --singletons hgsvc` and `--factor RNAPOL2 --singletons hgsvc`, then from the engine checkout on this branch `CONFIG_TAG=_trio_hgsvc OUT=out/xc_trio_ctcf_hgsvc CHROMS=chr1 sbatch --array=0-8 .../triosformer/slurm/trio_ensemble.sh` and the same with `CONFIG_TAG=_trio_rnapol2_hgsvc OUT=out/xc_trio_rnapol2_hgsvc`. The E arm: the ARCH path of each line's chr1 models.

- [ ] **Step 4: Commit** the script and the spec update. Tell the user the commands are ready; do not submit.

---

### Task 5: Machine readable battery and saddle

**Files:**
- Modify: `playground/validation_battery.py`, `playground/saddle_offline.py`
- Test: `harness/test_battery_bonds.py` is the existing battery test; add `harness/test_battery_json.py`

**Interfaces:**
- Produces: `--json PATH` on both scripts. The battery writes `[{"arm", "n", "pearson", "spearman", "scc", "multimm", "exponent", "e_short", "e_long", "rg", "wb_aa", "wb_sa", "xb"}]`, the saddle `[{"arm", "n", "saddle", "eig_r", "kappa"}]`. Text output unchanged.

- [ ] **Step 1: Write the failing test.** Build two tiny arm directories of three cif files each from `harness/test_battery_bonds.py`'s fixture helpers and a small synthetic mcool with `cooler.create_cooler`; run the battery main with `--json`; assert the file has two rows with every key and `n == 3`.

- [ ] **Step 2: Run to verify failure**, the flag is unknown and the file is not written.

- [ ] **Step 3: Implement.** Read the flag with the existing `_str_flag`, collect each arm's row where it is printed, write once at the end.

- [ ] **Step 4: Run to verify pass, then confirm the text output is byte identical** to the previous commit on the same input by diffing the two runs' stdout.

- [ ] **Step 5: Commit.**

---

### Task 6: Pile-up at RNAPII loops

**Files:**
- Create: `playground/rnapii/pileup.py`
- Test: `harness/test_pileup.py`

**Interfaces:**
- Consumes: `validation.metrics.hic.observed_hic`, `simulated_contacts`; Task 1 `read_bedpe`; the battery's `load` for cif files.
- Produces:
  - `pile(matrix: F64Array, bin_starts: I64Array, binsize: int, loops: list[Loop], flank: int, min_sep: int) -> tuple[F64Array, int]`, the mean over loops of the `(2 flank + 1)` square window of observed over expected by diagonal, and the number of loops used.
  - `apa_score(p: F64Array) -> float`, centre pixel over the mean of the 3 by 3 lower left corner.
  - CLI `pileup.py --json PATH <mcool> <region> <binsize> <loops.bedpe> <arm_dir>...`, rows `{"arm", "n_models", "n_loops", "apa"}` with `arm = "map"` for the target; piles saved beside the JSON as `.npz`.

Defaults (my call, recorded in the module docstring): binsize 10,000, flank 10, `min_sep` 5 bins, loops from the line's `_xc_rnapii_clusters_3+.bedpe`. The model matrix is the sum of `simulated_contacts` over the ensemble with the battery's contact radius rule.

- [ ] **Step 1: Write the failing tests**

```python
def test_planted_enrichment() -> None:
    n, starts = 200, np.arange(200) * 10_000
    m = np.ones((n, n))
    loops = [("chr1", 50 * 10_000, 50 * 10_000 + 100, "chr1", 80 * 10_000, 80 * 10_000 + 100, 5)]
    m[50, 80] = m[80, 50] = 5.0
    p, used = pile(m, starts, 10_000, loops, flank=10, min_sep=5)
    assert used == 1 and abs(apa_score(p) - 5.0) < 1e-9

def test_short_loops_skipped() -> None:
    ...  # a loop two bins apart is not used and used == 0
```

- [ ] **Step 2: Run to verify failure.** **Step 3: Implement.** **Step 4: Run to verify pass.**

- [ ] **Step 5: Smoke on real data.** On the workstation, `pileup.py` on the GM12878 deep map, chr1:1-60000000, with no arm directory. Expected: `n_loops` in the thousands and the map's APA above 1.5. If the map shows no enrichment, stop and report, since the readout would then measure nothing.

- [ ] **Step 6: Commit.**

---

### Task 7: Export to the enhancer3d layout, and E at ten models

**Files:**
- Create: `playground/rnapii/e3d_export.py`
- Test: `harness/test_e3d_export.py`

**Interfaces:**
- Consumes: the battery's `load` for gnome3d cif.
- Produces:
  - `to_grid(pos: F64Array, mids: I64Array, start: int, end: int, res: int = 1000, max_gap: int = 5000) -> tuple[F64Array, I64Array, float]`, grid coordinates with NaN where the nearest bead midpoint is further than `max_gap`, the grid positions, and the covered fraction.
  - `write_model(stem: Path, grid: F64Array, positions: I64Array) -> None` writing `<stem>.cif` and `<stem>.txt` in the format `enhancer3d/models.py` `load_ensemble` reads: the txt is the count line then `x y z position` per row as in `tools/hcm_to_smooth.py` `write_txt`, NaN written as `nan`.
  - CLI `e3d_export.py <arm_dir> <out_dir> <chrom> [--min-covered 0.5]`, exits naming the model when a covered fraction is under the floor.
  - CLI `e3d_export.py --subsample <E_dir> <out_dir> --n 10 --seed 20261111`, symlinks ten seeded choices of E's model files with their txt.

- [ ] **Step 1: Write the failing tests.** A chain of beads every 2 kb over 0 to 100 kb plus a 20 kb gap: assert grid NaN exactly inside the gap beyond 5 kb from either side, covered fraction as computed by hand, and that `write_model` then reading the txt back gives the same count and positions.

- [ ] **Step 2: Run to verify failure.** **Step 3: Implement.** **Step 4: Run to verify pass.**

- [ ] **Step 5: Check against enhancer3d's own loader**

Run: `cd ~/Desktop/enhancer3d_cross_cell_lines_analysis && <its python> -c "from enhancer3d.models import load_ensemble; c,f,r = load_ensemble('<exported dir>'); print(c.shape, f, r)"`
Expected: shape `(n, positions, 3)`, `r == 1000`. Confirm in `enhancer3d/distances.py` that a NaN distance drops the pair (`p["d"] > 0` is false for NaN); if any statistic path does not drop NaN, report it to the user rather than patching their repository.

- [ ] **Step 6: Commit.**

---

### Task 8: Collect readouts and draw the figure

**Files:**
- Create: `playground/rnapii/figure_rnapii.py`
- Test: `harness/test_figure_rnapii.py`

**Interfaces:**
- Consumes: Task 5 battery and saddle JSON, Task 6 pile-up JSON and npz, the enhancer3d statistics tables (cross line E1 to E3 and jump, written by their CLI with `--n-models 10`), the triosformer tables from `trio_arms_compare.py` and `trio_arms_jump.py`, and each line's `_xc_prep.json`.
- Produces: `collect(paths: Manifest) -> pandas.DataFrame` in long form, columns `dataset, kind (line|person), arm (E|C|R|M), metric, value, n`; `figure_main(df, piles, out: Path)` and `figure_grid(df, out: Path)`. CLI `figure_rnapii.py <manifest.json> <out_dir>` where the manifest maps each source to its file.

Main figure, panels per the spec: a inputs and model snapshots, b one locus as target, C and R contact maps with RNAPII loops marked, c pile-ups, d structure paired dots per line and person C to R to M, e expression paired dots with E where it applies, f change in SCC against change in expression signal from C, one point per line or person per RNAPII arm. Supplementary grid: rows datasets, columns metrics, each cell coloured by the better arm. Labels for HFFC6 read `HFFc6 (RNAPII from BJ)` when the prep record says so; arms with `n < 10` carry `n=<k>`. Follow the `dataviz` skill for colour and type.

- [ ] **Step 1: Write the failing tests** on a synthetic long table: `figure_grid` names every dataset and metric present; the HFFC6 label contains `BJ` when the record says BJ; an arm with `n = 7` is labelled `n=7`.

- [ ] **Step 2: Run to verify failure.** **Step 3: Implement.** **Step 4: Run to verify pass.**

- [ ] **Step 5: Candidate loci for panel b.** Add `--loci` printing the ten 1 Mb windows of chr1 where the map's pile-up at RNAPII loops is strongest and R's contact map differs most from C's. Hand the list to the user to pick.

- [ ] **Step 6: Commit.** Then hand the user the readout commands for after the runs finish: battery and saddle with `--json` per line on C, R, M and the subsampled E, pile-up per line, the export and enhancer3d CLI per arm, the trio readouts per arm, and the figure call. The trio structure readouts are the same battery, saddle and pile-up per person, against that person's HGSVC mcool under `$TRIOS_HGSVC/mcool/`, with that person's `_rnapol2_clusters_3+.bedpe` as the pile-up loops, on `out/xc_trio_ctcf_hgsvc` and `out/xc_trio_rnapol2_hgsvc`.

---

## After chr1

Read the chr1 numbers against the spec's rule for M before the autosomes. The autosome runs reuse `crosscell_arms.sh` with `CHROMS` set to the report's 20 autosomes, which the builder takes from the enhancer3d analysis log.
