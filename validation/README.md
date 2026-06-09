# `validation/` — structure-validation harness

Proves that 3dgnome runs (and our divergences: excluded volume, confinement, dynamic loop
density) produce **sensible** structure, using the resolution-independent toolkit from the
original 3D-GNOME papers — *not* MERFISH/250-kb imaging, which is too coarse for our
native-resolution models. Methodology and rationale: [`../docs/validation.md`](../docs/validation.md).

Everything runs through the **public** gnome3d modelling API (`Settings`, `ContactData`,
`simulate`) — what a user doing modelling calls — and scores only public output (`BeadOut`)
plus the user's own input contacts. No internal pipeline state.

## Modules

- [`metrics.py`](metrics.py) — pure functions over coordinates (unit-testable, no gnome3d MC):
  - `self_consistency` — **V1**: input contact score vs output 3D distance (Spearman; want < 0).
  - `distance_scaling` / `contact_probability` — **V2**: power-law exponents vs genomic separation.
  - `dab_matrix` / `ensemble_diversity` — **V3**: Szałaj-2016 mirror-insensitive inter-structure
    distance `d_AB` (deliberately *not* RMSD).
  - `overlap_fraction` — the **EV/confinement** test: fraction of non-bonded beads that
    interpenetrate, i.e. the "physically impossible overlaps" the 2016 paper admitted.
- [`cell_config.py`](cell_config.py) — canonical 3dgnome modelling params baked in; `settings_for_cell(cell)` builds a `Settings` via `from_dict` (no .ini, no `--data-dir`).
- [`validate.py`](validate.py) — CLI driver: runs ensembles and reports / `--prove` compares.
- [`sweep.py`](sweep.py) — EV/confinement hyperparameter search scored by Hi-C correlation (see [RUNBOOK.md](RUNBOOK.md)).
- [`dataloader.py`](dataloader.py) — fetch experiment files (epigenomic tracks, contacts,
  imaging) from 4DN / ENCODE via a declarative per-cell-line manifest
  ([`manifests/`](manifests/)); resolves accession → URL + md5, downloads with caching +
  checksum, writes a lockfile. Feeds V4/V5 cross-checks and the epitensor ML pipeline.

## Usage

Settings are wired from canonical params (`cell_config.py`, `Settings.from_dict`) — pass only
`--cell`; no config file, no `--data-dir`. `--quality {fast,balanced,full}` sets the schedule
(default `full`). Ensembles need **n ≥ 100** (3dgnome standard); small `n` is for quick checks.

Report one cell line's ensemble:

```bash
.venv/bin/python -m validation.validate --cell GM12878 \
    --region chr1:18288319-20307135 -n 100
```

Prove a divergence makes sense (runs flags-**OFF** vs **ON** on the *same* loaded data and
judges the difference):

```bash
.venv/bin/python -m validation.validate --cell GM12878 --quality fast \
    --region chr1:18288319-20307135 -n 100 --prove ev        # or: confinement | dynamic | all
```

Fetch experiment data (epigenomic tracks etc.) from a manifest:

```bash
.venv/bin/python -m validation.dataloader \
    --manifest validation/manifests/GM12878.json --out data/_epigenome --dry-run   # resolve only
```

`--prove ev`/`confinement` PASS when overlaps **drop** and V1/V2 don't degrade — operationalising
[`../docs/validation.md`](../docs/validation.md) §2 ("complete the original authors' own
excluded-volume / nuclear-membrane TODO"). Exit code is non-zero on FAIL (CI-friendly).

Notes:
- `--data-root` points at the dir holding `<cell>/` data (default `data`); the shipped configs'
  absolute `/Projects/...` `data_dir` is bypassed entirely — `cell_config.py` wires paths by
  convention.
- `--contact-radius` sets the overlap / contact threshold (default: median baseline bond length).
- Canonical modelling params live in [`cell_config.py`](cell_config.py) (`CANONICAL`), assembled
  via `Settings.from_dict` — edit there to change the validation base config.

## What this does *not* cover yet

- **V4** ChIA-PET↔Hi-C cross-consistency, **V5** FISH locus-pair distances, **V6** perturbation —
  need experimental data from the loader (see `../docs/validation.md` §1).
- The exact **V1 correlation statistic** from the 2016 Supplemental Material (we use Spearman of
  score-vs-distance as a faithful, public-data stand-in).
