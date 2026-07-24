# validation

Structure validation harness for 3dgnome. It checks that reconstructions and the tuned divergences
(excluded volume, confinement, dynamic loop density) produce sensible structure, using the
resolution-independent measures from the 3D-GNOME papers. Everything runs through the public gnome3d
modelling API (`Settings`, `ContactData`, `simulate`) and scores public output (`BeadOut`) plus the
user's own input contacts.

The metrics themselves are defined in [../docs/validation-metrics.md](../docs/validation-metrics.md).
Every command for running them on the CUDA box is in [RUNBOOK.md](RUNBOOK.md).

Everything runs through one entry point, `python -m validation <subcommand>`. Run
`python -m validation run` for the standard battery, compare, synthetic, self-corr, model-hic, in
one pass.

## Modules

- [core/config.py](core/config.py). The canonical modelling params. `settings_for_cell(cell)` builds
  a `Settings` from them with no .ini. All config edits go through the helpers here.
- [core/variants.py](core/variants.py). The one reference/parity/tuned reconstruction path every
  study shares.
- [core/regions.py](core/regions.py). Region parsing and sampling shared by the studies.
- [core/ensemble.py](core/ensemble.py). Ensemble reconstruction and summary metrics shared by the
  studies.
- [core/data.py](core/data.py). Contact loading shared by the studies.
- [core/report.py](core/report.py). Console report formatting shared by the studies.
- [metrics/structure.py](metrics/structure.py). Pure functions over coordinates. Self-consistency,
  distance and contact-decay power laws, ensemble-diversity inter-structure distance `d_AB`, overlap
  fraction for the excluded volume test, and the reconstruction-fidelity pair `contact_measure` and
  `rmsd_superpose`.
- [metrics/hic.py](metrics/hic.py). Hi-C contact-map correlation. Simulated map against experimental
  Hi-C (SCC and inverse-distance Pearson), and the cross-data correlation of ChIA-PET against Hi-C.
- [studies/report.py](studies/report.py). `report` subcommand. Runs an ensemble and reports.
- [studies/prove.py](studies/prove.py). `prove` subcommand. Proves a divergence with `--prove`.
- [studies/compare.py](studies/compare.py). `compare` subcommand. Reference against python parity
  against python tuned across several multi-block regions, paired per region. Reports overlaps,
  Hi-C correlation, cross-data correlation, and a separate scaling-law pass on one large region.
- [studies/self_corr.py](studies/self_corr.py). `self-corr` subcommand. Feeds experimental Hi-C into
  the engine as singletons and correlates the reconstruction against held-out Hi-C bin pairs, for
  all three variants.
- [studies/synthetic.py](studies/synthetic.py). `synthetic` subcommand. The paper's core model
  validation. Reconstructs a known synthetic structure and measures fidelity with RMSD and the
  contact measure, and sweeps heatmap noise for the robustness curve.
- [studies/model_hic.py](studies/model_hic.py). `model-hic` subcommand. Correlates each variant's
  reconstructed contact map against Hi-C at 1 Mb, the paper's Fig. 2B resolution.
- [studies/boundaries.py](studies/boundaries.py). `boundaries` subcommand. Calls TAD boundaries from
  an mcool with cooltools insulation and writes them as a breakpoints file.
- [studies/sweep.py](studies/sweep.py). `sweep` subcommand. Excluded-volume and confinement
  hyperparameter search scored by Hi-C correlation. See [RUNBOOK.md](RUNBOOK.md).
- [studies/tune.py](studies/tune.py). `tune` subcommand. Fast Hi-C correlation tuning on a fixed
  tuned ensemble, sweeping only the readout parameters.
- [studies/fetch.py](studies/fetch.py). `fetch` subcommand. Fetches experiment files from 4DN and
  ENCODE via a per-cell-line manifest, with caching and checksums.

## Usage

Settings come from the canonical params, so pass only `--cell`. There is no config file and no
`--data-dir`. `--quality {fast,balanced,full}` sets the schedule and defaults to full. Ensembles
need n at least 100, the 3dgnome standard. A small n is for quick checks.

Report one cell line's ensemble.

```bash
.venv/bin/python -m validation report --cell GM12878 \
    --region chr1:18288319-20307135 -n 100
```

Prove a divergence makes sense. This runs the flags off against on over the same loaded data and
judges the difference. It passes when overlaps drop and self-consistency and scaling laws do not
degrade, and exits non-zero on failure.

```bash
.venv/bin/python -m validation prove --cell GM12878 --quality fast \
    --region chr1:18288319-20307135 -n 100 --prove ev
```

## Notes

- `--data-root` points at the directory holding `<cell>/` data and defaults to `data`.
- `--contact-radius` sets the overlap and contact threshold. It defaults to the median baseline bond
  length.
- The canonical modelling params live in [core/config.py](core/config.py) under `CANONICAL`. Edit
  there to change the validation base config.
