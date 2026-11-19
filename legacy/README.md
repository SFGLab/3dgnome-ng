# Retired code

This folder holds code the project no longer develops. Everything in it still runs from the repository root.

| Folder | Content |
|--------|---------|
| `3dnome/` | The reference implementation, the Monte Carlo core of 3D-GNOME. Read only. `make -C legacy 3dnome` builds `legacy/3dnome/3dnome`. |
| `harness/` | `integration.py` runs the reference binary and the Python model on one region and compares the two ensembles. The `test_*.py` files test scripts under `playground/` and two modules of `validation/`. Each runs as a script, `python legacy/harness/test_pileup.py`. |
| `validation/` | The validation package: the metrics of `design/validation-metrics.md`, the studies, the fetch and tracks commands and the manifests of the input data. `python -m legacy.validation --help` lists the commands. Its dependencies are the `validation` extra, `pip install -e ".[validation]"`. |

The tests of the model are in `tests/` and the production configuration is `gnome3d/production.py`.

Scripts under `playground/` and `slurm/` import the metrics as `legacy.validation.metrics` and fetch data with
`python -m legacy.validation fetch`.
