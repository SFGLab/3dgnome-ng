# Validation sweep — CUDA runbook

Full EV/confinement sweep (Hi-C-correlation-scored) across the 3 cell lines. Run this on the
CUDA box: ensembles need **n ≥ 100** (3dgnome ensemble standard, per the 2016 paper), which is
why this is a GPU job. Plan: [../docs/validation-sweep-plan.md](../docs/validation-sweep-plan.md).

## 0. Install

```bash
pip install -e ".[validation,jax]" "jax[cuda12]"     # validation deps + JAX CUDA backend
```

The JAX backend is per-stage via `mc_executor_<stage>` (`batch` = JAX/GPU, `serial`/`threaded` =
numba). With JAX installed, the default `auto` **already routes smooth + estimate_dist to the GPU
batch path** (region-batches the n=100 restarts) — so on a CUDA box it's automatic. To force it
explicitly (incl. arcs), set in `validation/cell_config.py::CANONICAL["simulation_backend"]`:

```python
"simulation_backend": {"ib_workers": 1, "heatmap_chains": 1, "smooth_chains": 1,
                       "mc_executor_smooth": "batch", "mc_executor_estimate_dist": "batch",
                       "mc_executor_arcs": "batch"},
```

## 1. Contact targets (plain Hi-C — CTCF-inclusive, NOT RNA Pol II)

Each cell line's Hi-C is a restriction-enzyme **plain Hi-C** (captures CTCF-mediated
architecture); RNA Pol II ChIA-PET/HiChIP is deliberately excluded (the dataloader warns if an
accession resolves to Pol II). Enzyme/depth differs per line (a cross-cell caveat), so tune
**per cell line** against its own Hi-C:

| cell line | accession | assay | size |
|---|---|---|---|
| GM12878 | 4DNFIQ32RWCQ | in situ Hi-C (MboI) | 153 MB |
| H1ESC | 4DNFIHO3CXUQ | in situ Hi-C (AluI) | 297 MB |
| HFFC6 | 4DNFIDKNBPC3 | in situ Hi-C (AluI) | 920 MB |

## 2. Per-cell-line commands

Settings are wired from canonical params (`validation/cell_config.py`) — **no config file or
`--data-dir`**; just `--cell`. `--quality full` is the default (real schedule, 50 000 steps);
use `fast`/`balanced` only for quick checks.

```bash
# --- GM12878 ---
python -m validation.dataloader --manifest validation/manifests/GM12878_hic.json --out data/_hic
python -m validation.sweep --cell GM12878 \
    --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 -n 100 --out out/sweep/GM12878.json

# --- H1ESC ---
python -m validation.dataloader --manifest validation/manifests/H1ESC_hic.json --out data/_hic
python -m validation.sweep --cell H1ESC \
    --hic data/_hic/H1ESC/hic.4DNFIHO3CXUQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 -n 100 --out out/sweep/H1ESC.json

# --- HFFC6 ---
python -m validation.dataloader --manifest validation/manifests/HFFC6_hic.json --out data/_hic
python -m validation.sweep --cell HFFC6 \
    --hic data/_hic/HFFC6/hic.4DNFIDKNBPC3.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 -n 100 --out out/sweep/HFFC6.json
```

Each sweep is **resumable** (its `--out` JSON caches per (config, region)); re-running continues
where it stopped. The grid (`validation/sweep.py::GRID`, 15 configs) spans EV weight from the
real default 0.1 up to 2.0, confinement at engaging packing factors {1.0, 0.75, 0.5}, combos,
and two EV-radius probes. To bump cell-wide flags (e.g. CUDA JAX backend), edit
`validation/cell_config.py::CANONICAL["simulation_backend"]`.

## 3. Also worth running at n=100

```bash
# vs-reference (faithful port + EV/confinement beat reference on overlaps)
python -m validation.compare_reference --region chr1:18288319-20307135 -n 100

# isolate each divergence (overlaps for EV, extent for confinement, scaling for dynamic)
python -m validation.validate --cell GM12878 \
    --region chr1:18288319-20307135 -n 100 --prove all
```

## 4. Reading the output

The sweep prints a per-config median table (Hi-C SCC, Pearson, overlap, scaling, diversity) and
the **constrained max-SCC winner** (max median SCC among configs with overlaps ≤ baseline and
sane scaling). Compare winners across the 3 cell lines for a **generalizing** EV/confinement
default. At n=100 the SCC deltas should clear the noise floor the lean n=2 pass could not.

> Cross-cell-line note: the three Hi-C files use different enzymes/depths, so absolute SCC isn't
> comparable across lines — but the *winning config* (relative ranking within each line) should
> agree if the recommendation generalizes.
