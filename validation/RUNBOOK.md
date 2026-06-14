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
`--data-dir`**; just `--cell`.

**Use `--search` (successive halving), not the flat sweep.** A flat 15-config × 20-region ×
n=100 × full-schedule grid is ~30 000 models (days). `--search` does it in three rungs:

1. **screen** — all 15 configs × `--screen-regions` (5) × `--search-n` (30) at `--search-quality`
   (balanced) → rank, keep top `--keep` (4);
2. **expand** — survivors × all `--n-regions` (20) at the search budget → pick the winner;
3. **validate** — winner + baseline × all regions at `--final-n` (100) × `--final-quality`
   (full). Only this rung pays the full-schedule × n=100 cost, on 2 configs.

That's ~7× cheaper than the flat grid (≈ days → ≈ half a day) and surfaces the likely winner
in the first hour.

```bash
# --- GM12878 ---
python -m validation.dataloader --manifest validation/manifests/GM12878_hic.json --out data/_hic
python -m validation.sweep --cell GM12878 --search \
    --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/GM12878_search.json

# --- H1ESC ---
python -m validation.dataloader --manifest validation/manifests/H1ESC_hic.json --out data/_hic
python -m validation.sweep --cell H1ESC --search \
    --hic data/_hic/H1ESC/hic.4DNFIHO3CXUQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/H1ESC_search.json

# --- HFFC6 ---
python -m validation.dataloader --manifest validation/manifests/HFFC6_hic.json --out data/_hic
python -m validation.sweep --cell HFFC6 --search \
    --hic data/_hic/HFFC6/hic.4DNFIDKNBPC3.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/HFFC6_search.json
```

Each run is **resumable** (its `--out` JSON caches per (config, region, budget)); re-running
continues where it stopped. Tune `--screen-regions / --search-n / --search-quality / --keep /
--final-n / --final-quality` to trade speed vs rigor. The 15-config grid
(`validation/sweep.py::GRID`) spans EV weight 0.1→2.0, confinement packing {1.0,0.75,0.5},
combos, and two EV-radius probes. To force the CUDA JAX backend cell-wide, edit
`validation/cell_config.py::CANONICAL["simulation_backend"]` (see §0).

> Drop `--search` for the legacy flat sweep (all configs × all regions at one `--quality`/`-n`).

## 3. Also worth running at n=100

```bash
# vs-reference: reference vs python-parity vs python-tuned across several MULTI-IB regions,
# paired per region with a sign-test (tuned should have fewer overlaps in most/all regions).
# Also reports the genome-structure scaling laws (R(s)~s^β, P(s)~s^-α with log-log R² vs the
# literature bands) and — with --hic — Hi-C SCC + MultiMM inverse-distance Pearson (ref vs tuned).
python -m validation.compare_reference -n 100 --n-regions 8 --chroms chr1,chr2,chr8,chr17 \
    --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool --binsize 25000

# Quotable-against-MultiMM number: --multimm-mode fixes the geometry to MultiMM's (≈20 Mb regions,
# 20 kb bins, ~20 kb/bead) so our MultiMM inverse-distance Pearson is directly comparable to their
# ≈0.70 (random <0.40). Coarse beads => tractable even at 20 Mb.
python -m validation.compare_reference --multimm-mode -n 100 --n-regions 8 \
    --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool

# isolate each divergence (overlaps for EV, extent for confinement, scaling for dynamic)
python -m validation.validate --cell GM12878 \
    --region chr1:18288319-20307135 -n 100 --prove all
```

## 4. Objective & reading the output

**Default objective: `--objective overlap`** — minimise median overlap fraction (the physical-
sanity lever) subject to guardrails: Rg must not inflate beyond `--rg-tol` (default **0.30** =
moderate de-compaction — strong EV reduces overlaps mostly by *expanding* an over-compact
structure, which the guard bounds; the grid is gentle-centred {0.25,0.5,1.0,1.5} so the winner is
in the smart-rearrangement regime, not the blow-up regime), plus sane scaling + non-collapsed
diversity. **Hi-C
SCC is deliberately NOT gated** — the n=100 validation showed it's insensitive to EV/confinement
(per-region ΔSCC is a coin flip, ≪ region-to-region variance), so gating on it only injects noise.
EV, meanwhile, reduces overlaps consistently (20/20, 14/14 regions). SCC/Pearson are still printed
as info.

`--objective scc` keeps the old behaviour (max median SCC s.t. overlaps ≤ baseline) if you want it.

The sweep prints a per-config median table (overlap, SCC, Pearson, Rg, scaling, diversity) and
the winner under the objective. Compare winners across the 3 cell lines for a **generalizing**
default; the min-overlap objective gives far more consistent winners than SCC (which chased noise).

> Cross-cell note: the three Hi-C files differ in enzyme/depth, so absolute SCC isn't comparable
> across lines — but the *winning config* should agree if the recommendation generalizes.
