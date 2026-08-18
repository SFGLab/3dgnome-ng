# Validation runbook

Commands for running the validation harness on the CUDA box. Ensembles need n at least 100, the
3dgnome standard, which is why this is a GPU job. Metric definitions are in
[../docs/validation-metrics.md](../docs/validation-metrics.md). The sweep plan is in
[../docs/validation-sweep-plan.md](../docs/validation-sweep-plan.md).

## 0. Install

```bash
pip install -e ".[validation,jax]" "jax[cuda12]"     # validation deps plus the JAX CUDA backend
make -C 3dnome                                        # build the reference binary
```

The reference build is required. It adds the `-r` seed flag the parallel reference ensemble uses in
section 3. The JAX backend is selected per stage through `mc_executor_<stage>`. `batch` routes to
the GPU, `serial` and `threaded` stay on numba. With JAX installed the default `auto` already sends
smooth and estimate_dist to the GPU batch path, so a CUDA box is automatic. To force it explicitly
including arcs, set this in `validation/core/config.py::CANONICAL["simulation_backend"]`.

```python
"simulation_backend": {"ib_workers": 1, "heatmap_chains": 1, "smooth_chains": 1,
                       "mc_executor_smooth": "batch", "mc_executor_estimate_dist": "batch",
                       "mc_executor_arcs": "batch"},
```

## 1. Contact targets

Each cell line uses restriction-enzyme plain Hi-C, which captures CTCF-mediated architecture. RNA
Pol II ChIA-PET and HiChIP are excluded, and the dataloader warns if an accession resolves to Pol
II. Enzyme and depth differ per line, so tune per cell line against its own Hi-C.

| cell line | accession | assay | size |
|---|---|---|---|
| GM12878 | 4DNFIQ32RWCQ | in situ Hi-C, MboI | 153 MB |
| H1ESC | 4DNFIHO3CXUQ | in situ Hi-C, AluI | 297 MB |
| HFFC6 | 4DNFIDKNBPC3 | in situ Hi-C, AluI | 920 MB |

## 2. Excluded-volume and confinement sweep

Settings come from the canonical params, so pass only `--cell`. There is no config file and no
`--data-dir`. Use `--search`, the successive-halving path, not the flat sweep. A flat grid of 15
configs by 20 regions by n=100 at full schedule is about 30000 models and takes days. `--search`
does it in three rungs.

1. Screen. All 15 configs by `--screen-regions` 5 by `--search-n` 30 at `--search-quality` balanced,
   then rank and keep the top `--keep` 4.
2. Expand. Survivors by all `--n-regions` 20 at the search budget, then pick the winner.
3. Validate. Winner plus baseline by all regions at `--final-n` 100 and `--final-quality` full. Only
   this rung pays the full-schedule n=100 cost, on 2 configs.

That is about 7 times cheaper than the flat grid and surfaces the likely winner in the first hour.

```bash
# GM12878
python -m validation fetch --manifest validation/manifests/GM12878_hic.json --out data/_hic
python -m validation sweep --cell GM12878 --search \
    --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/GM12878_search.json

# H1ESC
python -m validation fetch --manifest validation/manifests/H1ESC_hic.json --out data/_hic
python -m validation sweep --cell H1ESC --search \
    --hic data/_hic/H1ESC/4DNFIHO3CXUQ.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/H1ESC_search.json

# HFFC6
python -m validation fetch --manifest validation/manifests/HFFC6_hic.json --out data/_hic
python -m validation sweep --cell HFFC6 --search \
    --hic data/_hic/HFFC6/4DNFIDKNBPC3.mcool \
    --chrom chr1 --n-regions 20 --binsize 10000 --out out/sweep/HFFC6_search.json
```

Each run is resumable. Its `--out` JSON caches per config, region, and budget, so re-running
continues where it stopped. Tune `--screen-regions`, `--search-n`, `--search-quality`, `--keep`,
`--final-n`, and `--final-quality` to trade speed against rigour. The grid in
`validation/studies/sweep.py::GRID` spans
the subordinate excluded-volume range 0.05 to 0.5, all at or below `dist_weight` 1.0 so it stays a
gentle correction, a radius probe, and gentle confinement combinations. All variants run the
canonical config. Drop `--search` for the flat sweep over all configs by all regions at one
`--quality` and `-n`.

## 3. Reference comparison

Reference against python parity against python tuned across several multi-block regions, paired per
region. The overlap win is reported with a sign test and a Wilcoxon signed-rank test. With `--hic`
it also reports Hi-C SCC and the inverse-distance Pearson, and the cross-data correlation.

Cross-data correlation runs at `--cross-data-binsize`, which defaults to 1 Mb, the paper's Fig. 2B resolution. At 1 Mb the raw
log-Pearson is about 0.76 across GM12878 regions, above the paper's 0.67. The observed-over-expected
number is reported alongside as the decay-stripped structure-only agreement. See
[../docs/validation-metrics.md](../docs/validation-metrics.md).

The scaling laws are measured in a separate pass on one large region of at least `--law-mb`, default
20 Mb, the only scale where the fractal-globule power-law window exists. They are not gated on the
small overlap and Hi-C regions, where the reference fails them too. The exponents are
resolution-normalized through `--law-resolution-bp`, default 25000. Both variants are coarse-grained
to a common grid before fitting, so the reference at about 4.5 kb beads and the tuned model at
dynamic 1 kb beads are comparable. Tune `--law-region`, `--law-mb`, and `--law-n`, or skip the pass
with `--no-laws`.

The reference ensemble is parallelised across cores through `--ref-workers`, default auto, which is
`min(n, cpu_count)`. Each worker generates a chunk with a distinct `-r` seed, so the n=100 reference
is about cpu_count times faster and statistically equivalent to the serial run, though not
byte-identical. Use `--ref-workers 1` for the serial path.

```bash
python -m validation compare -n 100 --n-regions 24 --chroms chr1,chr2,chr8,chr17 \
    --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool --binsize 25000 --cross-data-binsize 1000000
```

Isolate each divergence, overlaps for excluded volume, extent for confinement, scaling for dynamic
loop density.

```bash
python -m validation prove --cell GM12878 \
    --region chr1:18288319-20307135 -n 100 --prove all
```

## 4. Reading the sweep output

The default objective is `--objective overlap`, which minimises the median resolution-normalized
overlap `overlap_frac_norm` subject to a guardrail. Rg must not inflate beyond `--rg-tol`, default
0.30. Excluded volume cuts overlaps partly by expanding an over-compact structure, and the guard
bounds that. Hi-C SCC is not gated, because it is insensitive to excluded volume and confinement.
The per-region change in SCC is far smaller than the region-to-region variance. The MultiMM column
is the inverse-distance Pearson, consistent with the compare study. SCC, Pearson, and MultiMM
print as information. `--objective scc` keeps the older behaviour, maximising median SCC subject to
overlaps at or below baseline.

The sweep prints a per-config median table of overlap, SCC, Pearson, Rg, scaling, and diversity,
and the winner under the objective. Compare winners across the three cell lines for a default that
generalises. The three Hi-C files differ in enzyme and depth, so absolute SCC is not comparable
across lines, but the winning config should agree if the recommendation generalises.

## 5. Hi-C self-correlation

Feeds experimental Hi-C into the engine as singleton contacts, then correlates the reconstruction
against held-out Hi-C bin pairs. The held-out split is essential. Correlating against the fed-in
contacts would be inflated by construction. It runs three variants, reference and python parity and
python tuned, so a modest result is diagnosable. The `replace` mode is pure Hi-C-driven and
comparable to MultiMM. The `augment` mode adds Hi-C to the ChIA-PET, with Hi-C counts scaled to the
ChIA-PET singleton median so neither swamps the other.

```bash
python -m validation self-corr --cell GM12878 \
    --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool --hic-singletons replace \
    --n 100 --n-regions 4 --binsize 25000 --out out/sweep/GM12878_selfcorr_replace.json

python -m validation self-corr --cell GM12878 \
    --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool --hic-singletons augment \
    --n 100 --n-regions 4 --binsize 25000 --out out/sweep/GM12878_selfcorr_augment.json
```

The split is a deterministic per-bin-pair hash, 50/50, with `--seed` to vary it. The score is the
inverse-distance Pearson restricted to the test pairs. It also reports self-consistency per
variant against the Hi-C train contacts.

## 6. Synthetic ground-truth reconstruction and noise robustness

The paper's core model validation from supplement sections III and IV. It reconstructs a known
synthetic structure from its own heatmap and measures fidelity with RMSD and the contact measure,
for all three variants. A single noise level runs the ground-truth test. A list of levels runs the
robustness sweep. Both scores are far below the random-structure baseline printed at the end.

```bash
# ground-truth reconstruction, one noise level
python -m validation synthetic --nodes 100 -n 10 --noise 0.0

# robustness sweep across noise levels
python -m validation synthetic --nodes 100 -n 10 \
    --noise 0.0,0.1,0.25,0.5,1.0
```

## 7. Model against Hi-C at 1 Mb

Correlates each variant's reconstructed contact map against experimental Hi-C at 1 Mb, the paper's
Fig. 2B resolution. This is a harder test than the paper's Fig. 2, which compares input data to
Hi-C rather than the model to Hi-C. Scale A, the inter-chromosomal comparison, is unavailable here
because the GM12878 ChIA-PET has no inter-chromosomal contacts.

```bash
python -m validation model-hic --hic data/_hic/GM12878/4DNFIQ32RWCQ.mcool \
    --region chr3:1-30000000 --binsize 1000000 -n 10
```
