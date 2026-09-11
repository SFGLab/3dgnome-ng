# Conformational ensembles on eden

TAD blocks, CTCF anchors, Hi-C singletons, the distance-map terms, dynamic subanchors and
excluded volume. Epigenome terms off. Shape bucketing and `heat_min_reduction` on.

Two entry points:

| what | setup | run |
|---|---|---|
| chr1, GM12878 (the pilot) | `setup.sh` | `chr1_ensemble.sh` |
| whole genome, any of the three lines | `setup_cell.sh <CELL>` | `genome_ensemble.sh` |

Configs are generated, not hand-written. `make_configs.py` builds one `.ini` per cell line from
`validation.core.config.CANONICAL`, so the cluster runs and the validation harness cannot drift
apart. Change CANONICAL and regenerate.

## One-time setup per cell line

**1. Copy the ChIA-PET inputs (~12 MB per line).** The manifests under `validation/manifests/`
cover Hi-C and epigenomic signal only, so the anchors, loop clusters, segment breakpoints and
centromeres have no download path. This is the one step no script can do for you:

```bash
for C in GM12878 H1ESC HFFC6; do
  rsync -av --progress \
      "data/$C/${C}_anchors_3+_oriented.bed" \
      "data/$C/${C}_clusters_3+.bedpe" \
      "data/$C/ccds_all_hg38_merged100k_${C}.breakpoints.bed" \
      data/$C/hg38_centromeres.bed \
      eden:/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/data/$C/
done
```

**2. Run the setup job per line.** Installs the environment, fetches the Hi-C, calls TADs and
builds the genome-wide Hi-C singletons, then verifies every path the array will open.

```bash
mkdir -p slurm/ensemble/logs          # once, before the first submit
sbatch slurm/ensemble/setup_cell.sh GM12878
SKIP_INSTALL=1 sbatch slurm/ensemble/setup_cell.sh H1ESC
SKIP_INSTALL=1 sbatch slurm/ensemble/setup_cell.sh HFFC6
```

The three are independent and can run at once; only the first needs the install. Everything is
idempotent, so rerun after a partial failure. The singleton step is the long one, roughly 20-40
minutes for a genome at 25 kb, because it reads a dense contact matrix per chromosome.

## Running

```bash
CELL=GM12878 sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh
CELL=H1ESC   sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh
CELL=HFFC6   sbatch --array=0-229%6 slurm/ensemble/genome_ensemble.sh
```

One array task is one chromosome by a block of `PER_TASK` conformations, so the array length is
`n_chroms * ceil(N_MODELS / PER_TASK)`. With the defaults (`N_MODELS=100`, `PER_TASK=10`, 23
chromosomes) that is **230**. The script prints the length it expects at startup and refuses a
task past the end, naming the right `--array`.

Fewer conformations, or a different shard size:

```bash
N_MODELS=20 PER_TASK=5 CELL=H1ESC sbatch --array=0-91%6 slurm/ensemble/genome_ensemble.sh
```

Check the mapping without an allocation:

```bash
DRY_RUN=1 CELL=GM12878 SLURM_ARRAY_TASK_ID=57 bash slurm/ensemble/genome_ensemble.sh
```

Output lands in `out/<cell>_genome/<chrom>/<chrom>_s<N>.cif`. A member whose `.cif` exists is
skipped, so resubmitting the same array fills gaps and a requeued task resumes.

## Cost

Measured: one chr1 GM12878 conformation is about 17 minutes on an A100 at 93,492 beads. Scaling
by anchor count gives roughly

| cell line | anchors | per conformation | 100 conformations |
|---|---|---|---|
| GM12878 | 243,848 | ~2.6 GPU-h | ~260 GPU-h |
| H1ESC | 150,158 | ~1.6 GPU-h | ~160 GPU-h |
| HFFC6 | 121,402 | ~1.3 GPU-h | ~130 GPU-h |
| **all three** | | **~5.5 GPU-h** | **~550 GPU-h** |

At 6 concurrent GPUs that is about **3.8 days** for 100 conformations of all three lines, or
about 18 hours for 20 each. Sharding does not change the total, only how it packs; raise the
`%N` throttle to whatever the cluster will give you, since `sacctmgr` shows no per-user cap.

## Per chromosome, not one genome job

Each task reconstructs one chromosome. The downstream enhancer3D analysis is intra-chromosomal
(enhancer-promoter distances within a chromosome) and its published models are per-chromosome,
so nothing consumes inter-chromosomal placement. Sharding this way turns one ~2.6 h genome run
into 23 short independent tasks that backfill into free GPUs, and a failure costs one chromosome
rather than a whole genome.

The cost is that chromosomes are placed independently, with no chromosome-level MC between them,
so the output is not a single coherent nucleus. If you need that, run one task with
`--region ""` instead; the total GPU time is the same.

## Install note

The setup job deliberately avoids `pip install -e ".[validation]"`. That extra pulls `hicrep` and
`pyBigWig`, which build from sdists whose `setup.py` imports numpy, and that build fails. Neither
is needed: `hicrep` is declared in `pyproject.toml` but never imported anywhere in `validation/`,
and `pyBigWig` is used only by the ATAC signal path, which these runs skip. `scipy`, `cooler` and
`cooltools` are the real requirements.

Setup is a job rather than login-node commands because eden's login node CPU is below the
x86-64-v2 baseline current NumPy wheels need, so NumPy cannot import there at all. A dgx node is
fine; `setup_cell.sh` prints the node's microarchitecture level so a repeat is readable.
