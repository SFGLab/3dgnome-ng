# chr1 conformational ensembles on SLURM

GM12878 chr1 with TAD blocks, CTCF anchors, Hi-C singletons, the distance-map terms, dynamic
subanchors and excluded volume. The epigenome terms are off.

## One-time setup on the cluster

`data/` is gitignored, so a fresh checkout has no inputs at all.

**1. Copy the ChIA-PET inputs (~12 MB).** The manifests under `validation/manifests/` cover
Hi-C and epigenomic signal only, so the anchors, loop clusters, segment breakpoints and
centromeres have no download path. This is the one step no script can do for you:

```bash
rsync -av --progress \
    "data/GM12878/GM12878_anchors_3+_oriented.bed" \
    "data/GM12878/GM12878_clusters_3+.bedpe" \
    data/GM12878/ccds_all_hg38_merged100k_GM12878.breakpoints.bed \
    data/GM12878/hg38_centromeres.bed \
    <cluster>:/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/data/GM12878/
```

**2. Run the setup job.** It installs the environment, fetches the Hi-C, calls TADs, builds the
singletons and verifies every path the array job will open.

```bash
sbatch slurm/ensemble/setup.sh
```

Wait for it to finish before submitting the array. Everything it does is idempotent, so rerun
it after a partial failure; `SKIP_INSTALL=1` reruns only the data half.

### Why setup is a job and not a few login-node commands

Installing on the login node fails with

```
RuntimeError: NumPy was built with baseline optimizations: (X86_V2)
              but your machine doesn't support: (X86_V2)
```

The login node's CPU predates the x86-64-v2 baseline that current NumPy wheels are built
against, so NumPy cannot import there at all. That rules the login node out for the data
preparation as well as the install. A GPU compute node is new enough; `setup.sh` prints the
node's microarchitecture level so a repeat of this is readable rather than cryptic.

The install is also deliberately not `pip install -e ".[validation]"`. That extra pulls
`hicrep` and `pyBigWig`, which build from sdists whose `setup.py` imports numpy, and that build
is what died. Neither is needed: `hicrep` is declared in `pyproject.toml` but never imported
anywhere in `validation/`, and `pyBigWig` is imported lazily by the ATAC signal path this
ensemble skips. `scipy`, `cooler` and `cooltools` are the real requirements.

Setup is a separate job rather than a branch inside the array because up to 100 array tasks
share one venv, and concurrent pip installs into a shared prefix corrupt it.

## Submitting

```bash
sbatch --array=0-99%20 slurm/ensemble/chr1_ensemble.sh      # 100 conformations, 20 at a time
PER_TASK=4 sbatch --array=0-24 slurm/ensemble/chr1_ensemble.sh
ROOT=$HOME/3dgnome-ng sbatch --array=0-9 slurm/ensemble/chr1_ensemble.sh
```

`ROOT` defaults to `/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng`. Set it to run from another
checkout, for example a worktree or the homestation box.

A member whose `.cif` exists is skipped, so resubmitting the same array fills in whatever is
missing and a requeued task resumes.

## Sizing

One conformation measured 5 h 09 m at these settings on an RTX 4060 Ti, producing 93,492 beads.
Time splits roughly 58% `estimate_dist`, 39% `smooth`, 2% `arcs`. `--time` is 8 h per task;
raise it with `PER_TASK`.

`estimate_dist` is a dry pass that exists to set the heat targets, not the MC that produces the
structure, and it is the largest single cost. `[subanchor_heatmap] heat_min_reduction` skips it
where the achievable reduction is provably small. It is off by default and untested at this
scale, so it is worth measuring on one conformation before committing an allocation.

## One GPU per task, not one job across many

Conformations are independent, so a fixed pool of GPUs yields the most conformations per hour
when each GPU owns a whole conformation. Splitting one conformation over N GPUs shortens that
conformation without raising throughput, and loses a little to load imbalance.

Ask for several GPUs when latency on a single structure matters. `multigpu_mode=groups` in the
ini then runs whole batch groups side by side, projected 7.96x on 8 GPUs from the chr1 profile,
and it costs nothing on one GPU. See the multi-GPU entry in AGENTS.md.

## Config

`gm12878_chr1_hic_tads.ini` is generated from `validation.core.config.CANONICAL` plus the TAD,
Hi-C-singleton and multi-GPU overrides. Change the canonical params there rather than editing
the ini, so the validation harness and the cluster runs cannot drift apart.
