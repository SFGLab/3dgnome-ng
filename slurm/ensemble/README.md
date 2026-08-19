# chr1 conformational ensembles on SLURM

GM12878 chr1 with TAD blocks, CTCF anchors, Hi-C singletons, the distance-map terms, dynamic
subanchors and excluded volume. The epigenome terms are off.

## One-time setup on the cluster

`data/` is gitignored, so a fresh checkout has neither the Hi-C nor the derived tracks.

```bash
python -m validation fetch --manifest validation/manifests/GM12878_hic.json --out data/_hic
python -m validation tracks --cell GM12878          # writes GM12878_tads.bed
python slurm/ensemble/prep_singletons.py \
    --mcool data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \
    --region chr1 --binsize 25000 \
    --out data/GM12878/GM12878_chr1_hic_25kb_singletons.bedpe
```

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
