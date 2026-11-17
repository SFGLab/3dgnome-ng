# Genomes on the new production config, and 3dgnome against MultiMM and cudaMMC

Production changed on 2026-11-16 (commit 9cd217a, configs regenerated in ac98231):
`[excluded_volume] arcs_repulsion = no`, `auto_factor_arcs = 1.0`, `[springs] gyration_weight = 0.1`,
`[simulation_arcs] start = maxent`, `[relax] use_cross_block_relax = yes`. `design/gap-fill.md`.

## Genomes on eden

Ten conformations per genome, autosomes chr1 to chr22, the same arms as the cross line run of
2026-11-12: cell lines GM12878, H1ESC and HFFC6 on arms C, R and M, and the nine trio samples on
CTCF and on CTCF with RNAPOL2, HGSVC singletons.

Staged on eden at `/mnt/evafs/groups/sfglab/nkozlov/3dgnome-lr2-run`, a copy of commit 860a847
(`STAGED_COMMIT`), production with the within block relaxation, without `data/`, with `data` and
`.venv` linked to `../3dgnome-ng`. Trio configs were generated in the eden triosformer checkout
with the staged engine, `TRIOS_GNOME3D=<run dir> pipeline/trio_configs.py --factor
{CTCF,RNAPOL2} --singletons hgsvc --scope genome --out-dir slurm/configs_lr2`, and copied into
`slurm/configs/` as `<sample>_trio{,_rnapol2}_hgsvc_genome_lr2.ini`.

Superseded and left for the user to delete: `3dgnome-lr-run` (commit ac98231, before the within
block relaxation), and in the triosformer checkout `slurm/configs_lr/`, `slurm/configs_lrc/`,
`slurm/configs/*_genome_lr.ini` (read the old production config, `GNOME3D` set instead of
`TRIOS_GNOME3D`) and `slurm/configs/*_genome_lrc.ini`.

Dry runs of tasks 0 and 197 of the cell array and 197 of both trio arrays map and pass their
guards. Submit from the run directory:

```bash
cd /mnt/evafs/groups/sfglab/nkozlov/3dgnome-lr2-run
A="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22"
T=/mnt/evafs/groups/sfglab/nkozlov/triosformer
ROOT=$PWD OUT=$PWD/out/lr SPLIT=1 CHROMS="$A" sbatch --array=0-197%10 slurm/ensemble/crosscell_arms.sh
ROOT=$PWD TRIOS_ROOT=$T PYTHONPATH=$PWD CONFIG_TAG=_trio_hgsvc_genome_lr2 OUT=$PWD/out/lr_trio_ctcf \
  CHROMS="${A// /,}" sbatch -A sfglab -p long --array=0-197%8 $T/slurm/trio_ensemble.sh
ROOT=$PWD TRIOS_ROOT=$T PYTHONPATH=$PWD CONFIG_TAG=_trio_rnapol2_hgsvc_genome_lr2 OUT=$PWD/out/lr_trio_rnapol2 \
  CHROMS="${A// /,}" sbatch -A sfglab -p long --array=0-197%8 $T/slurm/trio_ensemble.sh
```

The trio script sets no account and the user's default association on eden, `null`, has a submit
limit of zero, so the trio arrays need `-A sfglab`; without it sbatch refuses them with
`AssocMaxSubmitJobLimit`. Submitted on 2026-11-16: cell array 1943267, trio CTCF 1943269, trio
CTCF and RNAPOL2 1943270.

Cell array task i runs line and arm i % 9 on chromosome i / 9; trio task i runs one chromosome of
one sample, the chromosome slowest. A member whose cif exists is skipped, so resubmitting fills
gaps. `PYTHONPATH` puts the staged engine before the shared venv's editable install.

Host memory, read on 2026-11-17. The maxent anchor fit holds dense anchor by anchor arrays, and
a chr1 task of array 1943270 peaked at 53.7 GB at 23,080 anchors. Tasks 5 and 6, HG00733 and
GM19239 on chr1 at 25,660 and 25,409 anchors, were stopped by the out of memory killer at the
script's 64 GB. Every other chr1 task of both arrays is at 23,080 anchors or fewer. The two
tasks resubmit with more memory from the same directory, with `A` and `T` as above:

```bash
ROOT=$PWD TRIOS_ROOT=$T PYTHONPATH=$PWD CONFIG_TAG=_trio_rnapol2_hgsvc_genome_lr2 OUT=$PWD/out/lr_trio_rnapol2 \
  CHROMS="${A// /,}" sbatch -A sfglab -p long --mem=96G --array=5,6 $T/slurm/trio_ensemble.sh
```

## 3dgnome against MultiMM and cudaMMC, on frodo

Same inputs for the three tools, MultiMM at our bead count, cudaMMC with its own defaults.

| Axis | Measure |
|---|---|
| Performance | wall time per structure and per ensemble on the RTX 3090, at matched beads |
| Physics | the battery's overlaps within and across blocks, Rg against the law, bond lengths, the distance exponent and its two bands |
| Hi-C | pooled ensemble SCC and Spearman, raw and balanced, over mapped bins, and MultiMM's own metric |

Scales: three 5 Mb regions and chr1 1 to 60 Mb per cell line, and one whole genome per tool for
the timing. Results under `playground/square_blocks/tools/` and a write up in `design/`.
