# Trio ingestion

Turns the private Drive folder of trio ChIA-PET into the per sample layout the modelling
pipeline already reads. Three families, each a father, a mother and a child.

| family | father | mother | child |
|---|---|---|---|
| CHS, Han Chinese | HG00512 | HG00513 | HG00514 |
| PUR, Puerto Rican | HG00731 | HG00732 | HG00733 |
| YRI, Yoruba | GM19239 | GM19238 | GM19240 |

## Sequence

```bash
# 1. list the Drive folder, once. Opens a consent URL, caches a token under ~/.config/3dgnome
python playground/trio/gdrive_inventory.py <folder-url> --json trio_inventory.json

# 2. download the modelling subset, about 5 GB of the folder's 34 GB
python playground/trio/trio_fetch.py --inventory trio_inventory.json --dry-run
python playground/trio/trio_fetch.py --inventory trio_inventory.json

# 3. loops, anchors with orientation, mcool
python playground/trio/trio_prepare.py

# 4. TAD boundaries, then blocks and segments from them
python -m validation tracks --cell HG00512 --skip-compartments --skip-signal
python playground/trio/trio_segments.py --samples HG00512

# 5. contact singletons at 25 kb. Balancing is required first and is not done by hic2cool
cooler balance -p 8 data/_hic/HG00512/HG00512.mcool::/resolutions/25000
python slurm/ensemble/prep_singletons.py \
    --mcool data/_hic/HG00512/HG00512.mcool --chroms chr1-chr22,chrX \
    --binsize 25000 --out data/HG00512/HG00512_hic_25kb_singletons.bedpe

# 6. config, from the same CANONICAL the three cell lines use
python playground/trio/trio_configs.py --samples HG00512
```


## Running on the cluster

The anchors and clusters are built on the laptop, because the motif track they need matches
`playground/**/*.gz` in .gitignore and does not travel. Everything heavier runs on eden.

```bash
# on the laptop, once per sample
python playground/trio/trio_fetch.py --inventory trio_inventory.json
python playground/trio/trio_prepare.py --skip-hic

# copy the text inputs and the raw contact file
for S in HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240; do
  rsync -av "data/$S/" "eden:/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/data/$S/"
  rsync -av "data/_trio/$S/${S}_allres.hic" \
        "eden:/mnt/evafs/groups/sfglab/nkozlov/3dgnome-ng/data/_trio/$S/"
done

# on eden, once per sample. Idempotent, so rerun after any failure
mkdir -p slurm/ensemble/logs
for S in HG00512 HG00513 HG00514 HG00731 HG00732 HG00733 GM19238 GM19239 GM19240; do
  sbatch slurm/ensemble/trio_setup.sh $S
done

# one chromosome across all nine samples, 90 tasks
CHROMS=chr1 sbatch --array=0-89%6 slurm/ensemble/trio_ensemble.sh
```

### Resuming

Nothing here redoes finished work. `trio_fetch` skips a file whose md5 matches Drive's,
`trio_prepare` skips a sample whose clusters and anchors are present, `prep_singletons` skips an
existing output, and `trio_ensemble.sh` exits before requesting GPU work when every member of
its block already has a non empty `.cif`. Text outputs are written through a temporary file and
moved into place, so an interrupted run cannot leave a half written input that would still
parse.

To see where things stand and get back exactly the tasks that are missing:

```bash
python playground/trio/trio_status.py
python playground/trio/trio_status.py --resubmit     # just the --array spec
sbatch --array=$(python playground/trio/trio_status.py --resubmit)%6 slurm/ensemble/trio_ensemble.sh
```

### Sizing a job

The `long` partition allows 5 days, so wall clock is not the binding constraint and PER_TASK is
free to be large. Two things bound it instead.

Total time does not change with PER_TASK. Nine samples by 100 conformations on one chromosome is
900 conformations however they are grouped, and at a throttle of `%6` that is 900 times the per
conformation time divided by 6 either way. PER_TASK only trades job count against how easily a
job is scheduled, since a short job backfills into gaps a long one cannot.

Overrunning is cheap. gnome3d.cli writes each structure as it finishes, so a job killed at the
wall keeps everything already done and loses only the one in flight. `trio_status.py --resubmit`
then asks for exactly the gap.

Per conformation time scales with a chromosome's anchor count. GM12878 chr1 measured 17.6
minutes at 26,603 anchors, and the depth matched trio samples hold 21,688 to 32,243 there, so
chr1 lands near 15 to 21 minutes if the contact singletons prune anchors the way GM12878's 4DN
Hi-C did. This contact map is far denser and will prune less, so treat 60 minutes as the
pessimistic end until one run measures it.

chr1 carries about 11% of a sample's anchors, so one PER_TASK for every chromosome leaves most
of the limit unused on the small ones. Submission is per chromosome anyway, so vary it:

```bash
CHROMS=chr1  PER_TASK=10 sbatch --array=0-89%6 --time=24:00:00 slurm/ensemble/trio_ensemble.sh
CHROMS=chr21 PER_TASK=50 sbatch --array=0-17%6 --time=24:00:00 slurm/ensemble/trio_ensemble.sh
```

`--time` on the command line overrides the script's directive.

### Why chromosome first

One array task is one chromosome by one sample by a block of conformations, and the chromosome
is the slowest varying dimension. The first 9 times CHUNKS tasks therefore cover one chromosome
across all nine samples, so a full trio comparison on that chromosome is possible while the rest
of the genome is still queued. Sample first ordering would finish one individual before starting
the next, and a comparison needs all nine.

The mapping lives in `trio_samples.py::shard`, and both the job script and `trio_status.py` call
it, so the array index cannot mean one thing to the scheduler and another to the progress report.

Nine samples by 23 chromosomes by 10 chunks is 2070 tasks, and Slurm's default MaxArraySize is
1001. Check with `scontrol show config | grep MaxArraySize`. Submitting one chromosome at a time
with `CHROMS=chrN` is 90 tasks and sidesteps the limit entirely, which is also the ordering that
returns preliminary results soonest.

## Choices that are not obvious from the code

**The loop input is the CTCF site filtered set, not all PET3+ loops.** The providers' `wyniki_*`
files keep PET3+ loops whose anchors both overlap a CTCF binding site from the family's peak
union. That is 369k anchors for HG00512 against 1.41M for the unfiltered set. The unfiltered set
is 5.8 times GM12878 and would cost roughly 98 days of six GPUs for nine samples at 100
conformations. The filter is also the right one for a model driven by CTCF.

**Anchors are derived from the loops, not from the peak files.** The peak files carry no strand.
The anchor set is the distinct anchor intervals of the loop file, which is how `GM12878_anchors_3+_oriented.bed`
relates to `GM12878_clusters_3+.bedpe` in this repo, verified line for line.

**Orientation comes from JASPAR MA0139.1 hits, plus strand to R and minus strand to L.** The
mapping was recovered from the GM12878 anchors rather than assumed. Rerun the check with
`python playground/trio/trio_orient.py --validate data/GM12878/GM12878_anchors_3+_oriented.bed`.
It assigns L or R to 77% of anchors where GM12878 has 93%, because the bundled motif track is
cut at score 800. The flip rate among assigned anchors is 1.4%. Anchors with no motif become N
and carry no orientation energy, so the shortfall is conservative rather than wrong, and it
applies identically to all nine samples. A looser scan against `playground/hg38.fa` would close
the gap.

**Blocks and segments come from one insulation call at two scales.** Blocks are TAD boundaries
thinned at 100 kb, which only merges TADs smaller than that. Segments are the same boundaries
thinned at 2 Mb. Segments must be far coarser than blocks, because under the default
`refine_scope = segment` a segment holding one block or fewer is skipped and nothing is
reported. On GM12878 the 2 Mb thinning gives 1292 boundaries against the 1298 of the CCDS
breakpoints file the existing cell lines use. `trio_segments.py` refuses to write files below
two blocks per segment.

**The contact map is not independent of the loops.** `hic_files/*.hic` are built by
`juicer_tools pre` from each sample's own ChIA-PET pairs, so TAD blocks and singletons derive
from the same assay the model fits. The three cell line arm used 4DN Hi-C, which was
independent. Expect weaker singleton pruning here, and read any contact agreement accordingly.

**Chromosomes are renamed during conversion.** The `.hic` files name chromosomes `1`, `2`, `3`
and carry an `ALL` pseudo chromosome, while every other input says `chr1`. Left alone that
mismatch yields empty results rather than an error.


## Measured, depth matched CTCF arm

Every sample is drawn down to its family's minimum loop count by `trio_downsample.py`, so a
parent against child comparison is not also a density comparison.

| sample | role | family | loops | anchors | on own peaks |
|---|---|---|---|---|---|
| HG00512 | father | CHS | 131,656 | 260,392 | 92.4% |
| HG00513 | mother | CHS | 131,656 | 256,104 | 89.7% |
| HG00514 | child | CHS | 131,656 | 258,740 | 94.0% |
| HG00731 | father | PUR | 170,440 | 330,436 | 92.4% |
| HG00732 | mother | PUR | 170,440 | 333,833 | 93.0% |
| HG00733 | child | PUR | 170,440 | 329,316 | 87.8% |
| GM19239 | father | YRI | 101,564 | 201,465 | 91.5% |
| GM19238 | mother | YRI | 101,564 | 200,131 | 91.7% |
| GM19240 | child | YRI | 101,564 | 199,850 | 87.9% |

Within family anchor spread is 1.017 for CHS, 1.014 for PUR and 1.008 for YRI. Between families
nothing is matched, CHS and PUR sitting near 260k and 331k against YRI's 200k, so a cross family
claim needs its own treatment. Orientation is stable across samples, N between 25.7% and 28.1%,
and L against R balanced to within 1% everywhere.

Total is 2.37M anchors. Scaled by GM12878's measured 2.7 GPU hours per genome conformation at
243,848 anchors, that is about 26 GPU hours for one conformation of all nine, so roughly 2600
GPU hours for 100 each, or about 18 days at six concurrent GPUs. Treat it as a floor. The
contact map here is far denser than GM12878's 4DN Hi-C, 1.54M chr1 singleton pairs against 232k,
so the Hi-C singleton pruning that cut GM12878's chr1 anchors from 26,603 to 9,125 will prune
much less. One chr1 run replaces this estimate with a measurement.

## Open items

**The PUR reference naming is settled, it is an artifact.** CHS names `HG00514` and YRI names
`GM19240`, both the child of their family, but PUR files read
`wyniki_HG00731.clean.PETs.GM12878.repmerged...piki_Puerto`. If the CTCF site union had really
come from GM12878, PUR anchors would sit on their own called peaks far less often than the other
families do. They do not. PUR reads 92.4%, 92.9% and 87.8% against 89.7% to 94.0% for CHS and
91.5% to 91.7% for YRI. The filename is a template that was not updated.

**Depth is matched within two families of three, and the gap is one sample.** Everything here
comes from `downsampling/`. The high quality set is a strict subset of `subsample_1`, verified
on HG00512 where all 187466 of its loops appear among the 724112 subsampled ones and none
outside, so the CTCF filter does not bypass the subsampling. The filter is also even handed,
keeping between 19.6% and 25.9% of loops across all nine.

The imbalance is in `subsample_1` itself and it is concentrated in one place.

| family | subsample_1 loops | spread |
|---|---|---|
| YRI | 485307, 481465, 490630 | 1.02 |
| PUR | 791259, 884349, 807730 | 1.12 |
| CHS | 724112, 673007, 1084950 | 1.61 |

YRI shows the procedure works when it works. CHS does not, because HG00514 was not brought down
to its family's level. Between families there is no matching at all, YRI sitting near 486k
against about 830k for the other two, which is consistent with the stripes README describing a
within family and a between family undersampling as separate strategies and only the within
family one having been applied to the loop files. Only `subsample_1` exists as a loop file, so
there is no alternative arm to switch to.

**Three peak lines carry float coordinates.** Two in HG00514 and one in GM19240 are written like
`1.51e+08`, which lands about 6 kb off the true position. They are coerced and counted rather
than dropped. Peaks feed only the quality check, not the model.
