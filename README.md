# 3D-GNOMEᴺᴳ

3D-GNOMEᴺᴳ predicts the 3D structure of chromosomes from ChIA-PET loops and Hi-C contacts. It returns an ensemble of
bead chains as mmCIF files, one bead per kilobase in the production configuration. A run covers one region, one
chromosome or the whole genome.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Three extras add optional parts.

| Extra | Content |
|-------|---------|
| `jax` | The GPU kernels. This extra installs JAX for the CPU. Add the build for your hardware, as in `pip install -e ".[jax]" "jax[cuda12]"`. |
| `extract` | cooler, for `gnome3d-ng extract`. |
| `dev` | ruff, pyright and pytest. |

## Run

```bash
gnome3d-ng --config slurm/ensemble/gm12878_hic_arcs.ini --region chr1:18288319-20307135 -n 10 --out out/
```

`--region` takes one region `chr1:18288319-20307135`, one chromosome `chr14`, a list `chr1,chr3,chrX` or a range
`chr1-chr22,chrX`. Without it the run covers chr1 to chr22 and chrX. `-n` is the number of structures. `--data-dir`
replaces `data_dir` of the config.

The run writes one mmCIF file and one bead table per structure, `chr1_18288319_20307135_s1.cif` and
`chr1_18288319_20307135_s1.beads.tsv` for the first structure of the command above. ChimeraX opens the CIF as one
chain. Loop anchors have the residue name ALA and the beads between them GLY.

`--members 0-9` selects structures by index in place of `-n`. Several jobs with disjoint sets produce the same ensemble
as one job. A job that starts again skips the structures that are already written.

The same run from Python:

```python
from gnome3d.io import write_cif
from gnome3d.simulate import run_region

structures = run_region(
    config_path="slurm/ensemble/gm12878_hic_arcs.ini",
    region="chr1:18288319-20307135",
    n_structures=10,
    data_dir="data/GM12878",
)
for i, beads in enumerate(structures, start=1):
    write_cif(f"structure_{i}.cif", beads, entry_id=f"structure_{i}")
```

A bead is a tuple `(start, end, x, y, z, kind)`. `run_genome` takes the same region forms as the command and returns
one dict of chromosome to bead list per structure.

## Configuration

A config is an ini file. [SETTINGS.md](SETTINGS.md) lists every key with its default, its production value and its
meaning. It also states the distance law. The production values are `CANONICAL` in
[gnome3d/production.py](gnome3d/production.py). The configs under `slurm/ensemble/` are generated from it by
`slurm/ensemble/make_configs.py`, one per cell line and input set.

## Input files

The `[data]` section of the config points to the input files. The repository does not hold them.

| Key | Format | Content |
|-----|--------|---------|
| `anchors` | BED, `chr start end orientation` | Loop anchors with the orientation of the CTCF motif. |
| `clusters` | BEDPE, `chr1 s1 e1 chr2 s2 e2 score` | Loops as PET clusters. Several files give several loop factors, CTCF first. |
| `singletons` | BEDPE | Contacts for the exponent fit and the coarse heatmaps. ChIA-PET singletons or Hi-C bin pairs. |
| `segment_split` | BED | Segment breakpoints. |

`gnome3d-ng extract` writes the bin pairs of one region of an mcool file in the format of `singletons`.

```bash
gnome3d-ng extract --mcool sample.mcool --region chr1:11500000-12600000 --out region.bedpe
```

The optional inputs are a compartment track, an anchor activity track, an anchor contact table and a contact map for
the stretches without loops. SETTINGS.md describes them.

## Tests

```bash
.venv/bin/pytest                # every test, about two and a half minutes on a laptop
.venv/bin/pytest -m "not slow"  # without the files that take over a minute
```

`tests/unit/` has one file per feature. `tests/integration/` runs the whole pipeline on synthetic inputs and on one
GM12878 region when its inputs are under `data/`.

## Repository

| Folder | Content |
|--------|---------|
| `gnome3d/` | The model. |
| `tests/` | The test suite. |
| `web/` | The website. |
| `slurm/ensemble/` | Run scripts and configs for a cluster and a workstation. |
| `playground/` | Research scripts, the validation battery and figures. |
| `design/` | Design notes. [design/features.md](design/features.md) has every feature with its measurements. |
| `legacy/` | Retired code, with 3D-GNOME under `legacy/3dnome/`. See [legacy/README.md](legacy/README.md). |

[AGENTS.md](AGENTS.md) describes the model and the rules for a change.

## Acknowledgements

3D-GNOMEᴺᴳ builds on the 3D-GNOME family of tools and on the work of their authors. Please cite these papers when you
use it in academic work.

- Tang Z., Luo O. J., Li X., Zheng M., Zhu J. J., Szalaj P., Trzaskoma P., Magalska A., Wlodarczyk J., Ruszczycki B.,
  Michalski P., Piecuch E., Wang P., Wang D., Tian S. Z., Penrad-Mobayed M., Sachs L. M., Ruan X., Wei C.-L., Liu E. T.,
  Wilczynski G. M., Plewczynski D., Li G., Ruan Y. (2015). **CTCF-Mediated Human 3D Genome Architecture Reveals
  Chromatin Topology for Transcription.** *Cell.*
  [doi:10.1016/j.cell.2015.11.024](https://doi.org/10.1016/j.cell.2015.11.024)

- Szałaj P., Tang Z., Michalski P., Pietal M. J., Luo O. J., Sadowski M., Li X., Radew K., Ruan Y., Plewczynski D.
  (2016). **An integrated 3-Dimensional Genome Modeling Engine for data-driven simulation of spatial genome
  organization.** *Genome Research.* [doi:10.1101/gr.205062.116](https://doi.org/10.1101/gr.205062.116)

- Wlasnowolski M., Sadowski M., Czarnota T., Jodkowska K., Szalaj P., Tang Z., Ruan Y., Plewczynski D. (2020).
  **3D-GNOME 2.0: a three-dimensional genome modeling engine for predicting structural variation-driven alterations of
  chromatin spatial structure in the human genome.** *Nucleic Acids Research,* 48(W1), W170–W176.
  [doi:10.1093/nar/gkaa388](https://doi.org/10.1093/nar/gkaa388)

- Wlasnowolski M., Kadlof M., Sengupta K., Plewczynski D. (2023). **3D-GNOME 3.0: a three-dimensional genome modelling
  engine for analysing changes of promoter-enhancer contacts in the human genome.** *Nucleic Acids Research,* 51(W1),
  W5–W10. [doi:10.1093/nar/gkad354](https://doi.org/10.1093/nar/gkad354)

- Wlasnowolski M., Grabowski P., Roszczyk D., Kaczmarski K., Plewczynski D. (2023). **cudaMMC: GPU-enhanced multiscale
  Monte Carlo chromatin 3D modelling.** *Bioinformatics.*
  [doi:10.1093/bioinformatics/btad588](https://doi.org/10.1093/bioinformatics/btad588)
