# 3dgnome-ng

Python reimplementation of the [3dgnome](https://bitbucket.org/3dome/3dgnome/src/master/a) Monte Carlo chromosome structure prediction algorithm.

The reference C++ implementation lives in `3dnome/` (read-only). The rewrite lives in `src/`.

---

## Requirements

- Python 3.10+
- NumPy >= 1.24
- Numba >= 0.59
- g++ (for building the C++ reference binary and the test scorer)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Install with dev tooling (ruff + pyright)

```bash
pip install -e ".[dev]"
```

Alternatively, runtime deps alone can be installed via `pip install -r requirements.txt`.

---

## Building the C++ reference

The C++ binary is used only for the integration test (comparing distributions). Build it once:

```bash
make 3dnome        # builds 3dnome/3dnome
make scorer        # builds harness/scorer (unit-test scorer)
make               # builds both
```

---

## Running the Python reimplementation

### Quick start - single region

```python
from src.simulate import run_region

region = "chr1:18288319-20307135" 
entry_id = region.replace(':', '_').replace('-', '_')

structures = run_region(
    config_path="data/GM12878/config.ini",
    region=region,
    n_structures=1,
    data_dir="data/GM12878",   # override the absolute path baked into config.ini
)

# Each structure is a list of (midpoint_bp, x, y, z) tuples
for i, s in enumerate(structures):
    print(f"structure {i+1}: {len(s)} beads, "
          f"first bead at bp={s[0][0]}, pos=({s[0][1]:.3f}, {s[0][2]:.3f}, {s[0][3]:.3f})")

# Save each structure as its own mmCIF file
from src.io import write_cif
for i, s in enumerate(structures, start=1):
    write_cif(f"{entry_id}_structure_{i}.cif", s, entry_id=f"{entry_id}_s{i}")
```

Each CIF file can be opened directly in **ChimeraX** or **Chimera**:

```bash
chimerax chr1_18288319_20307135_structure_1.cif
```

Beads are written as sequential ALA residues on chain A - ChimeraX connects them as a polymer chain automatically.

`data_dir` overrides the `data_dir` key in the config, which is useful because the bundled `config.ini` has it hardcoded to `/Projects/GM12878/`. Pass the actual local path instead.

### Fast run (lower quality, useful for testing)

Use a config with relaxed MC settings:

```ini
[simulation_arcs]
delta_temp = 0.995
stop_condition_steps = 1000
```

Or pass the `--fast` flag to the integration harness (see below).

---

## Data format

Input files live under `data/<cell_line>/` and are referenced by the config:

| File | Format | Purpose |
|------|--------|---------|
| `<cell_line>_anchors_3+_oriented.bed` | BED (chr start end orientation) | ChIA-PET loop anchors |
| `<cell_line>_clusters_3+.bedpe` | BEDPE (chr1 s1 e1 chr2 s2 e2 score) | PET cluster arcs |
| `<cell_line>_singletons_lessthan3.bedpe` | BEDPE | Singleton contacts for segment-level heatmap |
| `ccds_all_hg38_merged100k_<cell_line>.breakpoints.bed` | BED | Segment boundary breakpoints |

The region string uses `chr:start-end` format (colon + dash):

```
chr1:18288319-20307135
```

## Algorithm overview

3dgnome uses a **coarse-to-fine hierarchical Monte Carlo** approach:

```
Level 1 (Chromosome root)  - whole chromosome, singleton heatmap energy
Level 2 (Segment)          - ~100 kb–1 Mb segments, singleton heatmap energy
Level 3 (Anchor/IB)        - ChIA-PET loop anchors, arc spring energy
Level 4 (Subanchor)        - chain + angle MC on loop_density subanchors inserted between each anchor pair
```

Each level runs simulated annealing, then passes 3D positions down as starting points for the next level. The `-v 2` pipeline covers all four levels: Level 4 always runs as the last step inside Level 3's IB reconstruction (`reconstruct_arcs` / C++ `reconstructClustersArcsDistances`). Both C++ and Python produce `n_anchors + (n_anchors−1) × loop_density` beads per IB.

### Metropolis criterion

```
accept if score_new <= score_old
      or rand() < jump_scale * exp(-jump_coef * (score_new / score_old) / T)
```

Note: `jump_scale` defaults to 50, so the acceptance probability can exceed 1 (always accepted in that case).

### Key non-obvious details

- **Angle metric is NOT acos**: `angle(v1, v2) = 1 - (dot(norm(v1), norm(v2)) + 1) / 2`
- **Heatmap score double-counts**: every pair (i, j) contributes twice; MC update is `score += 2 * (local_new - local_old)`
- **Arc score does not double-count**: global score sums i < j once; MC update is `score = score - local_old + local_new`
- **Random displacement is a uniform cube**: each component drawn from `Uniform(-step, step)`, not a sphere or Gaussian
- **Anchor-level step size is hardcoded**: `noise_size_small = 0.005` regardless of config `noise_arcs`
