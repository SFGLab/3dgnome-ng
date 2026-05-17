# 3dgnome-torch

Python/PyTorch reimplementation of the [3dgnome](https://github.com/SFGLab/3dgnome) Monte Carlo chromosome structure prediction algorithm.

The reference C++ implementation lives in `3dnome/` (read-only). The rewrite lives in `src/`.

---

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- NumPy >= 1.24
- g++ (for building the C++ reference binary and the test scorer)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## GPU acceleration

The MC loops use vectorized PyTorch tensor operations and automatically select the best available device:

```
CUDA  →  NVIDIA GPU (fastest)
MPS   →  Apple Silicon GPU
CPU   →  fallback (still faster than pure NumPy due to vectorization)
```

The device is printed once at the start of each run:

```
[simulate] device: mps
```

No code changes are needed to enable GPU — PyTorch finds the device automatically.

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

### Quick start — single region

```python
from src.simulate import run_region

structures = run_region(
    config_path="data/GM12878/config.ini",
    region="chr1:18288319-20307135",
    n_structures=5,
    data_dir="data/GM12878",   # override the absolute path baked into config.ini
)

# Each structure is a list of (midpoint_bp, x, y, z) tuples
for i, s in enumerate(structures):
    print(f"structure {i+1}: {len(s)} beads, "
          f"first bead at bp={s[0][0]}, pos=({s[0][1]:.3f}, {s[0][2]:.3f}, {s[0][3]:.3f})")

# Save all structures as a multi-model mmCIF file
from src.io import write_cif
write_cif("chr1_structures.cif", structures, entry_id="chr1_18288319_20307135")
```

The CIF file can be opened directly in **ChimeraX** or **Chimera**:

```bash
chimerax chr1_structures.cif
```

Each structure is stored as a separate model (`pdbx_PDB_model_num`).  
The B-factor column contains the **genomic midpoint in Mb**, so you can color beads by genomic position:

```
# in ChimeraX command line:
color byattribute bfactor palette blue:red
```

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
| `GM12878_anchors_3+_oriented.bed` | BED (chr start end orientation) | ChIA-PET loop anchors |
| `GM12878_clusters_3+.bedpe` | BEDPE (chr1 s1 e1 chr2 s2 e2 score) | PET cluster arcs |
| `GM12878_singletons_lessthan3.bedpe` | BEDPE | Singleton contacts for segment-level heatmap |
| `ccds_all_hg38_merged100k_GM12878.breakpoints.bed` | BED | Segment boundary breakpoints |

The region string uses `chr:start-end` format (colon + dash):

```
chr1:18288319-20307135
```

---

## Correctness harness

### Unit tests — energy functions

Compares every scoring function in `src/energy.py` against the compiled C++ reference (within 1e-6 tolerance):

```bash
# Run all unit tests
python harness/compare.py

# Run a specific group
python harness/compare.py distfns
python harness/compare.py heatmap arcs smooth metropolis angle

# Print C++ reference values only (no Python impl needed)
python harness/compare.py --reference

# Force recompile the scorer binary
python harness/compare.py --rebuild
```

Current status: **22/22 tests pass**.

### Integration test — end-to-end distribution comparison

Runs both C++ and Python on the same small region (`chr1:18288319-20307135`, ~102 anchor beads, ~2 Mb) and compares the bead-position distributions with a 2-sample KS test.

```bash
# Full run (default: 5 structures, balanced quality)
python harness/integration.py

# Fast mode (~5 s/structure, good for CI)
python harness/integration.py --fast -n 3

# C++ reference only (no Python side)
python harness/integration.py --cpp-only

# Keep output .hcm files in a temp directory for inspection
python harness/integration.py --keep --fast -n 2
```

PASS criteria: KS statistic ≤ 0.3 and p-value ≥ 0.05 for radius of gyration, pairwise distances, and consecutive bond lengths.

---

## Module layout

```
src/
├── energy.py      # All scoring functions (heatmap, arc spring, smooth, angle, distance conversions)
├── settings.py    # INI config parser — mirrors C++ Settings class
├── io.py          # Load anchors, arcs, singletons, breakpoints from files
├── hierarchy.py   # Cluster dataclass + coarse-to-fine tree building
├── mc.py          # Simulated annealing MC loops (heatmap and arc variants)
├── solver.py      # LooperSolver: orchestrates data loading + MC reconstruction
└── simulate.py    # Public entry point: run_region()

harness/
├── scorer.cpp     # C++ mini-binary compiled against real 3dnome sources
├── compare.py     # Unit-level correctness harness
└── integration.py # End-to-end distribution comparison
```

---

## Algorithm overview

3dgnome uses a **coarse-to-fine hierarchical Monte Carlo** approach:

```
Level 1 (Chromosome root)  — whole chromosome, singleton heatmap energy
Level 2 (Segment)          — ~100 kb–1 Mb segments, singleton heatmap energy
Level 3 (Anchor/IB)        — ChIA-PET loop anchors, arc spring energy
Level 4 (Subanchor)        — fine-resolution loop bases (not yet implemented)
```

Each level runs simulated annealing, then passes 3D positions down as starting points for the next level. The `-v 2` pipeline (implemented here) covers levels 1–3 (segment heatmap + anchor arc spring MC).

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
