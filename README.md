# 3dgnome-ng

Python reimplementation of the [3dgnome](https://bitbucket.org/3dome/3dgnome/src/master/) Monte Carlo chromosome
structure prediction algorithm.

The reference C++ implementation lives in `3dnome/` (read-only). The rewrite lives in `src/`.

WARNING: this is a very experimental playground project. (sic: it doesn't even reimplement 3dgnome properly to the
letter). New features and optimizations will be added iteratively. Use at your own risk.

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
make 3dnome # builds 3dnome/3dnome
```

---

## Running the Python reimplementation

### Quick start

```python
from gnome3d.simulate import run_region

region = "chr1:18288319-20307135"
entry_id = region.replace(':', '_').replace('-', '_')

structures = run_region(
    config_path="data/GM12878/config.ini",
    region=region,
    n_structures=1,
    data_dir="data/GM12878",  # override the absolute path baked into config.ini
)

# Each structure is a list of BeadOut (start, end, x, y, z, kind) tuples
for i, s in enumerate(structures):
    print(f"structure {i + 1}: {len(s)} beads, first bead: {s.start} {s.end} {s.x:.2f} {s.y:.2f} {s.z:.2f} {s.kind}")

# Save each structure as its own mmCIF file
from gnome3d.io import write_cif

for i, s in enumerate(structures, start=1):
    write_cif(f"{entry_id}_structure_{i}.cif", s, entry_id=f"{entry_id}_s{i}")
```

Each CIF file can be opened directly in **ChimeraX** or **Chimera**:

```bash
chimerax chr1_18288319_20307135_structure_1.cif
```

Beads are written as sequential ALA residues on chain A - ChimeraX connects them as a polymer chain automatically.

`data_dir` overrides the `data_dir` key in the config, which is useful because the bundled `config.ini` has it hardcoded
to `/Projects/GM12878/`. Pass the actual local path instead.

---

## Data format

Input files live under `data/<cell_line>/` and are referenced by the config:

| File                                                   | Format                              | Purpose                                      |
|--------------------------------------------------------|-------------------------------------|----------------------------------------------|
| `<cell_line>_anchors_3+_oriented.bed`                  | BED (chr start end orientation)     | ChIA-PET loop anchors                        |
| `<cell_line>_clusters_3+.bedpe`                        | BEDPE (chr1 s1 e1 chr2 s2 e2 score) | PET cluster arcs                             |
| `<cell_line>_singletons_lessthan3.bedpe`               | BEDPE                               | Singleton contacts for segment-level heatmap |
| `ccds_all_hg38_merged100k_<cell_line>.breakpoints.bed` | BED                                 | Segment boundary breakpoints                 |

The region string uses `chr:start-end` format (colon + dash):

```
chr1:18288319-20307135
```
