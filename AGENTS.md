# AGENTS.md — 3dgnome-torch

This codebase is a **line-for-line PyTorch reimplementation of cudaMMC** (C++/CUDA, located
in `cudaMMC/`), which reconstructs 3D chromatin structure from ChIA-PET data. The C++/CUDA
source under `cudaMMC/src/` and `cudaMMC/include/` is the **single source of truth**.
Python code in `src/` exists to reproduce its numerics on PyTorch (CPU/CUDA/MPS) — nothing
more, nothing less.

## Prime directive: prove every change against cudaMMC

You MUST treat `cudaMMC/` as the specification. For every algorithmic change, addition, or
"fix", you are required to:

1. **Locate the upstream source first.** Open the matching file in `cudaMMC/src/` (or
   `cudaMMC/include/`, `cudaMMC/thirdparty/`) and read the exact lines you intend to mirror.
   Key files: `LooperSolver.cpp` (pipeline + all 3 MC phases + scoring),
   `ParallelMonteCarloHeatmap.cu`, `ParallelMarkArcs.cu`, `HierarchicalChromosome.cu`,
   `Settings.cpp` (defaults), `thirdparty/common.cpp` (RNG, spline, angle helpers).
2. **Cite the line(s) in code.** Every algorithmic line in `src/` carries a
   `# cudaMMC: cpp:LINE` (or `.cu:LINE`) annotation. New/changed code MUST add or update
   these citations. See `src/mc.py:35-65`, `src/scores.py:22-37` for the established style.
3. **Show the evidence in your reply.** When you finish a change, paste the relevant
   cudaMMC snippet alongside the new Python and explain in 1–3 lines why they are
   numerically equivalent (same formula, same iteration order, same RNG/temperature
   schedule, same dtype where it matters). A change without this proof is not done.
4. **Run `python verify_algorithm.py`.** This is the canonical regression harness — it
   contains a discrepancy table plus reference implementations transcribed from C++ with
   inline citations and asserts that `src/mc.py` / `src/scores.py` match. Update the
   discrepancy table if you intentionally introduce or close a gap.
5. **Do not invent behaviour.** If cudaMMC does X, do X — even if X looks suboptimal. If
   you believe cudaMMC is wrong, say so explicitly and ask before diverging. Never silently
   "improve" the algorithm, change a default, swap a formula (e.g. classical
   `exp(-ΔE/T)` for the ratio-based form), or reorder MC moves.
6. **Do not edit `cudaMMC/`.** It is a vendored CMake build of the reference and must stay
   pristine. Read-only.

If the upstream behaviour is genuinely unclear after reading the C++, stop and ask rather
than guessing — `docs/cudaMMC_algorithm.md` may also help.

## Architecture (read in this order, alongside the cudaMMC counterpart)

| Python (`src/`)            | cudaMMC counterpart                                              |
|---                          |---                                                               |
| `main.py`                   | `cudaMMC/src/main.cpp`                                           |
| `solver.py` `LooperSolver`  | `LooperSolver.cpp` `runLooper()`, `reconstructClusters*`         |
| `tree.py` `ChromosomeTree`  | `LooperSolver.cpp` `createTreeChromosome`, `findGaps`, `findSplit`, `HierarchicalChromosome.cu` |
| `mc.py`                     | `LooperSolver.cpp` `MonteCarloHeatmap`/`Arcs`/`ArcsSmooth` (cpp:421-3390), `ParallelMonteCarloHeatmap.cu` |
| `scores.py`                 | `LooperSolver.cpp` `calcScore*` family                            |
| `settings.py`               | `Settings.cpp` / `Settings.h`                                    |
| `heatmap.py`                | `Heatmap.cpp`                                                    |
| `data_loading.py`, `data_structures.py` | `Anchor.cpp`, `Cluster.cpp`, `InteractionArc.cpp`, `InteractionArcs.cpp` |
| `distances.py`              | inline helpers in `LooperSolver.cpp` (`genomicLengthToDistance`, `countToDistance`) |
| `regions.py`                | CLI region parsing (Python-only convenience, no upstream)         |

Pipeline (mirrors `runLooper()`):
`set_contact_data` → `load_heatmap` → `create_tree_genome` →
`reconstruct_clusters_heatmap` (Phase 1, all anchors per chrom) →
`reconstruct_clusters_arcs_distances` (Phase 2 arcs + Phase 3 smooth, **per IB**) → `save`.

Tree levels: **1=chrom root, 2=segment (~2 Mb), 3=interaction block (IB), 4=anchor leaf**.
IB boundaries come from `findGaps`/`findSplit` arc-sweep (see `find_all_gaps` docstring).

Data flow per chromosome: BED anchors + BEDPE clusters → `Anchor`/`InteractionArc` →
`ChromosomeTree.clusters[]` (flat list, parent/children indices) → per-IB tensors
`(arc_s, arc_e, arc_exp, chain_lengths, orientations)` → MC → write back via
`tree.set_positions_from_tensor()` → `_propagate_positions_up()`.

## Project-specific conventions (all enforced by cudaMMC parity)

- **Ratio-based Metropolis** `scale*exp(-coef*s_curr/s_prev/T)` (see `_with_chance_ratio`
  in `src/mc.py`), matching `LooperSolver.cpp:3114-3116` and
  `ParallelMonteCarloHeatmap.cu:241-244`. **Never** use classical `exp(-ΔE/T)`.
- **Temperature decays per bead move**, not per outer step (`Settings.cpp:217 dtTempHeatmap`
  and equivalents).
- **Defaults live in `src/settings.py`** with `cudaMMC Settings.cpp:LINE` comments giving
  the upstream value. INI overrides via `Settings.from_ini(path)` (see `data/*/config.ini`,
  `cudaMMC/config.ini`).
- **Multi-restart per IB**: `simulation_steps_level_anchor`/`_subanchor` (default 5),
  keeping `best_pos` by lowest score — see `solver.py:338-382` mirroring
  `LooperSolver.cpp:2836`.
- **Paired full/single score functions** (`score_X` / `score_X_single`) must produce
  identical deltas — they are used for incremental MC. Test both when editing either.
- **Local 0..n-1 indices** inside `_arc_tensors_ib`; arcs stored as `(lo, hi)` so
  `score_arcs_single` finds both orderings.
- **Singletons feed only the heatmap expected-distance matrix**, never the arc spring list
  (see `solver.py:78-80`; cudaMMC does the same).
- **Device handling**: `settings.device` is `cuda`/`mps`/`cpu`; tensors created with
  `device=self.device`. Heatmap stored **float16** (~1.4 GB for N≈26 k); cast to float32
  only inside score functions. (cudaMMC uses `half3` on GPU — same intent.)
- **RNG**: `_random_displacement` uses `(2*uniform-1)*step` per axis, matching
  `common.cpp` `random_vector` / `.cu:75-80 randomVector`. Do not switch to Gaussian.

## Workflows

```bash
# Install
pip install -r requirements.txt          # torch>=2.0, numpy>=1.24

# Run pipeline (small region for dev iteration)
python main.py --anchors data/GM12878/GM12878_anchors_3+_oriented.bed \
               --clusters data/GM12878/GM12878_clusters_3+_oriented.bedpe \
               --config data/GM12878/config.ini \
               --chromosomes chr14:1:2500000 --device mps --cif

# Verify Python matches cudaMMC reference — REQUIRED after touching mc.py/scores.py
python verify_algorithm.py

# Build/run the cudaMMC reference for ground-truth comparison
cd cudaMMC && mkdir -p build && cd build && cmake .. && make -j
./cudaMMC ../config.ini                  # produces .hcm / .smooth.txt

# Convert outputs (.3d/.hcm/.smooth.txt) for ChimeraX/PyMOL
python to_cif.py output/structure_chr14.3d
```

There is no pytest suite or linter config: `verify_algorithm.py` plus a hand-comparison
against a fresh cudaMMC run is the canonical proof of correctness. When in doubt, run both
and diff the outputs.

