# Run time against cudaMMC

The Results page of the website compares the run time of cudaMMC and 3D-GNOMEᴺᴳ. The values are placeholders until
this run is done. The comparison is between the two engines on the same inputs and the same GPU, not between versions
of 3D-GNOMEᴺᴳ.

## Machine

The workstation, `ssh home-workstation`, one RTX 4060 Ti, CUDA 13.3 in /usr/local/cuda-13.3. One GPU job at a time,
in a screen session. cudaMMC is not on the workstation yet.

## cudaMMC

Copy the source from ~/Desktop/bio/cudaMMC and build it with `cmake -B build -S . -DCUDA_ARCH="89"` and
`make --directory build`, with nvcc from /usr/local/cuda-13.3/bin. Use the project directory of the published
enhancer3D models, its config.ini, the GM12878 CTCF anchors, clusters and singletons, the segment split and the
centromeres. That directory is /mnt/raid/cudammc/projects on the machine that ran modelling/run_ccd_models.py in
~/Desktop/enhancer3d_cross_cell_lines_analysis. Run with `-h`, the flag that run used.

## 3D-GNOMEᴺᴳ

The production config of validation/core/config.py with CTCF loops only, the GM12878 inputs of
playground/rnapii/crosscell_prep.py, executors as in production.

## Cases

| Case | Region | Structures |
|---|---|---|
| One structure, 1 Mb | chr1:11,500,000-12,600,000 | 1 |
| Ten structures, 60 Mb | chr1:1-60,000,000 | 10 |
| One structure, whole chromosome | chr1 | 1 |

Each case runs twice per engine after a warm up run. The table records the wall time of the whole command from
`/usr/bin/time -v`, peak GPU memory from `nvidia-smi --query-gpu=memory.used` sampled every second, and the version
of each engine. JAX compile time is in the wall time of the first run and is reported separately.

## Output

A table under design/figures/runtime-2026-11/ with case, engine, run, wall seconds, peak memory and version, and the
command lines that produced it. The Results page takes its values from that table.
