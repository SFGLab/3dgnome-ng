# Several user jobs on one GPU

The website runs user jobs on the queue. The kernels are bound by launch latency: 32 chains in one
launch cost 1.24 times one chain. A job leaves the card idle 42 percent of the time, in its host
stages, and keeps a kernel running the rest, with most of the card's cores unused. Throughput
comes from several jobs on each card at once. Each job must stay inside its own memory, and a job must not
stop another.

The production server, team-aragorn, has two RTX 2080 SUPER cards of 8 GiB, compute capability
7.5, driver 580 with CUDA 13.0, 36 cores and 125 GB of memory. The MPS control daemon is
installed. Persistence mode is off. Each worker is pinned to one card with `CUDA_VISIBLE_DEVICES`. A worker that sees
both cards would spread one job over both in the multi GPU groups mode, and every job would then
compete for both cards.

## Stability

Each worker gets a fixed share of the card in `XLA_PYTHON_CLIENT_MEM_FRACTION`, taken from the
peak memory of the largest job the form allows: 20 Mb, 10 structures, CTCF and RNAPII loops. The
window limit bounds every job, and no job exceeds its share. The guard of `gnome3d/mc/jax/share.py`
refuses a share that does not fit the free memory, with the numbers. The shares add up to less
than 1 with 0.5 GiB per process for the CUDA context.

## Measurements

First on the workstation, RTX 4060 Ti, 16 GiB, in a directory apart from ~/3dgnome-ng, with the
production config. The peak memory of a job carries over to the 2080. The worker count and the MPS
result do not: the 2080 has 46 SMs of an older generation against 34 on the 4060 Ti. The
throughput step runs again on the production server before the defaults are set, and the server's
JAX build is checked for compute capability 7.5.

| Step | Measure |
|---|---|
| Memory floor | the largest web job alone at pools of 0.75, 1, 1.5, 2 and 3 GiB: whether it finishes, its wall time, and its structures against the job at the default pool. The kernels size their launches from the pool through `gnome3d/mc/jax/memory.py`, and a smaller pool gives narrower launches. The floor is the smallest pool with a wall time near the default's |
| Peak memory | the same job at the default pool, GPU memory sampled every second with nvidia-smi |
| Throughput | jobs per hour at 1, 2, 3 and 4 workers, each worker running the same queue of web sized jobs |
| MPS | the same at the best worker count with the NVIDIA MPS daemon running, which lets kernels of several processes run at once instead of in turns; MPS is not documented for GeForce cards and is tested here |
| CPU threads | numba threads per worker set to the cores over the workers, against no cap |

Measured 2026-11-12 on the workstation, GM12878, production config, 10 structures:

| Run | Wall | Notes |
|---|---|---|
| chr1:1-60 Mb alone | 438 s | |
| chr1:1-60 Mb, two processes at once | 956 s each | 9 percent fewer jobs per hour than one after the other; the second process at a 2.6 GiB pool; structures equal bit for bit to the job alone |
| chr1:1-20 Mb alone | 69 s | GPU utilization under 10 percent in 42 percent of samples and over 90 percent in 58 percent, smooth stage 42 s in one launch of 60 blocks, 12.4 GiB in use of an 11.7 GiB pool plus context |

Without MPS two processes take turns on the card, and the switching costs more than the idle
host stages it fills. Two routes remain: MPS, which runs kernels of several processes at once,
and batching across jobs in one worker, which merges the smooth launches of the queued jobs
the way one job merges its blocks, with the host stages of the next job on other threads.
MPS is tested first.

Measured 2026-11-12 on team-frodo, one RTX 3090, the same job with 10 structures over chr1:1-20 Mb,
each of K jobs at an equal share of 0.9 of the card, `playground/gpu_throughput.sh`:

| Jobs at once | Without MPS | With MPS |
|---|---|---|
| 1 | 39.3 jobs per hour, 92 s | 42.8 |
| 2 | 37.3 | 46.5 |
| 3 | 39.4 | 47.3 |

Every structure of the twelve jobs equals the job alone bit for bit. One job keeps the card busy
most of the time. MPS with two jobs fills the host stages of each with the other's kernels and
gains 18 percent over one job; a third job adds nothing. Batching across jobs in one worker is not
worth building on this evidence.

The floor, `playground/gpu_floor.sh`, the same job alone at a fixed pool:

| Pool | Result |
|---|---|
| 21, 3, 2, 1.5, 1 GiB | finishes in 80 to 81 s, 0.65 GiB peak in use |
| 0.75 GiB | finishes in 80 s, 0.73 GiB peak in use |
| 0.5 GiB | out of memory on one array of 638 MiB, a clean error |

All finished runs equal the default pool bit for bit. The 638 MiB array is N by N in float32 over
the 13,178 beads, and it grows with the square of the window: 2.6 GiB at 40 Mb. Memory does not
bound the worker count at the 20 Mb limit. A pool of 1.5 GiB per worker leaves twice the need.

Decision for the production server, to confirm there: two workers per card with MPS, each at a
pool of 1.5 GiB, each worker pinned to its card. One worker per card without MPS is the fallback
at about 85 percent of the throughput.

With the window limit every job fits the floor of the largest job, and a worker needs no
prediction per job. A prediction per job from the exact tensor bytes of `memory.py` is needed
only if the site later takes jobs of very different sizes, whole chromosomes for example.
A pool in GiB, `GNOME3D_GPU_POOL_GIB`, turned into the fraction of the card the process sees,
keeps one setting valid on the 16 GiB workstation and the 8 GiB 2080.

## Output

The worker count per card, the share per worker, whether MPS is used and the thread cap become
the defaults in web/docker-compose.yml, with one worker service per card pinned to its device, and are written in web/README.md with the numbers.
