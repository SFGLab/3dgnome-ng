# Several user jobs on one GPU

The website runs user jobs on the queue. One job leaves the GPU mostly idle: the kernels are bound
by launch latency, and 32 chains in one launch cost 1.24 times one chain. Throughput comes from
several jobs on the card at once. Each job must stay inside its own memory, and a job must not
stop another.

## Stability

Each worker gets a fixed share of the card in `XLA_PYTHON_CLIENT_MEM_FRACTION`, taken from the
peak memory of the largest job the form allows: 20 Mb, 10 structures, CTCF and RNAPII loops. The
window limit bounds every job, so no job exceeds its share. The guard of `gnome3d/mc/jax/share.py`
refuses a share that does not fit the free memory, with the numbers. The shares add up to less
than 1 with 0.5 GiB per process for the CUDA context.

## Measurements on the workstation

RTX 4060 Ti, 16 GiB, with the production config, on the workstation in a directory apart from
~/3dgnome-ng.

| Step | Measure |
|---|---|
| Peak memory | the largest web job alone, GPU memory sampled every second with nvidia-smi |
| Throughput | jobs per hour at 1, 2, 3 and 4 workers, each worker running the same queue of web sized jobs |
| MPS | the same at the best worker count with the NVIDIA MPS daemon running, which lets kernels of several processes run at once instead of in turns; MPS is not documented for GeForce cards and is tested here |
| CPU threads | numba threads per worker set to the cores over the workers, against no cap |

Measured 2026-11-12 before this plan: two 60 Mb jobs of 10 structures at once took 956 s each
and gave the structures of a job on a free card bit for bit; the job peaked near 2.8 GiB.

## Output

The worker count, the share per worker, whether MPS is used and the thread cap become the
defaults in web/docker-compose.yml and are written in web/README.md with the numbers.
