# Kernel performance

Where a reconstruction spends its wall time, what has been done about it, and what is left. The
work is ordered by measured cost rather than by which kernel is most interesting. Each step
records its outcome when it is finished, so this file is the record as well as the plan.

Status. The numba excluded volume scan is fixed. The JAX kernels and the arcs stage are not.

## How to measure

`playground/profile_run.py <run.log>` sums the dispatch lines a run writes and reports each
stage's share, plus the per launch detail for the batched stages. `playground/relax_profile.py`
times the numba local scorers against bead count. `playground/jax_step_cost.py` and
`playground/jax_batch_cost.py` time the JAX smooth kernel against bead count and batch width.
The last two need a GPU, so they run on the workstation.

## Where the time goes

Two real runs, both GM12878, both with the tuned configuration. They differ in which executor
the arcs stage used, and that changes the answer completely.

| stage | whole chr1, arcs on threaded numba | chr1:1-60 Mb, arcs on batched JAX |
|---|---|---|
| arcs | 1,542 s, 22 percent | 4,633 s, 79 percent |
| smooth | 3,337 s, 47 percent | 741 s, 13 percent |
| estimate_dist | 2,197 s, 31 percent | 514 s, 9 percent |
| coarse and densify | 16 s | 4 s |

The production configuration sets every one of arcs, estimate_dist and smooth to `batch`, which
is JAX. So production's time is almost entirely in JAX kernels.

## What the kernels cost per step

The excluded volume term sums over pairs closer than `r0`. Only about fifty beads are ever that
close whatever the structure's size, because that is a local density, so a scan over every bead
does hundreds of times more work than it needs.

numba smooth, measured on a finished 60 Mb region:

| beads | chain term | excluded volume | its share of a step | beads inside the radius |
|---|---|---|---|---|
| 2,000 | 0.36 us | 2.68 us | 88 percent | 53 |
| 10,000 | 0.38 us | 12.91 us | 97 percent | 52 |
| 42,480 | 0.52 us | 52.87 us | 99 percent | 57 |

JAX smooth on the workstation GPU is a different shape of problem. One chain costs about ten
microseconds a step whatever the bead count, because the chain is sequential and each step is
one small reduction on the device. The excluded volume is only a quarter of that at 16,384
beads. Batching many regions into one launch is supposed to share that latency, and it does
not share it well:

| regions in a launch | us per step, 4,096 beads | per region | us per step, 16,384 beads | per region |
|---|---|---|---|---|
| 1 | 13.45 | 13.45 | 25.21 | 25.21 |
| 4 | 25.43 | 6.36 | 69.63 | 17.41 |
| 8 | 40.02 | 5.00 | 126.42 | 15.80 |
| 16 | 69.08 | 4.32 | 237.76 | 14.86 |
| 32 | 128.40 | 4.01 | 464.16 | 14.51 |

Thirty two regions buy 3.4 times at 4,096 beads and 1.7 times at 16,384, not thirty two. The
kernel is therefore not latency bound at production batch widths, it is arithmetic bound, and
the excluded volume share rises with the batch width to 43 percent at 4,096 beads and 46
percent at 16,384. Two things follow. A neighbour list for JAX is worth building, because it
cuts the term that is nearly half the cost in the regime that matters. And the executor choice
itself is in question, since numba with the cell grid costs about seven microseconds per step
for one region where JAX costs 14.5 per region at its widest batch.

## Steps

### 1. Which executor the smooth stage should use. Not started

Head to head on the same real interaction blocks, threaded numba with the cell grid against
batched JAX, wall time for an identical dispatch. No code, one measurement. It either changes
`mc_executor_smooth` in the production configuration, which is a larger win than any kernel
change, or it confirms JAX and step 2 follows.

### 2. A neighbour list for the JAX excluded volume. Not started

Only if step 1 keeps JAX. A fixed size neighbour array per bead, rebuilt each round on the
device, so the term reads its neighbours instead of every bead. It targets 43 to 46 percent of
the cost.

This cannot be bit exact the way the numba grid is. The numba sum is built in ascending bead
index, the order the full scan uses, so it is identical. JAX sums by reduction over a padded
array and the order is the reduction's, so results will differ in the last bits and structures
will diverge over millions of steps. That means a flag, its own validation against the current
kernel, and a decision about whether a changed trajectory is acceptable. It is a different bar
from the numba work and should not be described as the same kind of change.

### 3. The arcs stage. Not started

Seventy nine percent of the production configuration's time. Two independent levers, and the
profiling has to come before either.

The arcless pairs carry a `1/d` repulsion, and `arcs_repulsion_cutoff_factor` truncates it at
three times the mean arc distance, so the term is zero beyond that and is distance limited
after all. A cell grid may apply exactly as it did for smooth. The neighbour count inside that
cutoff is what decides the size of the win, and it has not been measured.

Separately, the launches converge badly. One took 5,023 rounds and 251 million steps for two
regions. Another reported 86 percent of its time wasted waiting for its slowest chain. That is
a grouping and scheduling problem rather than an arithmetic one, and it may be the larger half.

### 4. The estimate_dist stage. Not started

Thirty one percent of the whole chromosome run and never examined. It had a host side quadratic
reduction once before, which was fixed, so it deserves its own look rather than an assumption.

### 5. Re-measure, then run. Not started

Repeat the stage breakdown with whatever the earlier steps changed, measure the cross block
relaxation's cost with the cell grid, and only then start the trios.

## Outcomes

### The numba excluded volume cell grid. Done

Beads are binned into a linked list per cell at cell size `r0`, so the twenty seven cells around
a bead hold everything within `r0`. An accepted move unlinks the bead from its old cell and
links it into its new one, which keeps the cells exactly `r0` wide with no margin for drift and
the grid exact after every move. Both the step term and the initial score go through it. The
initial score mattered as much as the step: it was a full pair scan, quadratic in the structure,
about two seconds per call at 42,480 beads, and it masked the step improvement until it was
fixed too.

Results are identical bit for bit, so no trajectory changes and the parity gate is unaffected.
Pairs beyond the radius contribute exactly zero, and the sum is built in ascending bead index.
A query that finds more neighbours than its buffer holds returns a sentinel and the caller falls
back to the full scan, so correctness never depends on a capacity guess.

| beads | before | after | speedup |
|---|---|---|---|
| 8,000 | 0.50 s | 0.09 s | 5.3 |
| 20,000 | 1.53 s | 0.14 s | 11.2 |
| 42,480 | 13.30 s | 0.58 s | 22.8 |

Positions and scores identical in every case. `[simulation_backend] neighbour_grid` turns it
off. Below 2,048 beads the full scan is already cheap and the grid is skipped. The JAX kernels
are untouched. Fifteen checks in `harness/test_cells.py`.
