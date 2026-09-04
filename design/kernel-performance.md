# Kernel performance

Where a reconstruction spends its wall time, what has been done about it, and what is left. The
work is ordered by measured cost rather than by which kernel is most interesting. Each step
records its outcome when it is finished, so this file is the record as well as the plan.

Scope. This file covers the cost of a step and how work is grouped into launches. Making one
chain converge in fewer steps, or use more of a device within a step, is a separate problem and
lives in [intra-chain-parallelism.md](intra-chain-parallelism.md).

Status. The numba excluded volume scan is fixed and both batched stages now merge their
launches, together 1.29x end to end. Arcs is the largest stage left, its wall is one straggler
block on one core, and the hybrid kernel that would address it re-anneals from full temperature
and is 4.3x slower than the CPU as wired. A cell grid for its repulsion was built and refuted:
its blocks are too small for one to pay.

## How to measure

Prefer a real run's own log to a synthetic benchmark. A production log records every launch's
shape, wall time and step count, which is the same measurement a benchmark tries to reconstruct,
on the real workload and for free.

`playground/profile_run.py <run.log>` sums the dispatch lines and reports each stage's share plus
the per launch detail. `playground/launch_width.py <run.log>` breaks the smooth launches down by
how many IBs were in each and reports the cost per step per IB, which is what shows whether the
launches are wide enough. `playground/relax_profile.py` times the numba local scorers against
bead count.

`playground/jax_overhead.py` separates a JAX call's fixed cost from its per step cost by timing
the same shape at several step budgets. Run it before trusting any JAX timing: the fixed cost is
seconds, so a short benchmark measures it rather than the kernel. `playground/jax_step_cost.py`
and `playground/jax_batch_cost.py` are the short benchmarks that got this wrong and are kept only
so the error stays visible. All of these need a GPU, so they run on the workstation.

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

JAX smooth is not the same shape of problem, and an earlier version of this file had it
backwards. The numbers here replace those.

`jax_batch_cost.py` timed one call of a fixed step budget and divided by the budget. A call
carries a large fixed cost, so that division charged the fixed cost to the steps and reported a
per step cost that fell as the budget rose. Measured at one shape, 32 regions of 4,096 beads:

| steps in the call | us per step, 32 x 4,096 | us per step, 32 x 16,384 |
|---|---|---|
| 20,000 | 132.79 | 465.47 |
| 100,000 | 34.34 | 105.16 |
| 500,000 | 15.56 | 32.40 |
| 2,000,000 | 12.05 | 18.72 |

The fixed cost is about 2.3 seconds at the first shape and 9.1 at the second, and the true per
step costs are 12.05 and 18.72 us. The old table ran 20,000 steps and reported 128.40 and
464.16, so it was measuring the fixed cost. Production runs millions of steps per call and never
pays it in any meaningful proportion.

With the fixed cost separated out, the kernel is latency bound rather than arithmetic bound. One
region of 4,096 beads costs 9.72 us per step and thirty two cost 12.05, so thirty two times the
work costs 1.24 times the time. The same holds in production. Every smooth launch of a real
chr1 GM12878 run, grouped by how many interaction blocks were in the launch:

| IBs in the launch | us per step, whole launch | us per step per IB | share of smooth time |
|---|---|---|---|
| 1 | 14 to 22 | 14 to 22 | 10.0 percent |
| 2 | 17 to 37 | 8.6 to 18.3 | 33.4 percent |
| 3 to 4 | 19 to 31 | 4.8 to 7.8 | 12.4 percent |
| 12 | 16 to 18 | 1.3 to 1.5 | 5.3 percent |
| 16 to 64 | 11 to 14 | 0.18 to 0.69 | 38.8 percent |

Cost per step barely moves with either the width or the bead count. Sixty four IBs of 6,400
beads cost 13.9 us per step and one IB of 12,800 costs 16.9. Adding sixty three chains is free.

The wide launches in that table are not smooth. They are `estimate_dist`, which is wide only
because it expands each IB into sixteen restart replicas, so four nodes become sixty four
chains. The real smooth launches are one, two, three, four or twelve IBs wide, and 55.8 percent
of smooth time is in launches of four or fewer, running about forty times worse per IB than the
estimate_dist launches a few lines below them in the same log.

They are narrow because `SmoothStage.batch_key` is `(heat, orn, comp, brdg, shape bucket)` and
the shape ladder splits an otherwise uniform set across many buckets. The first dispatch of that
run is 28 IBs that agree on every energy term flag, cut into five launches by bead count alone.

## Steps

### 1. Which executor the smooth stage should use. Answered

JAX, and by a wide margin, but not for the reason the question assumed. The executor is second
order. Cost per step is flat in launch width, so the question that matters is how many IBs are
in a launch, and the answer today is too few.

### 2. A neighbour list for the JAX excluded volume. Dropped

It targeted arithmetic. The kernel is latency bound at every production shape, so cutting the
arithmetic saves nothing. Reinstate this only if a later measurement shows a shape where the
arithmetic actually dominates.

### 3. Merge the smooth groups. Built and measured

Group by the energy term signature alone and pad the group to the ladder bucket of its largest
member, instead of splitting a uniform set across bead buckets.

Merging is bounded by device memory rather than by time. The heat target is one (B, B) float32
per chain and is the only input that grows with the square of the padded size, so merging every
heat carrying group of one real dispatch to its largest bucket would need 25.8 GB on a 16 GB
card. Groups therefore pack into as few launches as an 11 GiB budget allows, largest bucket
first, rather than into one. Modelled on two real runs by costing each merged launch at its
slowest chain's step count and a conservative 20 us per step:

| run | smooth now | launches per dispatch | merged | |
|---|---|---|---|---|
| GM12878 chr1 | 3,315 s | 5 to 9 becomes 1 to 2 | 884 s | 3.7x |
| H1ESC chr1 | 2,909 s | 4 to 11 becomes 1 | 823 s | 3.5x |

Two things had to be settled first, and neither was a performance question. Both are done.

A merged launch runs until every chain converges, and a converged chain was latched but not
frozen, so it kept taking steps and a small IB would have annealed for as long as the largest
one needed. A converged chain now holds its state while the rest of the launch runs on, which
costs nothing because the arithmetic is free.

A chain's stream came from `jax.random.split(iter_key, K)` at its slot, and `base_key` came from
`problems[0]`, so both depended on who was batched with whom. The launch key now comes from the
scope alone and each chain folds its own seed. The arcs and multi-chain-restart kernels still
split by slot, which is correct for them since their widths come from settings rather than from
grouping.

What is left is arithmetic. XLA vectorises across the chain axis, so a reduction does not
associate the same way at every width. Measured, the first difference is 7.06e-08 relative,
which is one float32 ulp, and Monte Carlo then amplifies it into a different structure. A run
reproduces at a fixed configuration, but structures cannot be compared bit for bit across launch
widths, and this change breaks existing JAX smooth results once.

Nine checks in `harness/test_batch_seeding.py`, which compare at equal width to separate the
algorithm from the arithmetic and measure the width effect on its own. Removing either fix makes
them fail.

The wiring is confirmed on a real chr1:1-8000000 reconstruction. Three launches of one, one and
two IBs became one launch of four, the description now reads "all bead sizes", and the structure
is the same to the float noise the width change implies, radius of gyration 28.29 against 28.30
over 4,306 beads.

That run was on JAX's CPU backend, where the merge is worth almost nothing, 10.2 seconds against
9.7. CPU JAX is arithmetic bound, so four chains cost about four times one and padding a small IB
up to a large bucket is paid in full. The whole case for merging rests on the GPU being latency
bound, so the end to end number has to come from the workstation and not from a laptop.

#### What the merges measured

On chr1:1-60000000, GM12878, the tuned configuration, same region and seed in every arm.

| arm | total wall | smooth | estimate_dist | arcs |
|---|---|---|---|---|
| off | 1789 s | 784 s | 489 s | 506 s |
| smooth merged | 1577 s | 573 s | 489 s | 508 s |
| both merged | 1382 s | 556 s | 313 s | 505 s |

1.29x end to end. The smooth stage splits into two very different cases. Its heat free dispatch
went from four launches at 135.6 s to one at 43.2 s, 3.1x, which is what the model predicted. Its
heat carrying dispatch went from four launches at 648 s to two at 530 s, 1.22x, because the heat
target is one (B, B) float32 per chain and at 16,384 beads that is 1.07 GB, so only three chains
fit. The model was optimistic on both counts: it assumed one launch, and it costed a merged
launch at 20 us per step where the real cost of that shape is 39.

The estimate stage had no such limit. Its four groups became one launch of eighty chains, 489 s
to 313 s.

That launch also bounds the flat cost claim. Eighty chains of 16,384 beads run at 23.86 us per
step against 14.24 at thirty two, so 2.5 times the work costs 1.68 times the time. Cost is flat
in width up to roughly half a million bead evaluations per step and becomes partly arithmetic
bound past that. Sublinear, not free.

### 4. The arcs stage. Batching has nothing left to give it

Arcs is now the largest stage. On this region it runs eleven interaction blocks on sixteen
threaded workers, so every block already has a core to itself and the 505 s is the slowest single
block on one core. More cores cannot touch it, and neither can a wider launch.

The kernel that could, `mc_executor_jax_arcs_kernel = hybrid`, was run end to end for the first
time and is at least 4.3 times slower: over 2,187 s of arcs against 505. Its checker initialiser
costs 53 s as promised and the whole cost is the sequential polish, which re-anneals from full
temperature. Diagnosis and what to do about it are in
[intra-chain-parallelism.md](intra-chain-parallelism.md).

### 5. The kernels are latency bound. Measured, and it closes the GPU side

The roofline, from what the kernel provably touches, since XLA cannot cost the loops.

| shape | K | B | us/step | GB/s | percent of DRAM | percent of arithmetic |
|---|---|---|---|---|---|---|
| smooth heat+orn | 3 | 16384 | 39.02 | 15 | 5 | 0.1 |
| smooth orn only | 6 | 2048 | 18.17 | 8 | 3 | 0.0 |
| estimate dry | 80 | 16384 | 23.86 | 659 | 229 | 2.5 |
| estimate dry | 32 | 16384 | 14.24 | 442 | 153 | 1.7 |

Nothing is memory bound. The wide launches exceed the card's DRAM bandwidth, which is only
possible because their 15.7 MB working set fits its 32 MB L2, so DRAM never binds, and they are
still at 2.5 percent of arithmetic peak. The narrow launches sit at a few percent of one ceiling
and a tenth of a percent of the other. Far from both means the limit is the dependent chain of
sequential steps: each accept or reject must resolve before the next begins. That agrees with the
width scan, which found thirty two times the chains for 1.24 times the time.

`playground/jax_roofline.py` reports this. Read its note on `cost_analysis`: XLA does not unroll
a `while_loop` over a `fori_loop` for static costing, so its flops and bytes are a small fraction
of the real work. Measured against a hand count it under-reported by about 128 times, so the
figures above are analytic rather than from that call.

One correction this forces. The heat carrying launch is not slow because of the heat term. Every
launch costs 18 to 39 us per step whatever its terms, so 39 us at three chains against 23.86 at
eighty is the latency floor divided by fewer chains. It is slow because it is three chains wide,
and it is three chains wide because of the heat matrix.

### 6. Grouping is done, and merging is the right policy

A per step comparison says a padded wide launch is 30 percent worse than two launches split by
size. That is the wrong question. Sequential launches add their rounds where a merged launch runs
the slowest chain's, and the round collapse dominates: the estimate stage's four launches were
191, 239, 160 and 184 rounds for 469.5 s, and merged they are 236 rounds for 281.5 s. Costing a
two way size split at real production rates gives 170 s plus 156 s against 281 s merged, so
merging still wins.

The heat carrying dispatch is the one that cannot merge further, and not for want of trying. The
heat buffer is `(K, B, B)`, so a 512 bead block padded into a 16,384 launch costs a full 1.07 GB.
Adding a small chain costs what adding a large one costs, which is why the width caps at three
against an 11.1 GiB budget and a 3.5 peak overhead factor. The packing already does the best
available thing, three biggest together and two smallest in a cheap 4096 launch. The remaining
84 s, about six percent of the run, needs a sparse heat target, which is a capacity fix and not
a bandwidth one.

### 7. The arcs cell grid. Built, measured, refuted, reverted

Measured on real converged blocks, how many anchors fall inside the repulsion cutoff:

| N | neighbours inside | fraction | grid does |
|---|---|---|---|
| 103 | 59 | 57 percent | 1.7x less |
| 462 | 100 | 22 percent | 4.6x less |
| 1146 | 118 | 10 percent | 9.7x less |
| 1227 | 112 | 9 percent | 11x less |

The neighbour count is flat at 60 to 120 whatever the block size, which is the local density
property that made the smooth grid 22.8 times faster and bit identical. So the advantage grows
with N, the 2,048 anchor blocks land near 15 to 18 times, and those are the blocks that set the
arcs wall, since eleven blocks on sixteen threaded workers means the wall is the slowest single
one. It works because arcs runs on the CPU, where cutting the per step scan converts directly to
wall time and there is no latency floor to hide behind.

Read this table with what follows it: the counts are right and the conclusion first drawn from
them was not. `playground/jax_roofline.py` and the capture script that produced the table both
run in minutes, which is the point. Several of this section's measurements were wrong before they
were right, each time because a synthetic benchmark measured a fixed cost, used a density no real
structure has, or answered a different question than the one that mattered. A real run's log did
not mislead once.

Built with the same linked list grid the smooth stage uses, plus a compressed row for the
springs, which are not distance limited and so cannot come from a grid. It reached bit
identical results end to end, but only after dropping `fastmath` from the arcs local scorer:
reassociation makes the sum depend on how LLVM vectorised it, so the full scan and the gridded
scan disagreed on 73 of 400 anchors at the 1e-14 level. That flag was measured at 18.1 ms
against 18.2 without it, so it was buying nothing and costing exactness.

At the density real blocks actually have, it loses.

| N | inside the cutoff | grid off | grid on | |
|---|---|---|---|---|
| 462 | 15 percent | 0.032 s | 0.106 s | 0.3x |
| 1227 | 6.6 percent | 0.089 s | 0.166 s | 0.5x |
| 2048 | 5.4 percent | 0.359 s | 0.295 s | 1.2x |

A first benchmark said 3.2 to 3.3 times, on structures far more spread out relative to the
cutoff than a real converged block. Corrected to the measured densities the win disappears.

Two reasons, and both say a grid can never pay here. Arcs blocks are small, so the full scan is
a tight cache friendly loop over a thousand anchors rather than the forty thousand that made the
smooth scan catastrophic, and a linked list's pointer chasing costs more per candidate than that
loop does. And a three by three by three box of cells is 6.4 times the volume of the sphere it
is approximating, so at 6.6 percent inside the cutoff the grid still examines about 42 percent
of the block.

The neighbour counts above are still correct. What was wrong was reading "11 times less
distance work" as an available speedup: it is an upper bound on a perfect
neighbour list, and neither the grid's geometry nor its constant factor gets close to it at this
block size. Reverted; nothing kept.

### 8. The estimate_dist stage. Not started

Thirty one percent of the whole chromosome run. Its launches are already wide and already run
at 0.18 to 0.69 us per step per IB, so the kernel is not the lever. Its cost is sixteen restart
replicas times the step budget, and that is what to look at.

### 9. Step size annealing. Built and measured, it does not help speed

The one schedule knob tried so far. The rest of the options for cutting steps, and what has
already failed at it, are in [intra-chain-parallelism.md](intra-chain-parallelism.md).

`[simulation_arcs]
step_decay`, and the same key under `[simulation_arcs_smooth]` and `[simulation_ib]`, default 1.0
which holds the step as before and is byte exact against the previous commit. The floor is
`[main] step_decay_floor`, 0.1 of the starting step, which cudaMMC does not need because it
anneals over tens of rounds where our arcs stage has taken 3,929.

Measured on blocks at the density real ones converge to, production arcs schedule, three seeds:

| decay | rounds | energy | Rg |
|---|---|---|---|
| 1.0 | 72 | 36,078 | 2.199 |
| 0.9999 | 83 | 35,922 | 2.205 |
| 0.999 | 106 | 35,854 | 2.201 |
| 0.99 | 94 | 35,731 | 2.203 |

It costs 15 to 45 percent more rounds and reaches an energy about one percent lower, with the
radius of gyration unchanged. So it is a quality knob, not a speed one, and anyone reaching for
it hoping for speed should stop here.

The reason is worth keeping. Convergence in this regime fires on `stop_when_ratio_above`, a
relative improvement below 0.01 percent in a round, and not on the plateau branch at all: accepts
run about 4,500 a round against a `stop_condition_successes_threshold` of 100, so
`n_ok < stop_successes` is never true. A finer step keeps finding small improvements, so the run
goes longer and ends lower.

Cooling faster does not recover it either. At decay 1.0 a `delta_temp` of 0.9995 took 104 rounds
and 0.999 took 89, against 72 for the production 0.9999, all reaching the same energy. Neither
the step size nor the cooling rate moves the round count much, which says the schedule levers
inside the current convergence test are spent. The test itself is the remaining knob:
`stop_when_ratio_above` is hardcoded at 0.9999 in `pipeline`'s arcs entry, and loosening it stops
the run earlier by construction. That is an explicit quality for speed trade rather than a free
win, and it has not been measured.

### 10. Arcs on the JAX executor. Not started

Arcs is 79 percent of the time in the configuration that puts it on JAX and 22 percent in the one
that leaves it on threaded numba, which is what production uses. Whether the arcs kernel is
latency bound the way the smooth kernel is has not been measured, so whether widening its
launches would pay is unknown. Its cell grid is refuted above and its convergence is covered in
[intra-chain-parallelism.md](intra-chain-parallelism.md).

### 11. Re-measure, then run. Not started

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
