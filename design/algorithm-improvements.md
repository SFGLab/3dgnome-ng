# Algorithm improvements

Ways to make the Monte Carlo itself faster, as opposed to making a step cheaper to execute or
grouping work better. Those two are in [kernel-performance.md](kernel-performance.md), and they
are close to spent. This file is the space that is left, what constrains it, and what has been
ruled out.

## What the measurements constrain

Five facts, all measured, and every option below is judged against them.

**Arcs is about ninety percent of a genome scale run.** A real trio run spends 89.6 percent of
its wall there, 9.0 in estimate_dist and 1.1 in smooth. Anything that does not touch arcs is
working on a tenth of the problem.

**The arcs energy is an all pairs N body problem.** About 99.6 percent of anchor pairs carry a
truncated `1/d` repulsion, about 0.4 percent carry springs, and there is a confinement term. A
step scores one anchor against every other, so it costs O(N) for N of 256 to 2,048.

**The landscape is a funnel.** Ten independent starts on a real block land within one percent of
the same energy, with a coefficient of variation of a third of a percent, and temperature makes
no difference to where a run converges. There are no basins to escape.

**Acceptance is healthy.** It runs from 49.6 percent early to 15.8 percent late, averaging 35.5
percent, which is where a well sized Monte Carlo should be. Proposals are not being wasted.

**And yet a 1,227 anchor block needs 41.5 million steps**, which is 34,000 steps per anchor, and
the round count grows about as N to the 1.7. That is an enormous amount of work to descend a
funnel, and it is the number every option here is trying to cut.

## Realistic

### Force bias Monte Carlo. Built and measured, a modest win Removed 2026-09-06, measured and not adopted.

Propose along the local descent direction rather than isotropically. The gradient rides the same
sweep as the score, and the displacement is drawn as before and only steered, so
`[simulation_arcs] force_bias = 0` is bit exact.

Measured on the largest real block, two seeds, each arm to its own convergence:

| bias | rounds | seconds | energy | Rg | acceptance |
|---|---|---|---|---|---|
| 0 | 793 | 174.7 | 11,580 | 13.38 | 36.0 percent |
| 0.10 | 764 | 266.3 | 11,538 | 13.59 | 40.0 |
| 0.25 | 612 | 214.5 | 11,481 | 13.60 | 51.0 |
| 0.50 | 450 | 153.2 | 11,424 | 14.39 | 71.8 |

Across all three blocks, and with the spread measured over six starts:

| bias | speed | energy | spread |
|---|---|---|---|
| 0 | 1.00x | 11,543 | 29.25 percent |
| 0.50 | 1.41x | 0.992 of it | 27.42 |
| 0.75 | 1.95x | 0.989 | 26.17 |
| 0.90 | 2.25x | 0.989 | 25.97 |

A clean monotone trade: more bias is faster and reaches a slightly lower energy, and costs
spread, about a tenth of it by 0.9. The wall lags the round count because the gradient sweep
costs more per step than the score alone even sharing the loop.

It works, and the solver above makes it the lesser option: fifty nine times against a little over
two, and the solver gains spread where this loses it.

`playground/force_bias_sweep.py` runs the sweep.

### A neighbour list for the truncated repulsion

Only nine to twenty two percent of anchors are inside the repulsion cutoff, and the rest
contribute exactly zero, so a step scans about five times more than it needs.

A cell grid was built for this and refuted: linked list pointer chasing costs more per candidate
than the tight scan it replaces, and a three by three by three box of cells is 6.4 times the
volume of the sphere it approximates, so at 6.6 percent inside it still examined 42 percent of
the block. A Verlet list has neither problem. It is a contiguous array of exactly the neighbours
within the cutoff plus a skin, rebuilt every twenty or so steps, and it is the standard answer in
molecular dynamics for the same reason. It is a different data structure rather than a retry.

Force bias needs one anyway, since a force is the same sum as a score.

### Solving the arcs stage. Built and wired, and it wins by a lot

`[simulation_arcs] solver = mc | lbfgs`, default `mc`, with `solver_iters` at 200.
`gnome3d/mc/numba/arcs_solver.py` carries the energy and its gradient over a whole structure,
checked against the initialisers the Monte Carlo builds its score from.

Measured on real captured blocks with the settings loaded from a production config, so the
energy is the one production minimises. Six starts per arm, each run to its own convergence:

| arm | wall | energy | Rg | spread | realised over target | inside cutoff | nearest |
|---|---|---|---|---|---|---|---|
| MC | 2,981 s | 11,593 | 13.40 | 29.72 percent | 1.92 | 9.2 percent | 0.849 |
| L-BFGS 200 | 50 s | 11,638, 1.004 of it | 14.25 | 31.10 | 1.93 | 9.1 | 0.847 |
| L-BFGS 2000 | 495 s | 11,347, 0.979 | 18.58 | 27.63 | 1.94 | 8.9 | 0.850 |

**Fifty nine times, for the same energy to within half a percent, with the ensemble spread
slightly wider rather than narrower and the geometry matching across its whole distribution**,
tenth and ninetieth percentiles included. Two thousand iterations go below the Monte Carlo's
energy and are still six times faster.

The iteration count is a dial between energy and spread that the Monte Carlo does not offer:
two thousand iterations reach two percent lower energy and give up about a tenth of the spread.

An earlier version of this measurement left the excluded volume off, because the setting is
`apply_to_arcs` under `[excluded_volume]` and a grep for `exclusion_apply_to_arcs` missed it.
Both arms were consistent with each other so nothing was wrong internally, but it was not the
production energy. Redone with it on, the ratio improved from thirty six to fifty nine, since the
extra term costs the annealer more than it costs the solver.

**Why a solver works here at all.** The landscape is a funnel. Ten Monte Carlo starts land within
one percent of each other and temperature is inert, so there is nothing for a stochastic search
to escape. The singularity in the repulsion and the kinks at the cutoff, the stretch to squeeze
crossover and the confinement radius did not stop it, and it walks in from a nearly collapsed
start at an energy of 1.4e8.

**It also says something about the Monte Carlo.** Those ten starts sit within 0.38 percent of
each other, and the solver reaches two percent below all of them, so the Monte Carlo is not
finding the minimum, it is stopping short where the improvement ratio test fires while it is
still descending. The properly converged structure is more expanded, 18.58 against 13.40. Since
`anchor-placement.md` exists because blocks come out too compact, some of that may be a stopping
rule rather than a modelling problem, and that is worth checking before more modelling goes in.

Not yet done: the ensemble comparison through the real pipeline, and the validation battery,
Hi-C correlation and the within block distance exponent, which is what should gate adoption
rather than an energy number.

### A better initial structure

Spectral embedding or multidimensional scaling from the target distances lands near the minimum
in one shot, and a short Monte Carlo polish finishes it. This is SMACOF used as an initialiser
rather than as the solver, which is a different proposition from the attempt that failed.

### The convergence threshold

Built and measured. `[simulation_arcs] stop_condition_ratio` at 0.9995 gives 1.2 to 1.5 times for
one to six percent of energy. It is not recommended: it also shrinks blocks five to fifteen
percent, and block over compaction is what `anchor-placement.md` exists to fight.

### What is left before either could be used

Both are measured offline on captured problems and neither is wired into the pipeline.

Force bias is the small change. One setting, the Monte Carlo otherwise untouched, bit exact when
off, and about 1.8 times. It needs the diversity check and then it is a configuration decision.

L-BFGS is the large one. Thirty six times on the stage that is ninety percent of a genome scale
run, with matching energy, spread and geometry, but it replaces the arcs kernel rather than
tuning it. It needs wiring as an opt in kernel beside `mc` and `hybrid`, and then the validation
battery rather than an energy number, since the structures differ: Hi-C correlation, the within
block distance exponent, the boundary measurements. Its iteration count is a real knob and the
right setting for it is a question about ensembles, not about speed.

The under convergence finding should be followed up on its own. If the Monte Carlo stops two
percent above the minimum and the properly converged structure is more expanded, then some of
what `anchor-placement.md` treats as a modelling problem may be a stopping rule, and that is
worth knowing before more modelling goes into it.

## Research, not engineering

### A warp per bead kernel

cudaMMC gives each bead a warp whose 32 threads run independent 512 step chains from the same
start and keeps the best. It is the one parallel scheme our own root cause analysis does not rule
out, because it keeps the consecutive steps per bead that the checkerboard lacked and that arcs
needs. It is also a large hand written CUDA build aimed at a device whose problem here is
latency rather than arithmetic. See [intra-chain-parallelism.md](intra-chain-parallelism.md).

### A learned initial structure

Predict anchor positions from the arc graph directly, which removes most of the descent instead
of accelerating it. The flow matching work on the `cnf` branch is adjacent. Months, not days.

### Parallel event chain Monte Carlo

Event chain Monte Carlo is rejection free, and Krauth's review states plainly that a road map for
multithreaded ECMC exists only for hard sphere systems and that parallel event driven ECMC for
generic potentials is an open research subject. That makes it a paper rather than a task.

Our system is an unusually good testbed for it. The obstacle in the literature is that events are
global and that domain decomposition leaves residual interactions which destroy translational
symmetry. Our repulsion is truncated, so the interaction range is finite and the conflict graph
between candidate events is local, which is the precondition that makes speculative parallel
event execution tractable. The defensible contribution is parallel ECMC for truncated finite
range potentials, driven by chromatin structure determination, rather than general purpose
parallel ECMC in competition with the people who invented it.

Read first: the review's section 5.1.3 on parallel hard disk ECMC and its section 7; the
JeLLyFysh papers, which are the reference implementation; and the parallel discrete event
literature it cites, Lubachevsky on billiards in parallel and Miller and Luding on event driven
molecular dynamics in parallel, which is where conservative bounded lookahead and optimistic
execution with rollback come from.

**Gate it on a serial prototype.** The whole benefit is rejection freedom, so the ceiling on our
problem is set by our rejection rate, which is 65 percent, giving at most about 2.9 times. If
serial ECMC does not beat serial Monte Carlo on this energy then parallel ECMC is moot, and that
costs a day to find out rather than a year.

A second paper is already in hand: the landscape built from real contact data is effectively a
funnel, ten starts landing within one percent with temperature provably inert. That is a non
obvious claim about the structure of the inverse problem, it is cheap to make rigorous across
cell lines and block sizes, and it justifies the algorithmic choice in the first paper.

## Ruled out

Population annealing and parallel tempering, because both traverse a rough landscape and this one
is a funnel, and population annealing's resampling weights degenerate at the temperature the
schedule spends its time at. A checkerboard over arcs, because its every bead per sweep update
lands in a worse basin and that is its defining property. Step size annealing, which costs 15 to
45 percent more rounds for one percent lower energy, and was removed with its keys on
2026-09-06. An adaptive step targeting an acceptance
rate, because acceptance is already 35 percent. All are recorded with their measurements in
[intra-chain-parallelism.md](intra-chain-parallelism.md).
