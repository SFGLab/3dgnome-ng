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

### Force bias Monte Carlo

Propose a displacement biased along the local force rather than isotropically. The gradient is
already implicit in the score, so one pass can produce both, and a proposal that points downhill
is accepted more often and moves further when it is.

This is also the right way to use the gradient on this particular energy, which is why it ranks
above solving with one. The `1/d` repulsion is singular, so a gradient method sees unbounded
forces whenever two anchors approach and has to damp hard. Monte Carlo is naturally robust to
that, because a move landing on top of another anchor is simply rejected. Force bias keeps the
rejection as the safety valve and takes the gradient only as a hint.

It must pay for itself. A gradient pass costs about what a local score costs, so unless the
score and the force are accumulated in one loop a step gets meaningfully more expensive, and the
step count then has to fall by more than that to win.

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

### Gradient descent or L-BFGS

The arithmetic is attractive. Reaching convergence on a 1,227 anchor block costs Monte Carlo
about 5.1e10 pair evaluations, where two thousand L-BFGS iterations cost 3e9, or 2.7e8 with a
neighbour list. A gradient is also a dense parallel reduction, which is what a device does well,
where the sequential Monte Carlo sits at 2.5 percent of arithmetic peak.

It is ranked below force bias because the objective fights it. The `1/d` repulsion is singular.
The energy is piecewise smooth with kinks at the repulsion cutoff, at the stretch to squeeze
crossover, and at the confinement radius. Quasi Newton methods assume smoothness and stall on
kinks.

Note what the existing evidence does and does not say. The SMACOF attempt reached energies five
to eleven times worse, but SMACOF majorises a multidimensional scaling stress, which is not this
energy, so that is a different algorithm failing rather than gradients failing. The earlier
gradient and Langevin refutation was about throughput, a gradient being O(N squared) per step
against a Monte Carlo step at O(N), and it did not consider that the step counts differ by orders
of magnitude. Worth one afternoon with the captured blocks to settle, not a rewrite on faith.

### A better initial structure

Spectral embedding or multidimensional scaling from the target distances lands near the minimum
in one shot, and a short Monte Carlo polish finishes it. This is SMACOF used as an initialiser
rather than as the solver, which is a different proposition from the attempt that failed.

### The convergence threshold

Built and measured. `[simulation_arcs] stop_condition_ratio` at 0.9995 gives 1.2 to 1.5 times for
one to six percent of energy. It is not recommended: it also shrinks blocks five to fifteen
percent, and block over compaction is what `anchor-placement.md` exists to fight.

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
45 percent more rounds for one percent lower energy. An adaptive step targeting an acceptance
rate, because acceptance is already 35 percent. All are recorded with their measurements in
[intra-chain-parallelism.md](intra-chain-parallelism.md).
