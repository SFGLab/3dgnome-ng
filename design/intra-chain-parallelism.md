# Intra chain parallelism

How to make one Monte Carlo chain converge in fewer steps, or use more of a device per step.
This is the half of the performance problem that batching cannot reach, and it is where the work
goes once [kernel-performance.md](kernel-performance.md) runs out.

Status. Nothing here is built. Four things have been tried and three failed, and event chain
Monte Carlo has been read and ruled out for this purpose, which is most of what this document is
for.

## Why this is the only thing left

The kernels are latency bound. Measured against the workstation's two ceilings, a production
launch sits at 3 to 5 percent of memory bandwidth and 0.1 to 2.5 percent of arithmetic peak, and
the wide launches exceed DRAM bandwidth outright because their working set fits the card's 32 MB
L2. Far from both means the limit is the dependent chain of accept or reject decisions: each one
must resolve before the next begins.

That has a sharp consequence. Adding chains to a launch is nearly free, which the merge work
already collected, but it does not make any single chain finish sooner. The arcs stage runs
eleven blocks on sixteen threaded workers, so every block already has a core and the wall is the
slowest single block. No amount of width touches it.

So the two levers are fewer steps, or more parallelism inside one chain's step.

## What has been tried

**Plain GPU region batching. Failed.** A single chain on the GPU costs 12.6 us a step against a
CPU core's 1.8, and width only recovers so much: throughput peaks around 1.7 million single bead
steps a second at K=32 to 64, below even a four core CPU's 2.2 million. Sequential single bead MC
is the wrong shape for the device.

**Checkerboard, many beads at once. Fast, and wrong.** Colour the interaction graph and move a
whole colour per sweep. The colour gather kernel reached 4.5 million bead moves a second at
N=1555, about 23 times a single CPU core, and decisively beat multi core CPU. It also converges
to a 15 to 26 percent over compact structure from a collapsed seed, and the exact energy is
higher, so it is a genuinely worse minimum rather than a benign difference.

The root cause was pinned by elimination, and it matters for everything below. Even fully
sequential, one bead per colour with zero staleness, the checker still lands in the wrong basin.
What distinguishes it from the working kernel is every bead once per sweep against random bead
with replacement. Replacement lets one bead take consecutive steps and walk out of the compact
attractor; a coherent sweep cannot. That is the checkerboard's defining property, so arcs cannot
be fixed as a checkerboard. The smooth stage is immune because its energy landscape is dominated
by bonds and heat and is far less degenerate.

**Hybrid, checker as an initialiser then a sequential polish. Correct, and slower.** Reaches the
right minimum, Rg 1.015 end to end. Run end to end for the first time it is at least 4.3 times
slower than threaded numba: over 2,187 s of arcs against 505. The initialiser costs 53 s as
promised and the whole cost is the polish, which re-anneals from the full `max_temp_arcs` because
`pipeline/ib/arcs.py` hands `mc_arcs_jax_batch` the same settings object, so it throws away what
it was given. The genomic floor carries `genomic_floor_polish_temp` for exactly this reason; the
hybrid has no such knob. A warm start polish is the obvious repair and is small. It is not a sure
thing: a sequential step on this GPU costs about 7 us against a CPU core's 1.8, so the straggler
must converge in under about thirty percent of its steps merely to break even.

**Step size annealing. Built, and it is a quality knob.** See
[kernel-performance.md](kernel-performance.md); it costs 15 to 45 percent more rounds for about
one percent lower energy.

## What cudaMMC does, and why it is not the checkerboard

cudaMMC is the closest comparable code, a GPU port of the same reference solver, source at
`~/Desktop/bio/cudaMMC`. Read the source rather than the paper: the paper says "32 threads try
random moves of that bead", which reads like cooperative scoring, and the code does something
else.

    warpIdx = (threadIndex / warpSize) % activeRegionSize;  // a warp owns one bead
    curr_vector = clusters_positions[warpIdx];              // all 32 threads, same start
    for (int i = 0; i < 512; ++i) { ...Metropolis... }      // each thread its OWN chain
    i_winner = __reduce_min_sync(FULL_MASK, i_score);       // best of 32 wins

Each warp owns one bead. Its 32 threads run independent 512 step chains from the same start, each
computing the whole O(N) score itself, and the best is kept. A population search that trades
redundant work for a better move. All beads' warps run concurrently and asynchronously, reading
neighbours mid update with only a `__threadfence()`.

Two things follow.

It is **not** a checkerboard. Each warp takes 512 consecutive moves on its own bead, so it keeps
the with replacement property that the failure analysis above identified as the missing
ingredient, while still being parallel across beads. It is the one parallel variant our own root
cause does not rule out. That is the strongest reason to look at it.

Its synchronisation is nevertheless **worse** than ours, not better: no colouring, no ordering,
just an accepted race. If the checkerboard's staleness worried anyone, this should worry them
more. The part worth taking is the best of K speculative chains per bead; the part to leave is
how it reconciles beads.

Also worth noting: cudaMMC GPU-ised only the heatmap level MC.
`parallelMonteCarloArcs` is declared at `include/LooperSolver.h:199` and never implemented or
called, so arcs and arcs smooth stayed on the CPU. The stage it optimised costs us 3.0 seconds.
Its 3 to 25 times figures are for a profile nothing like ours.

## Options, cheapest first

**Diagnose the straggler before redesigning for it.** One arcs launch ran 3,929 rounds where its
median chain needed 2. That is either one pathological block or a schedule badly matched to it,
and finding out which is a measurement, not a rewrite. Nothing below should start before this.

**The convergence test itself.** Round count is set by `stop_when_ratio_above`, hardcoded at
0.9999, a relative improvement below 0.01 percent in a round. Neither the step size nor the
cooling rate moves the round count much, so this is the one knob that demonstrably does.
Loosening it stops the run earlier by construction. An explicit quality for speed trade rather
than a free win, and unmeasured.

**Best of K speculative moves per bead.** cudaMMC's warp scheme without its racy bead
reconciliation: for the bead being moved, evaluate K candidate displacements and take the best,
or Metropolis among them. Spends idle width on a single chain, which is exactly the case width
cannot otherwise help, and it does not change who moves when, only how good the chosen move is.
It does make the search greedier, which is defensible because this is an optimiser and not a
sampler, but it changes structures and needs its own validation.

**Parallel tempering.** Replicas on a temperature ladder that swap, the standard cure for a chain
stuck in a basin, which is what a 3,929 round straggler looks like. Uses width, though only tens
of replicas rather than the thousands the device could hold. The population annealing papers put
the two at comparable efficiency, so the choice between them is about which fits the pipeline,
and population annealing is the one that scales with the width we have.

**Population annealing. Read the papers; the most promising of these, and better than an earlier
draft of this file said.** Start R replicas at infinite temperature, where equilibrating is
trivial. To step to the next inverse temperature, resample replica j with weight
`exp[-(b_i - b_{i-1}) E_j] / Q`, then run some MCMC rounds on each replica at the new
temperature, and repeat down the ladder.

Three things make it fit us better than expected.

It is reported as comparably efficient to parallel tempering and both as far more efficient than
simulated annealing, which is what we run. Its authors have also used it as a heuristic to find
spin glass ground states, so unlike event chain Monte Carlo it has a track record as an optimiser
and not only as a sampler.

The diversity worry is measurable rather than fatal. Resampling copies replicas, so the
population correlates, and the standard diagnostic for that is the mean square family size,
`rho_t = lim R * sum_i n_i^2` over the fractions `n_i` of the population descended from each
initial replica. The effective number of independent members is `R / rho_t`, and `rho_t = 1` when
every family is a singleton. So the question is not whether resampling collapses the population
but how large R has to be for `R / rho_t` to be the ensemble size we actually want, and that is
a number we can measure rather than a reason not to try.

And it is parallel across replicas, which is the width the kernels leave idle.

Three things against.

The 230 times figure needs its baseline. The paper is explicit that its CPU code parallelises
with close to perfect scaling and that a comparison against a parallel CPU is made by dividing by
the core count, so against our sixteen cores it is nearer 14 times. The further tenfold from
multi-spin coding is an Ising trick on bit packed spins and does not apply to continuous
positions at all.

The results are for discrete spin models. Dense fluids and complex biomolecules appear in the
conclusions as a prospect, not a result, and there are no polymer numbers.

It changes what an ensemble is. We currently produce a set of independent local minima, one per
pipeline run. Population annealing produces a properly weighted equilibrium ensemble at the final
temperature, with its spread set by that temperature and by R rather than by independent starting
noise. That is arguably the more principled object, and it is a change to the science and not
only to the implementation, so it is not a decision to make on performance grounds alone.

It is also a pipeline level change rather than a kernel one: the population has to be carried
together down a shared temperature ladder with a global resampling step between rungs, where our
stages anneal each block independently and hand back one structure per run.

**Event chain Monte Carlo. Read the papers; it does not serve this goal.** The idea was that a
rejection free algorithm removes the dependent chain that is the latency floor. It does not,
because ECMC has one active particle at a time.

What it actually is. The total potential is written as a sum over factors and the Metropolis
filter is applied to each factor separately, so a move is accepted by consensus and vetoed by any
single factor. In the infinitesimal limit the rejection probability becomes a sum over factors,
a veto is attributed to a unique factor and turned into a lifting, and the trajectory between
vetoes is deterministic. The lifted sample space is the configuration together with the index of
**the** active particle. That singular is the whole problem for us.

Krauth's review, section 7, is explicit: a road map for multithreaded ECMC exists at present only
for hard sphere systems, and genuine parallel event driven ECMC for generic potentials is an open
research subject. Our potentials are generic: springs on a target distance, a truncated `1/d`
repulsion, a soft excluded volume. So ECMC would be a large rewrite that leaves the sequential
dependency exactly where it is.

What it would buy instead is rejection freedom, and that is worth measuring rather than
dismissing, because our waste is large. Acceptance through a production arcs anneal:

| block | overall | first quarter | last quarter | final round |
|---|---|---|---|---|
| N=462 | 14.3 percent | 20.0 | 12.3 | 12.6 |
| N=1227 | 9.0 percent | 25.1 | 2.2 | 1.7 |

At the large block 91 percent of 4.3 million evaluations move nothing, and by the end of the
anneal 98 percent do not. Every one of those is a full O(N) local score.

Three further costs, if anyone reconsiders. Every term needs a per factor event time, the
displacement at which the accumulated positive part of its energy change exceeds an exponential
variable; the springs, the repulsion and the excluded volume are one dimensional along a ray and
invertible, but the orientation term is not a pairwise potential in the positions at all and does
not factorise. ECMC's correctness is for a fixed inverse temperature and we anneal. And it is a
sampler, so using it as an optimiser is off label, as our Metropolis already is.

**The cheap thing the paper actually points at.** If 91 percent of evaluations move nothing, the
step size is far too large for the temperature, and that is fixable with feedback rather than a
new algorithm. Adapting the step to hold a measured acceptance rate is standard practice, and it
is what the step decay in [kernel-performance.md](kernel-performance.md) was groping at and got
wrong: that is an open loop geometric schedule, this is closed loop on the measured rate.

One caveat carries over from that result and should be designed around rather than discovered
again. More productive steps improve the score faster per round, and the convergence test stops
on the score no longer improving, so the run ends lower rather than sooner. Recovering the waste
buys quality by default. Banking it as wall time needs the convergence threshold loosened in the
same experiment, which is the one knob measured to move round count.

## References

Population annealing on GPU, arXiv 1703.03676, and its theory, arXiv 1508.05647, both read.
Event chain Monte Carlo for dense polymer melts, arXiv 1502.06447, and Krauth's review, arXiv
2102.07217, both read. cudaMMC, Bioinformatics 39(10) btad588, source at
`~/Desktop/bio/cudaMMC`, read.
