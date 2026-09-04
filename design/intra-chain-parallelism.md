# Intra chain parallelism

What has been tried to parallelise one Monte Carlo chain, and why most of it failed. This is the
half of the performance problem that batching cannot reach.
[kernel-performance.md](kernel-performance.md) covers the cost of a step and the grouping of
work. [algorithm-improvements.md](algorithm-improvements.md) covers changing the Monte Carlo so
it needs fewer steps, which is the open space.

Status. Nothing here is built. Four things have been tried and three failed, event chain Monte
Carlo has been read and ruled out for this purpose, and population annealing has been ruled out
by a measurement of the landscape. That record is most of what this document is for.

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

## The landscape is a funnel, and that rules out two of the options

An earlier version of this section said temperature was harmful and that a real ladder was two to
four percent worse. Both claims came from matched budget comparisons, and a matched budget is
exactly the test that flatters greedy descent: it dives into the nearest basin and banks energy
early, so it wins at any fixed budget whether or not that basin is the right one. Temperature
exists to accept worsening moves and leave a basin, and a fixed budget cannot see that. Those
claims are withdrawn.

What the schedule does is still worth knowing, and it is arithmetic rather than a benchmark.
`delta_temp` is applied once per MC step, not per round, so at the reference's 0.9999 over 50,000
steps a round the temperature falls by 148 each round: from `max_temp` 5.0 it is 3e-2 after one
round and 1e-13 after six, where runs take seventy to ninety. Whatever escaping happens, happens
in the first one percent of the run. Those are the reference's own values in every stage.

The right test for whether that matters is the spread over many starts, not the mean at a fixed
budget. If temperature earns its keep, running without it leaves some starts stuck somewhere much
worse. On real captured arcs blocks, ten starts each, both arms run to their own convergence:

| block | temperature | mean | cv | worst | best to worst |
|---|---|---|---|---|---|
| N=1146 | production 5.0 | 11,871 | 0.38 percent | 11,937 | 1.012 |
| N=1146 | none | 11,850 | 0.52 percent | 11,957 | 1.017 |
| N=462 | production 5.0 | 4,040 | 0.34 percent | 4,057 | 1.010 |
| N=462 | none | 4,043 | 0.34 percent | 4,071 | 1.013 |

The distributions are indistinguishable. Every start lands within about one percent of the same
energy with or without temperature, and running to convergence rather than a budget the two agree
to within one percent on energy as well. So temperature is **inert** on the arcs objective at
these sizes, not harmful: the landscape is a funnel and the escape mechanism has nothing to
escape. Ten starts cannot rule out a rare trap, but a coefficient of variation of a third of a
percent in both arms is strong evidence against one.

That still rules out the two options that exist to traverse a rough landscape. Population
annealing resamples on the Boltzmann weight, and at the temperature this schedule spends its time
at those weights degenerate and the population collapses onto the lowest energy replica in a
single rung, `rho_t` going to R and the effective ensemble size to one. It would pay that price
to traverse a ladder worth nothing. Parallel tempering fails the same way: swapping with hotter
replicas helps only if they explore usefully, and here they do not.

It does not license removing the temperature. It is inert, the measurement is arcs only on three
blocks, and there is no reason to disturb a schedule inherited from the reference for no gain.

The smooth stage is **not** covered. Two attempts were discarded, one for the matched budget bias
above and one because the fixture dropped the orientation term entirely. A harness for it also
has to reproduce production behaviour before it can be believed: the current one converges in one
round where the real pipeline runs eighty three to a hundred and three on the same region.

`playground/trapping_spread.py` runs the spread measurement, on blocks from `capture_arcs.py`.

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

**Parallel tempering. Ruled out by the measurement above.** Replicas on a temperature ladder that
swap. It cures a chain stuck in a basin, and a ladder buys nothing here.

**Population annealing. Read the papers, then ruled out by the measurement above.** Start R
replicas at infinite temperature, resample each rung on the Boltzmann weight, run MCMC rounds,
repeat down the ladder. It is reported comparably efficient to parallel tempering and both far
more efficient than simulated annealing, its authors have used it to find spin glass ground
states so it has a track record as an optimiser, and it is parallel across replicas, which is the
width the kernels leave idle. Its resampling correlates the population, but that is measurable
rather than fatal: the mean square family size `rho_t = lim R * sum_i n_i^2` gives `R / rho_t` as
the effective number of independent members.

None of that survives a landscape where the ladder is worth nothing. At the effectively zero
temperature our schedule spends its time at, the resampling weights degenerate and the population
collapses to one effective member in a single rung.

Two things worth keeping from the reading anyway. The 230 times GPU figure is against a single
core: the paper says its CPU code parallelises with close to perfect scaling and that comparison
against a parallel CPU means dividing by the core count, so against our sixteen it is nearer 14,
and the further tenfold is Ising multi-spin coding which does nothing for continuous positions.
And the results are all discrete spin models; dense fluids and biomolecules appear in the
conclusions as a prospect, not a result.

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
