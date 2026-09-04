# Intra chain parallelism

How to make one Monte Carlo chain converge in fewer steps, or use more of a device per step.
This is the half of the performance problem that batching cannot reach, and it is where the work
goes once [kernel-performance.md](kernel-performance.md) runs out.

Status. Nothing here is built. Four things have been tried and three failed, which is most of
what this document is for.

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
of replicas rather than the thousands the device could hold.

**Population annealing.** A population carried through the temperature schedule, resampled toward
the Boltzmann weight at each step rather than taking the best at the end. The natural fit for a
latency bound kernel with idle width, and GPU implementations report around 230 times over a
serial CPU. The problem is that resampling kills replica diversity, and diversity is the product
here: the hybrid polish already homogenised an ensemble from 1.024 to 0.92 and needed re-noising
to recover it. This one fights what we are trying to make.

**Event chain Monte Carlo.** Rejection free. A displacement continues until an event under a
factorised Metropolis filter, instead of proposing, evaluating, accepting or rejecting. It
removes the dependent chain that is the latency floor rather than working around it. Developed
for dense polymer melts with excluded volume, which is our system, reaching speeds comparable to
optimised molecular dynamics with the gain growing at density. Also the largest rewrite, and a
sampler by construction, so running it under an annealing schedule is off label.

## References

Population annealing on GPU, arXiv 1703.03676, and its theory, arXiv 1508.05647. Event chain
Monte Carlo for dense polymer melts, arXiv 1502.06447. cudaMMC, Bioinformatics 39(10) btad588,
source at `~/Desktop/bio/cudaMMC`.
