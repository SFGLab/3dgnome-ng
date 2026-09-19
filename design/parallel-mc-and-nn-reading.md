# Parallel Monte Carlo and neural approaches, what was read and what applies

Started 2026-09-16. A reading list against the one question that matters for us: the smooth
stage is a Metropolis chain on one shared state, step N+1 depends on step N, and on the JAX
path a step costs a fixed latency whatever the launch width (`design/kernel-performance.md`).
Width across chains is already spent. What is left is width inside one chain, and
`design/intra-chain-parallelism.md` keeps exactly one such variant open, the best of K
speculative moves per bead. Every paper below is read against that.

## Applies

**Multiple try Metropolis on GPU.** Suchoski, Stage, Gurung and Baccam, 2022, "GPU accelerated
parallel processing for large scale Monte Carlo analysis: COVID-19 parameter estimation and new
case forecasting", Frontiers in Applied Mathematics and Statistics,
<https://doi.org/10.3389/fams.2022.818016>. Per iteration each of `Nc` chains draws `Nt`
proposals, all `Nc x Nt` likelihoods run at once on the GPU, one proposal per chain is chosen by
weight and accepted with the multiple try ratio, and the proposal covariance is adapted from the
chains. The parallelism sits above the likelihood, not inside it. 13.6x on one RTX 2080 Ti
against a 12 core CPU at 32 chains and 128 tries, 56x on eight A100s, and the gain is fewer
iterations at a fixed cost per iteration. That is the shape of our smooth kernel, so this is
the principled form of best of K. It pays most late in the anneal, where the reference log
showed 54 accepted of 50,000 proposals per milestone and `K` tries raise the chance of an
accepted move about `K` fold at unchanged cost per step. The multiple try ratio keeps detailed
balance for a sampler. We are an optimiser, so the greedy best of `K` is the version to build,
and it changes structures, so it needs the battery. The adaptive covariance is for a handful of
parameters and does not carry over to a per bead step. It gains nothing on the numba path, where
`K` tries cost `K` times the work.

The experiment: in the JAX smooth kernel draw `K` displacements per step instead of one, vmap
the local delta, take the minimum, run the existing accept. Measure rounds to convergence and
wall against `K = 1` on one 60 Mb structure, then the three cell battery if it wins.

**Metropolis within Gibbs over conditionally independent blocks.** Szalai-Gindl, Loredo,
Kelly, Csabai, Budavári and Dobos, "GPU accelerated hierarchical Bayesian inference with
application to modeling cosmic populations: CUDAHM", Astronomy and Computing 2018, arXiv
<https://arxiv.org/abs/2105.08026>. Plate level parameters are conditionally independent given
the upper level ones, so every plate is updated at once on the GPU by robust adaptive
Metropolis while the upper level is sampled on the host, 300,000 objects in about an hour on
one K40c. For a chain with local energy the same conditional independence holds between beads
that share no term, which is the graph coloured checkerboard. That was built and measured here:
it is fine for the smooth stage and, on the arcs stage from a collapsed seed, converges to a
worse compact minimum than the sequential update, which the hybrid then repairs at a cost that
made it slower end to end (`design/intra-chain-parallelism.md`, "What has been tried"). The
paper is the justification for the checkerboard, not a way round what was measured. With the
arcs stage on the L-BFGS solver in production the checkerboard's remaining home is the smooth
stage, where it was already fine, and the open question is only whether it beats best of `K`
there.

**Speculative prefetching, the exact form of width inside one chain.** Brockwell, "Parallel
Markov chain Monte Carlo simulation by pre-fetching", Journal of Computational and Graphical
Statistics 2006, and Angelino, Kohler, Waterland, Seltzer and Adams, "Accelerating MCMC via
parallel predictive prefetching", UAI 2014, <https://arxiv.org/abs/1403.7265>. The chain's next
`d` steps form a binary tree of accept and reject branches. Evaluate its nodes in parallel,
then walk the tree with the serial accept decisions, discarding what was not taken. The result
is identical to serial execution, and the speedup is `log2` of the lane count in general but
close to linear in the lanes where one branch dominates, which Angelino et al. exploit by
predicting acceptance. Our late anneal is that case: the reference log showed 54 accepted of
50,000 proposals per milestone, so the tree is a straight line of rejections from one state,
and evaluating the next `K` proposals from the current state in parallel and keeping the first
accepted in proposal order is the whole scheme. It is the same chain as serial, so it needs no
battery, only a wall measurement. It gains nothing early in the anneal where acceptance is high.
Against best of `K`: prefetching keeps the serial result and gains only where acceptance is
rare, best of `K` changes the result and gains everywhere. Build prefetching first, since it is
free of validation, then best of `K` on top if the early anneal is a large enough share of the
wall. The first measurement is the acceptance profile over the rounds of a real smooth run,
which decides how much either can give.

**Spatial checkerboard for off lattice particles.** Anderson, Jankowski, Grubb, Engel and
Glotzer, "Massively parallel Monte Carlo for many particle simulations on GPUs", Journal of
Computational Physics 2013, <https://arxiv.org/abs/1211.1646>, the scheme in HOOMD's HPMC.
Space is cut into cells wider than the interaction range, a sweep updates one cell of every
checkerboard colour at once with each move confined to its cell, and the grid is shifted at
random between sweeps. Detailed balance holds without a graph colouring, and it reached 148
times one CPU core on hard disks. Our earlier checkerboard was coloured on the interaction graph
and is recorded in `design/intra-chain-parallelism.md`, fine on the smooth stage and compacting
on the arcs stage from a collapsed seed. The spatial form is the same idea on the cell grid the
excluded volume already keeps, and the smooth stage's terms are all within a bond or two of a
bead except the heat term, which is off in production. It is the other candidate for the smooth
stage beside prefetching, and the two compose: cells in parallel, prefetching inside a cell.

**Rejection free selection.** Bortz, Kalos and Lebowitz 1975 and its continuous extension,
<https://arxiv.org/abs/cond-mat/0211164>. Every candidate move is weighted by its acceptance
probability and one is drawn, so every step moves. With continuous displacements the candidate
set has to be a finite draw, and drawing among `K` proposals by weight is multiple try
Metropolis again. Nothing separate to build.

**Event chain Monte Carlo.** Bernard, Krauth and Wilson 2009, and for polymers Kampmann, Boltz
and Kierfeld, "Parallelized event chain algorithm for dense hard sphere and polymer systems",
Journal of Computational Physics 2015, <https://arxiv.org/abs/1409.6948>. Rejection free and
lifted, one bead moves in a straight line until an event hands the motion to the next, with a
domain decomposed parallel version, and it reaches molecular dynamics speeds on hard sphere
polymer melts. It needs the energy factorised into pairwise terms and is a sampler at one
temperature. Our smooth energy has the cubed angle term and the orientation term, and the run
is an anneal. A rewrite of the stage, not a lever on it. Not now.

**Asynchronous updates.** Terenin, Simpson and Draper, "Asynchronous Gibbs sampling", AISTATS
2020, <https://proceedings.mlr.press/v108/terenin20a.html>. Run the sequential update in
parallel without synchronising, which is what cudaMMC's warps do to bead positions. It can
diverge, the exact variant needs a correction, and either way the result depends on thread
timing. The parity gate and the seeded reproducibility of every kernel rule it out here.

**Parallel across the sequence.** Zoltowski, Wu, Gonzalez, Kozachkov and Linderman,
"Parallelizing MCMC across the sequence length", 2025, <https://arxiv.org/abs/2508.18413>. A
whole trajectory of Gibbs, Langevin or Hamiltonian steps is solved as a fixed point with
parallel Newton iterations, tens of iterations for hundreds of thousands of samples. It needs a
smooth kernel. Metropolis accept and reject is a discontinuity in the state map, so it does not
apply to ours.

## Already done here, at the block level

**Graph partitioning across GPUs with boundary coordination.** Li, Landry and Mettu, "GPU
acceleration for Markov chain Monte Carlo sampling", Proceedings of the 4th International
Conference on AI-ML Systems, 2024, <https://doi.org/10.1145/3703412.3703428>, open access. The
interaction graph of a probabilistic model is partitioned, each subgraph is sampled on its own
GPU, information about the shared boundary is exchanged over NVLink during sampling and the
pieces are merged. Demonstrated on protein conformational stability, up to 4.0x on two A2000s and
2.4x to 2.7x on eight V100s over an adaptive Monte Carlo sampler. Only the abstract and the
reported numbers were readable; the ACM PDF is behind a script wall. For us the partition is the
interaction block, every block's chain already runs in parallel in one launch, and the boundary
is handled after sampling by the rigid stitch and the cross block relaxation rather than during
it. The chain inside a partition stays sequential in their scheme as in ours, so it does not
touch the one chain problem. What it suggests, unmeasured, is exchanging boundary information
during the smooth stage, a relaxation round between milestones rather than one pass at the end.
Their gain of 2.4x to 4x across several GPUs is below what per block batching on one GPU already
gives, so the refinement is about boundary quality, not speed.

## Does not apply

Three proton therapy dose codes, read because they were on a list. All three parallelise over
independent particle histories, one GPU thread per particle, and their contribution is
engineering against thread divergence. None touches a chain on a shared state.

- gPMC. Jia, Schuemann, Paganetti and Jiang, "GPU based fast Monte Carlo dose calculation for
  proton therapy", Physics in Medicine and Biology 2012,
  <https://doi.org/10.1088/0031-9155/57/23/7783>.
- VPMC. Shan et al., "Virtual particle Monte Carlo: a new concept to avoid simulating secondary
  particles in proton therapy dose calculation", Medical Physics 2022,
  <https://doi.org/10.1002/mp.15913>. Secondary particle spawning is replaced by pre tabulated
  virtual particles so every thread runs identical control flow on a fixed particle count. Our
  vmapped chains already run identical control flow, and our limit is latency per sequential
  step, not divergence.
- ARCHER. Chang et al., "ARCHER, a Monte Carlo code for multi particle radiotherapy through GPU
  accelerated simulation and DL based denoising", EPJ Nuclear Sciences and Technologies 2025,
  <https://doi.org/10.1051/epjn/2025008>. The one transferable idea is the denoiser, fewer
  histories and a learned model on the noisy output. For us that would mean stopping the smooth
  stage early and letting a network finish a structure, a different product, not built.

Parallel tempering and population annealing are ruled out by the funnel measurement in
`design/intra-chain-parallelism.md` and are not repeated here.

## Neural approaches

A 3D U-Net is an image to image network on a voxel grid. Nothing in the chromatin literature
uses one for structure, and our data is a chain and a contact matrix, not a volume. What exists
is of three kinds.

- **Direct reconstruction from a contact map.** HiC-GNN, Highsmith and Cheng, Computational and
  Structural Biotechnology Journal 2023, <https://doi.org/10.1016/j.csbj.2022.12.051>; an SO(3)
  equivariant graph network on single cell Hi-C, NAR Genomics and Bioinformatics 2025,
  <https://doi.org/10.1093/nargab/lqaf027>. One consensus structure per map at 100 kb to 1 Mb,
  no loops, no ensemble. Not comparable with what we produce.
- **Generative ensembles.** ChromoGen, Schuette, Lao and Zhang, Science Advances 2025,
  <https://doi.org/10.1126/sciadv.adr8265>, a diffusion model trained on 11 million simulated
  conformations, conditioned on sequence and DNase, a thousand structures in 20 minutes at
  20 kb over a 1.28 Mb window, and a diffusion transformer conditioned on Hi-C for E. coli,
  <https://arxiv.org/abs/2603.07472>. This is the only line that competes with what we do, and
  it is the direction on the cnf branch (`docs/flow-matching-ensembles.md`): a network learns
  from our own MC ensembles and then samples in seconds. It needs a large training set of our
  structures first, which the cell line and trio runs of September 2026 start to provide.
- **Contact map prediction from sequence or epigenome.** Enformer, Avsec et al., Nature Methods
  2021, <https://doi.org/10.1038/s41592-021-01252-x>; ChromNet, Advanced Science 2026,
  <https://doi.org/10.1002/advs.202508110>; CGLoop; HiCDiffusion. They predict maps, not
  structures, and could only feed us input. Survey: "Machine and deep learning methods for
  predicting 3D genome organization", <https://arxiv.org/abs/2403.03231>.

## The acceptance profile, measured 2026-09-19

`playground/accept_profile.py` on GM12878 chr1:1-8 Mb, five blocks of 48 to 2,011 beads, the
numba kernel from each block's own seed, 186 rounds of 50,000 proposals. Share of rounds by
acceptance: 0.35 under 0.1 percent, 0.58 between 0.1 and 1 percent, 0.08 between 1 and 10
percent, none above. Only the first round of a block accepts 5 to 8 percent; the median round
of the two large blocks accepts 0.2 percent. A batch of `K` proposals from one state, first
accepted kept, advances the chain about `K / (1 + pK)` steps at acceptance `p` for the cost of
about one step on the latency bound kernel: at 0.2 percent that is 14x at `K` 16, 25x at 32,
45x at 64, with the serial chain reproduced exactly. The stitch and relaxation ablation of the
same day is in `design/validation-2026-09.md`; both passes are off at chromosome scope, so the
gate before kernel work is passed.

## Prefetching built and measured, 2026-09-19

Commit c6f54ef, `[simulation_arcs_smooth] prefetch`, opt in at 1. At 1 the working tree is
byte identical to the commit before it on GM12878 chr1:1-8 Mb through the JAX executor, with
the temperature now carried per chain across the vmap. The smooth stage's wall on the RTX
4060 Ti, one structure, everything else production:

| region | K 1 | K 8 | K 16 | K 32 | K 64 |
|---|---|---|---|---|---|
| chr1:1-8 Mb, 5 blocks to 2,048 beads | 57.3 s | 16.0 s | 9.4 s | 6.2 s | 4.2 s |
| chr1:1-60 Mb, 11 blocks to 16,384 beads | 419.8 s | | | 116.1 s | 144.6 s |

13.6x on the small blocks at 64, 3.6x on the large ones at 32 and less at 64, because the JAX
kernel scans every bead for the excluded volume and a batch does that `K` times, so on a
16,384 bead block the batch stops being one step's latency somewhere between 32 and 64.
Rounds to convergence are level, 55 to 62 on the small region and 385 to 427 on the large.
Hi-C, the exponent and Rg are level on every arm; the single structure MultiMM number on the
large region is not comparable with the three structure arm it sits beside and is being
remeasured at three: 0.665 against 0.669 with every other number level, and K 16 on the
large region is 141 s, so 32 is the production value from 2026-09-19. The numba cell grid,
ported, would make the batch cost flat in `N` and recover the small block ratio on large
blocks; that is the next kernel lever after this one.

## Where this leaves the plan

**First, before any kernel work: measure the stitch and the relaxation at chromosome scope.**
Decided 2026-09-17. Both passes were built when every block's arcs were solved alone. The
joint solve has since removed the defect the stitch was built for, and the stitch's centroid
excluded volume, one term per block pair at the two radii of gyration added, is an inflation
force on a compact chromosome whose blocks interdigitate. The relaxation runs windowed at one
in production and is probably close to null. `slurm/ensemble/assemble_ablation.sh` runs four
arms on GM12878 chr1:1-60 Mb, production, stitch off, relaxation off, neither, three structures
each, then the battery and the boundary report. If neither holds Hi-C and the boundary ratio
at a smaller Rg, both passes are deleted per the project rule. Only then the kernel work.

Two parallel MC experiments on the JAX smooth kernel, in this order: speculative prefetching,
the first accepted of `K` proposals from the current state, which reproduces the serial chain
and needs only a wall measurement; then best of `K`, the greedy multiple try form, which changes
structures and needs the battery. The spatial checkerboard on the excluded volume cell grid is
the third candidate and composes with both. Before any of them, the acceptance profile over the
rounds of one real smooth run, since prefetching pays only where acceptance is rare. One neural direction, a ChromoGen style generator trained on our ensembles,
already planned on the cnf branch and waiting on training data. Nothing else on the list changes
what is built.
