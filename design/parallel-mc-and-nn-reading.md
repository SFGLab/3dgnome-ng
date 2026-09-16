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

## Where this leaves the plan

One parallel MC experiment, best of `K` in the JAX smooth kernel, with multiple try Metropolis
as its reference. One neural direction, a ChromoGen style generator trained on our ensembles,
already planned on the cnf branch and waiting on training data. Nothing else on the list changes
what is built.
