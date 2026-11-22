# The orientation term of the smooth stage

The term gives the chain at a CTCF anchor a direction that follows the motifs of its loops. This note has the
term as it is coded, the trace of 2026-11-22 of the damage it does to the chain and the open design.

## The term as coded

- A site is an anchor of the block with a CTCF loop to another anchor of the block. A loop of another factor
  gives no site.
- The vector of a site at bead `a` is `pos[a + 1] - pos[a - 1]`, scaled to length one. An anchor with the motif
  L has the vector reversed. An anchor without a motif has it as it is.
- A loop between the sites `i` and `j` has the energy `weight * w * ((1 - v_i . v_j) / 2)^2`, with `w` the square
  root of the PET count of the loop. The energy is zero for two equal vectors and largest for two opposite ones.
  A convergent pair then has the chain run in opposite directions through its two anchors.
- `[motif_orientation] weight` is 50 in production. A bond has `[springs] stretch_constant` 0.1, and
  `squeeze_constant` 0.1 on the default chain and 1.0 on the uniform chain.
- The kernels are `_local_score_orientation_nb` in `gnome3d/mc/numba/terms.py` and `_local_orientation_at` in
  `gnome3d/mc/jax/smooth.py`. The loops and their weights are built in `gnome3d/skeleton.py`.

The reference has the same form and the same three numbers: the vector over the two neighbours, the weight 50
and bonds at 0.1.

## Reports

The maintainer reported on 2026-11-21 two jobs of the site with long straight strands folded into sharp hairpins,
over about half of a model by eye, at chr3 134.39 Mb and at chr1 11.94 Mb.

The trace of 2026-11-19 in `design/features.md` on chr1 108.5 to 111 Mb counted 117 bonds over 3 beads on the
default chain against 3 with the term off.

## Trace of 2026-11-22

`playground/orientation_trace.py` records every call of the numba smooth kernel with its chain before and after
and reads the chain at the sites of the term. GM12878 chr1 11 to 13 Mb, 4 structures per arm, production configs,
before the cross block relaxation. The arm with the term off is read at the sites of the arm with it on.

| | default chain, on | default chain, off | uniform chain, on | uniform chain, off |
|---|---|---|---|---|
| sites | 368 | 368 | 228 | 228 |
| bonds of a site to its neighbours over 2 targets | 23.9 % | 1.5 % | 33.3 % | 5.3 % |
| bonds from those neighbours outwards over 2 targets | 22.7 % | 0.5 % | 30.3 % | 1.3 % |
| bonds three beads or more from a fixed bead over 2 targets | 0.6 % | 0.3 % | 0.9 % | 0.5 % |
| longest bond, in targets | 6.9 | 3.0 | 6.6 | 3.3 |
| bends sharper than 120 degrees within two beads of a site | 26.4 % | 4.6 % | 21.3 % | 5.7 % |
| bends sharper than 120 degrees three beads or more from a fixed bead | 0.4 % | 0.2 % | 1.1 % | 0.9 % |
| chord of a site over 1.5 | 16.6 % | 1.4 % | 36.4 % | 7.9 % |
| chord of a site under 0.5 | 13.3 % | 6.2 % | 7.5 % | 3.5 % |
| misfit of the term before the stage | 0.324 | 0.324 | 0.349 | 0.349 |
| misfit of the term after the stage | 0.076 | 0.338 | 0.007 | 0.319 |
| bond energy of the final chains | 146 | 19 | 167 | 35 |
| orientation energy of the final chains | 30,920 | 87,612 | 836 | 53,322 |
| beads within five beads of a site | 30.9 % | 30.9 % | 23.5 % | 23.5 % |

The chord of a site is the distance of its two neighbours over the sum of the two bond targets. The misfit is the
mean of `((1 - v_i . v_j) / 2)^2` over the loops. Two random vectors have 0.333.

- The damage is at the sites. A quarter to a third of the four bonds around a site are over twice their target
  with the term on. Away from the fixed beads the bonds are the same with the term on and off.
- The bonds of a site and the bonds one further out stretch by the same share. The term moves the two neighbours
  of a site one by one to the places that give the vector its direction, and both bonds of a moved neighbour
  stretch. The chord is both longer and shorter than without the term: a neighbour is pulled out along the
  direction or folded back beside the other one.
- The chain enters the stage with a random vector at every site: the misfit is that of two random vectors. No
  earlier stage gives the chain a direction at an anchor. The whole direction is made by two beads.
- The energy of the term is 200 times the energy of the bonds at the end of the stage on the default chain and
  5 times on the uniform chain. At the start of the stage it is 87,668 and 57,645. A move that turns the vector is
  taken at any cost in bonds.
- On the default chain the term ends at a misfit of 0.076 with 35 % of its energy left. On the uniform chain it
  ends at 0.007 with 1.4 %.
- A quarter to a third of the beads of a model are within five beads of a site.

## Open

The ablation on inputs the run did not read, 18 rows of twenty structures, the uniform chain with the term and
without it and the default chain without it, is queued on the workstation as `orn20`. The design of the term
follows that result and the maintainer's choice.
