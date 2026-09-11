# A/B compartments in 3dgnome, what was tried and what holds

Closed 2026-09-11. The compartment term is MultiMM's block copolymer affinity, ported in
August 2026 and measured through September. This note records where it stands and why.

## What holds

- **Chromosome scope for the arcs stage is production** (`[simulation_arcs] scope =
  chromosome`, `start = hilbert`, commits 7b2bed8 and 3c6dbda). Every anchor of a chromosome
  is solved as one problem, the anchors starting along a Hilbert curve scaled to the law's
  bond, since across blocks the solve holds no arc and the contact background 80 of 3.5
  million pairs, so the start is the long range arrangement; the curve gives it the cube
  root size growth the maps show beyond a megabase, with no sphere weight. Like for like on
  three cells against the walk start with a sphere at 10, which was production for a day:
  Pearson up on all, SCC level, MultiMM level on H1ESC and 0.01 to 0.04 down on the others,
  cross block overlaps 176, 173 and 143 to 163, 126 and 100 per thousand. Three cell gate on chr1:1-60 Mb against the deep
  maps: Pearson 0.271, 0.282 and 0.301 to 0.291, 0.318 and 0.304; MultiMM 0.607, 0.568 and
  0.652 to 0.674, 0.667 and 0.673; SCC level within 0.01; cross block overlaps 320, 416 and
  170 to 176, 173 and 143 per thousand beads. This came out of the compartment work and is
  its lasting result.
- **The compartment term is opt in, off in production.** `[compartments] use_compartments`
  at weight 0.5 with chromosome scope raises the compartment saddle on H1ESC 1.02 to 2.15
  and HFFC6 0.71 to 1.09, not on GM12878, and costs SCC and MultiMM 0.07 to 0.10 on every
  cell, on the balanced GM12878 chr2 window 0.14 and 0.22. A run that wants compartments can
  turn it on; the default does not.
- **The accessibility terms are off.** Fibre compaction moves the accessibility saddle away
  from experiment on four of four regions at 10 kb; bridging is inconsistent.

## What was learned, in order

1. **On 12 Mb windows the term works** (four of four GM12878 regions, saddle up 0.3 to 0.6,
   kappa up, Rg down 0 to 13 percent) once the baseline was corrected and the eigenvector came
   from the deep map. The B4 verdict of 2026-08 was made on an over compartmentalised
   baseline and a noisy track.
2. **On 60 Mb it did nothing under block scope**, because a compartment is a pattern over
   many blocks and each block's chain is solved alone. The boundary stitch and the cross block
   relaxation then separate blocks, and cross block compartments are interpenetrating blocks.
   Measured by a stage trace on one structure with the true block membership: after block
   placement 63,775 cross block bead pairs touch, after the stitch 4,595, after the
   relaxation 302, and the cross block saddle goes with them.
3. **Every fix inside the end passes failed.** A block affinity in the stitch, null to negative
   at every weight. The stitch repulsion at a quarter, no cross block gain, more overlaps.
   The relaxation radius at the overlap definition, half the compartments back at 60 percent
   more cross block overlaps. The term kept inside the relaxation, even as a chromosome wide
   anneal at forty times the weight, beads move a fifth of a bond because 4,096 anchors pin
   the chain.
4. **Solving the chromosome as one piece places compartments right and the size wrong.** The
   start decides the size, since the arcs energy has no term between far pairs: every anchor
   at one point collapses to Rg 14, a walk stays at 44, a walk inside the law's sphere at
   weight 10 lands at production's 22. That became chromosome scope.
5. **With chromosome scope the term acts across blocks**, GM12878 chr1 cross block saddle
   0.97 to 1.44 with the stitch and the relaxation on, at the SCC and MultiMM cost above.
6. **The anchor level term is not a lever.** A well between like anchors inside the joint
   solve (`[compartments] apply_to_arcs`, off): at 0.01 a mild gain at no cost on chr1, at
   0.05 a reversal, at 0.2 anchors piled into clusters; at 0.05 on the balanced chr2 window
   compaction to Rg 18 and overlaps up two and a half times, and at 0.2 a saddle of 1.60
   bought with anchor and cross block overlaps four times the no term value.
7. **GM12878 is not a data problem.** Its compartment track agrees with the other cells to a
   correlation of 0.7 to 0.8 up to sign, and its loops look like theirs in every summary. Its
   production run had compartment like structure on chr1 without any term, which chromosome
   scope removes as it does on every cell. On the balanced chr2 window the joint solve itself
   anti sorts, like blocks at 2.2 radii and unlike at 1.4, because compartments alternate
   along the genome and nothing pulls a block toward its own kind past the next one. The
   contact background should, and holds a few percent of far pairs only, anchors being 13 kb
   against 25 kb pixels.

## What would move it

A denser anchor level contact map, the deep map binned at 5 kb over each anchor's window, so
the joint solve places blocks relative to each other from the data. That is a data path
change, not a force, and the open item from `anchor-placement.md` as well.

## Instruments built on the way

`playground/saddle_offline.py` scores finished structures for the saddle, eigenvector
correlation and kappa; `playground/saddle_split.py` splits the saddle over within block and
cross block pairs, each class on its own decay; `playground/stage_trace.py` runs placement
in process with the true blocks, caches it, and reports what the stitch and the relaxation
do to compartments and to beads. Two instrument faults were found and fixed: the split had
shared one expectation between the classes, and the anchor heatmap was cut out of an N by N
matrix that production never read, 45 GB on a chromosome sized block.

## Accessibility, closed 2026-09-11

The two HiP-HoP mechanisms driven from ATAC, bridging between open beads and fibre
compaction of closed ones, were given their functional test: GM12878 whole chr1 on production
settings, accessibility in binary mode at the 80th percentile, five structures per arm, scored
by the enhancer3D pipeline's Spearman correlation between a gene's closest enhancer distance
in 3D and its expression over 813 genes. Off gives -0.189, bridging -0.194, fibre compaction
-0.131. Bridging is a null and fibre compaction weakens the signal by about 1.7 standard
errors. Together with the 10 kb accessibility saddle, where fibre moves away from experiment
on four regions of four and bridging is inconsistent, both terms stay off. The loops already
sit at open chromatin, so the track adds nothing the anchors do not carry.
