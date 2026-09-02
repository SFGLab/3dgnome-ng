# Anchor placement

Reconstructions come out as compact balls of anchors scattered in space and joined by thin
strands. This records what causes that, what has been measured, what has been ruled out, and
which changes are worth trying.

Status. Diagnosis complete and quantified. No fix implemented. Nothing here has been built.

## Symptom

A whole chromosome renders as roughly one ball per interaction block, each ball a tight tangle,
with long single strands between them. On chr1 that is about 52 balls for the arc gap block
partition and about 69 for the trio partition.

## What was measured

All numbers are from this repository, not from comparison against another implementation.

### Anchors inside a block collapse

The arc target does not depend on genomic separation. `freq_to_distance(freq)` is
`base_level + scale / exp(a * (freq + shift))`, a function of PET count alone, and at
`base_level = 0.2` with `a = 0.2` and `shift = 8` any arc above about twenty reads returns
approximately 0.2. Measured on H1ESC chr11, `playground/scale_clash.py`:

| genomic separation | arc target | chain target | realised |
|---|---|---|---|
| 5 to 20 kb | 0.200 | 5.14 | 0.359 |
| 20 to 50 kb | 0.202 | 8.24 | 0.369 |
| 50 to 100 kb | 0.213 | 13.64 | 0.404 |
| 100 to 300 kb | 0.249 | 25.51 | 0.450 |
| 300 kb to 1 Mb | 0.309 | 50.73 | 0.530 |

The arc target is flat across two orders of magnitude and the realised positions follow it, so
anchors sit 14 to 100 times closer than the chain expects.

### Almost no anchor pair is constrained at all

`calc_anchor_expected_distances` gives a pair either an arc target or -1, which the kernel scores
as an unbounded `1/d` repulsion. Measured on HG00512 chr1 over the twelve largest blocks,
5.17M anchor pairs, `playground/hic_coverage.py`:

| separation | pairs | arc | Hi-C only | neither |
|---|---|---|---|---|
| 0 to 5 kb | 20,942 | 0.0% | 0.0% | 100.0% |
| 5 to 20 kb | 23,156 | 5.2% | 0.0% | 94.7% |
| 50 to 100 kb | 62,236 | 3.2% | 0.2% | 96.6% |
| 100 to 300 kb | 220,539 | 1.9% | 0.2% | 97.9% |
| above 1 Mb | 4,173,583 | 0.0% | 0.1% | 99.9% |

Between 95 and 100 percent of pairs have nothing but the scale free repulsion. Arc springs pulling
everything to 0.2 against a repulsion with no sense of scale equilibrates at a uniform 0.4, which
is the ball.

### Anchors across a block boundary are pushed apart

The arcs MC runs per block, so a consecutive anchor pair split by a boundary has no term coupling
it. Both anchors are placed only through their own block centroid. Measured on HG00512 chr1,
`playground/boundary_gap.py`, matched on genomic gap between 64 and 700 kb:

| consecutive pairs | n | realised over target |
|---|---|---|
| inside one block | 793 | 0.074 |
| crossing a boundary | 35 | 1.965 |

The two errors run in opposite directions. Inside a block anchors are about thirteen times too
close. Across a boundary they are about twice too far. The boundary itself costs a factor of 27
independent of genomic distance. Note that n is only 35 for the crossing pairs, so the factor of
two is indicative rather than settled.

### The subanchor chain balloons

Between anchors 35 kb apart the chain is handed about 8.7 units of contour for a gap of 0.37,
roughly 36 times more than it needs. Anchors are held fixed during smooth MC, so the excess
relaxes into a free coil. That is the fuzz on each ball rather than a clean path.

### The result is a bimodal distance curve

Median spatial distance against genomic separation is flat near 1.8 from 1 kb to 400 kb, then
jumps to about 100 at 6.4 Mb. Realised over target is 1.9 at 5 kb and 0.57 at 50 kb, so the
structures do not follow `genomic_length_to_distance` at any scale and the error changes sign.

## What has been ruled out

Each of these was tested and is not the cause.

- Excluded volume, confinement and the CTCF motif weight. Ablated one at a time on a 10 Mb
  region. Confinement is inert, it never binds. Excluded volume accounts for about 23 percent of
  the short range inflation and motif for about 8 percent. The defect survives all three off.
- Smooth MC failing to converge. It converges, 222 rounds and 11.1M steps, one of one converged.
  It reaches a minimum that is not at the target distances.
- Ensemble depth. Five, ten and twenty conformations give the same answer to within 0.006 on a
  pinned gene set.
- Being worse than cudaMMC. Its leaf chain is equally flat. This is the algorithm's output, not a
  defect this port introduced. An earlier investigation chased four imaginary defects because it
  compared our raw beads against cudaMMC's `.smooth.txt`, which is the chain resampled onto a
  uniform 1 kb lattice and so manufactures short range structure that the model does not have.
- Hi-C supplying the missing targets. It reaches 0.1 percent of arcless pairs. Anchors cover
  little sequence, so a 25 kb bin rarely lands on two of them. Where Hi-C does exist its implied
  distance scales correctly, 2.57 below 50 kb against 39.86 above 300 kb, so the physics is right
  and only the sampling is too sparse.

## Solutions to explore

### A. Excluded volume radius that depends on genomic separation

Replace the constant `exclusion_radius_arcs` with `beta * genomic_length_to_distance(separation)`
for pairs with no arc. This is what a self avoiding polymer does and it is the mechanism that
produces contact probability decay. It addresses the collapse, which is the larger of the two
errors and affects almost every pair.

Keep it a floor rather than a spring. Arcs must still be able to pull distant loci together,
since that is the biology the model exists to capture. A two sided spring on every pair would
impose an ideal chain and erase the loops.

Cost is free. The kernel already visits every pair for the repulsion. It also retires the
unbounded `1/d`, which is the known cause of the small interaction block blow ups.

`beta` decides everything. Near one it forces ideal chain geometry and the structures inflate.
Calibrate it against a contact probability curve derived from the cell line's own Hi-C rather
than by eye.

### B. Stitch block edges together

Add a bond between the last anchor of one block and the first anchor of the next, targeting
`genomic_length_to_distance` of their genomic gap. About 68 boundaries on chr1, so one spring
each and no quadratic cost.

Those pairs are few but they set the global layout, so the effect on the scatter is out of
proportion to the count. This is also the narrow version of what `use_segment_arcs` attempted and
lost on. That change widened the arcs MC to whole segments, which dragged every arcless pair into
the unbounded repulsion and raised repulsion per arc by 2.06 times under arc gap blocks. Touching
only the boundary pairs avoids that entirely.

Placement needs a stage that sees both blocks. The natural shape is a short pass after anchors
are placed, adjusting each block rigidly to satisfy its edge bonds while leaving the intra block
arrangement the arcs MC produced untouched.

### C. Chain bonds between consecutive anchors in the arcs MC

The most direct statement of the missing constraint. Left last because it competes with the arc
springs and needs a weight balance that A and B do not.

### D. Finer blocks

Smaller blocks would let the hierarchy supply structure at shorter range, since it already places
block centroids by genomic distance. TAD blocks are about 340 kb against 4.8 Mb for arc gaps.
Rejected as a primary route because TAD boundaries orphan 43.6 percent of arcs, measured as 5437
of 12474 on GM12878 chr1 against exactly zero under arc gaps.

## Validation

The measurements to judge a change by already exist.

| gate | tool | current |
|---|---|---|
| realised over target | `playground/target_check.py` | 1.9 at 5 kb, 0.57 at 50 kb |
| distance against separation | `playground/scaling_curve.py` | 1.3 fold over 7 kb to 380 kb |
| boundary penalty | `playground/boundary_gap.py` | 26.7 times |
| enhancer promoter expression | `enhancer3d/playground/beyond_linear.py` | 86 percent of v4 |
| byte exact parity, flag off | see AGENTS.md parity gate | must stay identical |

The target for the second gate should come from a contact probability curve computed from the
cell line's own Hi-C, not from matching cudaMMC. Contact probability decaying near s to the minus
one implies distance growing near s to the one third, so a flat curve is wrong on the data's own
terms. That gives an objective goal and calibrates `beta`.

## Decisions worth making before building

- Both changes are divergences from the reference and from cudaMMC. They must be opt in, default
  off, and recorded in the AGENTS.md divergences section. The parity baseline does not move.
- Better geometry may not improve the expression statistic. If that happens it should be reported
  as evidence the statistic is limited by annotation, not used as a reason to tune `beta` until it
  moves. Agreeing this in advance is what keeps the exercise honest.
- The boundary factor of two rests on 35 pairs. Widen that measurement to a cell line with more
  boundaries before sizing B against it.

## Order

1. Derive the contact probability curve and calibrate `beta`. No GPU, uses mcools already fetched.
2. Widen the boundary measurement beyond 35 pairs.
3. Build B. It is smaller, cheaper and independent of `beta`.
4. Build A.
5. Only then consider C.
