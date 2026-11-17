# Models against Micro-C, loop by loop

Read on 2026-11-17. A whole map correlation with Micro-C at 5 kb is weak for every model, 0.08
to 0.14 in `design/2026-11-11-rnapii-release.md`, and ten structures give a map too sparse to
correlate at that bin size. Loops are the feature Micro-C resolves and the feature a ChIA-PET
model is built to place. The comparison is therefore made at loops called on the Micro-C map.

## Method

Micro-C of H1ESC and HFFC6, 4DN files 4DNFI2TK7L2F and 4DNFI18Q799K, chr1:0-60 Mb at 5 kb, the
slices `playground/rnapii/hic_slice.py` cut. GM12878 has no Micro-C.

- `playground/rnapii/microc_call_loops.py` writes a slice as a cooler, balances it and calls
  loops with the dots caller of cooltools 0.7.1, the HiCCUPS scheme, at 5 kb and at 10 kb, with
  the tool's defaults.
- `playground/rnapii/microc_loop_compare.py` scores an ensemble at each loop. The controls of a
  loop are the pairs at the same separation, its two bins moved together 3 to 10 bins either
  way along the diagonal. The distance ratio is the mean 3D distance between the loop's bins
  over the median of its controls. A loop is placed at a ratio of 0.8 or less, and a loop
  without beads at both ends is not placed. The contact enrichment is the loop pixel over the
  mean of its controls on the summed contact map, summed over loops.
- A loop's class is the input that supports it, a ChIA-PET loop with an anchor within one bin
  of each end. CTCF, RNAPII alone, or none.
- Every loop moved 30 bins along the diagonal is the null. A tool's placed share is read above
  its own null, since the null differs between tools.

Runs: `playground/rnapii/microc_loops.sh` on the workstation for the cell line arms, and
`playground/rnapii/microc_loops/tools/run.sh` on frodo for the three tools of
`out/toolpage60`. Tables in `playground/rnapii/microc_loops/cells` and `tools`.

## The instrument

On the moved loops the Micro-C map reads an enrichment of 0.94 to 1.00 and every arm a distance
ratio of 0.97 to 1.06. The moved loops pass the placed threshold 1 to 14 percent of the time in
cudaMMC and 3dgnome and 32 to 37 percent in MultiMM, whose ensembles differ more between
structures.

## Loops called

| | H1ESC | HFFC6 |
|---|---|---|
| loops at 5 kb | 605 | 969 |
| loops at 10 kb | 400 | 719 |
| at 5 kb, a CTCF loop supports | 263 | 417 |
| at 5 kb, an RNAPII loop alone supports | 40 | 62 |
| at 5 kb, no ChIA-PET loop within a bin | 302 | 490 |
| enrichment on the Micro-C map, all loops | 6.96 | 10.01 |

Half of the Micro-C loops have no ChIA-PET loop within a bin of both ends. Of the CTCF
ChIA-PET loops in the region, 353 of 2,088 and 604 of 2,253 are at a Micro-C loop.

## CTCF loops against CTCF and RNAPII loops

Ten structures, 5 kb. Placed share above the moved loops, and in brackets the chance that a
loop's ratio is under a moved loop's.

| Micro-C loops | Arm | H1ESC | HFFC6 |
|---|---|---|---|
| all | CTCF | +0.34 (0.76) | +0.34 (0.76) |
| all | CTCF and RNAPII | +0.37 (0.77) | +0.41 (0.78) |
| all | RNAPII ends merged | +0.35 (0.77) | +0.36 (0.76) |
| CTCF supports | CTCF | +0.61 (0.89) | +0.58 (0.87) |
| CTCF supports | CTCF and RNAPII | +0.64 (0.90) | +0.71 (0.93) |
| CTCF supports | RNAPII ends merged | +0.61 (0.89) | +0.60 (0.88) |
| RNAPII alone supports | CTCF | +0.22 (0.64) | +0.26 (0.66) |
| RNAPII alone supports | CTCF and RNAPII | +0.51 (0.92) | +0.64 (0.91) |
| RNAPII alone supports | RNAPII ends merged | +0.39 (0.81) | +0.58 (0.86) |
| no support | CTCF | +0.11 (0.65) | +0.14 (0.67) |
| no support | CTCF and RNAPII | +0.11 (0.63) | +0.12 (0.63) |

Contact enrichment at the loops CTCF supports, map 7.99 and 13.00: CTCF 3.10 and 2.51, CTCF
and RNAPII 6.00 and 8.39, merged 3.56 and 3.28.

RNAPII loops are confirmed by Micro-C. With them the loops only RNAPII supports go from a
quarter placed to a half and two thirds, and the loops CTCF supports gain as well. The 10 kb
calls agree, +0.33 to +0.73 and +0.22 to +0.56 on the RNAPII class.

## Three tools on the same CTCF loops

Ten structures, 5 kb, all Micro-C loops.

| Tool | Scored loops | Placed above moved | Under a moved loop | Contact enrichment |
|---|---|---|---|---|
| H1ESC cudaMMC | 132 of 605 | +0.18 | 0.93 | 30.9 |
| H1ESC MultiMM | 605 | +0.15 | 0.58 | 0.76 |
| H1ESC 3dgnome | 567 | +0.34 | 0.76 | 3.05 |
| HFFC6 cudaMMC | 186 of 969 | +0.13 | 0.83 | 40.6 |
| HFFC6 MultiMM | 969 | +0.11 | 0.59 | 0.45 |
| HFFC6 3dgnome | 913 | +0.34 | 0.76 | 2.31 |

cudaMMC has beads at a fifth of the loops. Where it has them it pulls the ends to a quarter of
the control distance, closer than the map's enrichment asks, and tells a loop from a moved one
better than the other two. MultiMM has beads everywhere and no contact enrichment at the loops.
At 10 kb the placed share above moved is +0.27, +0.21 and +0.38 on H1ESC and +0.24, +0.23 and
+0.35 on HFFC6.

## Open

- The cell line models place about a tenth of the loops no input supports above the null. A
  Micro-C loop list as a third loop input would be the direct test of that ceiling.
- 3dgnome's contact enrichment at CTCF loops is a third of the map's with CTCF loops alone.
- One region of one chromosome. The genome arrays on eden hold the same arms on chr1 to chr22.
