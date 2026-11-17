# Models against Micro-C, loop by loop

Read on 2026-11-17. A whole map correlation with Micro-C at 5 kb is weak for every model, 0.08
to 0.14 in `design/2026-11-11-rnapii-release.md`, and ten structures give a map too sparse to
correlate at that bin size. Loops are the feature Micro-C resolves and the feature a ChIA-PET
model is built to place. The comparison is therefore made at loops called on the Micro-C map.

## Maps

chr1:0-60 Mb at 5 kb, raw counts.

| Line | Map | Contacts in the region |
|---|---|---|
| H1ESC | 4DN 4DNFI2TK7L2F, `.hic`, cut by `playground/rnapii/hic_slice.py` | 95 million |
| HFFC6 | 4DN 4DNFI18Q799K, `.hic`, cut by `hic_slice.py` | 112 million |
| GM12878 | Hong et al., Zenodo 10.5281/zenodo.15303879, `GM12878_microc.50.cool`, CC BY 4.0, hg38, cut by `playground/rnapii/cool_slice.py` | 181 million |

4DN holds no Micro-C of GM12878. The three maps have the same decay, a slope of minus 1.25
between 20 kb and 1 Mb, and the same split of contacts by separation.

## Method

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
- A loop is kept at `min_z=15`, its count on the Micro-C map at least 15 standard deviations
  over its controls. The next section is the reason.

Runs: `playground/rnapii/microc_loops.sh` on the workstation for the cell line arms, the
`run.sh` beside each table on frodo for the three tools of `out/toolpage60`. Tables in
`playground/rnapii/microc_loops`: `cells_z15` and `tools_z15` with the floor, `cells` and
`tools` without it for H1ESC and HFFC6, `gm12878` the GM12878 calls, `check` the next section.

## The caller's defaults on the deep GM12878 map

The caller returned 5,979 loops on GM12878 against 605 and 969, with a median span of 1,040 kb
against 205 and 235 and a median count of 11 reads against 88 and 69. The reads of a call are
spread over as many 50 bp pixels as there are reads, so the calls are not duplicates.

`playground/rnapii/loops_in_map.py` reads each loop list in the deep Hi-C map of the same
cells, an independent experiment. Share of loops whose Hi-C count is over its controls, and
the summed enrichment in brackets:

| Span | H1ESC | HFFC6 | GM12878 |
|---|---|---|---|
| under 300 kb | 0.93 (2.4) | 0.96 (3.0) | 0.81 (2.3) |
| 300 kb to 1 Mb | 0.96 (2.6) | 0.96 (3.2) | 0.57 (1.5) |
| over 1 Mb | 0.83 (4.9) | 0.78 (4.2) | 0.51 (1.2) |

One half is chance. The H1ESC and HFFC6 loops are in the Hi-C map at every span and the
GM12878 calls beyond 300 kb are not. By read count the GM12878 calls under 20 reads, 4,341 of
5,979, read 1.05 to 1.09 in Hi-C. The same raw count is weaker evidence on a deeper map, so the
floor is on the count's excess over the controls in standard deviations.

| Floor of 15 | H1ESC | HFFC6 | GM12878 |
|---|---|---|---|
| loops kept at 5 kb | 385 of 605 | 639 of 969 | 902 of 5,979 |
| loops kept at 10 kb | 321 of 400 | 564 of 719 | 525 of 6,102 |
| share over controls in Hi-C | 0.97 | 0.97 | 0.86 |
| median span | 180 kb | 225 kb | 350 kb |
| a CTCF loop supports | 200 | 360 | 279 |
| an RNAPII loop alone supports | 21 | 38 | 65 |
| no ChIA-PET loop within a bin | 164 | 241 | 558 |
| enrichment on the Micro-C map | 8.5 | 12.4 | 7.7 |

One rule for the three lists. GM12878 still replicates less at 15, and its loops without a
ChIA-PET loop are 62 percent of the list against 43 and 38, so part of that class is chance
calls and its placed share reads low for that reason too.

## The instrument

On the moved loops the Micro-C map reads an enrichment of 0.96 to 1.01 and every arm a distance
ratio of 0.99 to 1.08. The moved loops pass the placed threshold 1 to 10 percent of the time in
cudaMMC and 3dgnome and 32 to 34 percent in MultiMM, whose ensembles differ more between
structures.

## CTCF loops against CTCF and RNAPII loops

Ten structures, 5 kb, floor of 15. Placed share above the moved loops, and in brackets the
chance that a loop's ratio is under a moved loop's.

| Micro-C loops | Arm | GM12878 | H1ESC | HFFC6 |
|---|---|---|---|---|
| all | original, cudaMMC | +0.14 (0.83) | +0.21 (0.93) | +0.15 (0.88) |
| all | CTCF | +0.21 (0.70) | +0.39 (0.80) | +0.44 (0.81) |
| all | CTCF and RNAPII | +0.30 (0.74) | +0.42 (0.80) | +0.50 (0.83) |
| all | RNAPII ends merged | +0.24 (0.70) | +0.38 (0.78) | +0.45 (0.80) |
| CTCF supports | CTCF | +0.61 (0.89) | +0.65 (0.90) | +0.62 (0.89) |
| CTCF supports | CTCF and RNAPII | +0.67 (0.93) | +0.68 (0.92) | +0.73 (0.95) |
| CTCF supports | RNAPII ends merged | +0.59 (0.89) | +0.62 (0.90) | +0.65 (0.90) |
| RNAPII alone supports | CTCF | +0.08 (0.57) | +0.32 (0.75) | +0.41 (0.73) |
| RNAPII alone supports | CTCF and RNAPII | +0.67 (0.92) | +0.48 (0.90) | +0.72 (0.95) |
| RNAPII alone supports | RNAPII ends merged | +0.37 (0.77) | +0.37 (0.80) | +0.67 (0.92) |
| no support | CTCF | +0.03 (0.61) | +0.09 (0.65) | +0.17 (0.69) |
| no support | CTCF and RNAPII | +0.07 (0.62) | +0.09 (0.63) | +0.12 (0.62) |

Contact enrichment at the loops CTCF supports, map 8.1, 9.0 and 14.0: CTCF 2.7, 3.6 and 2.7,
CTCF and RNAPII 5.1, 6.9 and 9.3, merged 2.7, 3.8 and 3.5.

RNAPII loops are confirmed by Micro-C on the three lines. With them the loops only RNAPII
supports are placed at +0.48 to +0.72 from +0.08 to +0.41, and the loops CTCF supports gain on
every line. Without the floor H1ESC and HFFC6 read the same way, +0.22 and +0.26 to +0.51 and
+0.64 on the RNAPII class.

## Three tools on the same CTCF loops

Ten structures, 5 kb, floor of 15, all Micro-C loops.

| Tool | Scored loops | Placed above moved | Under a moved loop | Contact enrichment |
|---|---|---|---|---|
| GM12878 cudaMMC | 227 of 902 | +0.14 | 0.84 | 33.0 |
| GM12878 MultiMM | 902 | +0.25 | 0.67 | 2.95 |
| GM12878 3dgnome | 876 | +0.26 | 0.70 | 2.75 |
| H1ESC cudaMMC | 99 of 385 | +0.21 | 0.93 | 39.9 |
| H1ESC MultiMM | 385 | +0.16 | 0.59 | 0.83 |
| H1ESC 3dgnome | 363 | +0.38 | 0.79 | 3.40 |
| HFFC6 cudaMMC | 139 of 639 | +0.16 | 0.87 | 43.4 |
| HFFC6 MultiMM | 639 | +0.15 | 0.62 | 0.44 |
| HFFC6 3dgnome | 615 | +0.45 | 0.82 | 2.60 |

cudaMMC has beads at a quarter of the loops. Where it has them it pulls the ends to a quarter
to a third of the control distance and tells a loop from a moved one better than the other two.
MultiMM is level with 3dgnome on GM12878 and under it on the other two lines. At 10 kb the
placed share above moved is +0.30, +0.39 and +0.33 on GM12878, MultiMM ahead, +0.29, +0.22 and
+0.40 on H1ESC and +0.27, +0.25 and +0.41 on HFFC6.

## Open

- The models place 3 to 17 percent of the loops no input supports above the null. A Micro-C
  loop list as a third loop input would be the direct test of that ceiling.
- 3dgnome's contact enrichment at CTCF loops is a third of the map's with CTCF loops alone.
- The floor of 15 is one choice. At 20 GM12878 keeps 561 loops and replicates at 0.89, at 10
  it keeps 1,659 at 0.77. A caller run at a stricter false discovery rate would be the
  caller's own way to the same end and is not tried.
- One region of one chromosome. The genome arrays on eden hold the same arms on chr1 to chr22.
