# personal3d

A person's 3dgnome-ng inputs from the person's genome, ideas 108 to 110 of
design/expression-from-structure.md. Every fold so far was built from a person's measured
loops, whose personal part is three tenths library. Here the inputs start from a panel's
shared loops and change only where the person's DNA changes them.

- `variants.py` reads a person's structural variants per haplotype from HGSVC freeze 4, the
  insertions and deletions with their sequences and the inversions, which are unphased and go
  on the first haplotype.
- `panel.py` builds a panel's consensus loops on the merged anchors of the nine, per factor,
  and places every person's measured loops on the same anchors.
- `rewrite.py` runs Sadowski's 2019 rules, vendored in `rules/` under their BSD licence, on a
  panel's network by one haplotype's variants. Each anchor keeps its reference interval, so a
  rewired loop names the reference anchors it joins.
- `hicdiffusion.py` builds a haplotype's 2 Mb window of sequence and runs HiCDiffusion on it in
  HiCDiffusion's own venv through `hicdiffusion_runner.py`. All haplotypes of a window share
  the sampling seeds. The deterministic map is the one to read, since two seeds of one
  sequence differ by more than two people do.
- `lift.py` maps reference positions to a haplotype's, through its deletions, insertions and
  inversions.
- `engine.py` writes a haplotype's engine inputs in its own coordinates, the rewritten loops,
  the base contact map lifted with a predicted window spliced in on the base map's grid and
  depth, and the arm's config with only `[data]` replaced, and runs gnome3d on a region.

The command line:

    python -m personal3d hic $TRIOS_FREEZE4 $TRIOS_REFERENCE chr1 108639238 out.npz
    python -m personal3d inputs $TRIOS_FREEZE4 $TRIOS_GNOME3D/data slurm/configs/hg00512_trio_rnapol2_hgsvc_conf_both.ini HG00512 chr1 108000000 111000000 out/ --hic out.npz

The locations come from config/env, `TRIOS_HICDIFFUSION`, `TRIOS_HICDIFFUSION_PYTHON`,
`TRIOS_HICDIFFUSION_MODELS`, `TRIOS_REFERENCE` and `TRIOS_FREEZE4`. The tests are
`tests/test_personal3d.py` and need `sortedcontainers`.
