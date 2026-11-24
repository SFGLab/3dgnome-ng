personal3d vendored from triosformer at 53044d1, 2026-11-11. Its rules/ carry their own BSD licence. Update by copying the package again.

Changed here on 2026-11-18 and not yet in triosformer: tandem duplications. `lift.py` has a DUP event with `copies` and `duplications`, `hicdiffusion.py` writes the duplicated bases twice, `engine.lift_bedpe` gives both copies their contacts, and `rewrite.py` keeps the reference interval on the anchors of a duplicated segment. `rules/` is untouched. Copying the package again loses these unless triosformer takes them first.

Changed here on 2026-11-24 and not yet in triosformer: `rules/threeDgenome.py`, `Genome.establish_connections`
reads the anchors of a segment in place. It copied the sorted list of the segment for each end of each loop, and
the build of the network of a chromosome was quadratic in its anchors: 2.2 s for chr1 of GM12878 against 0.2 s.
The copy was read and dropped. Under one hash seed the loops after deletions, insertions, inversions and
duplications on the CTCF loops of chr1 are the same before and after the change.

Changed here on 2026-11-24 and not yet in triosformer: `Genome.establish_connections` builds the network from
the loops in sorted order. It took them in the order of a set, which follows the hash seed of the process, and
two anchors of one start were resolved by that order: the same variants gave other loops in another process, 3
of 12 applications on chr1. `web/tests/test_variants.py` runs a network under three seeds.

