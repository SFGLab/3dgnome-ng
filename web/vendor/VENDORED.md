personal3d vendored from triosformer at 53044d1, 2026-11-11. Its rules/ carry their own BSD licence. Update by copying the package again.

Changed here on 2026-11-18 and not yet in triosformer: tandem duplications. `lift.py` has a DUP event with `copies` and `duplications`, `hicdiffusion.py` writes the duplicated bases twice, `engine.lift_bedpe` gives both copies their contacts, and `rewrite.py` keeps the reference interval on the anchors of a duplicated segment. `rules/` is untouched. Copying the package again loses these unless triosformer takes them first.
