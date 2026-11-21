"""The effect of the variants of a sample on the loops of a region, for the variants dialog."""

from __future__ import annotations

import re
from pathlib import Path

from personal3d import motifs, rewrite

from ..engine.inputs import loop_network
from .shapes import VariantCheck
from .vcf import sample_variants


_NAMES = {"DEL": "deletion", "INS": "insertion", "INV": "inversion", "DUP": "duplication", "BND": "breakend", "CNV": "copy number change"}


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _change(kind: str, before: dict, after_loops: list, factor: str) -> str:
    """The words for what one variant did to the loops of one factor."""
    keyed = {(lp.ref_a, lp.ref_b): lp for lp in after_loops if lp.ref_a and lp.ref_b}
    removed = [k for k in before if k not in keyed]
    kept_anchors = {r for lp in after_loops for r in (lp.ref_a, lp.ref_b) if r}
    lost_anchors = {r for k in removed for r in k if r not in kept_anchors}
    new_loops = [lp for lp in after_loops if lp.ref_a is None or lp.ref_b is None]
    new_anchors = {lp.a if lp.ref_a is None else lp.b for lp in new_loops}
    changed = [k for k, lp in keyed.items() if k in before and (lp.b[1] - lp.a[0]) != (before[k].b[1] - before[k].a[0])]
    flipped = {r for k, lp in keyed.items() if k in before for r, o0, o1 in ((k[0], before[k].orient[0], lp.orient[0]), (k[1], before[k].orient[1], lp.orient[1])) if o0 != o1}
    if kind == "DEL":
        if lost_anchors:
            return f"removes {_plural(len(lost_anchors), f'{factor} anchor')} and {'its' if len(lost_anchors) == 1 else 'their'} {_plural(len(removed), 'loop')}"
        return f"shortens {_plural(len(changed), 'loop')}" if changed else "no anchor inside"
    if kind == "INS":
        if new_anchors:
            return f"adds {_plural(len(new_anchors), f'{factor} anchor')}, {_plural(len(new_loops), 'new loop')}"
        return f"lengthens {_plural(len(changed), 'loop')}" if changed else f"no {factor} motif above the cutoff"
    if kind == "DUP":
        # A copy keeps the reference interval of its original. A copied loop has the key of a
        # loop that is there already and shows as one more loop than the keys count.
        copied = len(after_loops) - len(keyed) - len(new_loops)
        if copied > 0 or new_loops:
            return f"copies {_plural(max(copied, 0), 'loop')}, adds {_plural(len(new_loops), 'new loop')}" if new_loops else f"copies {_plural(copied, 'loop')}"
        return f"lengthens {_plural(len(changed), 'loop')}" if changed else "no anchor inside"
    if flipped or removed:
        return f"flips {_plural(len(flipped), 'anchor')}, replaces {_plural(len(removed) + len(new_loops), 'loop')}"
    return "no anchor inside"


def combine_changes(by_factor: list[tuple[str, str]]) -> str:
    """One sentence for the changes of several loop factors: one verb is summed with the split
    by factor, different changes are listed and a factor without change is left out."""
    real = [(f, w) for f, w in by_factor if not w.startswith("no ")]
    if not real:
        return by_factor[0][1] if by_factor else "no anchor inside"
    simple = [re.fullmatch(r"(\w+) (\d+) loops?", w) for _, w in real]
    if len(real) > 1 and all(simple) and len({m[1] for m in simple}) == 1:  # type: ignore[index]
        total = sum(int(m[2]) for m in simple)  # type: ignore[index]
        split = " and ".join(f"{m[2]} {f}" for (f, _), m in zip(real, simple, strict=True))  # type: ignore[index]
        return f"{simple[0][1]} {total} loop{'' if total == 1 else 's'}, {split}"  # type: ignore[index]
    return "; ".join(dict.fromkeys(w for _, w in real))


def variant_check(cell, region, factors: str, vcf: Path, sample: str, haplotype: int, own_loops: list[tuple[str, Path]] | None = None, own_anchors: Path | None = None) -> VariantCheck:  # noqa: ANN001, PLR0913
    """The effect of the variants of a sample on one haplotype on the loops of a region, one
    variant at a time, with the counts applied and skipped. Own loops replace the cell type's.

    @raises UnknownSample when the sample is not in the VCF
    """
    records, skipped_all = sample_variants(vcf, sample, region)
    mine = [r for r in records if r["haps"][haplotype]]
    chrom = region.chrom
    _, length, network = loop_network(cell, region, factors, own_loops or [], own_anchors)
    nets = []
    for factor, clusters in network:
        segs = rewrite.domains(clusters)
        base = rewrite.apply(clusters, segs, {chrom: length}, [], None, None, motifs.THRESHOLD)
        nets.append((factor, clusters, segs, {(lp.ref_a, lp.ref_b): lp for lp in base}))
    rows = []
    applied: dict[str, int] = {}
    for r in mine:
        sv = r["sv"]
        words = []
        for factor, clusters, segs, before in nets:
            after = rewrite.apply(clusters, segs, {chrom: length}, [sv], None, None, motifs.THRESHOLD)
            words.append((factor, _change(str(r["type"]), before, after, factor)))
        applied[str(r["type"])] = applied.get(str(r["type"]), 0) + 1
        rows.append({"type": r["type"], "pos": r["pos"], "end": sv.end - 1, "length": r["length"], "genotype": r["genotype"], "change": combine_changes(words)})
    hap = "first" if haplotype == 0 else "second"
    span = region.shown()
    n = len(rows)
    parts = [f"{_plural(n, 'variant')} of {sample} on the {hap} haplotype {'lies' if n == 1 else 'lie'} inside {span}."]
    if applied:
        parts.append(", ".join(_plural(c, _NAMES[k]) for k, c in applied.items()) + (" is" if n == 1 else " are") + " applied.")
    sk = sum(skipped_all.values())
    if sk:
        parts.append(f"{_plural(sk, 'record')} of other types {'is' if sk == 1 else 'are'} skipped, " + ", ".join(_plural(c, _NAMES.get(k, k.lower())) for k, c in skipped_all.items()) + ".")
    return {"summary": " ".join(parts), "applied": applied, "skipped": skipped_all, "variants": rows}
