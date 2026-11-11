"""SNVs that break a CTCF motif, idea 110, the most common personal change to an anchor.

The 2019 rules take structural variants only. A person's SNVs sit in CTCF sites far more
often. For each anchor with motifs, the windows of the rules' matrix scoring at least the
threshold on the reference, a haplotype's SNVs inside those windows are put in and the windows
rescored. An anchor whose every motif falls below the threshold on a haplotype is knocked out
on it, its loops removed and its partners rewired by the rules, `rewrite.apply(knockouts=)`.
SNVs shift no coordinate, so knockouts come before the structural variants.

The SNVs come from the 1000 Genomes high coverage phased panel cut to the nine, one VCF per
chromosome, read with bcftools.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .hicdiffusion import Reference
from .motifs import THRESHOLD, scores
from .variants import vcf_name


def read_snvs(vcf: Path, people: list[str], chrom: str) -> dict[str, tuple[dict[int, str], dict[int, str]]]:
    """Each person's alternate bases per haplotype, 0 based position to base."""
    names = [vcf_name(p) for p in people]
    q = subprocess.run(["bcftools", "query", "-s", ",".join(names), "-r", chrom, "-f", "%POS\t%REF\t%ALT[\t%GT]\n", str(vcf)], capture_output=True, text=True, check=True)
    out = {p: ({}, {}) for p in people}
    for line in q.stdout.splitlines():
        x = line.split("\t")
        pos, alt = int(x[0]) - 1, x[2]
        for p, gt in zip(people, x[3:], strict=True):
            a = gt.replace("/", "|").split("|")
            for h in (0, 1):
                if h < len(a) and a[h] == "1":
                    out[p][h][pos] = alt
    return out


def knockouts(ref: Reference, motifs_of: dict[tuple, list], alts: dict[int, str], pwm, threshold: float = THRESHOLD) -> tuple[set, dict]:
    """The anchors a haplotype's SNVs knock out, and for every anchor they touch the best motif
    score before and after."""
    P = pwm[["A", "C", "G", "T"]].to_numpy()
    if not alts:
        return set(), {}
    pos = np.array(sorted(alts))
    out, change = set(), {}
    for an, mot in motifs_of.items():
        if not mot:
            continue
        chrom, s, e = an
        lo, hi = np.searchsorted(pos, [s, e])
        if lo == hi:
            continue
        hit = [p for p in pos[lo:hi] if any(ms <= p < me for ms, me, _ in mot)]
        if not hit:
            continue
        seq = list(ref.fetch(chrom, s, e))
        for p in hit:
            seq[p - s] = alts[p]
        new = scores("".join(seq), P)
        after = max((new[ms - s] for ms, _, _ in mot if ms - s < len(new)), default=-np.inf)
        before = max(sc for _, _, sc in mot)
        change[an] = (before, float(after))
        if after < threshold:
            out.add(an)
    return out, change
