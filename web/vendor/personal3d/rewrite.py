"""Sadowski's 2019 rules applied to a panel's loops by one haplotype's structural variants.

The rules of personal3d/rules/threeDgenome.py take a reference interaction network, the
loops with each anchor's CTCF motif orientation and the contact domains they make, and
rewrite it by a list of variants. A deletion removes the anchors it touches with their loops,
an inversion flips and mirrors the anchors it holds, an insertion with a CTCF motif adds
anchors, and every anchor left without a loop is linked to the nearest convergent partner at
a distance between the 10th and 90th percentile of the loop lengths, at the median strength.

Here the reference is a panel, `panel.Panel`, and each anchor is tagged with its reference
interval before the rules run, so that a loop of the rewritten network names the reference
anchors it joins whatever the rules did to the coordinates. An anchor the rules made, from an
inserted motif or a duplicated copy, has no reference interval.

The rules call CTCF motifs with a per base loop over a pandas frame that fails on N. The
adapter swaps in a vectorised call of the same output, N scoring as the mean of the four, and
gives the rules' copy of an anchor, an insertion's split or a duplication's copy, the reference
interval of what it copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rules import motif as _motif
from .rules import threeDgenome as _tdg
from .rules import tools as _tools
from .variants import SV

PFM = Path(__file__).resolve().parent / "rules" / "MA0139.1.pfm"


def pwm():
    """The rules' CTCF matrix, JASPAR MA0139.1, as their log odds frame."""
    return _motif.pfm_to_pwm(_motif.read_raw_pfm(PFM))


def call_motifs(seq: str, pwm) -> list[tuple[int, str, float]]:
    """The rules' motif call, vectorised. A window's score is its better strand, R for the plus
    strand when it scores at least as well, and of positive windows within one motif length of
    each other the stronger is kept, the order of the rules' loop preserved."""
    P = pwm[["A", "C", "G", "T"]].to_numpy()
    m = len(P)
    if len(seq) < m:
        return []
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.array([code.get(c, 4) for c in seq.upper()])
    P5 = np.vstack([P.T, P.mean(axis=1)]).T
    win = np.lib.stride_tricks.sliding_window_view
    fwd = P5[np.arange(m), win(x, m)].sum(axis=1)
    rc = np.where(x == 4, 4, 3 - x)
    # The minus strand of window i is the reverse complement of x[i:i + m], scored against the
    # matrix in its own order.
    rev = P5[np.arange(m), win(rc, m)[:, ::-1]].sum(axis=1)
    best = np.maximum(fwd, rev)
    orient = np.where(fwd >= rev, "R", "L")
    out: list[tuple[int, str, float]] = []
    steps = m
    for i in np.flatnonzero(best > 0):
        # Positions between positive windows each add one step in the rules' loop.
        if out:
            steps += i - last - 1
        if steps < m and out:
            if best[i] > out[-1][2]:
                out[-1] = (int(i), str(orient[i]), float(best[i]))
                steps = 1
            else:
                steps += 1
        else:
            out.append((int(i), str(orient[i]), float(best[i])))
            steps = 1
        last = i
    return out


MOTIF_CUT = [0.0]


def _call_motifs(seq: str, pwm) -> list[tuple[int, str, float]]:
    """The call the rules make on an inserted sequence, at the cut apply was given."""
    return [m for m in call_motifs(seq, pwm) if m[2] > MOTIF_CUT[0]]


_tdg.call_motifs = _call_motifs
_duplicate = _tdg.Anchor.get_duplicate


def _get_duplicate(self):
    """The rules' copy of an anchor, an insertion's split or a duplication's copy, keeps the
    reference interval of the anchor it copies."""
    d = _duplicate(self)
    if hasattr(self, "ref"):
        d.ref = self.ref
    return d


_tdg.Anchor.get_duplicate = _get_duplicate


@dataclass
class Loop:
    """One loop of a rewritten network, in the haplotype's coordinates, with the reference
    intervals of its two anchors when they have one."""

    chrom: str
    a: tuple[int, int]
    b: tuple[int, int]
    orient: tuple[str, str]
    weight: float
    ref_a: tuple[int, int] | None
    ref_b: tuple[int, int] | None


def domains(clusters: list[list]) -> list[list]:
    """The contact domains of a loop set, the union of the spans of its loops, as the rules'
    segments. The 2019 paper took them from the CTCF loops the same way."""
    out = []
    by_chrom: dict[str, list] = {}
    for c in clusters:
        by_chrom.setdefault(c[0], []).append([min(c[1], c[4]), max(c[2], c[5])])
    for chrom, spans in by_chrom.items():
        out += [[chrom, s, e] for s, e in _tools.merge_intervals(spans)]
    return out


def apply(clusters: list[list], segments: list[list], chrom_lengths: dict[str, int], svs: list[SV], ref_of: dict | None = None, knockouts: set | None = None, motif_cut: float = 0.0) -> list[Loop]:
    """The network of clusters, nine columns, chrom start end twice, the two orientations and
    the strength, rewritten by svs. Variants that overlap one of another type are skipped as
    the rules skip them, and the rest are applied from the right so the coordinates of those
    still to come stay valid. ref_of gives the reference interval of an anchor the caller
    changed before the rules, as motifs.narrow clips them. knockouts are anchors, chrom and
    reference interval, removed with their loops before the variants, as snv.knockouts finds
    them, and their partners rewired by the rules. motif_cut is the score an inserted motif needs
    to become an anchor, the rules' own 0 by default."""
    MOTIF_CUT[0] = motif_cut
    g = _tdg.Genome(segments, clusters, dict(chrom_lengths), pwm())
    for chrom in g.segments:
        for seg in g.segments[chrom]:
            for an in seg.get_anchor_list():
                an.ref = (ref_of or {}).get((chrom, an.start, an.end), (an.start, an.end))
    if knockouts:
        for chrom in g.segments:
            gone = False
            for seg in g.segments[chrom]:
                for an in seg.get_anchor_list():
                    if (chrom, *an.ref) in knockouts:
                        seg.remove_anchor(an)
                        gone = True
            if gone:
                for seg in g.segments[chrom]:
                    seg.update_connectivity()
    by_chrom: dict[str, list] = {}
    for s in svs:
        by_chrom.setdefault(s.chrom, []).append([s.start, s.end, s.kind, s.seq or None])
    for chrom, rows in by_chrom.items():
        if chrom not in g.segments:
            continue
        for start, end, kind, seq in reversed(_tools.merge_svs(rows)):
            g.introduce_sv(chrom, start, end, kind, seq)
    loops = {}
    for chrom in g.segments:
        for seg in g.segments[chrom]:
            for an in seg.get_anchor_list():
                for other, w in an.get_connected_anchors().items():
                    x, y = (an, other) if an.start <= other.start else (other, an)
                    key = (chrom, x.start, x.end, y.start, y.end)
                    loops[key] = Loop(chrom, (x.start, x.end), (y.start, y.end), (x.motif_orient, y.motif_orient), float(w), getattr(x, "ref", None), getattr(y, "ref", None))
    return list(loops.values())
