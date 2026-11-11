"""personal3d's command line.

    python -m personal3d hic <freeze4> <fasta> <chrom> <start> <out.npz> [--people S ...] [--seeds N ...]
        HiCDiffusion's maps of every person's two haplotypes and the reference for the 2 Mb
        window from start, reference coordinates, with each haplotype's lift of the bin edges.

    python -m personal3d inputs <freeze4> <data dir> <template ini> <person> <chrom> <start> <end> <out dir>
        [--hic out.npz] [--base-singletons file] [--narrow fasta]
        The person's two haplotypes' engine inputs for a region, the panel being the other
        eight, and a config each. With --hic the window of that npz is spliced into the base
        singletons, by default the person's own conf arm map. With --narrow a deletion removes
        an anchor only when it cuts a CTCF motif of it or half of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import build, hicdiffusion, motifs, panel, rewrite, variants

PEOPLE = ["HG00512", "HG00513", "HG00514", "HG00731", "HG00732", "HG00733", "GM19238", "GM19239", "GM19240"]


def cmd_hic(a) -> None:
    ref = hicdiffusion.Reference(Path(a.fasta))
    hap = variants.haplotypes(Path(a.freeze4), a.people, a.chrom)
    names, windows = ["reference"], [hicdiffusion.haplotype_window(ref, a.chrom, a.start, [])]
    for p in a.people:
        for h in (0, 1):
            names.append(f"{p}_{h}")
            windows.append(hicdiffusion.haplotype_window(ref, a.chrom, a.start, hap[p][h]))
    cond, pred = hicdiffusion.predict(windows, tuple(a.seeds))
    edges = a.start + np.arange(257) * hicdiffusion.BIN
    # Where each reference bin edge falls in each haplotype's window, -1 if deleted.
    lifted = np.array([[(-1 if (x := w.lift(int(e))) is None else x) for e in edges] for w in windows])
    np.savez_compressed(a.out, names=np.array(names), cond=cond, pred=pred, edges=edges, lifted=lifted, chrom=a.chrom, start=a.start)
    print(f"wrote {a.out}, {len(names)} maps")


def cmd_inputs(a) -> None:
    data = Path(a.data)
    others = [q for q in PEOPLE if q != a.person]
    P = panel.build(data, PEOPLE, a.chrom)
    hap = variants.haplotypes(Path(a.freeze4), [a.person], a.chrom)[a.person]
    base_path = Path(a.base_singletons) if a.base_singletons else data / a.person / f"{a.person}_both_hic_25kb_singletons.bedpe"
    base = pd.read_csv(base_path, sep="\t", header=None, usecols=range(7))
    base = base[(base[0] == a.chrom) & (base[3] == a.chrom)]
    seg = pd.read_csv(data / a.person / f"{a.person}_segments.bed", sep="\t", header=None)
    seg = seg[seg[0] == a.chrom]
    cen = pd.read_csv(data / "GM12878" / "hg38_centromeres.bed", sep="\t", header=None)
    cen = cen[cen[0] == a.chrom]
    hic = np.load(a.hic) if a.hic else None
    mot = None
    if a.narrow:
        mot = motifs.anchor_motifs(hicdiffusion.Reference(Path(a.narrow)), P.anchors, rewrite.pwm())
    for h in (0, 1):
        name = f"{a.person}_{h}"
        pred = None
        if hic is not None:
            pred = (hic["cond"][list(hic["names"]).index(name)], int(hic["start"]))
        ini, region, _ = build.region_inputs(P, others, a.chrom, a.start, a.end, hap[h], base, seg, cen, Path(a.template), Path(a.out) / name, name, mot, pred)
        print(f"{name}: region {region}, {ini}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="personal3d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hic")
    h.add_argument("freeze4")
    h.add_argument("fasta")
    h.add_argument("chrom")
    h.add_argument("start", type=int)
    h.add_argument("out")
    h.add_argument("--people", nargs="+", default=PEOPLE)
    h.add_argument("--seeds", nargs="+", type=int, default=[1996])
    i = sub.add_parser("inputs")
    for x in ("freeze4", "data", "template", "person", "chrom"):
        i.add_argument(x)
    i.add_argument("start", type=int)
    i.add_argument("end", type=int)
    i.add_argument("out")
    i.add_argument("--hic")
    i.add_argument("--base-singletons")
    i.add_argument("--narrow", metavar="FASTA", help="the narrowed deletion rule, motifs read from this reference")
    a = ap.parse_args(argv)
    {"hic": cmd_hic, "inputs": cmd_inputs}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
