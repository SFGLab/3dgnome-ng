"""A person's contact map from their sequence, by HiCDiffusion, Chiliński and Plewczynski 2024.

HiCDiffusion predicts a Hi-C window of 2,097,152 bp at 8,192 bp from the sequence alone, and
its pretrained models were trained on GM12878, a lymphoblastoid line as the nine are. A
haplotype's window is the reference from a start position with the haplotype's variants in
it applied, deletions removed, insertions put in, inversions reverse complemented, until the
window is full. `Window.lift` gives the haplotype position of a reference position, so the
predicted map's bins can be read back in reference coordinates.

The model runs in its own environment, which personal3d.hicdiffusion_runner is started in,
named by TRIOS_HICDIFFUSION, the checkout, TRIOS_HICDIFFUSION_PYTHON, its interpreter, and
TRIOS_HICDIFFUSION_MODELS, the directory with the two checkpoints.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .variants import SV

SIZE = 2_097_152
BIN = SIZE // 256
CODE = np.full(256, 4, dtype=np.uint8)
for _c, _v in (("A", 0), ("C", 1), ("T", 2), ("G", 3)):
    CODE[ord(_c)] = CODE[ord(_c.lower())] = _v
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Reference:
    """A FASTA read through its .fai index, one region at a time."""

    def __init__(self, fasta: Path):
        self.fasta = Path(fasta)
        self.index = {}
        for line in Path(str(fasta) + ".fai").read_text().splitlines():
            name, length, offset, bases, width = line.split("\t")[:5]
            self.index[name] = (int(length), int(offset), int(bases), int(width))

    def length(self, chrom: str) -> int:
        return self.index[chrom][0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases start to end, 0 based, end exclusive, clipped to the chromosome."""
        length, offset, bases, width = self.index[chrom]
        start, end = max(0, start), min(end, length)
        if end <= start:
            return ""
        a = offset + (start // bases) * width + start % bases
        b = offset + (end // bases) * width + end % bases
        with self.fasta.open("rb") as f:
            f.seek(a)
            raw = f.read(b - a)
        return raw.replace(b"\n", b"").replace(b"\r", b"").decode()


@dataclass
class Window:
    """One haplotype's window, its sequence and the breakpoints of its lift."""

    chrom: str
    start: int
    seq: str
    # Pairs of reference position and haplotype offset in the window, from which on the lift is
    # a shift; a reference position inside a deletion has no haplotype position.
    steps: list[tuple[int, int]] = field(default_factory=list)
    deleted: list[tuple[int, int]] = field(default_factory=list)
    # Inverted reference intervals with the window offset their first base was put at.
    inverted: list[tuple[int, int, int]] = field(default_factory=list)

    def lift(self, pos: int) -> int | None:
        """The window offset of a 0 based reference position, None if deleted or outside."""
        for a, b in self.deleted:
            if a <= pos < b:
                return None
        for a, b, at in self.inverted:
            if a <= pos < b:
                return at + (b - 1 - pos)
        off = pos - self.start
        for ref, shift in self.steps:
            if pos >= ref:
                off = pos - ref + shift
        return off if 0 <= off < len(self.seq) else None


def haplotype_window(ref: Reference, chrom: str, start: int, svs: list[SV], size: int = SIZE) -> Window:
    """The haplotype's sequence of size bases from reference position start, 0 based.

    The variants are applied left to right, each one wholly after the last, one that overlaps
    the previous is skipped as the rules skip it. Deleted bases are skipped, an insertion is
    put in before its position, an inversion is reverse complemented in place. Reference
    bases are read until the window is full."""
    parts, steps, deleted, inverted = [], [], [], []
    pos, have = start, 0
    for s in sorted((v for v in svs if v.chrom == chrom), key=lambda v: v.start):
        a = s.start - 1
        if a < pos:
            continue
        if have + (a - pos) >= size:
            break
        parts.append(ref.fetch(chrom, pos, a))
        have += a - pos
        if s.kind == "DEL":
            pos = s.end - 1
            deleted.append((a, pos))
        elif s.kind == "INS":
            parts.append(s.seq)
            have += len(s.seq)
            pos = a
        elif s.kind == "INV":
            b = s.end - 1
            inverted.append((a, b, have))
            parts.append(ref.fetch(chrom, a, b).translate(COMPLEMENT)[::-1])
            have += b - a
            pos = b
        steps.append((pos, have))
    while have < size:
        chunk = ref.fetch(chrom, pos, pos + size - have)
        if not chunk:
            break
        parts.append(chunk)
        have += len(chunk)
        pos += len(chunk)
    seq = "".join(parts)[:size]
    return Window(chrom, start, seq.ljust(size, "N"), steps, deleted, inverted)


def encode(seq: str) -> np.ndarray:
    return CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def predict(windows: list[Window], seeds: tuple[int, ...] = (1996,)) -> tuple[np.ndarray, np.ndarray]:
    """HiCDiffusion's two maps for each window, the deterministic one and one per seed."""
    repo = Path(os.environ["TRIOS_HICDIFFUSION"])
    python = os.environ.get("TRIOS_HICDIFFUSION_PYTHON", str(repo / ".venv" / "bin" / "python"))
    models = Path(os.environ["TRIOS_HICDIFFUSION_MODELS"])
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(src, seqs=np.stack([encode(w.seq) for w in windows]))
        runner = Path(__file__).resolve().parent / "hicdiffusion_runner.py"
        subprocess.run([python, str(runner), str(src), str(dst), str(models / "hicdiffusion.ckpt"), str(models / "hicdiffusion_encoder_decoder.ckpt"), *map(str, seeds)], cwd=repo, check=True)
        out = np.load(dst)
        return out["cond"], out["pred"]
