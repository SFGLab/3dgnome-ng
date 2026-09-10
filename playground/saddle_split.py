"""The saddle split into within block and cross block pairs, for finished structures.

Says where a structure's compartmentalisation lives. Blocks are recovered from the densified
beads by the densification rule, bins take the block of most of their beads, and the saddle's
three enrichments are averaged over pairs inside one block and over pairs across two blocks
separately, each class normalised on its own distance decay so neither leaks into the other.
The experiment is split by the same block labels.

    python playground/saddle_split.py <mcool> <region> <binsize> <compartment bedGraph> <dir> [<dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.restitch_model import read_cif, split_blocks  # noqa: E402
from validation.metrics import hic as contacts  # noqa: E402
from validation.metrics import structure as smetrics  # noqa: E402
from validation.studies.epigenome import _track_on_bins  # noqa: E402

TARGET_BP = 5000


def bin_blocks(beads, bin_starts, binsize):
    blocks = split_blocks(beads, TARGET_BP)
    lab = np.full(len(bin_starts), -1, dtype=np.int64)
    votes: dict[int, dict[int, int]] = {}
    for k, blk in enumerate(blocks):
        for b in blk:
            i = int(((b.start + b.end) // 2 - bin_starts[0]) // binsize)
            if 0 <= i < len(bin_starts):
                votes.setdefault(i, {})[k] = votes.get(i, {}).get(k, 0) + 1
    for i, v in votes.items():
        lab[i] = max(v.items(), key=lambda kv: kv[1])[0]
    return lab


def _oe_masked(c, keep):
    """Observed over expected where the expectation on each diagonal is the mean over the
    kept pairs of that diagonal alone, so the within block and the cross block classes are
    each normalised on their own decay and neither leaks into the other."""
    n = c.shape[0]
    oe = np.zeros_like(c, dtype=np.float64)
    for d in range(1, n):
        idx = np.arange(n - d)
        diag = c[idx, idx + d]
        m = keep[idx, idx + d] & (diag > 0)
        if not m.any():
            continue
        v = np.where(keep[idx, idx + d], diag / float(diag[m].mean()), 0.0)
        oe[idx, idx + d] = v
        oe[idx + d, idx] = v
    return oe


def split_saddle(c, track, lab, n_quantiles=5):
    usable = (track != 0.0) & np.isfinite(track) & (c.sum(axis=1) > 0) & (lab >= 0)
    idx = np.flatnonzero(usable)
    order = idx[np.argsort(track[idx])]
    m = len(order) // n_quantiles
    lo, hi = order[:m], order[-m:]
    same = lab[:, None] == lab[None, :]
    out = {}
    for name, keep in (("within", same), ("cross", ~same)):
        oe = _oe_masked(c, keep)

        def mean_block(a, b):
            blk = oe[np.ix_(a, b)]
            vals = blk[blk > 0.0]
            return float(vals.mean()) if vals.size else float("nan")

        aa, bb, ab = mean_block(hi, hi), mean_block(lo, lo), mean_block(hi, lo)
        out[name] = ((aa + bb) / (2 * ab) if ab > 0 else float("nan"), aa, bb, ab)
    return out


def main():
    mcool, region, binsize, track = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    dirs = sys.argv[5:]
    chrom = region.split(":")[0]
    c_obs, bin_starts = contacts.observed_hic(mcool, region, binsize, balance=True)
    sort_track = _track_on_bins(track, chrom, bin_starts)
    print(f"{region} @ {binsize // 1000} kb   saddle as (AA+BB)/2AB, then AA BB AB, within block pairs | cross block pairs")
    radius = None
    lab = None
    for d in dirs:
        cifs = sorted(Path(d).glob("*.cif"))
        c_sim = np.zeros_like(c_obs)
        for p in cifs:
            beads = read_cif(str(p))
            if lab is None:
                lab = bin_blocks(beads, bin_starts, binsize)
                n_blk = len(set(lab[lab >= 0].tolist()))
                exp = split_saddle(c_obs, sort_track, lab)
                print(f"  {'experiment':<28} {n_blk} blocks   within {exp['within'][0]:.3f} ({exp['within'][1]:.2f} {exp['within'][2]:.2f} {exp['within'][3]:.2f}) | cross {exp['cross'][0]:.3f} ({exp['cross'][1]:.2f} {exp['cross'][2]:.2f} {exp['cross'][3]:.2f})")
            coords, mids = smetrics.to_arrays(beads)
            if radius is None:
                radius = float(np.median(smetrics.bond_lengths(coords)))
            c_sim += contacts.simulated_contacts(coords, mids, bin_starts, binsize, radius)
        r = split_saddle(c_sim, sort_track, lab)
        print(f"  {Path(d).name:<28}          within {r['within'][0]:.3f} ({r['within'][1]:.2f} {r['within'][2]:.2f} {r['within'][3]:.2f}) | cross {r['cross'][0]:.3f} ({r['cross'][1]:.2f} {r['cross'][2]:.2f} {r['cross'][3]:.2f})")


if __name__ == "__main__":
    main()
