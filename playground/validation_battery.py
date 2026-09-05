"""The validation battery for two arms of the same pipeline, on this project's own metrics.

Energy and Kolmogorov-Smirnov gates say two ensembles are interchangeable as distributions. They
do not say the structures agree with the data, and that is what should decide whether a change
is adopted. Three measures, all of which existing work in this repository already turns on.

Hi-C correlation, both the contact correlation for a single structure and the ensemble Pearson
that MultiMM actually reports, against the cell line's own contact map.

The realised distance exponent, how quickly distance grows with genomic separation. Contact
probability decays near s^-0.86 across three cell lines, so distance should grow near s^0.285,
and `anchor-placement.md` exists because our structures come out flatter than that.

The radius of gyration, since the solver's structures are systematically a little more
expanded and expansion is the direction that work has been trying to move.

And the bead overlap count, pairs closer than the excluded volume's own radius that are not
chain neighbours. Every other measure here is a fitted slope or a correlation, which a change
in overall size can move without any change in shape. An overlap is a count of violations of a
term the run already carries, which moves less freely, though not not at all since the radius
is derived from the structure's own chain bond.

Overlaps are reported in three columns because three different stages own them. A pair of
anchors inside one block is the arcs stage's doing and no later stage can move it, since the
smooth stage holds every anchor fixed. A pair involving a subanchor inside one block is the
smooth stage's own excluded volume failing to exclude. A pair across two blocks is block
placement, which the boundary stitch and the cross block relaxation own. Collapsing the three
lets an arm that improves block placement read as though it improved block shape.

    python playground/validation_battery.py <mcool> <region> <binsize> <arm_dir> [<arm_dir> ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.spatial import KDTree  # noqa: E402

from validation.metrics.hic import (  # noqa: E402
    hic_correlation,
    multimm_faithful_pearson,
    observed_hic,
)

NU_HIC = 0.285  # ps_curve.py, mean over three cell lines, 20 kb to 1 Mb


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positions, genomic starts, bead midpoints and an anchor mask, in genomic order.

    Starts are what the correlation and exponent measures have always used. Midpoints are what
    block inference needs, since the densification rule it inverts is stated on them.
    """
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float64)
    start = np.array([int(r[16]) for r in rows], dtype=np.int64)
    end = np.array([int(r[17]) for r in rows], dtype=np.int64)
    anchor = np.array([r[18] == "anchor" for r in rows], dtype=np.bool_)
    o = np.argsort(start)
    return pos[o], start[o], ((start + end) // 2)[o], anchor[o]


def block_owner(mid: np.ndarray, anchor: np.ndarray, target_bp: int) -> np.ndarray:
    """Which interaction block each bead belongs to, recovered from the densification rule.

    Takes bead midpoints, not starts. The rule is stated on midpoints and reading it off starts
    misses a boundary wherever the two anchors overlap, which merged 28 percent of the blocks of
    a real chromosome. The partition is checkable: the boundary stitch moves a block rigidly and
    the relaxation holds anchors fixed, so inside a true block every anchor pair distance
    survives both passes exactly. Under this rule all 1,494 blocks of a trio chr1 pass, with a
    largest drift of 3.4e-05 which is the cif's own float32 rounding.

    Densify puts `round(span / target_bp) - 1` subanchors between two consecutive anchors of one
    block, so a consecutive anchor pair further apart than one and a half times that target
    always has one between it. A consecutive anchor pair wider than that with nothing between it
    can therefore only be a block boundary. Pairs closer than the target are ambiguous and stay
    inside the block they follow, which merges rather than invents a boundary.

    Parameters
    ----------
    mid
        Genomic position of each bead, in genomic order.
    anchor
        True where that bead is an anchor.
    target_bp
        The run's `target_bp_per_subanchor`.
    """
    wide = np.diff(mid) > 1.5 * target_bp
    cut = anchor[:-1] & anchor[1:] & wide
    return np.concatenate(([0], np.cumsum(cut)))


def block_bonds(pos: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """The mean realised chain bond of each block, which is the scale its excluded volume used.

    One proxy is unavoidable here. The kernel takes the mean of the chain bond targets and a
    finished structure only carries the realised distances.

    Parameters
    ----------
    pos
        Bead positions in genomic order.
    owner
        Block index per bead.
    """
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    fallback = float(np.median(step))
    same = owner[:-1] == owner[1:]
    out = np.full(int(owner.max()) + 1, fallback)
    for k in range(out.size):
        inner = step[same & (owner[:-1] == k)]
        if inner.size:
            out[k] = float(inner.mean())
    return out


def overlaps(
    pos: np.ndarray, anchor: np.ndarray, owner: np.ndarray, rad: np.ndarray
) -> tuple[float, float, float]:
    """Overlapping pairs per thousand beads, split by which stage owns them.

    Returns the within block anchor anchor rate, the within block rate for pairs involving a
    subanchor, and the cross block rate.

    A pair overlaps when it is more than one bead apart along the chain, which is what
    `exclusion_skip_neighbors` skips, and closer than its block's radius. A cross block pair
    uses the mean of its two blocks' radii.

    The radii are passed in rather than derived per structure, and every arm is scored on one
    set, because they are derived from the chain bond and a structure that expands would
    otherwise be measured with a wider net. Measured on a real chromosome that is worth 4.8
    percent on the anchor column, against a true change of zero.

    Parameters
    ----------
    pos
        Bead positions in genomic order.
    anchor
        True where that bead is an anchor.
    owner
        Block index per bead.
    rad
        The excluded volume radius of each block.
    """
    n = len(pos)
    q = KDTree(pos).query_pairs(float(rad.max()), output_type="ndarray")
    if q.size:
        q = q[np.abs(q[:, 0] - q[:, 1]) > 1]
    if not q.size:
        return 0.0, 0.0, 0.0
    i, j = q[:, 0], q[:, 1]
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    q = q[d < 0.5 * (rad[owner[i]] + rad[owner[j]])]
    if not q.size:
        return 0.0, 0.0, 0.0
    i, j = q[:, 0], q[:, 1]
    cross = owner[i] != owner[j]
    aa = anchor[i] & anchor[j]
    k = 1000.0 / n
    return (
        k * float((~cross & aa).sum()),
        k * float((~cross & ~aa).sum()),
        k * float(cross.sum()),
    )


def exponent(pos: np.ndarray, mid: np.ndarray, lo: int = 20_000, hi: int = 1_000_000) -> float:
    """Slope of log distance against log genomic separation, over the band Hi-C is measured on."""
    n = len(mid)
    step = max(1, n // 700)  # bound the pair count on a whole region
    i = np.arange(0, n, step)
    a, b = np.meshgrid(i, i, indexing="ij")
    m = a < b
    a, b = a[m], b[m]
    sep = np.abs(mid[b] - mid[a]).astype(np.float64)
    d = np.linalg.norm(pos[b] - pos[a], axis=1)
    k = (sep >= lo) & (sep <= hi) & (d > 0)
    if k.sum() < 50:
        return float("nan")
    return float(np.polyfit(np.log(sep[k]), np.log(d[k]), 1)[0])


def _flag(name: str, default: float) -> float:
    """Read `--name value` out of argv and remove it, so the positionals keep their places."""
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    v = float(sys.argv[k + 1])
    del sys.argv[k : k + 2]
    return v


def main() -> None:
    target_bp = int(_flag("--target-bp", 1000))  # the run's target_bp_per_subanchor
    ev_factor = _flag("--ev-factor", 0.7)  # the run's exclusion_auto_factor_smooth
    mcool, region, binsize = sys.argv[1], sys.argv[2], int(sys.argv[3])
    c_obs, bin_starts = observed_hic(mcool, region, binsize, balance=True)
    print(f"observed Hi-C {region} at {binsize:,} bp: {c_obs.shape[0]} bins\n")
    rad: np.ndarray | None = None  # the first arm's first structure sets it for every arm
    print(
        f"  {'arm':>10s} {'n':>3s} {'pearson':>9s} {'spearman':>9s} {'SCC':>8s} "
        f"{'multimm':>9s} {'exponent':>9s} {'vs Hi-C':>8s} {'Rg':>8s} "
        f"{'wb-aa':>7s} {'wb-sa':>7s} {'xb':>7s}"
    )
    for d in sys.argv[4:]:
        cifs = sorted(Path(d).glob("*.cif"))
        if not cifs:
            print(f"  {Path(d).name:>10s}  no cif files")
            continue
        coords, mids = [], None
        pear, spear, scc, expo, rgs = [], [], [], [], []
        waa, wsa, xb = [], [], []
        for c in cifs:
            p, m, bmid, anchor = load(c)
            owner = block_owner(bmid, anchor, target_bp)
            if rad is None:
                rad = ev_factor * block_bonds(p, owner)
            coords.append(p)
            mids = m
            r = hic_correlation(p, m, mcool, region, binsize, contact_radius=2.0, balance=True)
            pear.append(r.get("pearson", float("nan")))
            spear.append(r.get("spearman", float("nan")))
            scc.append(r.get("scc", float("nan")))
            expo.append(exponent(p, m))
            rgs.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
            o = overlaps(p, anchor, owner, rad)
            waa.append(o[0])
            wsa.append(o[1])
            xb.append(o[2])
        mm = multimm_faithful_pearson(coords, mids, c_obs, bin_starts, binsize)
        e = float(np.nanmean(expo))
        print(
            f"  {Path(d).name:>10s} {len(cifs):>3d} {np.nanmean(pear):>9.3f} "
            f"{np.nanmean(spear):>9.3f} {np.nanmean(scc):>8.3f} {mm:>9.3f} "
            f"{e:>9.3f} {e / NU_HIC:>7.2f}x {np.mean(rgs):>8.2f} "
            f"{np.mean(waa):>7.1f} {np.mean(wsa):>7.1f} {np.mean(xb):>7.1f}",
            flush=True,
        )
    print(f"\n  exponent target is {NU_HIC} from the cell lines' own contact probability curves;")
    print("  the project's structures have measured flatter than that, so higher is better here.")
    print(f"  the overlap columns count pairs closer than {ev_factor} of their block's mean chain")
    print("  bond, per thousand beads, on radii the first arm sets for all of them. wb-aa is")
    print("  anchors inside one block, which only the arcs")
    print("  stage can move; wb-sa is the smooth stage's own excluded volume; xb is across two")
    print("  blocks, which the boundary stitch and the cross block relaxation own.")


if __name__ == "__main__":
    main()
