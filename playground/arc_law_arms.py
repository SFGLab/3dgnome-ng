"""Rebuild a real block's arcs target matrix under different distance laws and solve each.

The stage's matrix mixes two laws that disagree. An arc's target comes from its PET count, times
a separation factor under the separation aware law, and lands between 0.2 and 1.6 model units
over five kb to one Mb. A consecutive arcless anchor pair's target comes from the chain law and
lands between 4.0 and 135 over the same range. The two differ by eleven times at five kb and a
hundred at one Mb, so a pair an arc joins is asked to sit an order of magnitude closer than the
chain says two anchors that far apart should be. Measured on a finished chromosome, 93 percent
of arc joined anchor pairs end up closer than the excluded volume's own radius.

Blocks come from a finished model rather than a pipeline run. The model carries each anchor's
genomic range, the block partition is recoverable from the densification rule and checkable, and
the arcs come from the same bedpe the run read, so a block's inputs are fully determined. The
one thing not reproduced is the anchor heatmap scaling, which shrinks arc targets further, so
the production arm here is a conservative version of the real one.

Each arm is one target matrix solved from a common start, so the comparison is the law and
nothing else. The gates are the overlap rate, anchor pairs closer than 0.7 of the realised chain
bond, and the realised distance exponent against the 0.285 the contact probability curves give.

    python playground/arc_law_arms.py <model.cif> <arcs.bedpe> <config.ini> [chr]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnome3d.mc.numba.arcs_solver import solve_arcs
from gnome3d.pipeline.coarse.build import add_chain_bonds, arc_expected_matrix
from gnome3d.settings import Settings
from gnome3d.types import F64Array
from validation_battery import block_owner, load

NU_HIC = 0.285


def read_arcs(path: str, chrom: str) -> list[tuple[int, int, int, int]]:
    """(midpoint of end one, midpoint of end two, score) per loop on `chrom`, as bp."""
    out: list[tuple[int, int, int, int]] = []
    for ln in open(path):
        f = ln.split()
        if len(f) < 6 or f[0] != chrom or f[3] != chrom:
            continue
        sc = int(f[6]) if len(f) > 6 and f[6].lstrip("-").isdigit() else 1
        out.append(((int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2, sc, 0))
    return out


def unified_matrix(
    s: Settings,
    mids: list[int],
    arcs: list[tuple[int, int, int]],
    pull: float,
) -> F64Array:
    """One law for both families. A pair sits at the chain law distance for its separation, and
    an arc pulls its pair in from there by a factor between `pull` and 1 set by its PET rank.

    The rank rather than the raw count, so the pull does not depend on a library's depth.
    """
    n = len(mids)
    mat: F64Array = np.full((n, n), -1.0, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    if arcs:
        sc = np.array([a[2] for a in arcs], dtype=np.float64)
        rank = sc.argsort().argsort() / max(len(sc) - 1, 1)
        for (i, j, _), r in zip(arcs, rank, strict=True):
            bg = float(s.genomic_length_to_distance(max(abs(mids[j] - mids[i]), 1)))
            mat[i, j] = mat[j, i] = bg * (1.0 - r * (1.0 - pull))
    return add_chain_bonds(mat, mids, s)


def bead_size(s: Settings, mids: list[int]) -> float:
    """The chain law at the block's median consecutive anchor gap. That is the distance the
    chain holds two neighbouring anchors at, so it is the bead's own size in model units."""
    g = np.diff(np.sort(np.array(mids)))
    return float(s.genomic_length_to_distance(max(int(np.median(g)) if g.size else 1000, 1)))


def floored(s: Settings, mat: F64Array, mids: list[int], frac: float) -> F64Array:
    """The production matrix with every positive target floored at `frac` of the bead's size.

    An arc says two loci touch, and touching is one bead apart. A target below that asks two
    beads to occupy the same space, which no repulsion elsewhere can undo.
    """
    out = np.array(mat, dtype=np.float64, copy=True)
    m = out > 1e-6
    out[m] = np.maximum(out[m], frac * bead_size(s, mids))
    return out


def measure(pos: np.ndarray, mids: np.ndarray, mat: F64Array) -> dict[str, float]:
    """The exponent band is clipped into the block's own span, since a band wider than the block
    is fitted on nothing."""
    order = np.argsort(mids)
    p, m = pos[order], mids[order].astype(float)
    bond = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    iu = np.triu_indices(len(p), 1)
    d = np.linalg.norm(p[iu[0]] - p[iu[1]], axis=1)
    sep = np.abs(m[iu[0]] - m[iu[1]])
    span = float(m.max() - m.min())
    lo, hi = max(2_000.0, span / 50.0), max(20_000.0, span / 2.0)
    k = (sep >= lo) & (sep <= hi) & (d > 0)
    expo = float(np.polyfit(np.log(sep[k]), np.log(d[k]), 1)[0]) if k.sum() > 50 else float("nan")
    far = np.abs(iu[0] - iu[1]) > 1
    linked = mat[iu] > 1e-6
    # Do arcs still bring their pair closer than an unlinked pair at the same separation? If not
    # the law has stopped using the data.
    band = (sep >= lo) & (sep <= hi)
    la, na = linked & band, ~linked & band
    contrast = (
        float(np.median(d[la]) / np.median(d[na])) if la.sum() > 3 and na.sum() > 3 else float("nan")
    )
    return {
        "contrast": contrast,
        "bond": bond,
        "overlap": 1000.0 * float((d[far] < 0.7 * bond).sum()) / len(p),
        "arc_ovl": float((d[linked] < 0.7 * bond).mean()) if linked.any() else float("nan"),
        "exponent": expo,
        "rg": float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())),
    }


def main() -> None:
    model, bedpe, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
    chrom = sys.argv[4] if len(sys.argv) > 4 else "chr1"
    s = Settings()
    s.load_ini(cfg)

    pos, start, bmid, anchor = load(Path(model))
    owner = block_owner(bmid, anchor, int(s.target_bp_per_subanchor))
    ai = np.where(anchor)[0]
    loops = read_arcs(bedpe, chrom)
    # A loop end belongs to the anchor whose midpoint is nearest, within half a bead span.
    amid = bmid[ai]
    order = np.argsort(amid)
    asort = amid[order]

    def nearest(p: int) -> int:
        k = int(np.clip(np.searchsorted(asort, p), 0, len(asort) - 1))
        best, bd = -1, 1 << 62
        for t in (k - 1, k, k + 1):
            if 0 <= t < len(asort) and abs(int(asort[t]) - p) < bd:
                # order indexes the anchor list, ai maps that back to a bead index
                bd, best = abs(int(asort[t]) - p), int(ai[order[t]])
        return best if bd <= 20_000 else -1

    per_bead: dict[tuple[int, int], int] = {}
    for p1, p2, sc, _ in loops:
        u, v = nearest(p1), nearest(p2)
        if u >= 0 and v >= 0 and u != v:
            key = (min(u, v), max(u, v))
            per_bead[key] = max(per_bead.get(key, 0), sc)

    blocks = [np.where((owner == b) & anchor)[0] for b in np.unique(owner[ai])]
    blocks = sorted((b for b in blocks if len(b) >= 25), key=lambda b: -len(b))[:8]
    rng = np.random.default_rng(0)
    print(
        f"{'N':>5} {'arcs':>6} {'arm':>22} {'bond':>7} {'ovlp/k':>8} "
        f"{'arc ovl':>8} {'exponent':>9} {'vs HiC':>7} {'Rg':>8} {'arc/bg':>7}"
    )
    for blk in blocks:
        pos_in_blk = {int(g): k for k, g in enumerate(blk)}
        mids = [int(bmid[g]) for g in blk]
        arcs = [
            (pos_in_blk[u], pos_in_blk[v], sc)
            for (u, v), sc in per_bead.items()
            if u in pos_in_blk and v in pos_in_blk
        ]
        n = len(mids)
        start_pos = np.ascontiguousarray(rng.normal(0.0, 1.5, (n, 3)).astype(np.float32))
        prod = add_chain_bonds(arc_expected_matrix(s, mids, arcs), mids, s)
        arms: list[tuple[str, F64Array]] = [("production", prod)]
        arms.append(("floor 1.5 bead", floored(s, prod, mids, 1.5)))
        for pull in (0.9, 0.75, 0.6, 0.45, 0.3):
            arms.append((f"unified pull {pull}", unified_matrix(s, mids, arcs, pull)))
        for name, mat in arms:
            _, out = solve_arcs(start_pos.copy(), mat, s)
            r = measure(np.asarray(out, np.float64), np.array(mids), mat)
            print(
                f"{n:>5} {len(arcs):>6} {name:>22} {r['bond']:>7.2f} {r['overlap']:>8.1f} "
                f"{r['arc_ovl']:>7.0%} {r['exponent']:>9.3f} {r['exponent'] / NU_HIC:>6.2f}x "
                f"{r['rg']:>8.2f} {r['contrast']:>7.2f}"
            )
        print(f"  bead size {bead_size(s, mids):.2f}, block span "
              f"{(max(mids) - min(mids)) / 1000:.0f} kb, median anchor gap "
              f"{np.median(np.diff(np.sort(np.array(mids)))) / 1000:.1f} kb")
    print(f"\n  exponent target {NU_HIC}; ovlp/k is anchor pairs closer than 0.7 bonds per 1000")
    print("  arc ovl is the share of arc joined pairs that end up overlapping")


if __name__ == "__main__":
    main()
