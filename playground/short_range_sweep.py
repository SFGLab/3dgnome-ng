"""Choose the short range background's weight and range on real blocks.

An arcless anchor pair inside `background_range_bp` is held at the background for its separation
by a spring of `background_weight`; every other arcless pair keeps the repulsion that won the
battery. The all pairs version cured the kink in the realised exponent and lost every Hi-C
statistic, because a power law distance matrix at an exponent under a third cannot be embedded
in three dimensions over every pair. A band of it can, and this sweep asks how wide and how
strong. The gate is the two band exponents against the cell's measured input, with overlaps,
arcs realised over target, and the block's size alongside.

    python playground/short_range_sweep.py <model.cif> <arcs.bedpe> <nu> [chr]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnome3d.mc.numba.arcs_solver import solve_arcs
from gnome3d.pipeline.coarse.build import add_chain_bonds, arc_expected_matrix
from gnome3d.polymer import PolymerLaw, fit_arc_strength
from gnome3d.settings import Settings
from gnome3d.types import InteractionArc
from validation_battery import block_owner, load


def read_arcs(path: str, chrom: str) -> list[tuple[int, int, int]]:
    out = []
    for ln in open(path):
        f = ln.split()
        if len(f) < 6 or f[0] != chrom or f[3] != chrom:
            continue
        sc = int(f[6]) if len(f) > 6 and f[6].lstrip("-").isdigit() else 1
        out.append(((int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2, sc))
    return out


def band_exponent(pos: np.ndarray, mids: np.ndarray, lo: float, hi: float) -> float:
    iu = np.triu_indices(len(pos), 1)
    d = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    sep = np.abs(mids[iu[0]] - mids[iu[1]]).astype(float)
    k = (sep >= lo) & (sep <= hi) & (d > 0)
    return float(np.polyfit(np.log(sep[k]), np.log(d[k]), 1)[0]) if k.sum() > 40 else float("nan")


def main() -> None:
    model, bedpe, nu = sys.argv[1], sys.argv[2], float(sys.argv[3])
    chrom = sys.argv[4] if len(sys.argv) > 4 else "chr1"
    pos, start, bmid, anchor = load(Path(model))
    owner = block_owner(bmid, anchor, 1000)
    ai = np.where(anchor)[0]
    amid = bmid[ai]
    order = np.argsort(amid)
    asort = amid[order]

    def nearest(p: int) -> int:
        k = int(np.clip(np.searchsorted(asort, p), 0, len(asort) - 1))
        best, bd = -1, 1 << 62
        for t in (k - 1, k, k + 1):
            if 0 <= t < len(asort) and abs(int(asort[t]) - p) < bd:
                bd, best = abs(int(asort[t]) - p), int(ai[order[t]])
        return best if bd <= 20_000 else -1

    per_bead: dict[tuple[int, int], int] = {}
    for p1, p2, sc in read_arcs(bedpe, chrom):
        u, v = nearest(p1), nearest(p2)
        if u >= 0 and v >= 0 and u != v:
            key = (min(u, v), max(u, v))
            per_bead[key] = max(per_bead.get(key, 0), sc)
    all_arcs = [
        InteractionArc(0, 1, sc, genomic_start=int(bmid[u]), genomic_end=int(bmid[v]))
        for (u, v), sc in per_bead.items()
    ]
    strength = fit_arc_strength(all_arcs)

    blocks = [np.where((owner == b) & anchor)[0] for b in np.unique(owner[ai])]
    blocks = sorted((b for b in blocks if len(b) >= 40), key=lambda b: -len(b))[:8]
    rng = np.random.default_rng(0)
    print(f"{len(blocks)} blocks, nu {nu}, arc strength fit {'ok' if strength.ok else strength.reason}")
    print(
        f"{'weight':>7} {'range':>6} | {'20-100k':>8} {'100k-1M':>8} {'ovlp/k':>7} "
        f"{'arc d/t':>8} {'bg d/t':>7} {'Rg':>6}"
    )
    arms = [(0.0, 0)] + [(w, r) for r in (50_000, 100_000, 200_000) for w in (0.03, 0.1, 0.3, 1.0)]
    for w, r in arms:
        rows = []
        for blk in blocks:
            idx = {int(g): k for k, g in enumerate(blk)}
            mids = [int(bmid[g]) for g in blk]
            arcs = [(idx[u], idx[v], sc) for (u, v), sc in per_bead.items() if u in idx and v in idx]
            s = Settings()
            s.polymer = PolymerLaw(nu=nu, s0_bp=1000, q_half=1.0, arcs=strength)
            s.use_arcs_chain_bonds = True
            s.arcs_chain_bond_scale = 1.5
            s.arcs_repulsion_cutoff_factor = 3.0
            s.background_weight = w
            s.background_range_bp = r
            s.use_confinement = True
            s.confinement_apply_to_arcs = True
            s.use_excluded_volume = True
            s.exclusion_apply_to_arcs = True
            s.exclusion_weight = 0.1
            mat = add_chain_bonds(arc_expected_matrix(s, mids, arcs), mids, s)
            n = len(mids)
            start_pos = np.ascontiguousarray(rng.normal(0.0, 1.5, (n, 3)).astype(np.float32))
            _, out = solve_arcs(start_pos, mat, s)
            p = np.asarray(out, np.float64)
            m = np.array(mids, dtype=float)
            iu = np.triu_indices(n, 1)
            d = np.linalg.norm(p[iu[0]] - p[iu[1]], axis=1)
            e = mat[iu]
            bond = float(np.median(np.linalg.norm(np.diff(p[np.argsort(m)], axis=0), axis=1)))
            far = np.abs(iu[0] - iu[1]) > 1
            bgp = e <= -0.75
            rows.append(
                [
                    band_exponent(p, m, 2e4, 1e5),
                    band_exponent(p, m, 1e5, 1e6),
                    1000.0 * float((d[far] < 0.7 * bond).sum()) / n,
                    float(np.median(d[e > 1e-6] / e[e > 1e-6])) if (e > 1e-6).any() else np.nan,
                    float(np.median(d[bgp] / -e[bgp])) if bgp.any() else np.nan,
                    float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())),
                ]
            )
        rr = np.nanmean(rows, axis=0)
        label = "off" if w == 0.0 else f"{r // 1000}kb"
        print(
            f"{w:>7.2f} {label:>6} | {rr[0]:>8.3f} {rr[1]:>8.3f} {rr[2]:>7.1f} "
            f"{rr[3]:>8.2f} {rr[4]:>7.2f} {rr[5]:>6.1f}",
            flush=True,
        )
    print("\n  target for both bands is nu; arc d/t and bg d/t are realised over target, 1.0 when")
    print("  satisfied; ovlp/k is anchor pairs under 0.7 bonds per thousand; Rg the block's size.")


if __name__ == "__main__":
    main()
