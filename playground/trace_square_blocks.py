"""What holds anchor pairs between neighbouring bins in a finished job, by pair kind.

Rebuilds the joint arcs target matrix of a region the way the joint solve does, reads the
job's models, and for every anchor pair 25 kb to 400 kb apart reports its kind, the realised
distance against the law's background and the force each kind exerts at the realised
positions. Pairs are split by how many anchors their two 25 kb bins hold, so dense bins and
sparse bins can be compared at matched separation. Also prints the repulsion reach, which the
solver sets from the mean arc target, with every factor and with factor 0 alone.

    python playground/trace_square_blocks.py CONFIG.ini REGION MODELS_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse.build import _active_arcs
from gnome3d.settings import Settings
from gnome3d.skeleton import _collect_ib_work

BIN = 25_000
BANDS = ((25_000, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 400_000))


def read_anchors(cif: Path) -> dict[tuple[int, int], np.ndarray]:
    """Anchor bead positions of one model keyed by genomic start and end."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for line in cif.read_text().splitlines():
        if line.startswith("ATOM") and line.endswith("anchor"):
            f = line.split()
            out[(int(f[-3]), int(f[-2]))] = np.array([float(f[10]), float(f[11]), float(f[12])])
    return out


def main() -> None:
    config, region, models_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    s = Settings()
    s.load_ini(config)
    chrom = region.split(":")[0]
    chrs, reg = parse_chrs_arg(region)
    data = ContactData.from_files(s, chrs, reg)
    state = cb.build_state(s, data, chrs, reg)
    segs = [i for i, cl in enumerate(state.clusters) if int(cl.level) == 2]
    work = _collect_ib_work(state, chrom, {chrom: segs})
    active = [a for _, _, ar in work for a in ar]
    heat = None
    if s.use_anchor_heatmap and state.singletons:
        heat, _ = cb.build_contact_heatmaps(state, active, chrom, with_subanchor=False)
    m = cb.calc_anchor_expected_distances(state, active, chrom, heat)
    law = s.polymer_law()
    n = len(active)
    cl = state.clusters
    mids = np.array([cl[a].genomic_pos for a in active], dtype=np.int64)

    arc_t = m[m > 1e-6]
    _, arcs = _active_arcs(state, active, chrom, None)
    f0 = [s.arc_expected_distance(a[2], int(mids[a[1]] - mids[a[0]]), a[3] if len(a) > 3 else 0) for a in arcs if (a[3] if len(a) > 3 else 0) == 0]
    reach = float(s.arcs_repulsion_cutoff_factor) * float(arc_t.mean())
    print(f"{region}: {n} anchors in {len(work)} blocks, {len(arcs)} arcs, nu {law.nu:.3f}")
    print(f"  mean arc target {arc_t.mean():.2f} beads, repulsion reach {reach:.2f}; factor 0 alone mean {np.mean(f0):.2f}, reach {float(s.arcs_repulsion_cutoff_factor) * np.mean(f0):.2f}")
    for lo, hi in BANDS:
        sep = (lo + hi) // 2
        print(f"  law background at {sep // 1000} kb: {law.background(sep):.2f} beads")

    models = [read_anchors(p) for p in sorted(models_dir.glob("*.cif"))]
    keys = [(cl[a].start, cl[a].end) for a in active]
    have = [all(k in md for k in keys) for md in models]
    models = [md for md, ok in zip(models, have, strict=True) if ok]
    if not models:
        raise SystemExit("no model holds every active anchor; the start and end keys do not match")
    pos = np.stack([np.stack([md[k] for k in keys]) for md in models])  # (models, n, 3)
    print(f"  {len(models)} models matched on all {n} anchors")

    b = (mids - mids.min()) // BIN
    per_bin = np.bincount(b)
    density = per_bin[b]
    lo_d, hi_d = np.percentile(per_bin[per_bin > 0], [33, 67])
    iu = np.triu_indices(n, k=1)
    sep = np.abs(mids[iu[1]] - mids[iu[0]])
    t = m[iu]
    kind = np.where(t > 1e-6, "arc", np.where(t <= -0.75, "spring", "repulsion"))
    d = np.linalg.norm(pos[:, iu[0]] - pos[:, iu[1]], axis=-1)  # (models, pairs)
    dm = d.mean(axis=0)
    bgs = np.maximum(1.0, (sep / max(int(law.s0_bp), 1)) ** law.nu)
    rep_inv = 1.0 / reach if reach > 0 else 0.0
    # dE/dd per unordered pair at each model's realised distance, positive pulls together.
    dd = np.maximum(d, 1e-10)
    force = np.zeros_like(d)
    a = t > 1e-6
    rel = (d - t) / np.where(a, t, 1.0)
    k = np.where(rel >= 0, s.spring_stretch_arcs, s.spring_squeeze_arcs)
    force = np.where(a, 2.0 * k * rel / np.where(a, t, 1.0), force)
    sp = t <= -0.75
    bg = -t
    f_in = 2.0 * s.background_weight * (1.0 - bg / dd) * bg / dd**2
    f_out = 2.0 * s.background_weight * (d - bg) / np.where(sp, bg, 1.0) ** 2
    force = np.where(sp, np.where(d < bg, f_in, f_out), force)
    rp = (t < 0) & ~sp
    force = np.where(rp & (1.0 / dd > rep_inv), -1.0 / dd**2, force)
    fm = force.mean(axis=0)

    dense = (density[iu[0]] >= hi_d) & (density[iu[1]] >= hi_d)
    sparse = (density[iu[0]] <= lo_d) & (density[iu[1]] <= lo_d)
    print(f"  anchors per 25 kb bin: tercile edges {lo_d:.0f} and {hi_d:.0f}, max {per_bin.max()}")
    print(f"  {'band':>11} {'bins':>7} {'pairs':>6} {'arc':>5} {'spring':>6} {'repel':>6} {'d/law':>6} {'in reach':>8} {'F arc':>8} {'F spring':>8} {'F repel':>8} {'F net':>8}")
    for lo, hi in BANDS:
        for name, sel0 in (("dense", dense), ("sparse", sparse)):
            sel = sel0 & (sep >= lo) & (sep < hi)
            if not sel.any():
                continue
            kk = kind[sel]
            inr = (rp[sel] & (dm[sel] < reach)).sum() / max(rp[sel].sum(), 1)
            fa = fm[sel][kk == "arc"].sum() / sel.sum()
            fs = fm[sel][kk == "spring"].sum() / sel.sum()
            fr = fm[sel][kk == "repulsion"].sum() / sel.sum()
            print(
                f"  {lo // 1000:>4}-{hi // 1000:<4}kb {name:>7} {int(sel.sum()):>6} {(kk == 'arc').mean():>5.2f} {(kk == 'spring').mean():>6.2f} {(kk == 'repulsion').mean():>6.2f}"
                f" {np.median(dm[sel] / bgs[sel]):>6.2f} {inr:>8.2f} {fa:>8.3f} {fs:>8.3f} {fr:>8.3f} {fa + fs + fr:>8.3f}"
            )


if __name__ == "__main__":
    main()
