"""Large-region expansion ablation. Symptom: tuned Rg 26.95 >> ref 17.52 on large regions
(opposite of small regions where tuned compacts). Isolate which feature causes it.

parity vs tuned vs tuned-EV vs tuned-noDynamic on ONE large region.
  - tuned vs tuned-EV: SAME bead count -> isolates excluded-volume's Rg contribution.
  - tuned vs tuned-noDynamic: bead-count confound (dynamic sub-anchors ~2.3x beads).
"""
import numpy as np
import tempfile
from pathlib import Path

from gnome3d.settings import Settings
from gnome3d.data import ContactData
from validation import cell_config as cc
from validation import metrics
from validation.validate import _chrs_and_region, run_ensemble
import harness.integration as ig

REGION = "chr2:57942013-77730829"  # 19.8 Mb — the largest, where the median expansion should live
N = 2

tmp = Path(tempfile.mkdtemp(prefix="ablate_"))
config = tmp / "parity.ini"
ig.write_config(config, fast=False)
chrs_list, bed = _chrs_and_region(REGION)

s_par = Settings()
s_par.load_ini(str(config))
data = ContactData.from_files(s_par, chrs_list, bed)
span_mb = (bed.end - bed.start) / 1e6

s_tuned = cc.settings_for_cell("GM12878", "data", None)
s_noev = cc.apply_flags(s_tuned, {"use_excluded_volume": False})
s_nodyn = cc.apply_flags(s_tuned, {"use_dynamic_loop_density": False})

print(f"region {REGION}  span {span_mb:.1f} Mb")
for nm, s in [("use_excluded_volume", s_tuned), ("use_dynamic_loop_density", s_tuned)]:
    print(f"  tuned.{nm} = {getattr(s_tuned, nm, 'MISSING')}")


def measure(name, s):
    ens = run_ensemble(s, data, chrs_list, bed, N)
    rgs, ns, bonds, cvs, exts = [], [], [], [], []
    for beads in ens:
        coords, mids = metrics.to_arrays(beads)
        rgs.append(metrics.radius_of_gyration(coords))
        ns.append(len(coords))
        bl = metrics.bond_lengths(coords)
        bonds.append(float(np.median(bl)))
        cvs.append(float(bl.std() / bl.mean()))
        exts.append(metrics.max_extent(coords))
    rg, n = float(np.mean(rgs)), int(np.mean(ns))
    # Rg normalized by bead count (ideal-globule ~N^{1/3}) -> fair size across configs w/ diff N
    rg_norm = rg / (n ** (1 / 3))
    print(f"\n[{name}]  N={n}")
    print(f"  Rg            {rg:7.3f}")
    print(f"  Rg / N^(1/3)  {rg_norm:7.4f}   (bead-count-fair size)")
    print(f"  median bond   {np.mean(bonds):7.3f}")
    print(f"  bond CV       {np.mean(cvs):7.3f}")
    print(f"  max extent    {np.mean(exts):7.3f}")
    return rg, n, rg_norm


print("\n===== RESULTS =====")
measure("parity (features off)", s_par)
r_t, n_t, rn_t = measure("tuned (all on)", s_tuned)
r_e, n_e, rn_e = measure("tuned - EV", s_noev)
r_d, n_d, rn_d = measure("tuned - dynamic subanchors", s_nodyn)

print("\n===== VERDICT =====")
print(f"  tuned Rg {r_t:.2f} (N={n_t}) vs tuned-EV Rg {r_e:.2f} (N={n_e})  "
      f"-> EV adds {100*(r_t-r_e)/r_e:+.0f}% Rg at SAME N")
print(f"  tuned Rg {r_t:.2f} (N={n_t}) vs tuned-noDyn Rg {r_d:.2f} (N={n_d})  "
      f"-> dynamic beads add {100*(r_t-r_d)/r_d:+.0f}% Rg (N {n_t}->{n_d})")
