"""Localize the large-region expansion (NOT EV, NOT bead count). All at matched ~5k beads on the
19.8 Mb region, n=1. Reuse anchors: parity Rg~12.96, tuned-noDyn Rg~24.50 (from ablate_large2).

  B  parity + long MC schedule       -> is the expansion just CANONICAL annealing 10x longer?
  D  tuned-noDyn + strict anchors    -> does overlap_anchor_strict=True (parity's) recompact it?
  E  tuned-noDyn - heatmap targets   -> do subanchor/anchor-heatmap distance targets drive it?
  A  parity (re-baseline at n=1)
  C  tuned-noDyn (re-baseline at n=1)
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

REGION = "chr2:57942013-77730829"  # 19.8 Mb
N = 1

tmp = Path(tempfile.mkdtemp(prefix="ablate3_"))
config = tmp / "parity.ini"
ig.write_config(config, fast=False)
chrs_list, bed = _chrs_and_region(REGION)

s_par = Settings()
s_par.load_ini(str(config))
data = ContactData.from_files(s_par, chrs_list, bed)

s_tun = cc.settings_for_cell("GM12878", "data", None)
s_nodyn = cc.apply_flags(s_tun, {"use_dynamic_loop_density": False})

LONG = {"mc_stop_steps": 50000, "mc_stop_steps_heatmap": 50000,
        "mc_stop_steps_ib": 50000, "mc_stop_steps_smooth": 50000}
s_par_long = cc.apply_flags(s_par, LONG)
s_strict = cc.apply_flags(s_nodyn, {"overlap_anchor_strict": True})
s_noheat = cc.apply_flags(s_nodyn, {"use_subanchor_heatmap": False, "use_anchor_heatmap": False})


def measure(name, s):
    ens = run_ensemble(s, data, chrs_list, bed, N)
    coords, mids = metrics.to_arrays(ens[0])
    bl = metrics.bond_lengths(coords)
    rg = metrics.radius_of_gyration(coords)
    ext = metrics.max_extent(coords)
    print(f"[{name:32}] N={len(coords):5d}  Rg={rg:7.2f}  maxext={ext:7.2f}  "
          f"bond={float(np.median(bl)):5.3f}  bondCV={float(bl.std()/bl.mean()):5.3f}")
    return rg, ext


print(f"\nregion {REGION}  (n={N})\n" + "=" * 92)
measure("A parity", s_par)
measure("B parity + long schedule", s_par_long)
measure("C tuned-noDyn (baseline expanded)", s_nodyn)
measure("D tuned-noDyn + strict anchors", s_strict)
measure("E tuned-noDyn - heatmap targets", s_noheat)
print("=" * 92)
print("VERDICT: whichever of D/E collapses Rg toward A(parity) is the expander; if B~=A the schedule is innocent.")
