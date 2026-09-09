"""Compartment statistics of finished structures, so a battery run can be read for
compartmentalisation without a reconstruction.

Every arm is scored at the first arm's contact radius, the rule the epigenome study uses, so a
term that shortens bonds does not change the effective resolution of its own map.

    python playground/saddle_offline.py <mcool> <region> <binsize> <compartment bedGraph> <dir> [<dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.restitch_model import read_cif  # noqa: E402
from validation.metrics import hic as contacts  # noqa: E402
from validation.metrics import structure as smetrics  # noqa: E402
from validation.studies.epigenome import _track_on_bins  # noqa: E402


def main() -> None:
    mcool, region, binsize, track = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    dirs = sys.argv[5:]
    chrom = region.split(":")[0]
    c_obs, bin_starts = contacts.observed_hic(mcool, region, binsize, balance=True)
    sort_track = _track_on_bins(track, chrom, bin_starts)
    obs = contacts.compartment_saddle(c_obs, sort_track)["strength"]
    print(f"{region} @ {binsize // 1000} kb: experimental saddle {obs:.3f}")
    print(f"  {'arm':<28}{'saddle':>8}{'eig |r|':>9}{'kappa':>8}{'n':>4}")
    radius: float | None = None
    for d in dirs:
        cifs = sorted(Path(d).glob("*.cif"))
        c_sim = np.zeros_like(c_obs)
        for p in cifs:
            coords, mids = smetrics.to_arrays(read_cif(str(p)))
            if radius is None:
                radius = float(np.median(smetrics.bond_lengths(coords)))
            c_sim += contacts.simulated_contacts(coords, mids, bin_starts, binsize, radius)
        sad = contacts.compartment_saddle(c_sim, sort_track)["strength"]
        cc = contacts.compartment_correlation(c_sim, c_obs)
        print(
            f"  {Path(d).name:<28}{sad:>8.3f}{cc['eig_pearson_abs']:>9.3f}"
            f"{cc['agreement_kappa']:>8.3f}{len(cifs):>4}"
        )


if __name__ == "__main__":
    main()
