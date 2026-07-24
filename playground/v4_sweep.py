"""V4 methodology sweep (DATA-LEVEL, no model runs). Find the principled comparison config that
reaches the paper's ChIA-PET<->Hi-C band (rho 0.67-0.88, Fig. 2). Sweep resolution x Hi-C balance x
transform, median over regions. Report raw-Pearson(log1p), raw-Pearson, Spearman, and O/E-Pearson.
"""
import numpy as np
from scipy.stats import spearmanr

from validation.validate import load_chiapet_contacts, _chrs_and_region
from validation import contacts as C
from validation import cell_config as cc
import harness.integration as ig

s = cc.settings_for_cell("GM12878", "data", None)
mcool = "data/_hic/GM12878/4DNFIQ32RWCQ.mcool"

bp = str(ig.DATA_DIR / "ccds_all_hg38_merged100k_GM12878.breakpoints.bed")
from validation.compare_reference import enumerate_regions
REGIONS = enumerate_regions(bp, 12, chroms=["chr1", "chr2", "chr8", "chr17"],
                            min_ibs=4, max_ibs=15, max_mb=20)
RES = [100_000, 250_000, 500_000, 1_000_000]

# cache ChIA-PET contacts per region
chia_cache = {}
for r in REGIONS:
    cl, bed = _chrs_and_region(r)
    chia_cache[r] = load_chiapet_contacts(s, cl, bed)


def corrset(c_chia, c_obs, min_sep):
    n = c_obs.shape[0]
    if n < 4:
        return None
    iu = np.triu_indices(n, min_sep)
    a, b = c_chia[iu], c_obs[iu]
    both = (a > 0) & (b > 0)
    out = {}
    if a.std() > 1e-12 and b.std() > 1e-12:
        out["logP"] = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
        out["rawP"] = float(np.corrcoef(a, b)[0, 1])
        out["spear"] = float(spearmanr(a, b).statistic)
    # O/E (decay-stripped)
    oe = C._oe_correlation(C.observed_over_expected(c_chia, min_sep),
                           C.observed_over_expected(c_obs, min_sep), min_sep)
    out["oeP"] = oe["pearson_oe"]
    out["fill"] = float(both.mean())
    return out


print(f"{len(REGIONS)} regions, median over regions. paper Fig.2: rho 0.67-0.88\n")
for balance in (True, False):
    print(f"===== Hi-C balance={'ICE' if balance else 'raw'} =====")
    print(f"{'res':>7} {'diag':>5} | {'logP':>6} {'rawP':>6} {'spear':>6} {'oeP':>6} {'fill%':>6}")
    for res in RES:
        for min_sep in (1, 2):  # remove main diagonal vs a 2-bin band
            rows = []
            for r in REGIONS:
                try:
                    c_obs, starts = C.observed_hic(mcool, r, res, balance=balance)
                except Exception:
                    continue
                eff = int(starts[1] - starts[0]) if len(starts) > 1 else res
                c_chia = C.contact_list_heatmap(chia_cache[r], starts, eff)
                cs = corrset(c_chia, c_obs, min_sep)
                if cs:
                    rows.append(cs)
            if not rows:
                continue
            med = lambda k: float(np.nanmedian([x[k] for x in rows if k in x]))
            print(f"{res//1000:>6}k {min_sep:>5} | {med('logP'):>6.3f} {med('rawP'):>6.3f} "
                  f"{med('spear'):>6.3f} {med('oeP'):>6.3f} {100*med('fill'):>5.1f}%")
    print()
