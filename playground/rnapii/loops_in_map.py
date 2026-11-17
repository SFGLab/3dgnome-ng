"""Read a loop list in a contact map it was not called on, by loop span.

    python playground/rnapii/loops_in_map.py <slice.npz> <loops.bedpe> [<label>]

The slice is an npz of `hic_slice.py` or `cool_slice.py` at the loops' bin size or finer. Each
loop pixel is compared with its controls, the pixel moved 3 to 10 bins either way along the
diagonal, as `microc_loop_compare.py` does. Per span class the script prints the number of
loops, the summed loop count over the summed control mean, and the share of loops whose count
is over its control mean. A loop list that holds real loops reads above one in an independent
map of the same cells, and a list of chance calls reads one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SPANS = ((0, 300_000), (300_000, 1_000_000), (1_000_000, 10**10))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crosscell_prep import read_bedpe
    from microc_loop_compare import coarsen, count_enrichment
    from microc_readout import load_slice

    slice_path, loops_path = Path(sys.argv[1]), Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else slice_path.stem
    obs5, starts, b5 = load_slice(slice_path)
    lo = int(starts[0])
    loops = [lp for lp in read_bedpe(loops_path) if lp[1] >= lo]
    b = loops[0][2] - loops[0][1]
    obs = coarsen(obs5.astype(np.float64), b // b5)
    n = obs.shape[0]
    i = np.array([(lp[1] - lo) // b for lp in loops])
    j = np.array([(lp[4] - lo) // b for lp in loops])
    keep = (j < n) & (j - i >= 4)
    i, j = i[keep], j[keep]
    span = (j - i) * b
    cnt, ctl = count_enrichment(obs, i, j, obs.sum(axis=0) > 0)
    for a, c in SPANS:
        m = (span >= a) & (span < c) & np.isfinite(ctl) & (ctl > 0)
        if not m.any():
            continue
        print(
            f"{label:<28} span {a // 1000:>5} to {min(c, 10**7) // 1000:>5} kb  loops {int(m.sum()):>5}"
            f"  enrichment {cnt[m].sum() / ctl[m].sum():5.2f}  over control {(cnt[m] > ctl[m]).mean():.2f}"
            f"  median count {np.median(cnt[m]):6.1f}  median control {np.median(ctl[m]):6.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
