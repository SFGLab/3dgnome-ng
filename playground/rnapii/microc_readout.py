"""Fine scale readouts against Micro-C, for the arms of one line.

    python playground/rnapii/microc_readout.py --json out.json <slice.npz> <loops.bedpe> <arm_dir> ...

The map is a slice cut by `hic_slice.py`, raw counts at 5 kb. Two numbers per arm. The pile-up
at the line's RNAPII loops, as `pileup.py` computes it, on the Micro-C grid. And `r_fine`, the
agreement at the scale RNAPII acts on: the map and the model's summed contact map are cut into
2 Mb windows, each divided by its diagonal means, and the Pearson of the log observed over
expected over pairs 20 kb to 1 Mb apart is averaged over windows. Bulk Hi-C at 25 kb cannot
resolve this range; Micro-C can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_slice(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """The symmetric matrix, the bin starts and the bin size of a `hic_slice.py` npz."""
    z = np.load(path)
    n, b, a = int(z["nbins"]), int(z["binsize"]), int(z["start"])
    m = np.zeros((n, n), dtype=np.float32)
    m[z["bin1"], z["bin2"]] = z["count"]
    m[z["bin2"], z["bin1"]] = z["count"]
    return m, a + np.arange(n, dtype=np.int64) * b, b


def fine_agreement(obs: np.ndarray, sim: np.ndarray, window: int = 400, lo: int = 4, hi: int = 200) -> float:
    """Mean over windows of the Pearson of log1p observed over expected between two maps, over
    pairs `lo` to `hi` bins apart. Windows where either map is empty are skipped."""
    from pileup import observed_over_expected

    vals = []
    n = obs.shape[0]
    for a in range(0, n - window + 1, window):
        o = observed_over_expected(obs[a : a + window, a : a + window].astype(np.float64))
        s = observed_over_expected(sim[a : a + window, a : a + window].astype(np.float64))
        i, j = np.triu_indices(window, lo)
        keep = (j - i) <= hi
        x, y = np.log1p(o[i[keep], j[keep]]), np.log1p(s[i[keep], j[keep]])
        if x.std() > 0 and y.std() > 0:
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crosscell_prep import read_bedpe
    from pileup import apa_score, pile

    from playground.validation_battery import CONTACT_BEADS, load
    from legacy.validation.metrics.hic import simulated_contacts

    args = sys.argv[1:]
    k = args.index("--json")
    out = Path(args[k + 1])
    del args[k : k + 2]
    slice_path, loops_path, *arms = args
    obs, starts, b = load_slice(Path(slice_path))
    lo_bp, hi_bp = int(starts[0]), int(starts[-1]) + b
    loops = [lp for lp in read_bedpe(Path(loops_path)) if lp[0] == "chr1" and lp[1] >= lo_bp and lp[5] <= hi_bp]
    p, used = pile(obs.astype(np.float64), starts, b, loops, 10, 5)
    rows = [{"arm": "map", "n": 0, "n_loops": used, "apa_microc": apa_score(p), "r_fine": 1.0}]
    for d in arms:
        cifs = sorted(Path(d).glob("*.cif"))
        sim = np.zeros_like(obs)
        for c in cifs:
            pos, _, mid, _ = load(c)
            rad = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            sim += simulated_contacts(pos, mid, starts, b, rad).astype(np.float32)
        p, used = pile(sim.astype(np.float64), starts, b, loops, 10, 5)
        rows.append({"arm": Path(d).name, "n": len(cifs), "n_loops": used, "apa_microc": apa_score(p), "r_fine": fine_agreement(obs, sim)})
        print(f"  {Path(d).name:<24} n {len(cifs):>2}  APA {rows[-1]['apa_microc']:.3f}  r_fine {rows[-1]['r_fine']:.3f}", flush=True)
    print(f"  map APA {rows[0]['apa_microc']:.3f} over {rows[0]['n_loops']} loops")
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
