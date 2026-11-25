"""Run pyLoopSage on the loops of one of our arms and collect an ensemble for the validation battery.

    python playground/loopsage_arm.py LOOPS.bedpe REGION OUT_DIR [--genome hg38.fa] [--n 10]
                                      [--loopsage loopsage] [--platform CUDA] [--threads 8] [--seed 1]

The loops file is our seven column clusters bedpe, which pyLoopSage reads. It carries no
orientation of the CTCF sites, and with `--genome` pyLoopSage finds the motif in each anchor
itself, the path its documentation gives for ChIA-PET loops. Every other setting is its default:
one bead per 2 kb, 20,000 Monte Carlo steps of the loop extrusion with a snapshot every 200 and
one frame of molecular dynamics a snapshot. The figures, the report and the viewer are off.

pyLoopSage writes one structure a frame, `structures/ensemble/MDLE_<k>.cif`. The ensemble is the
`--n` frames evenly spaced from the first frame after the burn in to the last one. Its beads
carry no genomic coordinates. Each collected cif gives bead k of n the span
`[lo + (hi - lo) k / n, lo + (hi - lo) (k + 1) / n)` and marks every bead a subanchor, as
`multimm_arm.py` does. Every frame after the burn in goes to `<OUT_DIR>_all`, the ensemble that
pyLoopSage itself averages. The run stays in `<OUT_DIR>_run`.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multimm_arm import read_multimm_cif, uniform_beads  # noqa: E402

from gnome3d.io import write_cif  # noqa: E402


def _flag(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    v = sys.argv[k + 1]
    del sys.argv[k : k + 2]
    return v


def frames_after_burnin(run: Path) -> list[Path]:
    """Returns the frames of a run after the burn in of its resolved config, in time order."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read(run / "config_resolved.ini")
    first = int(cfg["monte_carlo"]["burnin"]) // int(cfg["monte_carlo"]["mc_step"]) + 1
    frames = sorted((run / "structures" / "ensemble").glob("MDLE_*.cif"), key=lambda p: int(p.stem.split("_")[1]))
    return [f for f in frames if int(f.stem.split("_")[1]) >= first]


def main() -> None:
    genome = _flag("--genome", "")
    n = int(_flag("--n", "10"))
    loopsage = _flag("--loopsage", "loopsage")
    platform = _flag("--platform", "CUDA")
    threads = _flag("--threads", "8")
    seed = _flag("--seed", "1")
    loops, region, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    lo, hi = (int(v) for v in region.split(":")[1].split("-"))
    work = out.parent / f"{out.name}_run"
    cmd = [loopsage, "-f", str(loops.resolve()), "-r", region, "-o", str(work.resolve()), "--platform", platform,
           "--seed", seed, "-t", threads, "--no-figures", "--no-report", "--no-render-3d"]  # fmt: skip
    if genome:
        cmd += ["--motif-genome", genome]
    print(" ".join(cmd), flush=True)
    work.mkdir(parents=True, exist_ok=True)
    with open(out.parent / f"{out.name}_run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    runs = [p.parent for p in work.glob("*/config_resolved.ini")]
    if rc != 0 or len(runs) != 1:
        raise SystemExit(f"pyLoopSage exit {rc}, {len(runs)} run folders under {work}")
    frames = frames_after_burnin(runs[0])
    if len(frames) < n:
        raise SystemExit(f"{len(frames)} frames after the burn in, {n} asked")
    picked = [frames[int(round(k))] for k in np.linspace(0, len(frames) - 1, n)]
    every = out.parent / f"{out.name}_all"
    for folder, chosen in ((out, picked), (every, frames)):
        folder.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(chosen):
            write_cif(str(folder / f"member_{k + 1}.cif"), uniform_beads(read_multimm_cif(frame), lo, hi))
    print(f"pyLoopSage {region}: {len(read_multimm_cif(picked[0]))} beads, frames {', '.join(f.stem.split('_')[1] for f in picked)} of {len(frames)} after the burn in")


if __name__ == "__main__":
    main()
