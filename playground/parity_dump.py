"""Reconstruct one region and dump its bead coordinates, for the byte exactness parity gate.

    python playground/parity_dump.py [--uniform] <tree_root> <out.npz> [region]

Run it once from a worktree at the previous commit and once from the working tree, with every
new flag off, and compare the two files exactly. Both runs use the same config and executor
settings, since batch grouping selects which chain gets which RNG stream.

Without `--uniform` the run reads `data/GM12878/config.ini` of the tree and the files it lists,
which git tracks. A worktree holds them.

With `--uniform` the run is the production config of GM12878 with `uniform_chain`,
`squeeze_constant` 1.0 and `loop_dropout`, two structures. It reads the data folder `data/` of
the directory the script is started in, as an absolute path, for every tree. A worktree has its
own `data/` with the tracked files alone, without the Hi-C singletons, and a run on it fits
another exponent. Start both runs in the checkout that holds the data. The line `polymer law`
of the two logs must be the same.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
uniform = sys.argv[1] == "--uniform"
if uniform:
    del sys.argv[1]
data_root = str(Path("data").resolve())
root = sys.argv[1]
out = str(Path(sys.argv[2]).resolve())
sys.path.insert(0, root)
os.chdir(root)

import numpy as np  # noqa: E402

region = sys.argv[3] if len(sys.argv) > 3 else "chr1:18288319-20307135"
if uniform:
    from gnome3d import log  # noqa: E402
    from gnome3d.data import ContactData  # noqa: E402
    from gnome3d.io import parse_region  # noqa: E402
    from gnome3d.production import settings_for_cell  # noqa: E402
    from gnome3d.simulate import simulate  # noqa: E402

    overrides = {
        "main": {"uniform_chain": "yes"},
        "springs": {"squeeze_constant": "1.0", "loop_dropout": "yes"},
    }
    s = settings_for_cell("GM12878", data_root=data_root, overrides=overrides)
    log.setup(s.output_level)
    bed = parse_region(region)
    assert bed is not None, region
    data = ContactData.from_files(s, [bed.chr], bed)
    structures = [per_chr[bed.chr] for per_chr in simulate(s, data, [bed.chr], 2, region=bed)]
else:
    from gnome3d.simulate import run_region  # noqa: E402

    structures = run_region("data/GM12878/config.ini", region, 1, data_dir="data/GM12878")
arr = np.array(
    [(k, b.start, b.end, b.x, b.y, b.z) for k, beads in enumerate(structures) for b in beads],
    dtype=np.float64,
)
np.savez(out, beads=arr[:, 1:] if not uniform else arr)
print(f"{root}: {len(arr)} beads -> {out}")
