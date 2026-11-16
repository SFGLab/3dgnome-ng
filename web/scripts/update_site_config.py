"""Give the cell types of an existing site data directory the current production config.

    python web/scripts/update_site_config.py SITE_DATA TAG [CELL=COOLER_URI ...]

For each cell type of the catalog that has `slurm/ensemble/<cell>_xc_C.ini`, that file with the
site's hole settings appended is written beside the old config as `config_<TAG>.ini`, and the
catalog entry is pointed at it. A `CELL=URI` argument sets the cell type's `contact_map`. The old
config stays, and the catalog as it was is kept as `catalog.json.before-<TAG>`. Nothing is
overwritten: an existing target stops the run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dataset import HOLES  # noqa: E402


def main() -> None:
    site, tag = Path(sys.argv[1]), sys.argv[2]
    maps = dict(a.split("=", 1) for a in sys.argv[3:])
    cat_path = site / "catalog.json"
    keep = site / f"catalog.json.before-{tag}"
    if keep.exists():
        raise SystemExit(f"{keep} exists; pick another tag")
    cat = json.loads(cat_path.read_text())
    for name, c in cat["cell_types"].items():
        src = ROOT / "slurm" / "ensemble" / f"{name.lower()}_xc_C.ini"
        if not src.is_file():
            print(f"{name}: no {src.name}, left as it is")
            continue
        dst = site / c["dir"] / f"config_{tag}.ini"
        if dst.exists():
            raise SystemExit(f"{dst} exists; pick another tag")
        dst.write_text(src.read_text() + HOLES)
        c["config"] = dst.name
        if name in maps:
            c["contact_map"] = maps[name]
        print(f"{name}: {dst.name}" + (f", contact map {maps[name]}" if name in maps else ""))
    shutil.copyfile(cat_path, keep)
    cat_path.write_text(json.dumps(cat, indent=2) + "\n")


if __name__ == "__main__":
    main()
