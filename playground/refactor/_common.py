"""Shared fixtures for the task-DAG refactor validations.

One small bundled region on the numba backend (no GPU needed), plus the
skeleton->Dag chain assembly the real driver will eventually use.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from gnome3d import log, skeleton  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_region  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

REGION = "chr1:18288319-20307135"
CONFIG = "data/GM12878/config_dryrun.ini"
DATA_DIR = "data/GM12878/"


def load_region(output_level: int = 0):
    """Settings + ContactData + skeleton seeds for the bundled test region."""
    log.setup(output_level)
    s = Settings()
    if not s.load_ini(CONFIG):
        raise SystemExit(f"cannot load {CONFIG} (run from repo root)")
    s.data_dir = DATA_DIR
    s.mc_backend = "numba"  # local, no GPU
    bed = parse_region(REGION)
    data = ContactData.from_files(s, [bed.chr], bed)
    seeds = skeleton.build_seeds(s, data, [bed.chr], bed)
    return s, bed, data, seeds
