"""Run one job outside the queue and print its final status.

    PYTHONPATH=web:web/vendor DATA_DIR=... python web/scripts/run_job.py <out dir> GM12878 chr1:11500000-12600000 \\
        --factors RNAPII --background none --structures 3 --set distance.polymer_exponent=0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, run_job
from gnome3d_web.settings import Settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("cell")
    ap.add_argument("region")
    ap.add_argument("--factors", default="CTCF+RNAPII")
    ap.add_argument("--background", default="hic")
    ap.add_argument("--structures", type=int, default=3)
    ap.add_argument("--set", action="append", default=[], help="section.key=value, repeated")
    a = ap.parse_args()
    req = JobRequest(
        cell_type=a.cell, region=a.region, factors=a.factors, background=a.background, n_models=a.structures,
        overrides=dict(s.split("=", 1) for s in a.set),
    )  # fmt: skip
    s = Settings.from_env()
    job = run_job(req, Catalog.load(s.data_dir), Path(a.out), engine=s.engine_cmd)
    status = json.loads((job / "status.json").read_text())
    print(json.dumps(status, indent=1))
    return 0 if status["state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
