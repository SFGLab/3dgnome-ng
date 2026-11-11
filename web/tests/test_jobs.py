"""One job end to end with a stand in engine, the cache key and the failure path.

    python web/tests/test_jobs.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data

from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, request_key, run_job

FAKE = [str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"), str(Path(__file__).resolve().parent / "fake_engine.py")]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        cat = Catalog.load(make_data(Path(d) / "data"))
        req = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=2)
        same = JobRequest(cell_type="GMX", region="chr1:0-3,000,000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=2)
        check("the cache key ignores how the region was typed", request_key(req, cat) == request_key(same, cat))
        other = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2)
        check("a different factor choice is a different key", request_key(req, cat) != request_key(other, cat))

        res = run_job(req, cat, Path(d) / "results" / request_key(req, cat), engine=FAKE)
        status = json.loads((res / "status.json").read_text())
        check("the job finishes", status["state"] == "done", str(status))
        check("two structures are written", len(list((res / "models").glob("*.cif"))) == 2)
        check("the SVs applied and skipped are reported", status["n_svs"] == 1 and status["skipped_svs"] == 1, str(status))
        ep = [ln.split("\t") for ln in (res / "ep.tsv").read_text().splitlines()]
        beta = next((r for r in ep if r[1] == "BETA"), None)
        check("the enhancer table has a row per marked gene", len(ep) == 3, str(ep))
        check(
            "a gene after the deletion is marked at its haplotype position",
            beta is not None and abs(int(beta[-1]) - 2_540_000) <= 5_000,
            str(beta),
        )
        marks = json.loads((res / "marks.json").read_text())
        check("marks list promoter and enhancer beads", marks["promoters"] and marks["enhancers"])

        bad = JobRequest(cell_type="GMX", region="chr9:0-1000000", factors="CTCF", n_models=1)
        res = run_job(bad, cat, Path(d) / "results" / "bad", engine=FAKE)
        status = json.loads((res / "status.json").read_text())
        check("an engine failure is a failed job with the log tail", status["state"] == "failed" and "engine exploded" in status["log_tail"], str(status)[:200])
    return done()


if __name__ == "__main__":
    sys.exit(main())
