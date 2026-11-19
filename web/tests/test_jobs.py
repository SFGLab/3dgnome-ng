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
        req = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=11)
        same = JobRequest(cell_type="GMX", region="chr1:0-3,000,000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=11)
        check("the cache key ignores how the region was typed", request_key(req, cat) == request_key(same, cat))
        other = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2)
        check("a different factor choice is a different key", request_key(req, cat) != request_key(other, cat))
        # A site finds the jobs it holds by this key. A change of the key, of the constant in it
        # or of the way a request is normalised, leaves every job of a running site unfound.
        pinned = request_key(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF"), cat)
        check("the key of a request is the one sites already hold", pinned == "c873ddce1421a87d", pinned)
        check("100 structures are kept", JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=100).n_models == 100)
        check("101 structures become 100", JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=101).n_models == 100)
        none = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, background="none")
        check("the contact background is in the key", request_key(none, cat) != request_key(other, cat))
        old = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, hicdiffusion=True)
        new = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, background="hicdiffusion")
        check("the old HiCDiffusion flag and the new background give one key", request_key(old, cat) == request_key(new, cat))

        res = run_job(req, cat, Path(d) / "results" / request_key(req, cat), engine=FAKE)
        status = json.loads((res / "status.json").read_text())
        check("the job finishes", status["state"] == "done", str(status))
        check("eleven structures are written", len(list((res / "models").glob("*.cif"))) == 11)
        order = json.loads((res / "marks.json").read_text())["models"]
        check("structures are listed in their own order, s2 before s10", order.index(next(m for m in order if m.endswith("_s2.cif"))) < order.index(next(m for m in order if m.endswith("_s10.cif"))), str(order[:3]))
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

        from gnome3d_web.jobs import _lifted_marks

        ct = cat.cell_types["GMX"]
        _, enh = _lifted_marks(ct, "chr1", [], None)
        check("a listed cell type marks its own enhancers", len(enh) == 2, str(enh))
        _, enh = _lifted_marks(ct, "chr1", [], None, cell_enhancers=False)
        check("your own cell type without an enhancers upload marks none", len(enh) == 0, str(enh))

        bad = JobRequest(cell_type="GMX", region="chr9:0-1000000", factors="CTCF", n_models=1)
        # A job the site's restart interrupted is run again in its own directory.
        part = Path(d) / "results" / "part"
        run_job(other, cat, part, engine=FAKE)
        first = (part / "engine.log").read_text()
        kept = sorted((part / "models").glob("*.cif"))[0]
        stamp = kept.stat().st_mtime_ns
        sorted((part / "models").glob("*.cif"))[1].rename(part / "models" / "gone")
        run_job(other, cat, part, engine=FAKE)
        log = (part / "engine.log").read_text()
        check("a job run again keeps the log it had and says where it goes on", log.startswith(first) and "goes on after 1 of 2 structures" in log, log)
        check("and ends with every structure", len(list((part / "models").glob("*.cif"))) == 2 and json.loads((part / "status.json").read_text())["state"] == "done")

        from gnome3d.io import write_cif
        from gnome3d.types import BeadOut

        class Stops(Exception):
            pass

        def beads():  # noqa: ANN202
            yield BeadOut(0, 10, 0.0, 0.0, 0.0, "anchor")
            raise Stops

        whole = Path(d) / "whole.cif"
        try:
            write_cif(str(whole), beads())  # type: ignore[arg-type]
        except Stops:
            pass
        check("a structure file stopped half way is not there under its name", not whole.exists())
        write_cif(str(whole), [BeadOut(0, 10, 0.0, 0.0, 0.0, "anchor")])
        check("a whole one is, with nothing left beside it", whole.exists() and [p.name for p in Path(d).glob("whole*")] == ["whole.cif"])

        res = run_job(bad, cat, Path(d) / "results" / "bad", engine=FAKE)
        status = json.loads((res / "status.json").read_text())
        check("an engine failure is a failed job with the log tail", status["state"] == "failed" and "engine exploded" in status["log_tail"], str(status)[:200])
    return done()


if __name__ == "__main__":
    sys.exit(main())
