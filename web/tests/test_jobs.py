"""One job end to end with a stand in engine, the cache key and the failure path.

    pytest web/tests/test_jobs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data

from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, request_key, run_job

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_engine.py")]


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    cat = Catalog.load(make_data(Path(d) / "data"))
    req = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=11)
    same = JobRequest(cell_type="GMX", region="chr1:0-3,000,000", factors="CTCF+RNAPII", sv_set="DEMO", sample="P1", haplotype=0, n_models=11)
    return SimpleNamespace(req=req, cat=cat, same=same, d=d)


def test_the_cache_key_ignores_how_the_region_was_typed(shared: SimpleNamespace) -> None:
    req, cat, same = shared.req, shared.cat, shared.same
    assert request_key(req, cat) == request_key(same, cat), "the cache key ignores how the region was typed"


@pytest.fixture(scope="module")
def other() -> Any:
    other = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2)
    return other


def test_a_different_factor_choice_is_a_different_key(other: Any, shared: SimpleNamespace) -> None:
    req, cat = shared.req, shared.cat
    assert request_key(req, cat) != request_key(other, cat), "a different factor choice is a different key"


def test_the_key_of_a_request_is_the_one_sites_already_hold(shared: SimpleNamespace) -> None:
    cat = shared.cat
    # A site finds the jobs it holds by this key. A change of the key, of the constant in it
    # or of the way a request is normalised, leaves every job of a running site unfound.
    pinned = request_key(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF"), cat)
    assert pinned == "c873ddce1421a87d", "the key of a request is the one sites already hold: " + str(pinned)
    assert JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=100).n_models == 100, "100 structures are kept"
    assert JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=101).n_models == 100, "101 structures become 100"


def test_the_contact_background_is_in_the_key(other: Any, shared: SimpleNamespace) -> None:
    cat = shared.cat
    none = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, background="none")
    assert request_key(none, cat) != request_key(other, cat), "the contact background is in the key"


def test_the_old_hicdiffusion_flag_and_the_new_background_give_one_key(shared: SimpleNamespace) -> None:
    cat = shared.cat
    old = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, hicdiffusion=True)
    new = JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", n_models=2, background="hicdiffusion")
    assert request_key(old, cat) == request_key(new, cat), "the old HiCDiffusion flag and the new background give one key"


@pytest.fixture(scope="module")
def status_and_res(shared: SimpleNamespace) -> SimpleNamespace:
    req, cat, d = shared.req, shared.cat, shared.d
    res = run_job(req, cat, Path(d) / "results" / request_key(req, cat), engine=FAKE)
    status = json.loads((res / "status.json").read_text())
    return SimpleNamespace(status=status, res=res)


def test_the_job_finishes(status_and_res: SimpleNamespace) -> None:
    status, res = status_and_res.status, status_and_res.res
    assert status["state"] == "done", "the job finishes: " + str(status)
    assert len(list((res / "models").glob("*.cif"))) == 11, "eleven structures are written"


def test_structures_are_listed_in_their_own_order_s2_before_s10(status_and_res: SimpleNamespace) -> None:
    res, status = status_and_res.res, status_and_res.status
    order = json.loads((res / "marks.json").read_text())["models"]
    assert order.index(next(m for m in order if m.endswith("_s2.cif"))) < order.index(next(m for m in order if m.endswith("_s10.cif"))), "structures are listed in their own order, s2 before s10: " + str(order[:3])
    assert status["n_svs"] == 1 and status["skipped_svs"] == 1, "the SVs applied and skipped are reported: " + str(status)


def test_the_enhancer_table_has_a_row_per_marked_gene(status_and_res: SimpleNamespace) -> None:
    res = status_and_res.res
    ep = [ln.split("\t") for ln in (res / "ep.tsv").read_text().splitlines()]
    beta = next((r for r in ep if r[1] == "BETA"), None)
    assert len(ep) == 3, "the enhancer table has a row per marked gene: " + str(ep)
    assert beta is not None and abs(int(beta[-1]) - 2_540_000) <= 5_000, "a gene after the deletion is marked at its haplotype position: " + str(beta)


def test_marks_list_promoter_and_enhancer_beads(status_and_res: SimpleNamespace) -> None:
    res = status_and_res.res
    marks = json.loads((res / "marks.json").read_text())
    assert marks["promoters"] and marks["enhancers"], "marks list promoter and enhancer beads"


def test_a_listed_cell_type_marks_its_own_enhancers(shared: SimpleNamespace) -> None:
    cat = shared.cat
    from gnome3d_web.jobs import _lifted_marks

    ct = cat.cell_types["GMX"]
    _, enh = _lifted_marks(ct, "chr1", [], None)
    assert len(enh) == 2, "a listed cell type marks its own enhancers: " + str(enh)
    _, enh = _lifted_marks(ct, "chr1", [], None, cell_enhancers=False)
    assert len(enh) == 0, "your own cell type without an enhancers upload marks none: " + str(enh)


@pytest.fixture(scope="module")
def log_and_first_and_part(other: Any, shared: SimpleNamespace) -> SimpleNamespace:
    d, cat = shared.d, shared.cat
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
    return SimpleNamespace(log=log, first=first, part=part, bad=bad)


def test_a_job_run_again_keeps_the_log_it_had_and_says_where_it_goes_on(log_and_first_and_part: SimpleNamespace) -> None:
    log, first, part = log_and_first_and_part.log, log_and_first_and_part.first, log_and_first_and_part.part
    assert log.startswith(first) and "goes on after 1 of 2 structures" in log, "a job run again keeps the log it had and says where it goes on: " + str(log)
    assert len(list((part / "models").glob("*.cif"))) == 2 and json.loads((part / "status.json").read_text())["state"] == "done", "and ends with every structure"


def test_a_structure_file_stopped_half_way_is_not_there_under_its_name(shared: SimpleNamespace) -> None:
    d = shared.d
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
    assert not whole.exists(), "a structure file stopped half way is not there under its name"
    write_cif(str(whole), [BeadOut(0, 10, 0.0, 0.0, 0.0, "anchor")])
    assert whole.exists() and [p.name for p in Path(d).glob("whole*")] == ["whole.cif"], "a whole one is, with nothing left beside it"


def test_an_engine_failure_is_a_failed_job_with_the_log_tail(log_and_first_and_part: SimpleNamespace, shared: SimpleNamespace) -> None:
    bad = log_and_first_and_part.bad
    cat, d = shared.cat, shared.d
    res = run_job(bad, cat, Path(d) / "results" / "bad", engine=FAKE)
    status = json.loads((res / "status.json").read_text())
    assert status["state"] == "failed" and "engine exploded" in status["log_tail"], "an engine failure is a failed job with the log tail: " + str(status)[:200]
