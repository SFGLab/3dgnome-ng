"""Uploads of own data: storage with a size cap, the check of each kind, the preview and the sweep.

    pytest web/tests/test_owndata.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data


@pytest.fixture(scope="module", autouse=True)
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    root = Path(d)
    data = make_data(root / "data")
    os.environ.update(
        DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), UPLOAD_MAX_MB="1",
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    ct = data / "cell_types" / "GMX"

    def up(kind: str, name: str, body: bytes, **fields: str):  # noqa: ANN202
        return c.post("/api/uploads", data={"kind": kind, **fields}, files={"file": (name, body)})

    r = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes(), cell="GMX", region="chr1:0-1000000")
    j = r.json()
    return SimpleNamespace(r=r, j=j, up=up, ct=ct, data=data, root=root, c=c)


def test_a_loops_file_is_stored_and_checked(site: SimpleNamespace) -> None:
    r, j = site.r, site.j
    assert r.status_code == 201 and j["ok"] and j["summary"].startswith("14 loops read.") and "are in the region" in j["summary"], "a loops file is stored and checked: " + r.text


def test_the_summary_counts_loops_between_chromosomes_and_over_1_mb(site: SimpleNamespace) -> None:
    up = site.up
    mixed = "chr1\t100000\t101000\tchr1\t300000\t301000\t5\nchr1\t1000000\t1001000\tchr2\t300000\t301000\t3\nchr1\t100000\t101000\tchr1\t1500000\t1501000\t4\n"
    j = up("loops", "mixed.bedpe", mixed.encode()).json()
    assert j["summary"] == "3 loops read. 1 loop between chromosomes is skipped and 1 loop longer than 1 Mb goes to the coarse contact map.", "the summary counts loops between chromosomes and over 1 Mb: " + str(j["summary"])


def test_a_ten_column_loop_file_is_refused_with_the_design_s_message(site: SimpleNamespace) -> None:
    up, ct, data, root, c = site.up, site.ct, site.data, site.root, site.c
    old = "chr1\t100000\t101000\tchr1\t300000\t301000\tR\tL\tCTCF\t5\n"
    j = up("loops", "hichip_loops.txt", old.encode()).json()
    assert not j["ok"] and j["error"]["line"] == 1 and "This is the loop format of 3D-GNOME 3.0" in j["error"]["found"] + j["error"]["fix"], "a ten column loop file is refused with the design's message: " + str(j)

    crlf = "# chr1 s1 e1 chr2 s2 e2 count\r\nchr1\t100000\t101000\tchr1\t300000\t301000\t5\r\n\r\n"
    j = up("loops", "crlf.bedpe", crlf.encode()).json()
    assert j["ok"] and j["summary"].startswith("1 loop read."), "Windows line ends and a header line are read: " + str(j)

    j = up("loops", "bad.bedpe", b"chr1\t100000\t101000\tchr1\t300000\t301000\t5\nchr1\t1\t2\tchr1\t3\t4\t2.5\n").json()
    assert not j["ok"] and j["error"]["line"] == 2 and "whole number" in j["error"]["fix"], "a count that is not a whole number names its line: " + str(j)

    j = up("anchors", "a.bed", b"chr1\t1\t2\tR\nchr1\t5\t6\nchr1\t9\t10\tL\n").json()
    assert not j["ok"] and j["error"]["line"] == 2 and "every line or on none" in j["error"]["fix"], "anchors with orientation on some lines only name the line: " + str(j)
    j = up("anchors", "a.bed", (ct / "anchors_ctcf.bed").read_bytes()).json()
    assert j["ok"] and j["summary"] == "15 anchors read, with orientations.", "anchors with orientations read: " + str(j["summary"])

    j = up("contacts", "c.bedpe", (ct / "singletons.bedpe").read_bytes(), cell="GMX", region="chr1:0-1000000").json()
    assert j["ok"] and j["summary"].startswith("39 contact pairs read.") and "in the region" in j["summary"], "contacts count their pairs and the pairs in the region: " + str(j["summary"])
    j = up("enhancers", "e.bed", (ct / "enhancers.bed").read_bytes()).json()
    assert j["ok"] and j["summary"] == "2 enhancers read.", "enhancers count: " + str(j["summary"])

    j = up("vcf", "demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes()).json()
    assert j["ok"] and j["samples"] == ["P1", "P2"], "a VCF lists its samples: " + str(j)
    j = up("vcf", "plain.vcf", b"##fileformat=VCFv4.2\n#CHROM\tPOS\n").json()
    assert not j["ok"] and "bgzip" in j["error"]["fix"], "a VCF that is not compressed is refused with the fix: " + str(j)

    big = b"chr1\t1\t2\tchr1\t3\t4\t5\n" * 60000
    r = up("loops", "big.bedpe", big)
    uploads = root / "results" / "uploads"
    assert r.status_code == 413 and "1 MB" in r.json()["error"], "a file over the cap is a 413 with its size: " + r.text[:200]
    assert not any(p.name == "big.bedpe" for p in uploads.rglob("*")), "nothing of the refused file stays"

    lid = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes()).json()["id"]
    prev = c.get(f"/api/uploads/{lid}/preview", params={"region": "chr1:0-1000000"}).json()
    assert len(prev) >= 1 and all(len(x) == 3 for x in prev), "the preview lists the loops in the region: " + str(prev[:2])
    assert c.get("/api/uploads/nothing12345/preview", params={"region": "chr1:0-1000"}).status_code == 404, "an unknown upload is a 404"

    loops_a = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes()).json()["id"]
    loops_b = up("loops", "rnapii.bedpe", (ct / "rnapii.bedpe").read_bytes()).json()["id"]
    contacts = up("contacts", "c.bedpe", (ct / "singletons.bedpe").read_bytes()).json()["id"]
    enh = up("enhancers", "e.bed", b"chr1\t1300000\t1302000\n").json()["id"]
    own = {"loops": [{"factor": "CTCF", "upload": loops_a}, {"factor": "RNAPII", "upload": loops_b}], "contacts": contacts, "enhancers": enh}
    form = {"cell_type": "own", "region": "chr1:0-3000000", "n_models": "2", "own": json.dumps(own)}
    r = c.post("/api/jobs", data=form)
    url = r.json().get("url", "")
    assert r.status_code == 201 and url.startswith("/p/"), "an own data job is private: " + r.text
    s = c.get(f"/api{url}/summary").json()
    assert s["state"] == "done" and s["loops"] == {"CTCF": 14, "RNAPII": 1}, "an own data job runs to done: " + str(s)[:300]
    assert s["request"]["own"] == {"loops": ["CTCF", "RNAPII"], "anchors": False, "contacts": True, "enhancers": True}, "the summary names the own files without server paths: " + str(s["request"].get("own"))
    marks = json.loads((root / "results" / "private" / url.split("/")[-1] / "marks.json").read_text())
    enh_marks = [e for e in marks["elements"] if e["kind"] == "enhancer"]
    assert [(e["start"], e["end"]) for e in enh_marks] == [(1300000, 1302000)], "own enhancers replace the cell type's: " + str(enh_marks)
    def enhancer_spans(u: str) -> list[tuple[int, int]]:
        m = json.loads((root / "results" / "private" / u.split("/")[-1] / "marks.json").read_text())
        return [(e["start"], e["end"]) for e in m["elements"] if e["kind"] == "enhancer"]

    borrow = {"cell_type": "own", "region": "chr1:0-3000000", "n_models": "2", "factors": "CTCF"}
    r = c.post("/api/jobs", data=borrow | {"own": json.dumps({"contacts": contacts, "loops_from": "GMX", "enhancers_from": "GMX"})})
    u = r.json().get("url", "")
    s = c.get(f"/api{u}/summary").json() if u else {}
    assert r.status_code == 201 and s.get("state") == "done" and s.get("loops") == {"CTCF": 14}, "your own cell type takes a listed cell type's loops: " + (r.text + str(s)[:300])
    assert u and enhancer_spans(u) == [(1490000, 1492000), (2700000, 2702000)], "and its enhancers when asked: " + str(u and enhancer_spans(u))
    assert s.get("request", {}).get("own", {}).get("loops_from") == "GMX" and s["request"]["own"].get("enhancers_from") == "GMX", "the summary names where loops and enhancers come from: " + str(s.get("request"))
    r = c.post("/api/jobs", data=borrow | {"own": json.dumps({"contacts": contacts, "loops_from": "GMX", "enhancers_from": "none"})})
    u = r.json().get("url", "")
    assert r.status_code == 201 and enhancer_spans(u) == [], "enhancers from none marks none: " + r.text
    for key in ("loops_from", "enhancers_from"):
        r = c.post("/api/jobs", data=borrow | {"own": json.dumps({"contacts": contacts, "loops_from": "GMX", key: "NOSUCH"})})
        assert r.status_code == 400 and r.json()["field"] == "own" and "NOSUCH" in r.json()["error"], f"an unknown cell type in {key} is a 400 on own" + ": " + r.text
    r = c.post("/api/jobs", data=borrow | {"own": json.dumps({"contacts": contacts})})
    assert r.status_code == 400 and r.json()["field"] == "loops", "your own cell type with neither loops nor a listed set is a 400 on loops: " + r.text

    r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": own["loops"]})})
    assert r.status_code == 400 and r.json()["field"] == "contacts", "an own data job without contacts on cell type own is a 400 on contacts: " + r.text
    r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": [{"factor": "CTCF", "upload": "nothing12345"}], "contacts": contacts})})
    assert r.status_code == 400 and r.json()["field"] == "own", "an unknown upload is a 400 on own: " + r.text
    bad = up("loops", "old.txt", old.encode()).json()["id"]
    r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": [{"factor": "CTCF", "upload": bad}], "contacts": contacts})})
    assert r.status_code == 400 and "check" in r.json()["error"], "an upload that failed its check is a 400: " + r.text
    r = c.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-3000000", "n_models": "2", "own": json.dumps({"loops": [{"factor": "CTCF", "upload": loops_a}]})})
    assert r.status_code == 201 and c.get(f"/api{r.json()['url']}/summary").json()["state"] == "done", "a hosted cell type with own loops keeps its contacts and runs: " + r.text

    v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "sv_set": "DEMO"})
    j = v.json()
    assert v.status_code == 200 and j["applied"] == {"DEL": 1} and j["skipped"] == {"CNV": 1}, "the variant check counts what applies and what is skipped: " + v.text[:300]

    dele = next(x for x in j["variants"] if x["type"] == "DEL")
    assert dele["change"] == "removes 1 CTCF anchor and its 2 loops" and dele["genotype"] == "1|0", "a deletion's change names the anchor and loops it removes: " + str(dele)
    assert j["summary"].startswith("1 variant of P1 on the first haplotype lies inside chr1:0-3,000,000."), "the summary sentence names the sample and haplotype: " + str(j["summary"])

    v1 = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "1", "sv_set": "DEMO"}).json()
    assert v1["applied"] == {"INS": 1}, "the second haplotype carries the insertion: " + str(v1)[:200]

    vid = up("vcf", "demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes()).json()["id"]
    v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "upload": vid})
    assert v.status_code == 200 and v.json()["applied"] == {"DEL": 1}, "an uploaded VCF is checked the same way: " + v.text[:200]

    job = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2", "sample": "P1", "haplotype": "0"}
    r = c.post("/api/jobs", data=job | {"vcf_upload": vid})
    url = r.json().get("url", "")
    s = c.get(f"/api{url}/summary").json() if url else {}
    assert r.status_code == 201 and url.startswith("/p/") and s.get("state") == "done" and s.get("n_svs") == 1, "a job takes its VCF by upload id and is private: " + (r.text + str(s)[:200])
    for bad_id in ("nosuchupload1", loops_a):
        r = c.post("/api/jobs", data=job | {"vcf_upload": bad_id})
        assert r.status_code == 400 and r.json()["field"] == "vcf_upload", f"a VCF upload id that is unknown or of another kind is a 400 ({bad_id[:6]})" + ": " + r.text

    notvcf = up("vcf", "plain.vcf", b"#CHROM\tPOS\n").json()["id"]
    r = c.post("/api/jobs", data=job | {"vcf_upload": notvcf})
    assert r.status_code == 400 and r.json()["field"] == "vcf_upload", "a VCF upload that failed its check is a 400: " + r.text

    v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "upload": loops_a})
    assert v.status_code == 400 and v.json()["field"] == "upload", "a variant check on an upload that is not a VCF is a 400 naming it: " + v.text
    r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": ["not an object"], "contacts": contacts})})
    assert r.status_code == 400 and r.json()["field"] == "own", "an own loops entry that is not an object is a 400 on own: " + r.text
    v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "NOBODY", "haplotype": "0", "upload": vid})
    assert v.status_code == 400 and v.json()["field"] == "sample", "a sample not in the VCF is a 400 on the sample field: " + v.text

    mine = up("loops", "mine.bedpe", b"chr1\t900000\t901000\tchr1\t1300000\t1301000\t5\n").json()["id"]
    own = json.dumps({"loops": [{"factor": "H3K27ac", "upload": mine}]})
    v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "sv_set": "DEMO", "own": own})
    dele = next((x for x in v.json().get("variants", []) if x["type"] == "DEL"), {})
    assert "H3K27ac" in dele.get("change", "") and "CTCF" not in dele.get("change", ""), "with own loops the check describes the change to those loops: " + v.text[:300]

    from gnome3d_web.variants.effect import combine_changes

    assert combine_changes([("CTCF", "lengthens 9 loops"), ("RNAPII", "lengthens 2 loops")]) == "lengthens 11 loops, 9 CTCF and 2 RNAPII", "one verb over two factors is summed with the split"
    assert combine_changes([("CTCF", "removes 1 CTCF anchor and its 2 loops"), ("RNAPII", "shortens 1 loop")]) == "removes 1 CTCF anchor and its 2 loops; shortens 1 loop", "different changes are listed by factor"
    assert combine_changes([("CTCF", "shortens 3 loops"), ("RNAPII", "no anchor inside")]) == "shortens 3 loops", "no change in one factor is left out"
    assert combine_changes([("CTCF", "no anchor inside"), ("RNAPII", "no anchor inside")]) == "no anchor inside", "no change in every factor is said once"

    from gnome3d_web.jobs.workers import sweep_private

    stale = uploads / lid
    old_t = time.time() - 30 * 86400
    os.utime(stale, (old_t, old_t))
    gone = sweep_private(root / "results", 14)
    assert lid in gone and not stale.exists(), "uploads older than the retention period are swept: " + str(gone)
    meta = json.loads((next(p for p in uploads.iterdir()) / "meta.json").read_text())
    assert {"kind", "name", "size"} <= set(meta), "an upload keeps its kind and name: " + str(meta)
