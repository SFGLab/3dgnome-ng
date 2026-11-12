"""Uploads of own data: storage with a size cap, the check of each kind, the preview and the sweep.

    python web/tests/test_owndata.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from _check import check, done
from fixture import make_data


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = make_data(root / "data")
        os.environ.update(
            DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), UPLOAD_MAX_MB="1",
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )  # fmt: skip
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        ct = data / "cell_types" / "GMX"

        def up(kind: str, name: str, body: bytes, **fields: str):  # noqa: ANN202
            return c.post("/api/uploads", data={"kind": kind, **fields}, files={"file": (name, body)})

        r = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes(), cell="GMX", region="chr1:0-1000000")
        j = r.json()
        check("a loops file is stored and checked", r.status_code == 201 and j["ok"] and j["summary"].startswith("14 loops read.") and "are in the region" in j["summary"], r.text)

        mixed = "chr1\t100000\t101000\tchr1\t300000\t301000\t5\nchr1\t1000000\t1001000\tchr2\t300000\t301000\t3\nchr1\t100000\t101000\tchr1\t1500000\t1501000\t4\n"
        j = up("loops", "mixed.bedpe", mixed.encode()).json()
        check("the summary counts loops between chromosomes and over 1 Mb", j["summary"] == "3 loops read. 1 loop between chromosomes is skipped, and 1 loop longer than 1 Mb goes to the coarse contact map.", j["summary"])

        old = "chr1\t100000\t101000\tchr1\t300000\t301000\tR\tL\tCTCF\t5\n"
        j = up("loops", "hichip_loops.txt", old.encode()).json()
        check("a ten column loop file is refused with the design's message", not j["ok"] and j["error"]["line"] == 1 and "This is the loop format of 3D-GNOME 3.0" in j["error"]["found"] + j["error"]["fix"], str(j))

        crlf = "# chr1 s1 e1 chr2 s2 e2 count\r\nchr1\t100000\t101000\tchr1\t300000\t301000\t5\r\n\r\n"
        j = up("loops", "crlf.bedpe", crlf.encode()).json()
        check("Windows line ends and a header line are read", j["ok"] and j["summary"].startswith("1 loop read."), str(j))

        j = up("loops", "bad.bedpe", b"chr1\t100000\t101000\tchr1\t300000\t301000\t5\nchr1\t1\t2\tchr1\t3\t4\t2.5\n").json()
        check("a count that is not a whole number names its line", not j["ok"] and j["error"]["line"] == 2 and "whole number" in j["error"]["fix"], str(j))

        j = up("anchors", "a.bed", b"chr1\t1\t2\tR\nchr1\t5\t6\nchr1\t9\t10\tL\n").json()
        check("anchors with orientation on some lines only name the line", not j["ok"] and j["error"]["line"] == 2 and "every line or on none" in j["error"]["fix"], str(j))
        j = up("anchors", "a.bed", (ct / "anchors_ctcf.bed").read_bytes()).json()
        check("anchors with orientations read", j["ok"] and j["summary"] == "15 anchors read, with orientations.", j["summary"])

        j = up("contacts", "c.bedpe", (ct / "singletons.bedpe").read_bytes(), cell="GMX", region="chr1:0-1000000").json()
        check("contacts count their pairs and the pairs in the region", j["ok"] and j["summary"].startswith("39 contact pairs read.") and "in the region" in j["summary"], j["summary"])
        j = up("enhancers", "e.bed", (ct / "enhancers.bed").read_bytes()).json()
        check("enhancers count", j["ok"] and j["summary"] == "2 enhancers read.", j["summary"])

        j = up("vcf", "demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes()).json()
        check("a VCF lists its samples", j["ok"] and j["samples"] == ["P1", "P2"], str(j))
        j = up("vcf", "plain.vcf", b"##fileformat=VCFv4.2\n#CHROM\tPOS\n").json()
        check("a VCF that is not compressed is refused with the fix", not j["ok"] and "bgzip" in j["error"]["fix"], str(j))

        big = b"chr1\t1\t2\tchr1\t3\t4\t5\n" * 60000
        r = up("loops", "big.bedpe", big)
        uploads = root / "results" / "uploads"
        check("a file over the cap is a 413 with its size", r.status_code == 413 and "1 MB" in r.json()["error"], r.text[:200])
        check("nothing of the refused file stays", not any(p.name == "big.bedpe" for p in uploads.rglob("*")))

        lid = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes()).json()["id"]
        prev = c.get(f"/api/uploads/{lid}/preview", params={"region": "chr1:0-1000000"}).json()
        check("the preview lists the loops in the region", len(prev) >= 1 and all(len(x) == 3 for x in prev), str(prev[:2]))
        check("an unknown upload is a 404", c.get("/api/uploads/nothing12345/preview", params={"region": "chr1:0-1000"}).status_code == 404)

        loops_a = up("loops", "ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes()).json()["id"]
        loops_b = up("loops", "rnapii.bedpe", (ct / "rnapii.bedpe").read_bytes()).json()["id"]
        contacts = up("contacts", "c.bedpe", (ct / "singletons.bedpe").read_bytes()).json()["id"]
        enh = up("enhancers", "e.bed", b"chr1\t1300000\t1302000\n").json()["id"]
        own = {"loops": [{"factor": "CTCF", "upload": loops_a}, {"factor": "RNAPII", "upload": loops_b}], "contacts": contacts, "enhancers": enh}
        form = {"cell_type": "own", "region": "chr1:0-3000000", "n_models": "2", "own": json.dumps(own)}
        r = c.post("/api/jobs", data=form)
        url = r.json().get("url", "")
        check("an own data job is private", r.status_code == 201 and url.startswith("/p/"), r.text)
        s = c.get(f"/api{url}/summary").json()
        check("an own data job runs to done", s["state"] == "done" and s["loops"] == {"CTCF": 14, "RNAPII": 1}, str(s)[:300])
        check("the summary names the own files without server paths", s["request"]["own"] == {"loops": ["CTCF", "RNAPII"], "anchors": False, "contacts": True, "enhancers": True}, str(s["request"].get("own")))
        marks = json.loads((root / "results" / "private" / url.split("/")[-1] / "marks.json").read_text())
        enh_marks = [e for e in marks["elements"] if e["kind"] == "enhancer"]
        check("own enhancers replace the cell type's", [(e["start"], e["end"]) for e in enh_marks] == [(1300000, 1302000)], str(enh_marks))
        r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": own["loops"]})})
        check("an own data job without contacts on cell type own is a 400 on contacts", r.status_code == 400 and r.json()["field"] == "contacts", r.text)
        r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": [{"factor": "CTCF", "upload": "nothing12345"}], "contacts": contacts})})
        check("an unknown upload is a 400 on own", r.status_code == 400 and r.json()["field"] == "own", r.text)
        bad = up("loops", "old.txt", old.encode()).json()["id"]
        r = c.post("/api/jobs", data=form | {"own": json.dumps({"loops": [{"factor": "CTCF", "upload": bad}], "contacts": contacts})})
        check("an upload that failed its check is a 400", r.status_code == 400 and "check" in r.json()["error"], r.text)
        r = c.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-3000000", "n_models": "2", "own": json.dumps({"loops": [{"factor": "CTCF", "upload": loops_a}]})})
        check("a hosted cell type with own loops keeps its contacts and runs", r.status_code == 201 and c.get(f"/api{r.json()['url']}/summary").json()["state"] == "done", r.text)

        v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "sv_set": "DEMO"})
        j = v.json()
        check("the variant check counts what applies and what is skipped", v.status_code == 200 and j["applied"] == {"DEL": 1} and j["skipped"] == {"CNV": 1}, v.text[:300])
        dele = next(x for x in j["variants"] if x["type"] == "DEL")
        check("a deletion's change names the anchor and loops it removes", dele["change"] == "removes 1 CTCF anchor and its 2 loops" and dele["genotype"] == "1|0", str(dele))
        check("the summary sentence names the sample and haplotype", j["summary"].startswith("1 variant of P1 on the first haplotype lies inside chr1:0-3,000,000."), j["summary"])
        v1 = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "1", "sv_set": "DEMO"}).json()
        check("the second haplotype carries the insertion", v1["applied"] == {"INS": 1}, str(v1)[:200])
        vid = up("vcf", "demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes()).json()["id"]
        v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "P1", "haplotype": "0", "upload": vid})
        check("an uploaded VCF is checked the same way", v.status_code == 200 and v.json()["applied"] == {"DEL": 1}, v.text[:200])
        v = c.post("/api/variants/check", data={"cell": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "sample": "NOBODY", "haplotype": "0", "upload": vid})
        check("a sample not in the VCF is a 400 on the sample field", v.status_code == 400 and v.json()["field"] == "sample", v.text)

        from gnome3d_web.owndata import combine_changes

        check("one verb over two factors is summed with the split", combine_changes([("CTCF", "lengthens 9 loops"), ("RNAPII", "lengthens 2 loops")]) == "lengthens 11 loops, 9 CTCF and 2 RNAPII")
        check("different changes are listed by factor", combine_changes([("CTCF", "removes 1 CTCF anchor and its 2 loops"), ("RNAPII", "shortens 1 loop")]) == "removes 1 CTCF anchor and its 2 loops; shortens 1 loop")
        check("no change in one factor is left out", combine_changes([("CTCF", "shortens 3 loops"), ("RNAPII", "no anchor inside")]) == "shortens 3 loops")
        check("no change in every factor is said once", combine_changes([("CTCF", "no anchor inside"), ("RNAPII", "no anchor inside")]) == "no anchor inside")

        from gnome3d_web.queue import sweep_private

        stale = uploads / lid
        old_t = time.time() - 30 * 86400
        os.utime(stale, (old_t, old_t))
        gone = sweep_private(root / "results", 14)
        check("uploads older than the retention period are swept", lid in gone and not stale.exists(), str(gone))
        meta = json.loads((next(p for p in uploads.iterdir()) / "meta.json").read_text())
        check("an upload keeps its kind and name", {"kind", "name", "size"} <= set(meta), str(meta))
    return done()


if __name__ == "__main__":
    sys.exit(main())
