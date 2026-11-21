"""Gene expression beside a job: the tables, the level of a gene and the contrast of two jobs.

    pytest web/tests/test_expression.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixture import make_data

from gnome3d_web import expression
from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, normalised

HAS_DESEQ = importlib.util.find_spec("pydeseq2") is not None


def counts_table(rng: np.random.Generator, columns: dict[str, float], up: dict[str, float]) -> pd.DataFrame:
    """Counts of 400 genes named by name. ALPHA and BETA are the genes of the test region. `up`
    multiplies the mean of a gene in the columns whose scale is not one."""
    names = ["ALPHA", "BETA"] + [f"GENE{i}" for i in range(398)]
    mean = rng.uniform(20, 2000, len(names))
    data = {}
    for col, scale in columns.items():
        m = mean * np.array([up.get(n, 1.0) if scale != 1.0 else 1.0 for n in names])
        data[col] = rng.negative_binomial(20, 20 / (20 + m))
    out = pd.DataFrame(data, index=pd.Index(names, name="gene"))
    out.insert(0, "length", 1000 + np.arange(len(names)) * 10)
    return out


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = Path(str(tmp_path_factory.mktemp("work")))
    data = make_data(root / "data")
    rng = np.random.default_rng(5)
    (data / "expression").mkdir()
    # Two tables of counts for the cell type, the second with ALPHA at eight times and BETA at an
    # eighth, one of counts for the people of the call set and one of normalised values.
    counts_table(rng, {"a1": 1.0, "a2": 1.0}, {}).to_csv(data / "expression" / "lines.tsv", sep="\t")
    counts_table(np.random.default_rng(5), {"b1": 2.0, "b2": 2.0}, {"ALPHA": 8.0, "BETA": 0.125}).to_csv(data / "expression" / "other.tsv", sep="\t")
    counts_table(rng, {"P1": 1.0, "P2": 1.0}, {}).drop(columns="length").to_csv(data / "expression" / "people.tsv", sep="\t")
    (counts_table(rng, {"P1": 1.0}, {}).drop(columns="length") / 7.3).to_csv(data / "expression" / "tmm.tsv", sep="\t")
    catalog = json.loads((data / "catalog.json").read_text())
    catalog["cell_types"]["FBX"] = {**catalog["cell_types"]["GMX"], "label": "FBX, fibroblast", "lymphoblastoid": False}
    catalog["expression"] = {
        "LINES": {"label": "Lines RNA-seq", "file": "expression/lines.tsv", "kind": "counts", "cell_types": {"GMX": ["a1", "a2"]}},
        "OTHER": {"label": "Other RNA-seq", "file": "expression/other.tsv", "kind": "counts", "cell_types": {"GMX": ["b1", "b2"], "FBX": ["b1", "b2"]}},
        "PEOPLE": {"label": "People mRNA-seq", "file": "expression/people.tsv", "kind": "counts", "people": True, "lymphoblastoid": True},
        "TMM": {"label": "Panel", "file": "expression/tmm.tsv", "kind": "values", "unit": "TMM", "people": True, "lymphoblastoid": True},
    }
    (data / "catalog.json").write_text(json.dumps(catalog))
    os.environ.update(
        DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}

    def job(**more: str) -> str:
        r = c.post("/api/jobs", data={**form, **more})
        assert r.status_code == 201, "the job is accepted: " + r.text
        return r.json()["url"]

    return SimpleNamespace(c=c, data=data, root=root, form=form, job=job)


def test_a_table_of_featurecounts_is_read_as_it_is(tmp_path: Path) -> None:
    p = tmp_path / "counts.txt"
    p.write_text(
        '# Program:featureCounts v2.1.1; Command:"featureCounts" "-a" "x.gtf"\n'
        "Geneid\tChr\tStart\tEnd\tStrand\tLength\trep1.bam\trep2.bam\n"
        "ENSG00000223972.5\tchr1;chr1\t11869;12010\t12227;12057\t+;+\t1735\t0\t3\n"
        "ENSG00000227232.5\tchr1\t14404\t29570\t-\t1351\t12\t20\n"
    )
    t = expression.read_table(p)
    assert t.key == "gene_id" and list(t.data.index) == ["ENSG00000223972", "ENSG00000227232"], "genes are Ensembl ids without their version: " + str(list(t.data.index))
    assert list(t.data.columns) == ["rep1.bam", "rep2.bam"], "the samples are the columns after the place of the gene: " + str(list(t.data.columns))
    assert t.length is not None and list(t.length) == [1735, 1351], "the length of each gene is kept"


def test_a_plain_table_names_its_genes_in_the_first_column(tmp_path: Path) -> None:
    p = tmp_path / "counts.csv"
    p.write_text("gene,s1,s2,note\nALPHA,5,7,x\nBETA,0,2,y\nALPHA,9,9,z\n")
    t = expression.read_table(p)
    assert t.key == "name" and list(t.data.index) == ["ALPHA", "BETA"], "genes are names and a gene listed twice is read once: " + str(list(t.data.index))
    assert list(t.data.columns) == ["s1", "s2"] and t.length is None, "a column that holds no numbers is no sample"
    (tmp_path / "bad.tsv").write_text("gene\tnote\nALPHA\tx\n")
    with pytest.raises(expression.ExpressionError, match="no column of numbers"):
        expression.read_table(tmp_path / "bad.tsv")


def test_counts_must_be_whole_numbers(tmp_path: Path) -> None:
    p = tmp_path / "values.tsv"
    p.write_text("gene\ts1\nALPHA\t5.5\nBETA\t2\n")
    t = expression.read_table(p)
    with pytest.raises(expression.ExpressionError, match="not whole numbers"):
        expression.check_counts(t, ["s1"], "values.tsv")


def test_the_level_of_a_gene_is_tpm_with_lengths_and_cpm_without() -> None:
    data = pd.DataFrame({"s1": [100, 300, 600], "s2": [200, 600, 1200]}, index=pd.Index(["A", "B", "C"], name="name"))
    tpm, unit = expression.level(expression.Table("name", data, pd.Series([1000, 3000, 3000], index=data.index)), "counts")
    assert unit == "TPM" and abs(tpm.sum() - 1e6) < 1e-6, "TPM sums to a million"
    assert abs(tpm["A"] - tpm["B"]) < 1e-6 and abs(tpm["C"] - 2 * tpm["A"]) < 1e-6, "TPM is the count over the length: " + str(tpm.round(1).to_dict())
    cpm, unit = expression.level(expression.Table("name", data), "counts")
    assert unit == "CPM" and abs(cpm["A"] - 1e5) < 1e-6, "without lengths the level is counts per million: " + str(cpm.to_dict())
    values, unit = expression.level(expression.Table("name", data / 10), "values")
    assert unit == "" and values["A"] == 15.0, "normalised values are averaged over the samples"


def test_the_size_factor_of_a_sample_follows_its_depth() -> None:
    rng = np.random.default_rng(1)
    base = pd.Series(rng.integers(50, 500, 200))
    f = expression.size_factors(pd.DataFrame({"x": base, "y": base * 3}))
    assert abs(f["y"] / f["x"] - 3) < 1e-9 and abs(f["x"] * f["y"] - 1) < 1e-9, "a sample three times as deep has three times the factor: " + str(f.to_dict())


def test_a_cell_type_is_offered_its_replicates_and_a_sample_its_libraries(site: SimpleNamespace) -> None:
    cat = Catalog.load(site.data)
    alone = cat.expression_options("GMX")
    assert [(o["id"], o["label"]) for o in alone] == [("LINES", "Lines RNA-seq of GMX, 2 replicates"), ("OTHER", "Other RNA-seq of GMX, 2 replicates")], "a cell type has its replicates: " + str(alone)
    person = {o["id"]: o for o in cat.expression_options("GMX", "P1")}
    assert person["PEOPLE"]["label"] == "People mRNA-seq of P1, 1 library" and person["PEOPLE"]["columns"] == ["P1"], "a hosted sample has its own library: " + str(person["PEOPLE"])
    assert person["TMM"]["kind"] == "values" and person["TMM"]["unit"] == "TMM", "a table of normalised values states its unit"
    assert not person["PEOPLE"]["other_cell_type"], "a lymphoblastoid line beside a lymphoblastoid cell type needs no warning"
    other = {o["id"]: o for o in cat.expression_options("FBX", "P1")}
    assert other["PEOPLE"]["other_cell_type"] and not other["OTHER"]["other_cell_type"], "a person's lymphoblastoid line beside another cell type is marked: " + str(other)
    assert "TMM" not in {o["id"] for o in cat.expression_options("GMX", "P2")}, "a sample a table lacks is not offered that table"
    r = site.c.get("/api/expression", params={"cell": "GMX", "sv_set": "DEMO", "sample": "P1"})
    assert r.status_code == 200 and {o["id"] for o in r.json()["options"]} == {"LINES", "OTHER", "PEOPLE", "TMM"}, "the API lists the same options: " + r.text[:200]
    assert {o["id"] for o in site.c.get("/api/expression", params={"cell": "GMX", "sv_set": "DEMO", "sample": "nobody"}).json()["options"]} == {"LINES", "OTHER"}, "a sample the call set lacks gets the cell type alone"


def test_a_request_without_expression_keeps_its_key(site: SimpleNamespace) -> None:
    cat = Catalog.load(site.data)
    plain = normalised(JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF"), cat)
    assert "expression" not in plain, "a request without expression has the fields it had before: " + str(sorted(plain))
    assert "expression" in asdict(JobRequest(cell_type="GMX", region="chr1:0-3000000", factors="CTCF", expression={"source": "LINES"})), "a request with expression carries it"


def test_a_job_with_expression_shows_it_in_its_summary_its_genes_and_its_files(site: SimpleNamespace) -> None:
    c = site.c
    url = site.job(expression="LINES")
    s = c.get(f"/api{url}/summary").json()
    assert s["expression"] == {"label": "Lines RNA-seq of GMX, 2 replicates", "kind": "counts", "unit": "TPM", "samples": 2, "source": "LINES"}, "the summary states the expression: " + str(s["expression"])
    assert s["request"]["expression"] == "LINES" and "path" not in json.dumps(s["request"]), "the request names the source and no path of the server"
    rows = {r["gene"]: r for r in c.get(f"/api{url}/genes").json()}
    table = pd.read_csv(site.data / "expression" / "lines.tsv", sep="\t", index_col=0)
    rate = table[["a1", "a2"]].div(table["length"], axis=0)
    want = float((rate / rate.sum() * 1e6).mean(axis=1)["ALPHA"])
    assert rows and all(r["expression"] is not None for r in rows.values()), "every gene of the table has its level: " + str({g: r["expression"] for g, r in rows.items()})
    assert abs(rows["ALPHA"]["expression"] - want) < 0.01, f"the level of a gene is its TPM over the replicates: {rows['ALPHA']['expression']} against {want}"
    kept = c.get(url + "/files/expression.tsv")
    assert kept.status_code == 200 and kept.text.splitlines()[0].split("\t") == ["name", "length", "a1", "a2"], "the job keeps the samples it used, with the genes under the key of the table: " + kept.text[:60]
    bare = site.job()
    assert c.get(f"/api{bare}/summary").json()["expression"] is None, "a job without expression states none"
    assert all(r["expression"] is None for r in c.get(f"/api{bare}/genes").json()), "its gene table has no level"
    r = c.post("/api/jobs", data={**site.form, "expression": "NOPE"})
    assert r.status_code == 400 and r.json()["field"] == "expression", "an unknown source is refused at its field: " + r.text


def test_two_jobs_with_replicates_are_compared_with_deseq2(site: SimpleNamespace) -> None:
    c = site.c
    a, b = site.job(expression="LINES"), site.job(expression="OTHER")
    r = c.get("/api/compare/expression", params={"a": a, "b": b})
    out = r.json()
    assert r.status_code == 200 and out["label_a"].startswith("Lines") and out["label_b"].startswith("Other"), "the answer names both sides: " + r.text[:200]
    rows = {x["gene"]: x for x in out["rows"]}
    assert set(rows) == {"ALPHA", "BETA"}, "the rows are the genes both jobs have: " + str(sorted(rows))
    assert 2.5 < rows["ALPHA"]["log2fc"] < 3.5 and -3.5 < rows["BETA"]["log2fc"] < -2.5, "a gene at eight times reads a log2 fold change near 3 at any depth: " + str({g: x["log2fc"] for g, x in rows.items()})
    if HAS_DESEQ:
        assert out["mode"] == "deseq2" and out["reason"] == "", "two replicates on each side are tested with DESeq2: " + str(out["mode"])
        assert rows["ALPHA"]["padj"] is not None and rows["ALPHA"]["padj"] < 0.01, "the changed gene is significant: " + str(rows["ALPHA"]["padj"])
    else:
        assert out["mode"] == "ratio" and "PyDESeq2 is not installed" in out["reason"], "without PyDESeq2 the answer is a ratio and says why"
    assert out["rho"] is None, "two genes give no correlation"
    again = c.get("/api/compare/expression", params={"a": a, "b": b}).json()
    assert again["rows"] == out["rows"] and list((site.root / "results" / "expression").glob("*.json")), "the contrast of a pair of tables is computed once and kept"


def test_one_library_a_side_gives_a_ratio_and_no_test(site: SimpleNamespace) -> None:
    c = site.c
    a = site.job(expression="LINES")
    b = site.job(expression="PEOPLE", sv_set="DEMO", sample="P1")
    out = c.get("/api/compare/expression", params={"a": a, "b": b}).json()
    assert out["mode"] == "ratio" and "two samples or more on each side" in out["reason"], "one library is compared by a ratio: " + str(out.get("reason"))
    assert out["rows"] and all(x["padj"] is None for x in out["rows"]), "a ratio has no p-value"
    values = site.job(expression="TMM", sv_set="DEMO", sample="P1")
    mixed = c.get("/api/compare/expression", params={"a": a, "b": values}).json()
    assert mixed["mode"] == "none" and "read counts and the other normalised values" in mixed["reason"] and mixed["rows"] == [], "counts and normalised values are not compared: " + str(mixed)
    same = c.get("/api/compare/expression", params={"a": a, "b": a}).json()
    assert same["mode"] == "none" and "same samples" in same["reason"], "a job against itself has no difference: " + str(same)
    none = c.get("/api/compare/expression", params={"a": a, "b": site.job()}).json()
    assert none["mode"] == "none" and "no expression data" in none["reason"], "a job without expression is said to have none"


def upload_table(site: SimpleNamespace, table: pd.DataFrame, name: str) -> dict:
    r = site.c.post("/api/uploads", data={"kind": "expression"}, files={"file": (name, table.to_csv(sep="\t").encode())})
    assert r.status_code == 201, r.text
    return r.json()


def test_an_uploaded_table_is_checked_and_gives_a_private_job_with_the_chosen_samples(site: SimpleNamespace) -> None:
    table = counts_table(np.random.default_rng(9), {"s1": 1.0, "s2": 1.0, "s3": 1.0}, {})
    up = upload_table(site, table, "my counts.tsv")
    assert up["ok"], "a table of genes by samples passes its check"
    assert up["samples"] == ["s1", "s2", "s3"], "the check lists the samples and leaves the length column out"
    assert up["counts"], "a table of whole numbers is read as counts"
    assert "400 genes by name and 3 samples" in up["summary"], "the summary counts the genes and the samples"
    url = site.job(expression="own", expression_upload=up["id"], expression_columns=json.dumps(["s1", "s3"]), expression_kind="counts")
    assert url.startswith("/p/"), "a job with an uploaded table is private"
    summary = site.c.get(f"/api{url}/summary").json()
    assert summary["expression"]["samples"] == 2, "the job has the two chosen samples"
    assert summary["expression"]["unit"] == "TPM", "counts with gene lengths give TPM"
    assert summary["expression"]["label"] == "Your own table my_counts.tsv, 2 samples", "the label names the file and its samples"
    assert summary["request"]["expression"] == "upload", "the request states an upload and no path"
    genes = {g["gene"]: g for g in site.c.get(f"/api{url}/genes").json()}
    assert genes["ALPHA"]["expression"] > 0, "a gene of the region has a level from the uploaded table"


def test_normalised_values_are_refused_as_counts_and_a_request_without_a_table_is_refused(site: SimpleNamespace) -> None:
    table = counts_table(np.random.default_rng(9), {"s1": 1.0, "s2": 1.0}, {}).drop(columns="length") / 3.7
    up = upload_table(site, table, "tmm.tsv")
    assert up["ok"] and not up["counts"], "a table with fractions passes and is not read as counts"
    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2", "expression": "own"}
    r = site.c.post("/api/jobs", data={**form, "expression_upload": up["id"], "expression_kind": "counts"})
    assert r.status_code == 400 and r.json()["field"] == "expression", "fractions stated as counts are refused at the expression step"
    assert "not whole numbers" in r.json()["error"], "the refusal says what read counts are"
    r = site.c.post("/api/jobs", data={**form, "expression_upload": up["id"], "expression_kind": "values"})
    assert r.status_code == 201, "the same table is taken as normalised values"
    r = site.c.post("/api/jobs", data=form)
    assert r.status_code == 400 and "needs a table" in r.json()["error"], "your own expression without an upload is refused"
    bad = site.c.post("/api/uploads", data={"kind": "expression"}, files={"file": ("genes.txt", b"gene\nALPHA\nBETA\n")}).json()
    assert not bad["ok"] and "one column or more of samples" in bad["error"]["found"], "a table without samples fails its check with the reason"
