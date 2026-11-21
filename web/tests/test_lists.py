"""Job lists shared by a sync code: the code, the merge and the endpoints.

    pytest web/tests/test_lists.py
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data


def job(key: str, at: float, name: str = "", edited: float | None = None, kind: str = "jobs") -> dict[str, object]:
    out: dict[str, object] = {"url": f"/{kind}/{key}", "label": f"GMX {key}", "name": name, "at": at}
    if edited is not None:
        out["edited"] = edited
    return out


@pytest.fixture(scope="module", autouse=True)
def shared() -> SimpleNamespace:
    from gnome3d_web.lists import store as lists

    code = lists.new_code()
    return SimpleNamespace(code=code, lists=lists)


def test_a_code_is_four_groups_of_four_from_the_alphabet(shared: SimpleNamespace) -> None:
    code, lists = shared.code, shared.lists
    assert bool(__import__("re").fullmatch(r"([0-9A-HJKMNP-TV-Z]{4}-){3}[0-9A-HJKMNP-TV-Z]{4}", code)), "a code is four groups of four from the alphabet: " + str(code)
    assert lists.new_code() != code, "two codes differ"
    assert lists.normalise(" k7m2 9qxd4thp-w3rz ") == "K7M2-9QXD-4THP-W3RZ", "a typed code ignores case, spaces and dashes"
    assert lists.normalise("OIL0-1111-2222-3333") == "0110-1111-2222-3333", "O, I and L read as the digits they resemble"
    assert lists.normalise("K7M2-9QXD") is None, "a short code is no code"
    assert lists.normalise("U7M2-9QXD-4TLP-W3RZ") is None, "a code with a letter outside the alphabet is no code"


def test_clean_keeps_job_routes_with_a_time_newest_first_and_no_private_job(shared: SimpleNamespace) -> None:
    lists = shared.lists
    rows = lists.clean([job("a", 1), job("b", 3), job("x", 2, kind="p"), {"url": "/api/catalog", "at": 9}, {"url": "/jobs/c"}, "text", job("d", True)], private=False)  # type: ignore[arg-type]
    assert [r["url"] for r in rows] == ["/jobs/b", "/jobs/a"], "clean keeps job routes with a time, newest first and no private job: " + str(rows)
    rows = lists.clean([job("a", 1), job("x", 2, kind="p")], private=True)
    assert [r["url"] for r in rows] == ["/p/x", "/jobs/a"], "a private list keeps private jobs"
    assert len(lists.clean([{"url": "/jobs/a", "at": 1, "name": "n" * 500}], False)[0]["name"]) == lists.MAX_TEXT, "clean cuts long text"
    assert [r["at"] for r in lists.clean([job(str(i), i) for i in range(60)], False)][:2] == [59, 58] and len(lists.clean([job(str(i), i) for i in range(60)], False)) == lists.MAX_JOBS, "clean keeps at most MAX_JOBS, the newest"


def test_merge_is_the_union_newest_first(shared: SimpleNamespace) -> None:
    lists = shared.lists
    merged = lists.merge([job("a", 1, "old", edited=5), job("b", 2)], [job("a", 1, "new", edited=9), job("c", 3)])
    assert [r["url"] for r in merged] == ["/jobs/c", "/jobs/b", "/jobs/a"], "merge is the union, newest first"
    assert merged[2]["name"] == "new", "merge keeps the name edited last"
    assert lists.merge([job("a", 1, "new", edited=9)], [job("a", 1, "old", edited=5)])[0]["name"] == "new", "merge keeps the later edit from either side"


def test_a_new_list_answers_its_code_and_its_jobs_without_the_private_one(shared: SimpleNamespace, tmp_path: Path) -> None:
    lists = shared.lists
    d = str(tmp_path)
    root = Path(d)
    make_data(root / "data")
    os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"))
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    r = c.post("/api/lists", json={"jobs": [job("a", 1), job("x", 2, kind="p")], "private": False})
    made = r.json()
    assert r.status_code == 201 and lists.normalise(made["code"]) == made["code"] and [j["url"] for j in made["jobs"]] == ["/jobs/a"] and made["private"] is False, "a new list answers its code and its jobs without the private one: " + r.text[:200]

    code = made["code"]
    r = c.get(f"/api/lists/{code.lower().replace('-', ' ')}")
    assert r.status_code == 200 and r.json()["code"] == code and len(r.json()["jobs"]) == 1, "the list is read by its code as typed: " + r.text[:120]

    r = c.put(f"/api/lists/{code}", json={"jobs": [job("b", 5), job("y", 6, kind="p"), job("a", 1, "renamed", edited=7)]})
    assert r.status_code == 200 and [(j["url"], j["name"]) for j in r.json()["jobs"]] == [("/jobs/b", ""), ("/jobs/a", "renamed")], "a second browser's jobs merge in: " + r.text[:200]
    assert [j["url"] for j in c.get(f"/api/lists/{code}").json()["jobs"]] == ["/jobs/b", "/jobs/a"], "the merged list is what the first browser reads"

    r = c.put(f"/api/lists/{code}", json={"jobs": []})
    assert len(r.json()["jobs"]) == 2, "an empty write removes nothing"

    r = c.get("/api/lists/0000-0000-0000-0000")
    assert r.status_code == 404 and "No list has this code" in r.json()["error"], "an unknown code is a 404 with its reason"

    r = c.get("/api/lists/short")
    assert r.status_code == 400 and "16 letters and digits" in r.json()["error"], "a malformed code is a 400 that shows the form of a code"
    assert c.put("/api/lists/0000-0000-0000-0000", json={"jobs": [job("a", 1)]}).status_code == 404 and c.get("/api/lists/0000-0000-0000-0000").status_code == 404, "a write to an unknown code makes no list"

    r = c.post("/api/lists", json={"jobs": [job("x", 2, kind="p")], "private": True})
    assert r.json()["private"] is True and r.json()["jobs"][0]["url"] == "/p/x", "a list made with private jobs holds them"
    assert c.post("/api/lists", content=b"[" + b"1," * 60_000 + b"1]").status_code == 413, "a body that is too large is refused"
    assert c.post("/api/lists", content=b"{not json").status_code == 400, "a body that is not JSON is refused"

    names = [p.name for p in (root / "results" / "lists").iterdir()]
    assert len(names) == 2 and all(code.replace("-", "") not in n and code not in n for n in names), "the files are not named by the codes: " + str(names)
    assert code not in (root / "results" / "lists" / names[0]).read_text(), "a list's file holds no code"
