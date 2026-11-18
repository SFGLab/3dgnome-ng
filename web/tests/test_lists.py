"""Job lists shared by a sync code: the code, the merge and the endpoints.

    python web/tests/test_lists.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data


def job(key: str, at: float, name: str = "", edited: float | None = None, kind: str = "jobs") -> dict[str, object]:
    out: dict[str, object] = {"url": f"/{kind}/{key}", "label": f"GMX {key}", "name": name, "at": at}
    if edited is not None:
        out["edited"] = edited
    return out


def main() -> int:
    from gnome3d_web import lists

    code = lists.new_code()
    check("a code is four groups of four from the alphabet", bool(__import__("re").fullmatch(r"([0-9A-HJKMNP-TV-Z]{4}-){3}[0-9A-HJKMNP-TV-Z]{4}", code)), code)
    check("two codes differ", lists.new_code() != code)
    check("a typed code ignores case, spaces and dashes", lists.normalise(" k7m2 9qxd4thp-w3rz ") == "K7M2-9QXD-4THP-W3RZ")
    check("O, I and L read as the digits they resemble", lists.normalise("OIL0-1111-2222-3333") == "0110-1111-2222-3333")
    check("a short code is no code", lists.normalise("K7M2-9QXD") is None)
    check("a code with a letter outside the alphabet is no code", lists.normalise("U7M2-9QXD-4TLP-W3RZ") is None)

    rows = lists.clean([job("a", 1), job("b", 3), job("x", 2, kind="p"), {"url": "/api/catalog", "at": 9}, {"url": "/jobs/c"}, "text", job("d", True)], private=False)  # type: ignore[arg-type]
    check("clean keeps job routes with a time, newest first and no private job", [r["url"] for r in rows] == ["/jobs/b", "/jobs/a"], str(rows))
    rows = lists.clean([job("a", 1), job("x", 2, kind="p")], private=True)
    check("a private list keeps private jobs", [r["url"] for r in rows] == ["/p/x", "/jobs/a"])
    check("clean cuts long text", len(lists.clean([{"url": "/jobs/a", "at": 1, "name": "n" * 500}], False)[0]["name"]) == lists.MAX_TEXT)
    check("clean keeps at most MAX_JOBS, the newest", [r["at"] for r in lists.clean([job(str(i), i) for i in range(60)], False)][:2] == [59, 58] and len(lists.clean([job(str(i), i) for i in range(60)], False)) == lists.MAX_JOBS)

    merged = lists.merge([job("a", 1, "old", edited=5), job("b", 2)], [job("a", 1, "new", edited=9), job("c", 3)])
    check("merge is the union, newest first", [r["url"] for r in merged] == ["/jobs/c", "/jobs/b", "/jobs/a"])
    check("merge keeps the name edited last", merged[2]["name"] == "new")
    check("merge keeps the later edit from either side", lists.merge([job("a", 1, "new", edited=9)], [job("a", 1, "old", edited=5)])[0]["name"] == "new")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_data(root / "data")
        os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"))
        from fastapi.testclient import TestClient
        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        r = c.post("/api/lists", json={"jobs": [job("a", 1), job("x", 2, kind="p")], "private": False})
        made = r.json()
        check("a new list answers its code and its jobs without the private one", r.status_code == 201 and lists.normalise(made["code"]) == made["code"] and [j["url"] for j in made["jobs"]] == ["/jobs/a"] and made["private"] is False, r.text[:200])
        code = made["code"]
        r = c.get(f"/api/lists/{code.lower().replace('-', ' ')}")
        check("the list is read by its code as typed", r.status_code == 200 and r.json()["code"] == code and len(r.json()["jobs"]) == 1, r.text[:120])
        r = c.put(f"/api/lists/{code}", json={"jobs": [job("b", 5), job("y", 6, kind="p"), job("a", 1, "renamed", edited=7)]})
        check("a second browser's jobs merge in", r.status_code == 200 and [(j["url"], j["name"]) for j in r.json()["jobs"]] == [("/jobs/b", ""), ("/jobs/a", "renamed")], r.text[:200])
        check("the merged list is what the first browser reads", [j["url"] for j in c.get(f"/api/lists/{code}").json()["jobs"]] == ["/jobs/b", "/jobs/a"])
        r = c.put(f"/api/lists/{code}", json={"jobs": []})
        check("an empty write removes nothing", len(r.json()["jobs"]) == 2)
        r = c.get("/api/lists/0000-0000-0000-0000")
        check("an unknown code is a 404 with its reason", r.status_code == 404 and "No list has this code" in r.json()["error"])
        r = c.get("/api/lists/short")
        check("a malformed code is a 400 that shows the form of a code", r.status_code == 400 and "16 letters and digits" in r.json()["error"])
        check("a write to an unknown code makes no list", c.put("/api/lists/0000-0000-0000-0000", json={"jobs": [job("a", 1)]}).status_code == 404 and c.get("/api/lists/0000-0000-0000-0000").status_code == 404)
        r = c.post("/api/lists", json={"jobs": [job("x", 2, kind="p")], "private": True})
        check("a list made with private jobs holds them", r.json()["private"] is True and r.json()["jobs"][0]["url"] == "/p/x")
        check("a body that is too large is refused", c.post("/api/lists", content=b"[" + b"1," * 60_000 + b"1]").status_code == 413)
        check("a body that is not JSON is refused", c.post("/api/lists", content=b"{not json").status_code == 400)
        names = [p.name for p in (root / "results" / "lists").iterdir()]
        check("the files are not named by the codes", len(names) == 2 and all(code.replace("-", "") not in n and code not in n for n in names), str(names))
        check("a list's file holds no code", code not in (root / "results" / "lists" / names[0]).read_text())
    return done()


if __name__ == "__main__":
    sys.exit(main())
