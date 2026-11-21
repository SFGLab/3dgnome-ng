"""The routes of the shared job lists."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..answers import REFUSALS
from ..context import Here
from .shapes import SharedList
from .store import MAX_BODY, normalise

BAD_CODE = "A sync code has 16 letters and digits, like K7M2-9QXD-4THP-W3RZ."
NO_LIST = "No list has this code. Compare it with the computer that shows it."
router = APIRouter(responses=REFUSALS)


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise HTTPException(413)
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(400) from None
    return body if isinstance(body, dict) else {}


def _answer(code: str, record: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse({"code": code, "private": record["private"], "jobs": record["jobs"]}, status)


@router.post("/api/lists", response_model=SharedList, status_code=201)
async def list_create(site: Here, request: Request) -> JSONResponse:
    body = await _body(request)
    code, record = site.lists.create(body.get("jobs"), body.get("private") is True)
    return _answer(code, record, 201)


@router.get("/api/lists/{code}", response_model=SharedList)
def list_read(site: Here, code: str) -> JSONResponse:
    shown = normalise(code)
    if shown is None:
        return JSONResponse({"error": BAD_CODE}, 400)
    record = site.lists.read(shown)
    return _answer(shown, record) if record else JSONResponse({"error": NO_LIST}, 404)


@router.put("/api/lists/{code}", response_model=SharedList)
async def list_update(site: Here, code: str, request: Request) -> JSONResponse:
    shown = normalise(code)
    if shown is None:
        return JSONResponse({"error": BAD_CODE}, 400)
    record = site.lists.update(shown, (await _body(request)).get("jobs"))
    return _answer(shown, record) if record else JSONResponse({"error": NO_LIST}, 404)
