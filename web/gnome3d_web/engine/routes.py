"""The routes of the settings a request may change."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ..site import Here
from .settings import parse_ini, schema

router = APIRouter()


@router.get("/api/settings")
def api_settings(site: Here, cell: str) -> JSONResponse:
    return JSONResponse(schema(site.cell(cell).path("config")))


@router.post("/api/settings/ini")
async def api_settings_ini(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    overrides, refused = parse_ini((await file.read(1_000_000)).decode("utf-8", "replace"))
    return JSONResponse({"overrides": overrides, "refused": refused})
