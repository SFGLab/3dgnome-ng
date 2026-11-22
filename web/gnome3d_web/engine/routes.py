"""The routes of the settings a request may change."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ..answers import REFUSALS
from ..context import Here
from .settings import parse_ini, schema
from .shapes import IniOverrides, SettingGroup

router = APIRouter(responses=REFUSALS)


@router.get("/api/settings", response_model=list[SettingGroup])
def api_settings(site: Here, cell: str, chain: str = "anchors") -> JSONResponse:
    return JSONResponse(schema(site.cell(cell).path("config"), chain))


@router.post("/api/settings/ini", response_model=IniOverrides)
async def api_settings_ini(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    overrides, refused = parse_ini((await file.read(1_000_000)).decode("utf-8", "replace"))
    return JSONResponse({"overrides": overrides, "refused": refused})
