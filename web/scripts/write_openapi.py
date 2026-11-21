"""Writes the description of the interface of the site, from which the types of the frontend are generated.

    PYTHONPATH=web:web/vendor python web/scripts/write_openapi.py
    cd web/frontend && npm run api

The first command writes web/frontend/src/api/openapi.json from the routes of the server. The
second writes src/api/schema.d.ts from it. Run both after a change of a route or of a shape.
web/tests/test_contract.py fails while the file is not the one the server gives.
"""

from __future__ import annotations

import json
from pathlib import Path

from gnome3d_web.app import interface

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api" / "openapi.json"


def described() -> str:
    """The description as the file has it."""
    return json.dumps(interface().openapi(), indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(described())
    print(f"wrote {OUT}")
