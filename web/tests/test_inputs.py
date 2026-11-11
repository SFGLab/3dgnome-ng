"""Engine inputs for one request.

    python web/tests/test_inputs.py
"""

from __future__ import annotations

import configparser
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data
from personal3d.variants import SV

from gnome3d_web.catalog import Catalog
from gnome3d_web.inputs import build_inputs
from gnome3d_web.regions import Region


def rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        cat = Catalog.load(make_data(Path(d) / "data"))
        ct = cat.cell_types["GMX"]
        region = Region("chr1", 0, 3_000_000)

        inp = build_inputs(ct, region, "CTCF", [], Path(d) / "plain")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        data = cfg["data"]
        check("a CTCF run reads one loop file and names no factors", "," not in data["clusters"] and "factors" not in data)
        loops = rows(Path(data["data_dir"]) / data["clusters"])
        check("the loops are those inside the region", len(loops) == 14, str(len(loops)))
        check("the region is unchanged without variants", inp.region == "chr1:0-3000000")
        check("the rest of the config is the cell type's", cfg["main"]["max_pet_length"] == "1000000")

        inp = build_inputs(ct, region, "CTCF+RNAPII", [], Path(d) / "both")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        check(
            "a CTCF and RNAPII run reads two loop files, CTCF first",
            cfg["data"]["factors"] == "CTCF,RNAPOL2" and cfg["data"]["clusters"].split(",")[0].endswith("ctcf_clusters.bedpe"),
        )

        dele = SV("chr1", 900_001, 960_001, "DEL")
        inp = build_inputs(ct, region, "CTCF", [dele], Path(d) / "del")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        anchors = rows(Path(cfg["data"]["data_dir"]) / cfg["data"]["anchors"])
        starts = sorted(int(a[1]) for a in anchors)
        check("a deletion shifts the anchors after it by its length", 1_040_000 in starts and 1_100_000 not in starts, str(starts[:8]))
        check("the region shrinks by the deletion", inp.region == "chr1:0-2940000", inp.region)
        check("the variants applied are counted", inp.n_svs == 1)
    return done()


if __name__ == "__main__":
    sys.exit(main())
